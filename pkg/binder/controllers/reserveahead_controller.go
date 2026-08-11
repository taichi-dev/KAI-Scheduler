// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package controllers

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	schedulingv1alpha2 "github.com/kai-scheduler/KAI-scheduler/pkg/apis/scheduling/v1alpha2"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/binding/resourcereservation"
	bindercommon "github.com/kai-scheduler/KAI-scheduler/pkg/binder/common"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/resources"
)

const (
	// defaultReserveAheadDelay is how long a fraction must stay Pending+unschedulable before we
	// reserve a whole GPU ahead for it. It gives the scheduler a chance to pack it onto a freed
	// GPU first (the common, cheaper outcome) before we enqueue a whole-GPU reservation to Slurm.
	defaultReserveAheadDelay = 15 * time.Second
	// reserveAheadPollInterval is how often we re-check a placed-but-not-yet-running reservation.
	reserveAheadPollInterval = 10 * time.Second
)

// ReserveAheadReconciler enqueues a whole-GPU reservation for a GPU-fraction pod that has been
// Pending and unschedulable for a while (it could not pack onto an existing reserved GPU). The
// reservation is placed by the external scheduler (slurm-bridge), which may preempt lower-QoS
// Slurm jobs to make room — KAI does not decide whether the cluster is "full".
//
// The fraction is left Pending so the KAI scheduler keeps trying to pack it onto a GPU that frees
// up; whichever happens first wins the race:
//   - scheduler packs it onto a freed reserved GPU -> its GPU-group label flips, the reserve-ahead
//     reservation is orphaned and torn down by the reservation sync;
//   - the reserve-ahead reservation gets placed -> we create the BindRequest to bind the fraction
//     onto it.
//
// The fraction is labeled with the reserve-ahead GPU group so the existing reservation sync keeps
// the parked reservation alive while the fraction references it and reaps it once it does not.
type ReserveAheadReconciler struct {
	Client              client.Client
	Scheme              *runtime.Scheme
	ResourceReservation resourcereservation.Interface
	// SchedulerName is the fraction pods' schedulerName (the KAI scheduler).
	SchedulerName string
	// Enabled is true only in external-reservation mode (reservationSchedulerName set); otherwise
	// there is nothing that can place a node-less reservation, so the reconciler no-ops.
	Enabled bool
	// Delay is how long a fraction must be Pending+unschedulable before reserving ahead.
	Delay time.Duration
}

func (r *ReserveAheadReconciler) delay() time.Duration {
	if r.Delay <= 0 {
		return defaultReserveAheadDelay
	}
	return r.Delay
}

// reserveAheadGroup derives a stable GPU-group id for a fraction from its identity, so repeated
// reconciles (and reconcile-on-delete, which only has the namespaced name) address the same
// reservation. Hashed to stay within label-value constraints.
func reserveAheadGroup(namespace, name string) string {
	sum := sha256.Sum256([]byte(namespace + "/" + name))
	return "reserve-ahead-" + hex.EncodeToString(sum[:8])
}

func (r *ReserveAheadReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if !r.Enabled {
		return ctrl.Result{}, nil
	}
	logger := log.FromContext(ctx)
	gpuGroup := reserveAheadGroup(req.Namespace, req.Name)

	pod := &corev1.Pod{}
	if err := r.Client.Get(ctx, req.NamespacedName, pod); err != nil {
		if apierrors.IsNotFound(err) {
			// Fraction gone: tear down any reserve-ahead reservation we created for it.
			if syncErr := r.ResourceReservation.SyncForGpuGroup(ctx, gpuGroup); syncErr != nil {
				logger.Error(syncErr, "failed to sync reserve-ahead reservation after fraction deletion",
					"gpuGroup", gpuGroup)
			}
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Only act on GPU-fraction pods managed by our scheduler.
	if pod.Spec.SchedulerName != r.SchedulerName || !resources.RequestsGPUFraction(pod) {
		return ctrl.Result{}, nil
	}

	// Once a BindRequest exists, the fraction's placement is decided — the reconciler MUST stop
	// touching it (re-ensuring or re-labeling here would race the binder and corrupt accounting).
	br := &schedulingv1alpha2.BindRequest{}
	brErr := r.Client.Get(ctx, req.NamespacedName, br)
	if brErr == nil {
		if bindRequestUsesGroup(br, gpuGroup) {
			// The reserve-ahead reservation won the race and is being bound onto — keep it.
			return ctrl.Result{}, nil
		}
		// The scheduler packed the fraction onto a different GPU that freed up (pack won the
		// race). Point the pod's group label at the real bound group so accounting is correct,
		// then tear down our now-orphaned reserve-ahead reservation.
		if podHasGpuGroup(pod, gpuGroup) && len(br.Spec.SelectedGPUGroups) > 0 {
			if err := r.labelFractionGpuGroup(ctx, pod, br.Spec.SelectedGPUGroups[0]); err != nil {
				return ctrl.Result{}, err
			}
		}
		if err := r.ResourceReservation.SyncForGpuGroup(ctx, gpuGroup); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to tear down reserve-ahead reservation for <%s/%s>: %w",
				pod.Namespace, pod.Name, err)
		}
		return ctrl.Result{}, nil
	} else if !apierrors.IsNotFound(brErr) {
		return ctrl.Result{}, brErr
	}

	// No BindRequest. If the pod is already bound/terminal/terminating (without a BindRequest, e.g.
	// completed), reap any reserve-ahead reservation it no longer references.
	if pod.Spec.NodeName != "" || pod.Status.Phase != corev1.PodPending || pod.DeletionTimestamp != nil {
		if !podHasGpuGroup(pod, gpuGroup) {
			if syncErr := r.ResourceReservation.SyncForGpuGroup(ctx, gpuGroup); syncErr != nil {
				logger.Error(syncErr, "failed to sync reserve-ahead reservation for settled fraction",
					"gpuGroup", gpuGroup)
			}
		}
		return ctrl.Result{}, nil
	}

	// Give the scheduler time to pack onto a freed GPU before reserving ahead.
	if wait := r.delay() - time.Since(pod.CreationTimestamp.Time); wait > 0 {
		return ctrl.Result{RequeueAfter: wait}, nil
	}

	// If the reservation is already placed and running, bind the fraction onto it.
	ready, nodeName, err := r.ResourceReservation.ReservationPlacement(ctx, gpuGroup)
	if err != nil {
		return ctrl.Result{}, err
	}
	if ready {
		return ctrl.Result{}, r.bindFractionToReservation(ctx, pod, nodeName, gpuGroup)
	}

	// Ensure exactly one parked reservation exists for this fraction, and tie the fraction to its
	// group so the reservation sync keeps it alive (and reaps it once the fraction moves on).
	if _, err := r.ResourceReservation.EnsureReservation(ctx, pod, gpuGroup); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to ensure reserve-ahead reservation for <%s/%s>: %w",
			pod.Namespace, pod.Name, err)
	}
	if err := r.labelFractionGpuGroup(ctx, pod, gpuGroup); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("Reserved a whole GPU ahead for pending fraction; waiting for external scheduler to place it",
		"pod", pod.Name, "namespace", pod.Namespace, "gpuGroup", gpuGroup)
	return ctrl.Result{RequeueAfter: reserveAheadPollInterval}, nil
}

func (r *ReserveAheadReconciler) bindFractionToReservation(
	ctx context.Context, pod *corev1.Pod, nodeName, gpuGroup string,
) error {
	logger := log.FromContext(ctx)
	portion, err := resources.GetGPUFraction(pod)
	if err != nil {
		return fmt.Errorf("failed to read gpu-fraction for <%s/%s>: %w", pod.Namespace, pod.Name, err)
	}
	bindRequest := &schedulingv1alpha2.BindRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       pod.Name,
				UID:        pod.UID,
			}},
		},
		Spec: schedulingv1alpha2.BindRequestSpec{
			PodName:              pod.Name,
			SelectedNode:         nodeName,
			SelectedGPUGroups:    []string{gpuGroup},
			ReceivedResourceType: bindercommon.ReceivedTypeFraction,
			ReceivedGPU: &schedulingv1alpha2.ReceivedGPU{
				Count:   1,
				Portion: fmt.Sprintf("%.2f", portion),
			},
			BackoffLimit: ptrInt32(defaultReserveAheadBackoffLimit),
		},
	}
	if err := r.Client.Create(ctx, bindRequest); err != nil {
		if apierrors.IsAlreadyExists(err) {
			// The scheduler (or a previous reconcile) already created a BindRequest for this pod —
			// e.g. it packed the fraction onto a GPU that freed up first. Let that one proceed; our
			// now-orphaned reservation is reaped by the sync once the fraction settles elsewhere.
			logger.Info("BindRequest already exists for fraction; not overriding",
				"pod", pod.Name, "namespace", pod.Namespace)
			return nil
		}
		return fmt.Errorf("failed to create BindRequest for reserve-ahead fraction <%s/%s>: %w",
			pod.Namespace, pod.Name, err)
	}
	logger.Info("Bound pending fraction onto its reserve-ahead reservation",
		"pod", pod.Name, "namespace", pod.Namespace, "node", nodeName, "gpuGroup", gpuGroup)
	return nil
}

func (r *ReserveAheadReconciler) labelFractionGpuGroup(ctx context.Context, pod *corev1.Pod, gpuGroup string) error {
	if podHasGpuGroup(pod, gpuGroup) {
		return nil
	}
	updated := pod.DeepCopy()
	if updated.Labels == nil {
		updated.Labels = map[string]string{}
	}
	updated.Labels[constants.GPUGroup] = gpuGroup
	if err := r.Client.Patch(ctx, updated, client.MergeFrom(pod)); err != nil {
		return fmt.Errorf("failed to label fraction <%s/%s> with reserve-ahead group: %w",
			pod.Namespace, pod.Name, err)
	}
	return nil
}

func podHasGpuGroup(pod *corev1.Pod, gpuGroup string) bool {
	for _, g := range resources.GetGpuGroups(pod) {
		if g == gpuGroup {
			return true
		}
	}
	return false
}

func bindRequestUsesGroup(br *schedulingv1alpha2.BindRequest, gpuGroup string) bool {
	for _, g := range br.Spec.SelectedGPUGroups {
		if g == gpuGroup {
			return true
		}
	}
	return false
}

// SetupWithManager sets up the controller with the Manager.
func (r *ReserveAheadReconciler) SetupWithManager(mgr ctrl.Manager, params *ReconcilerParams) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("reserveahead").
		For(&corev1.Pod{}).
		Watches(&corev1.Pod{}, r.eventHandlers()).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: params.MaxConcurrentReconciles,
			RateLimiter: workqueue.NewTypedItemExponentialFailureRateLimiter[ctrl.Request](
				time.Duration(params.RateLimiterBaseDelaySeconds)*time.Second,
				time.Duration(params.RateLimiterMaxDelaySeconds)*time.Second,
			),
			SkipNameValidation: &[]bool{true}[0],
		}).
		Complete(r)
}

func (r *ReserveAheadReconciler) eventHandlers() handler.Funcs {
	enqueue := func(ctx context.Context, obj client.Object, q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
		pod, ok := obj.(*corev1.Pod)
		if !ok || pod.Spec.SchedulerName != r.SchedulerName || !resources.RequestsGPUFraction(pod) {
			return
		}
		q.Add(reconcile.Request{NamespacedName: client.ObjectKeyFromObject(pod)})
	}
	return handler.Funcs{
		CreateFunc: func(ctx context.Context, e event.CreateEvent, q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
			enqueue(ctx, e.Object, q)
		},
		UpdateFunc: func(ctx context.Context, e event.UpdateEvent, q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
			enqueue(ctx, e.ObjectNew, q)
		},
		DeleteFunc: func(ctx context.Context, e event.DeleteEvent, q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
			enqueue(ctx, e.Object, q)
		},
	}
}

const defaultReserveAheadBackoffLimit int32 = 100

func ptrInt32(v int32) *int32 { return &v }
