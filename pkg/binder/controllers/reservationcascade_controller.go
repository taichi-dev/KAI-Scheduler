// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package controllers

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/binding/resourcereservation"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
)

// reservationCascadePollInterval is how often we re-check that the fractions backed by a
// deleting reservation have terminated before releasing the reservation's finalizer.
const reservationCascadePollInterval = 5 * time.Second

// ReservationCascadeReconciler makes deleting a reservation ("capacity") pod cascade to the
// GPU-fraction workloads it backs. slurm-bridge frees the whole GPU (cancels the GRES job) the
// moment the reservation pod terminates, so a reservation must never outlive-in-reverse its
// fractions: if it is deleted, the fractions on that GPU have to go too, or they run on a GPU
// Slurm now considers free (silent double-booking).
//
// The reservation pod carries ReservationCascadeFinalizer. On deletion, this controller deletes
// the fraction pods sharing its GPU group and holds the finalizer until they are actually gone —
// so the reservation stays Terminating until the workloads have drained, then is released.
type ReservationCascadeReconciler struct {
	Client               client.Client
	Scheme               *runtime.Scheme
	ReservationNamespace string
	// Enabled only in external-reservation mode (slurm-bridge), where the binder stamps the
	// finalizer; otherwise this controller would strand pods with a finalizer nothing removes.
	Enabled bool
}

func (r *ReservationCascadeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if !r.Enabled {
		return ctrl.Result{}, nil
	}
	logger := log.FromContext(ctx)

	pod := &corev1.Pod{}
	if err := r.Client.Get(ctx, req.NamespacedName, pod); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if pod.Namespace != r.ReservationNamespace {
		return ctrl.Result{}, nil
	}
	group := pod.Labels[constants.GPUGroup]

	if pod.DeletionTimestamp.IsZero() {
		// Defensive: ensure the finalizer is present (the binder normally stamps it at creation).
		if group != "" && controllerutil.AddFinalizer(pod, resourcereservation.ReservationCascadeFinalizer) {
			if err := r.Client.Update(ctx, pod); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	// Reservation is being deleted.
	if !controllerutil.ContainsFinalizer(pod, resourcereservation.ReservationCascadeFinalizer) {
		return ctrl.Result{}, nil
	}
	if group == "" {
		// Nothing to cascade — release immediately.
		return ctrl.Result{}, r.releaseFinalizer(ctx, pod)
	}

	remaining, err := r.deleteFractionsForGroup(ctx, group)
	if err != nil {
		return ctrl.Result{}, err
	}
	if remaining > 0 {
		logger.Info("Reservation deleted; deleting backed fractions and waiting for them to terminate",
			"gpuGroup", group, "remaining", remaining)
		return ctrl.Result{RequeueAfter: reservationCascadePollInterval}, nil
	}

	logger.Info("All fractions backed by the reservation have terminated; releasing reservation",
		"gpuGroup", group)
	return ctrl.Result{}, r.releaseFinalizer(ctx, pod)
}

// deleteFractionsForGroup deletes all GPU-fraction pods (outside the reservation namespace) that
// share the given GPU group and returns how many still exist (including those still terminating).
func (r *ReservationCascadeReconciler) deleteFractionsForGroup(ctx context.Context, group string) (int, error) {
	pods := &corev1.PodList{}
	if err := r.Client.List(ctx, pods, client.MatchingLabels{constants.GPUGroup: group}); err != nil {
		return 0, err
	}
	remaining := 0
	for i := range pods.Items {
		p := &pods.Items[i]
		if p.Namespace == r.ReservationNamespace {
			continue // skip reservation pods themselves
		}
		remaining++
		if p.DeletionTimestamp.IsZero() {
			if err := r.Client.Delete(ctx, p); err != nil && !apierrors.IsNotFound(err) {
				return remaining, err
			}
		}
	}
	return remaining, nil
}

func (r *ReservationCascadeReconciler) releaseFinalizer(ctx context.Context, pod *corev1.Pod) error {
	if controllerutil.RemoveFinalizer(pod, resourcereservation.ReservationCascadeFinalizer) {
		return r.Client.Update(ctx, pod)
	}
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ReservationCascadeReconciler) SetupWithManager(mgr ctrl.Manager, params *ReconcilerParams) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("reservationcascade").
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

func (r *ReservationCascadeReconciler) eventHandlers() handler.Funcs {
	enqueue := func(_ context.Context, obj client.Object, q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
		if obj.GetNamespace() != r.ReservationNamespace {
			return
		}
		q.Add(reconcile.Request{NamespacedName: client.ObjectKeyFromObject(obj)})
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
