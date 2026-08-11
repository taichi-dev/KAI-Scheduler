// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package resourcereservation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	"golang.org/x/exp/slices"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	karpenterv1 "sigs.k8s.io/karpenter/pkg/apis/v1"

	schedulingv1alpha2 "github.com/kai-scheduler/KAI-scheduler/pkg/apis/scheduling/v1alpha2"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/binding/resourcereservation/group_mutex"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/resources"
)

type Interface interface {
	Sync(ctx context.Context) error
	SyncForNode(ctx context.Context, nodeName string) error
	SyncForGpuGroup(ctx context.Context, gpuGroup string) error
	ReserveGpuDevice(ctx context.Context, pod *v1.Pod, nodeName string, gpuGroup string) (string, error)
	RemovePodGpuGroupsConnection(ctx context.Context, pod *v1.Pod) error
	// EnsureReservation creates (idempotently, by gpuGroup) a whole-GPU reservation pod for the
	// given gpuGroup, WITHOUT binding it to a node — the external scheduler (slurm-bridge) places
	// it. Used by the reserve-ahead controller to enqueue a reservation for a Pending fraction
	// that cannot yet pack, so Slurm can make room (preempt) while the fraction waits. External
	// reservation mode only.
	EnsureReservation(ctx context.Context, sourcePod *v1.Pod, gpuGroup string) (*v1.Pod, error)
	// ReservationPlacement reports whether the reservation pod for gpuGroup has been placed and is
	// ready to back a fraction (Running, scheduled to a node, and has self-reported its GPU index),
	// along with the node it landed on.
	ReservationPlacement(ctx context.Context, gpuGroup string) (ready bool, nodeName string, err error)
}

const (
	resourceReservation     = "resource-reservation"
	gpuReservationPodPrefix = "gpu-reservation"
	gpuIndexAnnotationName  = "run.ai/reserve_for_gpu_index"
	numberOfGPUsToReserve   = 1
	unknownGpuIndicator     = "-1"
	// pendingGpuIndicator is returned by waitForGPUReservationPodAllocation when the
	// reservation pod did not allocate within the timeout but is still Pending (e.g. an
	// external scheduler such as slurm-bridge has enqueued its Slurm job and is waiting
	// for a whole GPU to free). Distinct from unknownGpuIndicator, which means the
	// reservation genuinely failed and should be reaped.
	pendingGpuIndicator = "-2"
	// ReservationCascadeFinalizer is placed on externally-scheduled reservation pods so that
	// deleting a reservation cascades to the GPU-fraction workloads it backs: a controller
	// deletes those fractions and only removes this finalizer once they are gone. Without it,
	// deleting the reservation frees the whole GPU in Slurm (slurm-bridge cancels the GRES job
	// on pod termination) while the fractions keep running on it — a silent double-booking.
	ReservationCascadeFinalizer = "kai.scheduler/reservation-cascade"
)

// ErrReservationPending signals that a GPU reservation pod was created and is still
// Pending (parked, e.g. waiting in an external scheduler's queue) rather than failed.
// The BindRequest reconciler treats it as "retry later" — it must NOT roll back the
// fraction pod's GPU-group binding or mark the BindRequest Failed, so the parked
// reservation keeps its place in the queue and is reused on the next reconcile. The
// parked reservation is protected from GC while its BindRequest stays non-terminal
// (see hasActiveBindRequestsForGpuGroup) and is torn down by the reservation sync once
// the fraction binds elsewhere (packs onto a freed GPU) and its BindRequest goes terminal.
var ErrReservationPending = errors.New("gpu reservation pod is still pending allocation")

type service struct {
	fakeGPuNodes                        bool
	kubeClient                          client.WithWatch
	reservationPodImage                 string
	allocationTimeout                   time.Duration
	gpuGroupMutex                       *group_mutex.GroupMutex
	namespace                           string
	serviceAccountName                  string
	appLabelValue                       string
	scalingPodNamespace                 string
	runtimeClassName                    string
	podResources                        *v1.ResourceRequirements
	reservationPodSecurityContext       *v1.PodSecurityContext
	reservationContainerSecurityContext *v1.SecurityContext
	reservationSchedulerName            string
	reservationGpuResourceName          string
	reservationPodAnnotations           map[string]string
	mpsEnabled                          bool
	mpsBaseDir                          string
}

func NewService(
	fakeGPuNodes bool,
	kubeClient client.WithWatch,
	reservationPodImage string,
	allocationTimeout time.Duration,
	namespace string,
	serviceAccountName string,
	appLabelValue string,
	scalingPodNamespace string,
	runtimeClassName string,
	podResources *v1.ResourceRequirements,
	reservationPodSecurityContext *v1.PodSecurityContext,
	reservationContainerSecurityContext *v1.SecurityContext,
) *service {
	return &service{
		fakeGPuNodes:                        fakeGPuNodes,
		kubeClient:                          kubeClient,
		reservationPodImage:                 reservationPodImage,
		allocationTimeout:                   allocationTimeout,
		gpuGroupMutex:                       group_mutex.NewGroupMutex(),
		namespace:                           namespace,
		serviceAccountName:                  serviceAccountName,
		appLabelValue:                       appLabelValue,
		scalingPodNamespace:                 scalingPodNamespace,
		runtimeClassName:                    runtimeClassName,
		podResources:                        podResources,
		reservationPodSecurityContext:       reservationPodSecurityContext,
		reservationContainerSecurityContext: reservationContainerSecurityContext,
	}
}

// WithExternalReservation switches the service into "external reservation" mode: reservation
// pods are created unbound with the given spec.schedulerName (so an external scheduler places
// them) instead of being pre-bound via spec.nodeName. They request gpuResourceName for the whole
// GPU (defaulting to nvidia.com/gpu when empty) and carry the given extra annotations. Passing an
// empty schedulerName leaves the service in the default pinned mode.
func (rsc *service) WithExternalReservation(schedulerName, gpuResourceName string, podAnnotations map[string]string) *service {
	rsc.reservationSchedulerName = schedulerName
	rsc.reservationGpuResourceName = gpuResourceName
	rsc.reservationPodAnnotations = podAnnotations
	return rsc
}

// WithMPS makes each reservation pod also host a per-GPU nvidia-cuda-mps-control
// daemon (the reservation pod already holds the whole physical GPU and its
// lifecycle equals "this GPU is in fractional use"). The daemon serves the
// fraction (client) pods that share the GPU; a shared hostPath under baseDir
// carries the MPS pipe/log directory keyed by gpu-group. Passing an empty
// baseDir uses the default. Reservation pods themselves do NOT become MPS
// clients — they keep using the device directly.
func (rsc *service) WithMPS(baseDir string) *service {
	rsc.mpsEnabled = true
	if baseDir == "" {
		baseDir = mps.DefaultBaseDir
	}
	rsc.mpsBaseDir = baseDir
	return rsc
}

func (rsc *service) externalReservationMode() bool {
	return rsc.reservationSchedulerName != ""
}

func (rsc *service) gpuResourceName() v1.ResourceName {
	if rsc.reservationGpuResourceName != "" {
		return v1.ResourceName(rsc.reservationGpuResourceName)
	}
	return constants.NvidiaGpuResource
}

// reservationPodAnnotationsMap returns the annotations to stamp on a reservation pod: always the
// karpenter do-not-disrupt annotation, plus any configured external-mode annotations.
func (rsc *service) reservationPodAnnotationsMap() map[string]string {
	annotations := map[string]string{
		karpenterv1.DoNotDisruptAnnotationKey: "true",
	}
	maps.Copy(annotations, rsc.reservationPodAnnotations)
	return annotations
}

func (rsc *service) Sync(ctx context.Context) error {
	podsList := &v1.PodList{}
	err := rsc.kubeClient.List(ctx, podsList,
		client.HasLabels{constants.GPUGroup},
	)
	if err != nil {
		return err
	}

	return rsc.SyncForPodsList(ctx, podsList)
}

func (rsc *service) SyncForNode(ctx context.Context, nodeName string) error {
	podsList := &v1.PodList{}
	err := rsc.kubeClient.List(ctx, podsList,
		client.HasLabels{constants.GPUGroup},
		client.MatchingFields{"spec.nodeName": nodeName},
	)
	if err != nil {
		return err
	}

	return rsc.SyncForPodsList(ctx, podsList)
}

func (rsc *service) SyncForPodsList(ctx context.Context, podsList *v1.PodList) error {
	gpuGroupsMap := map[string]bool{}
	for _, pod := range podsList.Items {
		gpuGroups := resources.GetGpuGroups(&pod)
		for _, gpuGroup := range gpuGroups {
			gpuGroupsMap[gpuGroup] = true
		}
	}

	for gpuGroup := range gpuGroupsMap {
		err := rsc.SyncForGpuGroup(ctx, gpuGroup)
		if err != nil {
			return err
		}
	}

	return nil
}

func (rsc *service) SyncForGpuGroup(ctx context.Context, gpuGroup string) error {
	rsc.gpuGroupMutex.LockMutexForGroup(gpuGroup)
	defer rsc.gpuGroupMutex.ReleaseMutex(gpuGroup)

	return rsc.syncForGpuGroupWithLock(ctx, gpuGroup)
}

func (rsc *service) syncForGpuGroupWithLock(ctx context.Context, gpuGroup string) error {
	podsList := &v1.PodList{}
	err := rsc.kubeClient.List(ctx, podsList,
		client.MatchingLabels{constants.GPUGroup: gpuGroup},
	)
	if err != nil {
		return err
	}

	multiFractionsPodsList := &v1.PodList{}
	multiGroupKey, multiGroupValue := resources.GetMultiFractionGpuGroupLabel(gpuGroup)
	err = rsc.kubeClient.List(ctx, multiFractionsPodsList,
		client.MatchingLabels{multiGroupKey: multiGroupValue},
	)
	if err != nil {
		return err
	}

	pods := []*v1.Pod{}
	for index := range len(podsList.Items) {
		pods = append(pods, &podsList.Items[index])
	}
	for index := range len(multiFractionsPodsList.Items) {
		pods = append(pods, &multiFractionsPodsList.Items[index])
	}

	return rsc.syncForPods(ctx, pods, gpuGroup)
}

func (rsc *service) syncForPods(ctx context.Context, pods []*v1.Pod, gpuGroupToSync string) error {
	logger := log.FromContext(ctx)
	reservationPods := map[string]*v1.Pod{}
	fractionPods := map[string][]*v1.Pod{}

	for _, pod := range pods {
		if pod.Namespace == rsc.namespace {
			reservationPods[gpuGroupToSync] = pod
			continue
		}

		if slices.Contains(
			[]v1.PodPhase{v1.PodRunning, v1.PodPending},
			pod.Status.Phase,
		) {
			fractionPods[gpuGroupToSync] = append(fractionPods[gpuGroupToSync], pod)
		}
	}

	for gpuGroup, pods := range fractionPods {
		if _, found := reservationPods[gpuGroup]; !found {
			err := rsc.deleteNonReservedPods(ctx, gpuGroup, pods)
			if err != nil {
				return err
			}
		}
	}

	for gpuGroup, reservationPod := range reservationPods {
		if _, found := fractionPods[gpuGroup]; !found {
			hasActive, err := rsc.hasActiveBindRequestsForGpuGroup(ctx, gpuGroup)
			if err != nil {
				logger.Error(err, "Failed to check active BindRequests for gpu group",
					"gpuGroup", gpuGroup)
				return err
			}
			if hasActive {
				logger.Info("Skipping reservation pod deletion, active BindRequests exist",
					"gpuGroup", gpuGroup)
				continue
			}
			logger.Info("Did not find fraction pod for gpu group, deleting reservation pod",
				"gpuGroup", gpuGroup)
			err = rsc.deleteReservationPod(ctx, reservationPod)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// hasActiveBindRequestsForGpuGroup checks if any non-terminal BindRequests reference
// the given GPU group. This prevents premature reservation pod deletion when the
// informer cache has not yet propagated GPU group labels on recently-bound fraction pods.
func (rsc *service) hasActiveBindRequestsForGpuGroup(ctx context.Context, gpuGroup string) (bool, error) {
	bindRequestList := &schedulingv1alpha2.BindRequestList{}
	if err := rsc.kubeClient.List(ctx, bindRequestList); err != nil {
		return false, fmt.Errorf("failed to list BindRequests: %w", err)
	}

	for _, br := range bindRequestList.Items {
		if br.Status.Phase == schedulingv1alpha2.BindRequestPhaseSucceeded ||
			br.Status.Phase == schedulingv1alpha2.BindRequestPhaseFailed {
			continue
		}
		if slices.Contains(br.Spec.SelectedGPUGroups, gpuGroup) {
			return true, nil
		}
	}
	return false, nil
}
func (rsc *service) ReserveGpuDevice(ctx context.Context, pod *v1.Pod, nodeName string, gpuGroup string) (string, error) {
	logger := log.FromContext(ctx)

	rsc.gpuGroupMutex.LockMutexForGroup(gpuGroup)
	defer rsc.gpuGroupMutex.ReleaseMutex(gpuGroup)

	gpuIndex, err := rsc.acquireGPUIndexByGroup(ctx, pod, nodeName, gpuGroup)
	if err != nil {
		return unknownGpuIndicator, err
	}

	err = rsc.updatePodGPUGroup(ctx, pod, nodeName, gpuGroup)
	if err != nil {
		logger.Error(err, "Failed to update GPU group annotation on pod",
			"namespace", pod.Namespace, "name", pod.Name, "node", nodeName)
		syncErr := rsc.syncForGpuGroupWithLock(ctx, gpuGroup)
		if syncErr != nil {
			logger.Error(err, "Failed to sync reservation for gpu group", "gpuGroup", gpuGroup)
		}
		return unknownGpuIndicator, err
	}

	return gpuIndex, nil
}

func (rsc *service) updatePodGPUGroup(
	ctx context.Context, pod *v1.Pod, nodeName string, gpuGroup string) error {
	logger := log.FromContext(ctx)
	logger.Info("Updating GPU group annotation for pod",
		"namespace", pod.Namespace, "name", pod.Name, "node", nodeName,
		"gpu-group", gpuGroup)
	originalPod := pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	isMultiFraction, err := resources.IsMultiFraction(pod)
	if err != nil {
		return fmt.Errorf(
			"failed to determine if pod <%s/%s> is a multi fractional pod while setting gpu group label. %w",
			pod.Namespace, pod.Name, err)
	}
	if isMultiFraction {
		labelKey, labelValue := resources.GetMultiFractionGpuGroupLabel(gpuGroup)
		pod.Labels[labelKey] = labelValue
	} else {
		pod.Labels[constants.GPUGroup] = gpuGroup
	}

	err = rsc.kubeClient.Patch(ctx, pod, client.MergeFrom(originalPod))
	if err != nil {
		return fmt.Errorf("failed to patch pod <%s/%s> with GPU group label: %v", pod.Namespace, pod.Name, err)
	}

	return nil
}

func (rsc *service) RemovePodGpuGroupsConnection(ctx context.Context, pod *v1.Pod) error {
	var patch []map[string]string
	for labelKey := range pod.Labels {
		if labelKey == constants.GPUGroup || strings.HasPrefix(labelKey, constants.MultiGpuGroupLabelPrefix) {
			patch = append(patch, map[string]string{
				"op":   "remove",
				"path": fmt.Sprintf("/metadata/labels/%s", escapeJSONPointer(labelKey)),
			})
		}
	}

	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return fmt.Errorf("failed to generate a patch for pod gpu-group removal. %w", err)
	}

	if err := rsc.kubeClient.Patch(ctx, pod, client.RawPatch(types.JSONPatchType, patchBytes)); err != nil {
		return err
	}
	return nil
}

func (rsc *service) EnsureReservation(ctx context.Context, sourcePod *v1.Pod, gpuGroup string) (*v1.Pod, error) {
	if !rsc.externalReservationMode() {
		return nil, fmt.Errorf("reserve-ahead requires external reservation mode (reservationSchedulerName)")
	}
	// nodeName "" — the external scheduler (slurm-bridge) picks the node and places the pod.
	return rsc.createGPUReservationPod(ctx, sourcePod, "", gpuGroup)
}

func (rsc *service) ReservationPlacement(ctx context.Context, gpuGroup string) (bool, string, error) {
	gpuIndex, err := rsc.findGPUIndexByGroup(gpuGroup)
	if err != nil {
		// The reservation pod exists but has not self-reported its GPU index yet — not ready.
		return false, "", nil
	}
	if gpuIndex == "" {
		// No reservation pod for this group.
		return false, "", nil
	}
	pods := &v1.PodList{}
	if listErr := rsc.kubeClient.List(ctx, pods,
		client.InNamespace(rsc.namespace),
		client.MatchingLabels{constants.GPUGroup: gpuGroup}); listErr != nil {
		return false, "", listErr
	}
	if len(pods.Items) == 0 {
		return false, "", nil
	}
	pod := pods.Items[0]
	ready := pod.Status.Phase == v1.PodRunning && pod.Spec.NodeName != ""
	return ready, pod.Spec.NodeName, nil
}

// escapeJSONPointer escapes a string for use in a JSON Pointer path (RFC 6901).
// ~ must be escaped as ~0, and / must be escaped as ~1.
func escapeJSONPointer(s string) string {
	s = strings.ReplaceAll(s, "~", "~0")
	s = strings.ReplaceAll(s, "/", "~1")
	return s
}

func (rsc *service) acquireGPUIndexByGroup(
	ctx context.Context, sourcePod *v1.Pod, nodeName, gpuGroup string,
) (string, error) {
	gpuIndex, err := rsc.findGPUIndexByGroup(gpuGroup)
	if err != nil {
		return "", err
	}
	if gpuIndex != "" {
		return gpuIndex, err
	}
	return rsc.createGPUReservationPodAndGetIndex(ctx, sourcePod, nodeName, gpuGroup)
}

func (rsc *service) findGPUIndexByGroup(gpuGroup string) (
	gpuIndex string, err error,
) {
	pods := &v1.PodList{}
	err = rsc.kubeClient.List(context.Background(), pods,
		client.InNamespace(rsc.namespace),
		client.MatchingLabels{constants.GPUGroup: gpuGroup})
	if err != nil {
		return "", err
	}

	if len(pods.Items) == 0 {
		return "", nil
	}

	gpuIndex, found := pods.Items[0].Annotations[gpuIndexAnnotationName]
	if !found {
		return "", fmt.Errorf("failed to find annotation on reservation pod %s", pods.Items[0].Name)
	}

	return gpuIndex, nil
}

func (rsc *service) createGPUReservationPodAndGetIndex(
	ctx context.Context, sourcePod *v1.Pod, nodeName, gpuGroup string,
) (
	gpuIndex string, err error) {
	logger := log.FromContext(ctx)
	pod, err := rsc.createGPUReservationPod(ctx, sourcePod, nodeName, gpuGroup)
	if err != nil {
		return unknownGpuIndicator, err
	}

	gpuIndex = rsc.waitForGPUReservationPodAllocation(ctx, nodeName, pod.Name)
	if gpuIndex == pendingGpuIndicator {
		// The reservation is still Pending (parked): do NOT delete it. Keep it in place
		// so it holds its spot in the external scheduler's queue and is reused on the next
		// reconcile. The caller (BindRequest reconciler) requeues without rollback.
		logger.Info("GPU reservation pod is still pending, parking it for retry",
			"name", pod.Name, "gpuGroup", gpuGroup)
		return unknownGpuIndicator, fmt.Errorf(
			"gpu reservation pod %v/%v: %w", rsc.namespace, pod.Name, ErrReservationPending)
	}
	if gpuIndex == unknownGpuIndicator {
		deleteErr := rsc.deleteReservationPod(ctx, pod)
		if deleteErr != nil {
			logger.Error(deleteErr, "failed to delete reservation pod", "name", pod.Name)
		}
		return unknownGpuIndicator, fmt.Errorf(
			"failed waiting for GPU reservation pod to allocate: %v/%v", rsc.namespace, pod.Name)
	}

	if rsc.externalReservationMode() && nodeName != "" {
		// The reservation pod is running now (it self-reported its GPU index), so the external
		// scheduler has set its nodeName. In v1 (static single-GPU-node pool) the fraction pod is
		// bound to the scheduler-selected node, so the reservation must have landed there too;
		// otherwise the fraction would share a GPU that lives on a different node. Fail loudly
		// instead of binding to the wrong node. True multi-node "follow the reservation" is a
		// Phase-2 change to the bind path.
		placed := &v1.Pod{}
		if getErr := rsc.kubeClient.Get(ctx,
			types.NamespacedName{Namespace: rsc.namespace, Name: pod.Name}, placed); getErr != nil {
			return unknownGpuIndicator, fmt.Errorf(
				"failed to read externally-scheduled reservation pod %v/%v: %w", rsc.namespace, pod.Name, getErr)
		}
		if placed.Spec.NodeName != "" && placed.Spec.NodeName != nodeName {
			if deleteErr := rsc.deleteReservationPod(ctx, placed); deleteErr != nil {
				logger.Error(deleteErr, "failed to delete diverged reservation pod", "name", pod.Name)
			}
			return unknownGpuIndicator, fmt.Errorf(
				"external reservation pod %v/%v was placed on node %q but the scheduler selected %q; "+
					"multi-node external placement is not supported in v1 (static single-GPU-node pool)",
				rsc.namespace, pod.Name, placed.Spec.NodeName, nodeName)
		}
	}

	return gpuIndex, err
}

func (rsc *service) deleteNonReservedPods(ctx context.Context, gpuGroup string, pods []*v1.Pod) error {
	logger := log.FromContext(ctx)
	for _, pod := range pods {
		if pod.Status.Phase != v1.PodRunning {
			continue
		}
		logger.Info("Warning: Found pod without reservation, deleting", "name", pod.Name,
			"namespace", pod.Namespace, "gpuGroup", gpuGroup)
		err := rsc.kubeClient.Delete(ctx, pod)
		if err != nil {
			logger.Error(err, "Failed to delete non-reserved pods", "name", pod.Name,
				"namespace", pod.Namespace)
			return err
		}
	}
	return nil
}

func (rsc *service) deleteReservationPod(ctx context.Context, pod *v1.Pod) error {
	logger := log.FromContext(ctx)
	// This is the INTERNAL reservation deleter (reap-on-allocation-timeout, node-divergence
	// guard, and orphan cleanup in syncForPods). Strip our cascade finalizer first so this
	// system-initiated deletion does NOT cascade to the backed fraction workloads — otherwise a
	// transient/racy internal reservation delete would kill healthy fractions. Only an EXTERNAL
	// delete (which leaves the finalizer intact) should trigger the cascade.
	if slices.Contains(pod.Finalizers, ReservationCascadeFinalizer) {
		original := pod.DeepCopy()
		kept := make([]string, 0, len(pod.Finalizers))
		for _, f := range pod.Finalizers {
			if f != ReservationCascadeFinalizer {
				kept = append(kept, f)
			}
		}
		pod.Finalizers = kept
		if patchErr := rsc.kubeClient.Patch(ctx, pod, client.MergeFrom(original)); patchErr != nil &&
			!apierrors.IsNotFound(patchErr) {
			return fmt.Errorf("failed to strip cascade finalizer from reservation pod %s: %w", pod.Name, patchErr)
		}
	}
	logger.Info("Deleting reservation pod", "name", pod.Name)
	err := rsc.kubeClient.Delete(ctx,
		pod,
		client.GracePeriodSeconds(0),
	)
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete reservation pod: %w", err)
		}
		logger.Info("Reservation pod not found, skipping deletion", "name", pod.Name)
	}
	return nil
}

func (rsc *service) createGPUReservationPod(
	ctx context.Context, sourcePod *v1.Pod, nodeName, gpuGroup string,
) (*v1.Pod, error) {
	logger := log.FromContext(ctx)
	if rsc.isScalingUp(ctx) {
		return nil, fmt.Errorf("cluster is scaling up, could not create reservation pod")
	}

	// In external-reservation mode the pod is not pre-bound, so its name must not depend on a
	// node (there isn't one yet) — key it on the gpu-group only so concurrent/retried creates
	// still collide on a single object.
	nameNode := nodeName
	if rsc.externalReservationMode() {
		nameNode = ""
	}
	podName := reservationPodName(nameNode, gpuGroup)

	// Build resource requirements starting with the whole-GPU request. The resource key is
	// configurable (e.g. deviceclass.resource.kubernetes.io/gpu.nvidia.com for a DRA-backed
	// extended resource under slurm-bridge) and defaults to nvidia.com/gpu.
	gpuResource := rsc.gpuResourceName()
	resources := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			gpuResource: *resource.NewQuantity(numberOfGPUsToReserve, resource.DecimalSI),
		},
		Requests: v1.ResourceList{
			gpuResource: *resource.NewQuantity(numberOfGPUsToReserve, resource.DecimalSI),
		},
	}

	if rsc.podResources != nil {
		if rsc.podResources.Limits != nil {
			delete(rsc.podResources.Limits, gpuResource)
			maps.Copy(resources.Limits, rsc.podResources.Limits)
		}
		if rsc.podResources.Requests != nil {
			delete(rsc.podResources.Requests, gpuResource)
			maps.Copy(resources.Requests, rsc.podResources.Requests)
		}
	}

	pod, err := rsc.createResourceReservationPod(sourcePod, nodeName, gpuGroup, podName, resources)
	if err != nil {
		// The reservation pod name is deterministic per (node, gpu-group). AlreadyExists
		// means another actor (a concurrent bind, a retry, or another binder replica)
		// already created the reservation pod for this gpu-group, so reuse it rather than
		// creating a duplicate on a different physical GPU.
		if apierrors.IsAlreadyExists(err) {
			logger.Info("GPU reservation pod already exists for gpu group, reusing",
				"nodeName", nodeName, "namespace", rsc.namespace, "name", podName, "gpuGroup", gpuGroup)
			return pod, nil
		}
		logger.Error(err, "Failed to create GPU reservation pod on node",
			"nodeName", nodeName, "namespace", rsc.namespace, "name", podName)
		return nil, err
	}

	logger.Info(
		"Successfully created GPU resource reservation pod",
		"nodeName", nodeName, "namespace", rsc.namespace, "name", podName)
	return pod, nil
}

func (rsc *service) waitForGPUReservationPodAllocation(
	ctx context.Context, nodeName, gpuReservationPodName string,
) string {
	logger := log.FromContext(ctx)
	pods := &v1.PodList{}
	watcher, err := rsc.kubeClient.Watch(
		ctx, pods,
		client.InNamespace(rsc.namespace),
		client.MatchingFields{"metadata.name": gpuReservationPodName},
	)
	if err != nil {
		logger.Error(err, "Failed to watch for GPU reservation pod")
		return unknownGpuIndicator
	}
	defer watcher.Stop()

	timeout := time.After(rsc.allocationTimeout)
	for {
		select {
		case <-ctx.Done():
			logger.Error(ctx.Err(),
				"Context done while waiting for GPU reservation pod to be allocated",
				"nodeName", nodeName, "name", gpuReservationPodName)
			return unknownGpuIndicator
		case <-timeout:
			logger.Error(fmt.Errorf("timeout"),
				"Reached timeout while waiting for GPU reservation pod to be allocated",
				"nodeName", nodeName, "name", gpuReservationPodName)
			// A reservation that is still Pending after the timeout has not failed — an
			// external scheduler (slurm-bridge) may have enqueued it and be waiting for a
			// whole GPU. Signal "parked" so the caller keeps it instead of reaping it.
			return rsc.timeoutAllocationIndicator(ctx, gpuReservationPodName)
		case event, ok := <-watcher.ResultChan():
			if !ok {
				logger.Error(nil,
					"GPU reservation pod watch channel closed while waiting for allocation",
					"nodeName", nodeName, "name", gpuReservationPodName)
				return unknownGpuIndicator
			}

			if event.Type == watch.Error {
				logger.Error(fmt.Errorf("watch error"),
					"Error event received while waiting for GPU reservation pod allocation",
					"nodeName", nodeName, "name", gpuReservationPodName, "event", fmt.Sprintf("%v", event.Object))
				return unknownGpuIndicator
			}

			pod, ok := event.Object.(*v1.Pod)
			if pod.Annotations != nil && pod.Annotations[gpuIndexAnnotationName] != "" {
				return pod.Annotations[gpuIndexAnnotationName]
			}
		}
	}
}

// timeoutAllocationIndicator classifies a reservation pod that did not report its GPU
// index within the allocation timeout: pendingGpuIndicator if it still exists and is
// Pending (parked, e.g. waiting in slurm-bridge's queue) so the caller keeps it, or
// unknownGpuIndicator otherwise (gone / failed) so the caller reaps it.
func (rsc *service) timeoutAllocationIndicator(ctx context.Context, gpuReservationPodName string) string {
	pod := &v1.Pod{}
	if err := rsc.kubeClient.Get(ctx,
		types.NamespacedName{Namespace: rsc.namespace, Name: gpuReservationPodName}, pod); err != nil {
		// Cannot confirm it is parked (e.g. not found): treat as failed so it is reaped.
		return unknownGpuIndicator
	}
	if pod.Status.Phase == v1.PodPending {
		return pendingGpuIndicator
	}
	return unknownGpuIndicator
}

func (rsc *service) createResourceReservationPod(
	sourcePod *v1.Pod, nodeName, gpuGroup, podName string, resources v1.ResourceRequirements,
) (*v1.Pod, error) {
	var tolerations []v1.Toleration
	if sourcePod != nil {
		sourcePodSpec := sourcePod.Spec.DeepCopy()
		tolerations = sourcePodSpec.Tolerations
	}

	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: rsc.namespace,
			Labels: map[string]string{
				constants.AppLabelName: rsc.appLabelValue,
				constants.GPUGroup:     gpuGroup,
			},
			Annotations: rsc.reservationPodAnnotationsMap(),
		},
		Spec: v1.PodSpec{
			Tolerations: tolerations,
			RuntimeClassName: func() *string {
				if len(rsc.runtimeClassName) == 0 {
					return nil
				}
				return &rsc.runtimeClassName
			}(),
			SecurityContext:    rsc.reservationPodSecurityContext,
			ServiceAccountName: rsc.serviceAccountName,
			Containers: []v1.Container{
				{
					Name:            resourceReservation,
					Image:           rsc.reservationPodImage,
					ImagePullPolicy: v1.PullIfNotPresent,
					Resources:       resources,
					SecurityContext: rsc.reservationContainerSecurityContext,
					Env: []v1.EnvVar{
						{
							Name: "POD_NAME",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "metadata.name",
								},
							},
						},
						{
							Name: "POD_NAMESPACE",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "metadata.namespace",
								},
							},
						},
					},
				},
			},
		},
	}

	if rsc.externalReservationMode() {
		// Leave the pod unbound (no nodeName, no faked PodScheduled) so the external scheduler
		// places it; the whole-GPU request keeps it from double-booking.
		podSpec.Spec.SchedulerName = rsc.reservationSchedulerName
		// Cascade-delete the backed fractions if this reservation is deleted (see finalizer doc).
		podSpec.Finalizers = append(podSpec.Finalizers, ReservationCascadeFinalizer)
	} else {
		// Default behavior: pre-bind the reservation pod to the selected node and mark it
		// PodScheduled so the kubelet/device-plugin admits it without any scheduling pass.
		podSpec.Spec.NodeName = nodeName
		podSpec.Status = v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionTrue,
				},
			},
		}
	}

	if rsc.mpsEnabled {
		rsc.applyMPSHosting(podSpec, gpuGroup)
	}

	if rsc.fakeGPuNodes {
		podSpec.Spec.Containers[0].Image = "ubuntu"
		podSpec.Spec.Containers[0].Args = []string{"sleep", "infinity"}
	}

	return podSpec, rsc.kubeClient.Create(context.Background(), podSpec)
}

// applyMPSHosting wires the reservation pod to host the per-GPU MPS control
// daemon: a hostPath mount of the node-local MPS base directory, plus the env
// the reservation entrypoint reads to start nvidia-cuda-mps-control for this
// gpu-group. The reservation pod's own workload uses the GPU directly (it only
// runs NVML discovery); it never sets CUDA_MPS_PIPE_DIRECTORY for itself, so it
// is not an MPS client.
func (rsc *service) applyMPSHosting(podSpec *v1.Pod, gpuGroup string) {
	hostPathType := v1.HostPathDirectoryOrCreate
	podSpec.Spec.Volumes = append(podSpec.Spec.Volumes, v1.Volume{
		Name: mps.VolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: rsc.mpsBaseDir,
				Type: &hostPathType,
			},
		},
	})

	container := &podSpec.Spec.Containers[0]
	// The MPS control daemon must create its per-GPU pipe/log dirs on the shared (root-owned)
	// hostPath and run nvidia-cuda-mps-control to serve client contexts, so an MPS-hosting
	// reservation container runs as root — overriding the default non-root reservation security
	// context. Only reservations that host MPS are elevated; non-MPS reservations are untouched.
	root := int64(0)
	nonRoot := false
	allowPrivEsc := true
	container.SecurityContext = &v1.SecurityContext{
		RunAsUser:                &root,
		RunAsNonRoot:             &nonRoot,
		AllowPrivilegeEscalation: &allowPrivEsc,
	}
	container.VolumeMounts = append(container.VolumeMounts, v1.VolumeMount{
		Name:      mps.VolumeName,
		MountPath: rsc.mpsBaseDir,
	})
	container.Env = append(container.Env,
		v1.EnvVar{Name: "KAI_MPS_ENABLED", Value: "true"},
		v1.EnvVar{Name: "KAI_MPS_BASE_DIR", Value: rsc.mpsBaseDir},
		// Ensure the NVIDIA container runtime exposes the MPS control/server
		// binaries and compute libraries in the reservation container.
		v1.EnvVar{Name: "NVIDIA_DRIVER_CAPABILITIES", Value: "compute,utility"},
		v1.EnvVar{Name: "GPU_GROUP", ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: fmt.Sprintf("metadata.labels['%s']", constants.GPUGroup),
			},
		}},
	)
}

func (rsc *service) isScalingUp(ctx context.Context) bool {
	logger := log.FromContext(ctx)
	pods := &v1.PodList{}
	err := rsc.kubeClient.List(ctx, pods,
		client.InNamespace(rsc.scalingPodNamespace),
	)
	if err != nil {
		logger.Error(err, "Could not list scaling pods")
		return false
	}

	for _, pod := range pods.Items {
		podStatus := pod.Status
		for _, condition := range podStatus.Conditions {
			if condition.Type == v1.PodScheduled && condition.Status != v1.ConditionTrue {
				logger.Error(fmt.Errorf("not found"),
					"Unscheduled scaling pod was found", "name", pod.Name)
				return true
			}
		}
	}
	return false
}

func IsGPUReservationPod(pod *v1.Pod) bool {
	return strings.HasPrefix(pod.Name, gpuReservationPodPrefix)
}

// reservationPodName derives a deterministic reservation pod name from the node and
// gpu-group, so concurrent or retried creates for the same gpu-group collide on a
// single API-server object (AlreadyExists) instead of producing duplicate reservation
// pods on different physical GPUs.
func reservationPodName(nodeName, gpuGroup string) string {
	hash := sha256.Sum256([]byte(nodeName + "/" + gpuGroup))
	return fmt.Sprintf("%s-%s", gpuReservationPodPrefix, hex.EncodeToString(hash[:8]))
}
