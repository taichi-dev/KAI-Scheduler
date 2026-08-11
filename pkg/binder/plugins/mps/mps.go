// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package mps is the binder side of MPS-based VRAM enforcement for fractional
// GPU pods. It mirrors the hamicore binder plugin: at PreBind, for opted-in
// fraction pods, it computes the per-fraction MPS values and writes them into
// the shared-gpu capabilities ConfigMap (whose keys the mps admission plugin
// referenced as env vars on the fraction container):
//
//   - CUDA_MPS_PIPE_DIRECTORY           -> the pipe dir for this pod's gpu-group
//   - CUDA_MPS_PINNED_DEVICE_MEM_LIMIT  -> fraction x node VRAM, in MiB
//   - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -> optional, ~ fraction x 100
//
// It must run AFTER the gpusharing binder plugin, which creates the capabilities
// ConfigMap; the factory enforces this via priority + dependency validation.
package mps

import (
	"context"
	"fmt"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	admissionmps "github.com/kai-scheduler/KAI-scheduler/pkg/admission/webhook/v1alpha2/mps"
	"github.com/kai-scheduler/KAI-scheduler/pkg/apis/scheduling/v1alpha2"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/common"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/common/gpusharingconfigmap"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/plugins/state"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

type Plugin struct {
	kubeClient             client.Client
	baseDir                string
	activeThreadPercentage bool
}

func New(kubeClient client.Client, baseDir string, activeThreadPercentage bool) *Plugin {
	if baseDir == "" {
		baseDir = mps.DefaultBaseDir
	}
	return &Plugin{
		kubeClient:             kubeClient,
		baseDir:                baseDir,
		activeThreadPercentage: activeThreadPercentage,
	}
}

func (p *Plugin) Name() string {
	return "mps"
}

func (p *Plugin) PreBind(
	ctx context.Context, pod *v1.Pod, node *v1.Node, bindRequest *v1alpha2.BindRequest, _ *state.BindingState,
) error {
	// Two independent gates: the allocation must be a fraction AND the pod must
	// have opted into MPS. Reservation/slurm/whole-GPU pods fail both.
	if !common.IsSharedGPUAllocation(bindRequest) || !admissionmps.EnforceMPS(pod) {
		return nil
	}

	gpuGroup, err := selectedGpuGroup(bindRequest)
	if err != nil {
		return err
	}

	memLimitMib, err := allocatedMemoryMib(node, bindRequest)
	if err != nil {
		return err
	}

	portion, err := strconv.ParseFloat(bindRequest.Spec.ReceivedGPU.Portion, 64)
	if err != nil {
		return fmt.Errorf("invalid received gpu portion %q: %w", bindRequest.Spec.ReceivedGPU.Portion, err)
	}

	containerRef, err := common.GetFractionContainerRef(pod)
	if err != nil {
		return fmt.Errorf("failed to get fraction container ref: %w", err)
	}

	capabilitiesConfigMapName, err := gpusharingconfigmap.ExtractCapabilitiesConfigMapName(pod, containerRef)
	if err != nil {
		return err
	}

	pipeDir := mps.PipeDir(p.baseDir, gpuGroup)
	memValue := mps.PinnedDeviceMemLimitValue(memLimitMib)

	// Determine CUDA_MPS_ACTIVE_THREAD_PERCENTAGE with per-pod precedence:
	//   1. explicit gpu-sm-percent annotation -> always honored (decouples SM%
	//      from the VRAM fraction), regardless of the global flag; an invalid
	//      value fails the bind loudly rather than being silently ignored;
	//   2. else the portion-derived value when the activeThreadPercentage flag
	//      is on;
	//   3. else leave the key unset.
	smPercentValue := ""
	if raw, ok := pod.Annotations[constants.GpuSmPercent]; ok {
		pct, err := mps.ParseActiveThreadPercentage(raw)
		if err != nil {
			return fmt.Errorf("pod %s/%s has invalid %s annotation: %w",
				pod.Namespace, pod.Name, constants.GpuSmPercent, err)
		}
		smPercentValue = mps.ActiveThreadPercentageValueFromPercent(pct)
	} else if p.activeThreadPercentage {
		smPercentValue = mps.ActiveThreadPercentageValue(portion)
	}

	log.FromContext(ctx).Info("configuring MPS for fraction pod",
		"namespace", pod.Namespace, "name", pod.Name, "gpuGroup", gpuGroup,
		"pipeDir", pipeDir, "pinnedMemLimit", memValue, "activeThreadPercentage", smPercentValue)

	return common.UpdateConfigMapEnvironmentVariable(ctx, p.kubeClient, pod, capabilitiesConfigMapName,
		func(data map[string]string) error {
			data[mps.PipeDirectoryEnvVar] = pipeDir
			data[mps.PinnedDeviceMemLimitEnvVar] = memValue
			if smPercentValue != "" {
				data[mps.ActiveThreadPercentageEnvVar] = smPercentValue
			}
			return nil
		})
}

func selectedGpuGroup(bindRequest *v1alpha2.BindRequest) (string, error) {
	if bindRequest == nil || len(bindRequest.Spec.SelectedGPUGroups) == 0 {
		return "", fmt.Errorf("bind request has no selected gpu group for MPS pipe directory")
	}
	// A fraction pod backed by MPS shares a single physical GPU, i.e. a single
	// reservation / gpu-group. Multi-fraction (multi-device) pods are out of
	// scope for this prototype.
	return bindRequest.Spec.SelectedGPUGroups[0], nil
}

// allocatedMemoryMib computes fraction x node VRAM (MiB), mirroring the hamicore
// binder plugin's CUDA_DEVICE_MEMORY_LIMIT calculation.
func allocatedMemoryMib(node *v1.Node, bindRequest *v1alpha2.BindRequest) (int64, error) {
	if node == nil || bindRequest == nil || bindRequest.Spec.ReceivedGPU == nil {
		return 0, fmt.Errorf("missing data for MPS memory limit calculation")
	}

	memoryLabel, found := node.Labels[constants.NvidiaGpuMemory]
	if !found {
		return 0, fmt.Errorf("node does not include %s label", constants.NvidiaGpuMemory)
	}
	totalGPUMemoryMib, err := strconv.ParseInt(memoryLabel, 10, 64)
	if err != nil || totalGPUMemoryMib <= 0 {
		return 0, fmt.Errorf("invalid %s label value %q", constants.NvidiaGpuMemory, memoryLabel)
	}

	portion, err := strconv.ParseFloat(bindRequest.Spec.ReceivedGPU.Portion, 64)
	if err != nil || portion <= 0 {
		return 0, fmt.Errorf("invalid received gpu portion %q", bindRequest.Spec.ReceivedGPU.Portion)
	}

	allocated := int64(float64(totalGPUMemoryMib) * portion)
	if allocated <= 0 {
		return 0, fmt.Errorf("calculated allocated gpu memory is zero")
	}
	return allocated, nil
}

func (p *Plugin) PostBind(
	context.Context, *v1.Pod, *v1.Node, *v1alpha2.BindRequest, *state.BindingState,
) {
}

// Rollback is a no-op: the capabilities ConfigMap (and thus every MPS key in it)
// is owned and deleted by the gpusharing plugin's Rollback.
func (p *Plugin) Rollback(
	context.Context, *v1.Pod, *v1.Node, *v1alpha2.BindRequest, *state.BindingState,
) error {
	return nil
}
