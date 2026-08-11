// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package mps is the admission side of MPS-based VRAM enforcement for
// fractional GPU pods. It mirrors the hamicore admission plugin: for fraction
// pods that opt in (annotation mps: "true") it injects CUDA_MPS_* env var
// references (ConfigMapKeyRef into the shared-gpu capabilities ConfigMap, whose
// values the mps binder plugin fills at bind time) plus the hostPath volume
// that carries the per-GPU MPS pipe/log directory.
//
// It must run AFTER the gpusharing admission plugin, which sets the shared-gpu
// ConfigMap-name annotation this plugin reads.
package mps

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"

	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/common"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/common/gpusharingconfigmap"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/resources"
)

type MPS struct{}

func New() *MPS {
	return &MPS{}
}

func (p *MPS) Name() string {
	return "mps"
}

// Validate rejects an invalid gpu-sm-percent annotation early, so a malformed
// explicit SM% request is caught at admission rather than only failing the bind
// loop later. A pod that omits the annotation is unaffected.
func (p *MPS) Validate(pod *v1.Pod) error {
	if raw, ok := pod.Annotations[constants.GpuSmPercent]; ok {
		if _, err := mps.ParseActiveThreadPercentage(raw); err != nil {
			return fmt.Errorf("invalid %s annotation: %w", constants.GpuSmPercent, err)
		}
	}
	return nil
}

// Mutate injects the CUDA_MPS_* env var references and the shared MPS pipe-dir
// hostPath mount into the fraction container. Only fraction pods that opt into
// MPS (annotation mps: "true") are touched; reservation and Slurm pods never
// carry that annotation nor a fraction request, so they are never given MPS
// env and keep using the physical device directly.
func (p *MPS) Mutate(pod *v1.Pod) error {
	if len(pod.Spec.Containers) == 0 {
		return nil
	}

	if !EnforceMPS(pod) {
		return nil
	}

	containerRef, err := common.GetFractionContainerRef(pod)
	if err != nil {
		return err
	}

	capabilitiesConfigMapName, err := gpusharingconfigmap.ExtractCapabilitiesConfigMapName(pod, containerRef)
	if err != nil {
		return err
	}

	// Values are written into the ConfigMap by the mps binder plugin at bind
	// time (it knows the node's VRAM and the assigned gpu-group). Optional=true
	// so the container starts fine before/if a value is absent.
	for _, key := range []string{
		mps.PipeDirectoryEnvVar,
		mps.PinnedDeviceMemLimitEnvVar,
		mps.ActiveThreadPercentageEnvVar,
	} {
		common.AddEnvVarToContainer(containerRef.Container, v1.EnvVar{
			Name: key,
			ValueFrom: &v1.EnvVarSource{
				ConfigMapKeyRef: &v1.ConfigMapKeySelector{
					Key: key,
					LocalObjectReference: v1.LocalObjectReference{
						Name: capabilitiesConfigMapName,
					},
					Optional: ptr.To(true),
				},
			},
		})
	}

	// Mount the node-local MPS base directory. The exact per-GPU subdirectory
	// is selected by the CUDA_MPS_PIPE_DIRECTORY value (set at bind time), which
	// resolves inside this mount.
	common.SetHostPathVolume(pod, containerRef.Container, mps.VolumeName, mps.DefaultBaseDir, mps.DefaultBaseDir)

	return nil
}

// EnforceMPS reports whether MPS VRAM enforcement applies to this pod: it must
// be a fractional GPU request AND explicitly opt in via the mps annotation.
// This is the single gate shared by the admission and binder plugins so that
// reservation/slurm/whole-GPU pods are provably never affected.
func EnforceMPS(pod *v1.Pod) bool {
	if !resources.RequestsGPUFraction(pod) {
		return false
	}
	return pod.Annotations[constants.MpsAnnotation] == "true"
}
