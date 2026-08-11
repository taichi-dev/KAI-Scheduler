// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package resourcereservation

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

func reservationPodShell(gpuGroup string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gpu-reservation-abc",
			Namespace: constants.DefaultResourceReservationName,
			Labels:    map[string]string{constants.GPUGroup: gpuGroup},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "resource-reservation"}},
		},
	}
}

func TestApplyMPSHosting(t *testing.T) {
	rsc := &service{mpsEnabled: true, mpsBaseDir: mps.DefaultBaseDir}
	pod := reservationPodShell("grp-42")

	rsc.applyMPSHosting(pod, "grp-42")

	// hostPath volume for the shared MPS base dir.
	var vol *v1.Volume
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].Name == mps.VolumeName {
			vol = &pod.Spec.Volumes[i]
		}
	}
	if vol == nil {
		t.Fatalf("expected MPS hostPath volume %q", mps.VolumeName)
	}
	if vol.HostPath == nil || vol.HostPath.Path != mps.DefaultBaseDir {
		t.Errorf("hostPath = %+v, want path %q", vol.HostPath, mps.DefaultBaseDir)
	}
	if vol.HostPath.Type == nil || *vol.HostPath.Type != v1.HostPathDirectoryOrCreate {
		t.Errorf("expected HostPathDirectoryOrCreate")
	}

	c := pod.Spec.Containers[0]

	// Mount at the base dir.
	mounted := false
	for _, m := range c.VolumeMounts {
		if m.Name == mps.VolumeName && m.MountPath == mps.DefaultBaseDir {
			mounted = true
		}
	}
	if !mounted {
		t.Errorf("expected MPS volume mount at %q", mps.DefaultBaseDir)
	}

	// Env: KAI_MPS_ENABLED, KAI_MPS_BASE_DIR, NVIDIA_DRIVER_CAPABILITIES, and
	// GPU_GROUP via downward API (never a literal, so it tracks the label).
	env := map[string]v1.EnvVar{}
	for _, e := range c.Env {
		env[e.Name] = e
	}
	if env["KAI_MPS_ENABLED"].Value != "true" {
		t.Errorf("KAI_MPS_ENABLED = %q, want true", env["KAI_MPS_ENABLED"].Value)
	}
	if env["KAI_MPS_BASE_DIR"].Value != mps.DefaultBaseDir {
		t.Errorf("KAI_MPS_BASE_DIR = %q, want %q", env["KAI_MPS_BASE_DIR"].Value, mps.DefaultBaseDir)
	}
	if env["NVIDIA_DRIVER_CAPABILITIES"].Value != "compute,utility" {
		t.Errorf("NVIDIA_DRIVER_CAPABILITIES = %q", env["NVIDIA_DRIVER_CAPABILITIES"].Value)
	}
	gg, ok := env["GPU_GROUP"]
	if !ok {
		t.Fatalf("expected GPU_GROUP env")
	}
	if gg.ValueFrom == nil || gg.ValueFrom.FieldRef == nil {
		t.Errorf("GPU_GROUP should be a downward-API fieldRef, got %+v", gg)
	} else if gg.ValueFrom.FieldRef.FieldPath != "metadata.labels['"+constants.GPUGroup+"']" {
		t.Errorf("GPU_GROUP fieldPath = %q", gg.ValueFrom.FieldRef.FieldPath)
	}
}
