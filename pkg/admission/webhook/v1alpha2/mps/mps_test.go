// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package mps

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	commonmps "github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

func fractionPod(annotations map[string]string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", Annotations: annotations},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
	}
}

func TestMutateGating(t *testing.T) {
	tests := []struct {
		name        string
		pod         *v1.Pod
		expectError bool
		expectMPS   bool
	}{
		{
			name: "no containers",
			pod:  &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{}}},
		},
		{
			name: "non-fraction pod skipped",
			pod:  fractionPod(map[string]string{constants.MpsAnnotation: "true"}),
		},
		{
			name: "fraction pod without mps annotation skipped",
			pod:  fractionPod(map[string]string{constants.GpuFraction: "0.5"}),
		},
		{
			name: "fraction pod with mps=false skipped",
			pod: fractionPod(map[string]string{
				constants.GpuFraction: "0.5", constants.MpsAnnotation: "false",
			}),
		},
		{
			name: "opted-in fraction pod without configmap annotation errors",
			pod: fractionPod(map[string]string{
				constants.GpuFraction: "0.5", constants.MpsAnnotation: "true",
			}),
			expectError: true,
		},
		{
			name: "opted-in fraction pod gets MPS env and mount",
			pod: fractionPod(map[string]string{
				constants.GpuFraction:                   "0.5",
				constants.MpsAnnotation:                 "true",
				constants.GpuSharingConfigMapAnnotation: "pod-shared-gpu",
			}),
			expectMPS: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := New().Mutate(tt.pod)
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			hasEnv := containerHasEnv(tt.pod, commonmps.PipeDirectoryEnvVar)
			if hasEnv != tt.expectMPS {
				t.Fatalf("MPS env presence = %v, want %v", hasEnv, tt.expectMPS)
			}
			if !tt.expectMPS {
				if podHasVolume(tt.pod, commonmps.VolumeName) {
					t.Errorf("did not expect MPS hostPath volume")
				}
				return
			}

			assertOptionalConfigMapEnv(t, tt.pod, commonmps.PipeDirectoryEnvVar)
			assertOptionalConfigMapEnv(t, tt.pod, commonmps.PinnedDeviceMemLimitEnvVar)
			assertOptionalConfigMapEnv(t, tt.pod, commonmps.ActiveThreadPercentageEnvVar)

			if !podHasVolume(tt.pod, commonmps.VolumeName) {
				t.Errorf("expected MPS hostPath volume %q", commonmps.VolumeName)
			}
			vol := getVolume(tt.pod, commonmps.VolumeName)
			if vol == nil || vol.HostPath == nil || vol.HostPath.Path != commonmps.DefaultBaseDir {
				t.Errorf("expected hostPath %q, got %+v", commonmps.DefaultBaseDir, vol)
			}
			if !containerHasMount(tt.pod, commonmps.VolumeName, commonmps.DefaultBaseDir) {
				t.Errorf("expected volume mount at %q", commonmps.DefaultBaseDir)
			}
		})
	}
}

func TestMutateIdempotent(t *testing.T) {
	pod := fractionPod(map[string]string{
		constants.GpuFraction:                   "0.5",
		constants.MpsAnnotation:                 "true",
		constants.GpuSharingConfigMapAnnotation: "pod-shared-gpu",
	})
	if err := New().Mutate(pod); err != nil {
		t.Fatalf("first mutate: %v", err)
	}
	if err := New().Mutate(pod); err != nil {
		t.Fatalf("second mutate: %v", err)
	}
	// Exactly one volume and one mount, one env of each name.
	if n := countVolumes(pod, commonmps.VolumeName); n != 1 {
		t.Errorf("expected 1 MPS volume after double-mutate, got %d", n)
	}
	if n := countEnv(pod, commonmps.PipeDirectoryEnvVar); n != 1 {
		t.Errorf("expected 1 pipe-dir env after double-mutate, got %d", n)
	}
	if n := countMounts(pod, commonmps.VolumeName); n != 1 {
		t.Errorf("expected 1 MPS mount after double-mutate, got %d", n)
	}
}

func TestValidateSmPercent(t *testing.T) {
	tests := []struct {
		name        string
		smPercent   string
		present     bool
		expectError bool
	}{
		{name: "no annotation ok", present: false},
		{name: "valid value ok", present: true, smPercent: "80"},
		{name: "non-integer errors", present: true, smPercent: "abc", expectError: true},
		{name: "out of range errors", present: true, smPercent: "150", expectError: true},
		{name: "empty errors", present: true, smPercent: "", expectError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ann := map[string]string{constants.GpuFraction: "0.5", constants.MpsAnnotation: "true"}
			if tt.present {
				ann[constants.GpuSmPercent] = tt.smPercent
			}
			err := New().Validate(fractionPod(ann))
			if tt.expectError && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func assertOptionalConfigMapEnv(t *testing.T, pod *v1.Pod, name string) {
	t.Helper()
	for _, e := range pod.Spec.Containers[0].Env {
		if e.Name != name {
			continue
		}
		if e.ValueFrom == nil || e.ValueFrom.ConfigMapKeyRef == nil {
			t.Errorf("%s: expected ConfigMapKeyRef", name)
			return
		}
		ref := e.ValueFrom.ConfigMapKeyRef
		if ref.Key != name {
			t.Errorf("%s: key = %q", name, ref.Key)
		}
		if ref.Optional == nil || !*ref.Optional {
			t.Errorf("%s: expected Optional=true", name)
		}
		return
	}
	t.Errorf("env %s not found", name)
}

func containerHasEnv(pod *v1.Pod, name string) bool { return countEnv(pod, name) > 0 }

func countEnv(pod *v1.Pod, name string) int {
	n := 0
	if len(pod.Spec.Containers) == 0 {
		return 0
	}
	for _, e := range pod.Spec.Containers[0].Env {
		if e.Name == name {
			n++
		}
	}
	return n
}

func podHasVolume(pod *v1.Pod, name string) bool { return countVolumes(pod, name) > 0 }

func countVolumes(pod *v1.Pod, name string) int {
	n := 0
	for _, v := range pod.Spec.Volumes {
		if v.Name == name {
			n++
		}
	}
	return n
}

func getVolume(pod *v1.Pod, name string) *v1.Volume {
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].Name == name {
			return &pod.Spec.Volumes[i]
		}
	}
	return nil
}

func containerHasMount(pod *v1.Pod, name, path string) bool {
	if len(pod.Spec.Containers) == 0 {
		return false
	}
	for _, m := range pod.Spec.Containers[0].VolumeMounts {
		if m.Name == name && m.MountPath == path {
			return true
		}
	}
	return false
}

func countMounts(pod *v1.Pod, name string) int {
	n := 0
	if len(pod.Spec.Containers) == 0 {
		return 0
	}
	for _, m := range pod.Spec.Containers[0].VolumeMounts {
		if m.Name == name {
			n++
		}
	}
	return n
}
