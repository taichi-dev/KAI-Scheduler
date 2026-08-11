// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package mps

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/kai-scheduler/KAI-scheduler/pkg/apis/scheduling/v1alpha2"
	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/common"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	commonmps "github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

func t4Node() *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "gpu-node",
			Labels: map[string]string{constants.NvidiaGpuMemory: "15360"},
		},
	}
}

func fractionBindRequest(portion string, groups ...string) *v1alpha2.BindRequest {
	return &v1alpha2.BindRequest{
		Spec: v1alpha2.BindRequestSpec{
			ReceivedResourceType: common.ReceivedTypeFraction,
			ReceivedGPU:          &v1alpha2.ReceivedGPU{Portion: portion},
			SelectedGPUGroups:    groups,
		},
	}
}

func TestAllocatedMemoryMib(t *testing.T) {
	got, err := allocatedMemoryMib(t4Node(), fractionBindRequest("0.5", "g"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 7680 {
		t.Errorf("allocatedMemoryMib = %d, want 7680", got)
	}
}

func TestSelectedGpuGroupRequiresGroup(t *testing.T) {
	if _, err := selectedGpuGroup(fractionBindRequest("0.5")); err == nil {
		t.Errorf("expected error when no gpu group is selected")
	}
	got, err := selectedGpuGroup(fractionBindRequest("0.5", "group-x"))
	if err != nil || got != "group-x" {
		t.Errorf("selectedGpuGroup = (%q,%v), want (group-x,nil)", got, err)
	}
}

func TestPreBindGatedOut(t *testing.T) {
	p := New(nil, "", false) // nil client is safe: gated-out returns before use

	// Non-fraction allocation.
	regular := &v1alpha2.BindRequest{Spec: v1alpha2.BindRequestSpec{ReceivedResourceType: common.ReceivedTypeRegular}}
	if err := p.PreBind(context.Background(), &v1.Pod{}, t4Node(), regular, nil); err != nil {
		t.Errorf("regular allocation should be a no-op, got %v", err)
	}

	// Fraction allocation but the pod did not opt into MPS.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{constants.GpuFraction: "0.5"}, // no mps: "true"
		},
		Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
	}
	if err := p.PreBind(context.Background(), pod, t4Node(), fractionBindRequest("0.5", "g"), nil); err != nil {
		t.Errorf("non-opted-in fraction should be a no-op, got %v", err)
	}
}

func TestPreBindWritesConfigMap(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = v1.AddToScheme(scheme)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "trainer",
			Namespace: "team-a",
			Annotations: map[string]string{
				constants.GpuFraction:                   "0.5",
				constants.MpsAnnotation:                 "true",
				constants.GpuSharingConfigMapAnnotation: "trainer-shared-gpu",
			},
		},
		Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
	}
	// The gpusharing plugin creates this ConfigMap earlier in PreBind; the
	// capabilities name for container index 0 is "<prefix>-0".
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "trainer-shared-gpu-0", Namespace: "team-a"},
		Data:       map[string]string{},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod, cm).Build()

	p := New(c, commonmps.DefaultBaseDir, true) // activeThreadPercentage on
	br := fractionBindRequest("0.5", "grp-777")

	if err := p.PreBind(context.Background(), pod, t4Node(), br, nil); err != nil {
		t.Fatalf("PreBind: %v", err)
	}

	got := &v1.ConfigMap{}
	if err := c.Get(context.Background(),
		types.NamespacedName{Namespace: "team-a", Name: "trainer-shared-gpu-0"}, got); err != nil {
		t.Fatalf("get configmap: %v", err)
	}

	wantPipe := commonmps.PipeDir(commonmps.DefaultBaseDir, "grp-777")
	if got.Data[commonmps.PipeDirectoryEnvVar] != wantPipe {
		t.Errorf("%s = %q, want %q", commonmps.PipeDirectoryEnvVar,
			got.Data[commonmps.PipeDirectoryEnvVar], wantPipe)
	}
	if got.Data[commonmps.PinnedDeviceMemLimitEnvVar] != "0=7680M" {
		t.Errorf("%s = %q, want 0=7680M", commonmps.PinnedDeviceMemLimitEnvVar,
			got.Data[commonmps.PinnedDeviceMemLimitEnvVar])
	}
	if got.Data[commonmps.ActiveThreadPercentageEnvVar] != "50" {
		t.Errorf("%s = %q, want 50", commonmps.ActiveThreadPercentageEnvVar,
			got.Data[commonmps.ActiveThreadPercentageEnvVar])
	}
}

// mpsPodWithSmPercent builds an opted-in fraction pod carrying an optional
// gpu-sm-percent annotation (empty smPercent omits it) plus the shared-gpu
// ConfigMap the gpusharing plugin would have created.
func mpsPodWithSmPercent(smPercent string) (*v1.Pod, *v1.ConfigMap) {
	ann := map[string]string{
		constants.GpuFraction:                   "0.5",
		constants.MpsAnnotation:                 "true",
		constants.GpuSharingConfigMapAnnotation: "trainer-shared-gpu",
	}
	if smPercent != "" {
		ann[constants.GpuSmPercent] = smPercent
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "trainer", Namespace: "team-a", Annotations: ann},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
	}
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "trainer-shared-gpu-0", Namespace: "team-a"},
		Data:       map[string]string{},
	}
	return pod, cm
}

func newFakeClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	_ = v1.AddToScheme(scheme)
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

// The explicit gpu-sm-percent annotation must be honored even when the global
// activeThreadPercentage flag is OFF, and must override the portion-derived
// value (0.5 -> "50") when the flag is ON.
func TestPreBindSmPercentAnnotationOverridesFlag(t *testing.T) {
	for _, flag := range []bool{false, true} {
		pod, cm := mpsPodWithSmPercent("80")
		c := newFakeClient(pod, cm)
		p := New(c, commonmps.DefaultBaseDir, flag)

		if err := p.PreBind(context.Background(), pod, t4Node(), fractionBindRequest("0.5", "grp-777"), nil); err != nil {
			t.Fatalf("flag=%v: PreBind: %v", flag, err)
		}

		got := &v1.ConfigMap{}
		if err := c.Get(context.Background(),
			types.NamespacedName{Namespace: "team-a", Name: "trainer-shared-gpu-0"}, got); err != nil {
			t.Fatalf("flag=%v: get configmap: %v", flag, err)
		}
		if got.Data[commonmps.ActiveThreadPercentageEnvVar] != "80" {
			t.Errorf("flag=%v: %s = %q, want 80 (annotation wins)", flag,
				commonmps.ActiveThreadPercentageEnvVar, got.Data[commonmps.ActiveThreadPercentageEnvVar])
		}
	}
}

// An invalid gpu-sm-percent annotation must fail the bind loudly.
func TestPreBindSmPercentInvalidErrors(t *testing.T) {
	for _, bad := range []string{"abc", "0", "150", "50.5", ""} {
		pod, cm := mpsPodWithSmPercent("placeholder")
		pod.Annotations[constants.GpuSmPercent] = bad // include the empty-string case
		c := newFakeClient(pod, cm)
		p := New(c, commonmps.DefaultBaseDir, false)

		if err := p.PreBind(context.Background(), pod, t4Node(), fractionBindRequest("0.5", "grp-777"), nil); err == nil {
			t.Errorf("gpu-sm-percent=%q: expected PreBind error, got nil", bad)
		}
	}
}

// With no annotation and the flag OFF, the SM% key must be absent (case 3).
func TestPreBindSmPercentAbsentWhenFlagOff(t *testing.T) {
	pod, cm := mpsPodWithSmPercent("")
	c := newFakeClient(pod, cm)
	p := New(c, commonmps.DefaultBaseDir, false)

	if err := p.PreBind(context.Background(), pod, t4Node(), fractionBindRequest("0.5", "grp-777"), nil); err != nil {
		t.Fatalf("PreBind: %v", err)
	}
	got := &v1.ConfigMap{}
	if err := c.Get(context.Background(),
		types.NamespacedName{Namespace: "team-a", Name: "trainer-shared-gpu-0"}, got); err != nil {
		t.Fatalf("get configmap: %v", err)
	}
	if _, ok := got.Data[commonmps.ActiveThreadPercentageEnvVar]; ok {
		t.Errorf("expected %s to be unset when no annotation and flag off, got %q",
			commonmps.ActiveThreadPercentageEnvVar, got.Data[commonmps.ActiveThreadPercentageEnvVar])
	}
}
