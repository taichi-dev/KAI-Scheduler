// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package controllers

import (
	"context"
	"testing"
	"time"

	"go.uber.org/mock/gomock"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	kubeaischedulerscheme "github.com/kai-scheduler/KAI-scheduler/pkg/apis/client/clientset/versioned/scheme"
	schedulingv1alpha2 "github.com/kai-scheduler/KAI-scheduler/pkg/apis/scheduling/v1alpha2"
	mock_resourcereservation "github.com/kai-scheduler/KAI-scheduler/pkg/binder/binding/resourcereservation/mock"
	bindercommon "github.com/kai-scheduler/KAI-scheduler/pkg/binder/common"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
)

const raSchedulerName = "kai-scheduler"

func raTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := v1.AddToScheme(s); err != nil {
		t.Fatalf("add corev1 to scheme: %v", err)
	}
	if err := kubeaischedulerscheme.AddToScheme(s); err != nil {
		t.Fatalf("add scheduling scheme: %v", err)
	}
	return s
}

func raFraction(name string, age time.Duration) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         "team",
			UID:               types.UID(name + "-uid"),
			CreationTimestamp: metav1.NewTime(time.Now().Add(-age)),
			Annotations:       map[string]string{constants.GpuFraction: "0.5"},
		},
		Spec:   v1.PodSpec{SchedulerName: raSchedulerName},
		Status: v1.PodStatus{Phase: v1.PodPending},
	}
}

func TestReserveAheadGroupIsStableAndDeterministic(t *testing.T) {
	a := reserveAheadGroup("team", "frac-1")
	b := reserveAheadGroup("team", "frac-1")
	c := reserveAheadGroup("team", "frac-2")
	if a != b {
		t.Errorf("group not deterministic: %q vs %q", a, b)
	}
	if a == c {
		t.Errorf("distinct fractions collided on group %q", a)
	}
	if len(a) > 63 {
		t.Errorf("group %q exceeds label-value length", a)
	}
}

func TestReserveAheadDisabledIsNoop(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	// No mock expectations set -> any call fails the test.
	pod := raFraction("frac-1", time.Hour)
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: false}

	res, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.RequeueAfter != 0 {
		t.Errorf("disabled reconciler should not requeue, got %v", res.RequeueAfter)
	}
}

func TestReserveAheadWaitsBeforeReserving(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	// Fresh pod (age 0) -> must requeue without touching the reservation service.
	pod := raFraction("frac-1", 0)
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: true, Delay: 15 * time.Second}

	res, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.RequeueAfter <= 0 {
		t.Errorf("expected a requeue while waiting, got %v", res.RequeueAfter)
	}
}

func TestReserveAheadEnsuresReservationAndLabelsFraction(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	pod := raFraction("frac-1", time.Hour)
	group := reserveAheadGroup("team", "frac-1")
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: true, Delay: 15 * time.Second}

	rrs.EXPECT().ReservationPlacement(gomock.Any(), group).Return(false, "", nil)
	rrs.EXPECT().EnsureReservation(gomock.Any(), gomock.Any(), group).Return(pod, nil)

	res, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.RequeueAfter != reserveAheadPollInterval {
		t.Errorf("expected poll requeue %v, got %v", reserveAheadPollInterval, res.RequeueAfter)
	}

	got := &v1.Pod{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "team", Name: "frac-1"}, got); err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if got.Labels[constants.GPUGroup] != group {
		t.Errorf("fraction not labeled with reserve-ahead group: got %q want %q", got.Labels[constants.GPUGroup], group)
	}
}

func TestReserveAheadPackWonTearsDownReservation(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	group := reserveAheadGroup("team", "frac-1")
	// Fraction was reserve-ahead labeled, but the scheduler packed it onto a DIFFERENT freed group.
	pod := raFraction("frac-1", time.Hour)
	pod.Labels = map[string]string{constants.GPUGroup: group}
	br := &schedulingv1alpha2.BindRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "frac-1", Namespace: "team"},
		Spec:       schedulingv1alpha2.BindRequestSpec{PodName: "frac-1", SelectedNode: "node-y", SelectedGPUGroups: []string{"real-group-abc"}},
	}
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod, br).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: true, Delay: 15 * time.Second}

	// Teardown of the orphaned reserve-ahead reservation; NO EnsureReservation.
	rrs.EXPECT().SyncForGpuGroup(gomock.Any(), group).Return(nil)

	if _, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := &v1.Pod{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "team", Name: "frac-1"}, got); err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if got.Labels[constants.GPUGroup] != "real-group-abc" {
		t.Errorf("pod group label = %q, want the real bound group real-group-abc", got.Labels[constants.GPUGroup])
	}
}

func TestReserveAheadReservationWonIsKept(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	group := reserveAheadGroup("team", "frac-1")
	pod := raFraction("frac-1", time.Hour)
	// BindRequest binds onto OUR reserve-ahead group (the reservation won the race).
	br := &schedulingv1alpha2.BindRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "frac-1", Namespace: "team"},
		Spec:       schedulingv1alpha2.BindRequestSpec{PodName: "frac-1", SelectedNode: "node-x", SelectedGPUGroups: []string{group}},
	}
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod, br).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: true, Delay: 15 * time.Second}

	// No reservation-service calls at all — the reservation is in use, nothing to do.
	if _, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestReserveAheadBindsWhenReservationPlaced(t *testing.T) {
	ctrlm := gomock.NewController(t)
	rrs := mock_resourcereservation.NewMockInterface(ctrlm)
	pod := raFraction("frac-1", time.Hour)
	group := reserveAheadGroup("team", "frac-1")
	c := fake.NewClientBuilder().WithScheme(raTestScheme(t)).WithObjects(pod).Build()
	r := &ReserveAheadReconciler{Client: c, ResourceReservation: rrs, SchedulerName: raSchedulerName, Enabled: true, Delay: 15 * time.Second}

	rrs.EXPECT().ReservationPlacement(gomock.Any(), group).Return(true, "node-x", nil)
	// EnsureReservation must NOT be called on the ready path.

	if _, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "team", Name: "frac-1"}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	br := &schedulingv1alpha2.BindRequest{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "team", Name: "frac-1"}, br); err != nil {
		t.Fatalf("expected a BindRequest to be created: %v", err)
	}
	if br.Spec.SelectedNode != "node-x" {
		t.Errorf("BindRequest node = %q, want node-x", br.Spec.SelectedNode)
	}
	if len(br.Spec.SelectedGPUGroups) != 1 || br.Spec.SelectedGPUGroups[0] != group {
		t.Errorf("BindRequest groups = %v, want [%s]", br.Spec.SelectedGPUGroups, group)
	}
	if br.Spec.ReceivedResourceType != bindercommon.ReceivedTypeFraction {
		t.Errorf("BindRequest resource type = %q, want %q", br.Spec.ReceivedResourceType, bindercommon.ReceivedTypeFraction)
	}
}
