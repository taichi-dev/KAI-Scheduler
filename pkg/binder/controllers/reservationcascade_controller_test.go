// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package controllers

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/kai-scheduler/KAI-scheduler/pkg/binder/binding/resourcereservation"
	"github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
)

const rcResNs = "kai-resource-reservation"

func rcScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := v1.AddToScheme(s); err != nil {
		t.Fatalf("add corev1: %v", err)
	}
	return s
}

func rcReservation(group string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "gpu-reservation-x",
			Namespace:  rcResNs,
			Labels:     map[string]string{constants.GPUGroup: group},
			Finalizers: []string{resourcereservation.ReservationCascadeFinalizer},
		},
	}
}

func rcFraction(name, group string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "team",
			Labels:    map[string]string{constants.GPUGroup: group},
		},
	}
}

func rcReq() ctrl.Request {
	return ctrl.Request{NamespacedName: types.NamespacedName{Namespace: rcResNs, Name: "gpu-reservation-x"}}
}

func TestReservationCascadeDeletesFractionsAndWaits(t *testing.T) {
	group := "g-1"
	resv := rcReservation(group)
	f1 := rcFraction("frac-1", group)
	f2 := rcFraction("frac-2", group)
	c := fake.NewClientBuilder().WithScheme(rcScheme(t)).WithObjects(resv, f1, f2).Build()
	r := &ReservationCascadeReconciler{Client: c, ReservationNamespace: rcResNs, Enabled: true}
	ctx := context.Background()

	// Delete the reservation -> finalizer keeps it Terminating.
	if err := c.Delete(ctx, resv); err != nil {
		t.Fatalf("delete reservation: %v", err)
	}

	// First reconcile: fractions deleted, finalizer retained (requeue while they drain).
	res, err := r.Reconcile(ctx, rcReq())
	if err != nil {
		t.Fatalf("reconcile 1: %v", err)
	}
	if res.RequeueAfter != reservationCascadePollInterval {
		t.Errorf("expected requeue %v, got %v", reservationCascadePollInterval, res.RequeueAfter)
	}
	for _, n := range []string{"frac-1", "frac-2"} {
		if err := c.Get(ctx, types.NamespacedName{Namespace: "team", Name: n}, &v1.Pod{}); !apierrors.IsNotFound(err) {
			t.Errorf("fraction %s should have been deleted, err=%v", n, err)
		}
	}
	if err := c.Get(ctx, rcReq().NamespacedName, &v1.Pod{}); err != nil {
		t.Errorf("reservation should still exist (finalizer held) while fractions drain, err=%v", err)
	}

	// Second reconcile: no fractions remain -> finalizer removed -> reservation GC'd.
	if _, err := r.Reconcile(ctx, rcReq()); err != nil {
		t.Fatalf("reconcile 2: %v", err)
	}
	if err := c.Get(ctx, rcReq().NamespacedName, &v1.Pod{}); !apierrors.IsNotFound(err) {
		t.Errorf("reservation should be released once fractions are gone, err=%v", err)
	}
}

func TestReservationCascadeNoFractionsReleasesImmediately(t *testing.T) {
	resv := rcReservation("g-empty")
	c := fake.NewClientBuilder().WithScheme(rcScheme(t)).WithObjects(resv).Build()
	r := &ReservationCascadeReconciler{Client: c, ReservationNamespace: rcResNs, Enabled: true}
	ctx := context.Background()
	if err := c.Delete(ctx, resv); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := r.Reconcile(ctx, rcReq()); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if err := c.Get(ctx, rcReq().NamespacedName, &v1.Pod{}); !apierrors.IsNotFound(err) {
		t.Errorf("reservation with no backed fractions should be released immediately, err=%v", err)
	}
}

func TestReservationCascadeDisabledIsNoop(t *testing.T) {
	resv := rcReservation("g-1")
	f1 := rcFraction("frac-1", "g-1")
	c := fake.NewClientBuilder().WithScheme(rcScheme(t)).WithObjects(resv, f1).Build()
	r := &ReservationCascadeReconciler{Client: c, ReservationNamespace: rcResNs, Enabled: false}
	ctx := context.Background()
	_ = c.Delete(ctx, resv)
	if _, err := r.Reconcile(ctx, rcReq()); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	// Fraction must be untouched when disabled.
	if err := c.Get(ctx, types.NamespacedName{Namespace: "team", Name: "frac-1"}, &v1.Pod{}); err != nil {
		t.Errorf("fraction should be untouched when disabled, err=%v", err)
	}
}
