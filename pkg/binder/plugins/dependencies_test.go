// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package plugins

import (
	"testing"

	"k8s.io/utils/ptr"
)

func TestValidateDependentPluginsHamiCoreRequiresGPUSharing(t *testing.T) {
	cfg := Config{
		GPUSharingPluginName: {
			Enabled: ptr.To(false),
		},
		HamiCorePluginName: {
			Enabled: ptr.To(true),
		},
	}

	if err := validateDependentPlugins(cfg); err == nil {
		t.Fatalf("expected dependency validation error")
	}
}

func TestValidateDependentPluginsHamiCoreRequiresOrdering(t *testing.T) {
	cfg := Config{
		GPUSharingPluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(50),
		},
		HamiCorePluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(100),
		},
	}

	if err := validateDependentPlugins(cfg); err == nil {
		t.Fatalf("expected ordering validation error")
	}
}

func TestValidateDependentPluginsMPSRequiresGPUSharing(t *testing.T) {
	cfg := Config{
		GPUSharingPluginName: {
			Enabled: ptr.To(false),
		},
		MPSPluginName: {
			Enabled: ptr.To(true),
		},
	}

	if err := validateDependentPlugins(cfg); err == nil {
		t.Fatalf("expected dependency validation error")
	}
}

func TestValidateDependentPluginsMPSRequiresOrdering(t *testing.T) {
	cfg := Config{
		GPUSharingPluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(50),
		},
		MPSPluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(100),
		},
	}

	if err := validateDependentPlugins(cfg); err == nil {
		t.Fatalf("expected ordering validation error")
	}
}

func TestValidateDependentPluginsMPSValidWhenOrderedAfterGPUSharing(t *testing.T) {
	cfg := Config{
		GPUSharingPluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(100),
		},
		MPSPluginName: {
			Enabled:  ptr.To(true),
			Priority: ptr.To(40),
		},
	}

	if err := validateDependentPlugins(cfg); err != nil {
		t.Fatalf("expected valid config, got %v", err)
	}
}
