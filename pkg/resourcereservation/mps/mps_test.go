// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package mps

import (
	"os"
	"path/filepath"
	"testing"

	commonmps "github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

func TestConfigFromEnv(t *testing.T) {
	t.Setenv("KAI_MPS_ENABLED", "true")
	t.Setenv("KAI_MPS_BASE_DIR", "/custom/base")
	t.Setenv("GPU_GROUP", "grp-1")

	cfg := ConfigFromEnv()
	if !cfg.Enabled || cfg.BaseDir != "/custom/base" || cfg.GPUGroup != "grp-1" {
		t.Fatalf("ConfigFromEnv = %+v", cfg)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
	if cfg.PipeDir() != "/custom/base/grp-1/pipe" {
		t.Errorf("PipeDir = %q", cfg.PipeDir())
	}
	if cfg.LogDir() != "/custom/base/grp-1/log" {
		t.Errorf("LogDir = %q", cfg.LogDir())
	}
}

func TestConfigFromEnvDefaultsBaseDir(t *testing.T) {
	t.Setenv("KAI_MPS_ENABLED", "true")
	t.Setenv("KAI_MPS_BASE_DIR", "")
	t.Setenv("GPU_GROUP", "grp-1")

	if got := ConfigFromEnv().BaseDir; got != commonmps.DefaultBaseDir {
		t.Errorf("BaseDir = %q, want default %q", got, commonmps.DefaultBaseDir)
	}
}

func TestValidateRequiresGroupWhenEnabled(t *testing.T) {
	if err := (Config{Enabled: true, BaseDir: "/x"}).Validate(); err == nil {
		t.Errorf("expected error when enabled without GPU_GROUP")
	}
	if err := (Config{Enabled: false}).Validate(); err != nil {
		t.Errorf("disabled config should validate, got %v", err)
	}
}

func TestEnsureDirsCreatesPipeAndLog(t *testing.T) {
	base := t.TempDir()
	cfg := Config{Enabled: true, BaseDir: base, GPUGroup: "grp-abc"}
	m := NewManager(cfg)

	if err := m.ensureDirs(); err != nil {
		t.Fatalf("ensureDirs: %v", err)
	}
	for _, dir := range []string{cfg.PipeDir(), cfg.LogDir()} {
		info, err := os.Stat(dir)
		if err != nil {
			t.Fatalf("stat %q: %v", dir, err)
		}
		if !info.IsDir() {
			t.Errorf("%q is not a directory", dir)
		}
	}
	// Both dirs live under the per-group dir under base.
	if got := filepath.Dir(cfg.PipeDir()); got != commonmps.GroupDir(base, "grp-abc") {
		t.Errorf("pipe dir parent = %q, want %q", got, commonmps.GroupDir(base, "grp-abc"))
	}
}

func TestDaemonEnvSetsPipeAndLog(t *testing.T) {
	cfg := Config{Enabled: true, BaseDir: "/b", GPUGroup: "g"}
	env := NewManager(cfg).cfg.daemonEnv()

	wantPipe := commonmps.PipeDirectoryEnvVar + "=/b/g/pipe"
	wantLog := commonmps.LogDirectoryEnvVar + "=/b/g/log"
	var hasPipe, hasLog bool
	for _, e := range env {
		if e == wantPipe {
			hasPipe = true
		}
		if e == wantLog {
			hasLog = true
		}
	}
	if !hasPipe || !hasLog {
		t.Errorf("daemonEnv missing pipe/log: hasPipe=%v hasLog=%v", hasPipe, hasLog)
	}
}
