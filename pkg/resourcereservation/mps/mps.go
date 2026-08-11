// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package mps runs, inside the GPU reservation pod, the per-GPU
// nvidia-cuda-mps-control daemon that the fraction (client) pods connect to.
//
// The reservation pod is the natural host for the daemon: it already holds the
// whole physical GPU (so Slurm accounts the GPU as one gres/gpu job and no
// Slurm job runs on it), and its lifetime is exactly "this GPU is in fractional
// use". When the reservation is torn down the daemon dies with it and the GPU
// returns to Slurm as a whole device.
//
// The daemon creates its pipe/log directories under a hostPath base shared with
// the client pods, keyed by gpu-group so multiple fractional GPUs on one node
// never collide. GPU compute mode is left at DEFAULT: MPS is opt-in per client
// via CUDA_MPS_PIPE_DIRECTORY, so the reservation pod's own direct use of the
// device and any (non-fractional) direct users are unaffected.
package mps

import (
	"context"
	"fmt"
	"os"
	"os/exec"

	"sigs.k8s.io/controller-runtime/pkg/log"

	commonmps "github.com/kai-scheduler/KAI-scheduler/pkg/common/mps"
)

const (
	controlBinary = "nvidia-cuda-mps-control"
	dirPerm       = 0o777
)

// Config is resolved from the environment the binder injects onto the
// reservation pod (see resourcereservation.applyMPSHosting).
type Config struct {
	Enabled  bool
	BaseDir  string
	GPUGroup string
}

// ConfigFromEnv reads the MPS hosting configuration from the pod's environment.
func ConfigFromEnv() Config {
	cfg := Config{
		Enabled:  os.Getenv("KAI_MPS_ENABLED") == "true",
		BaseDir:  os.Getenv("KAI_MPS_BASE_DIR"),
		GPUGroup: os.Getenv("GPU_GROUP"),
	}
	if cfg.BaseDir == "" {
		cfg.BaseDir = commonmps.DefaultBaseDir
	}
	return cfg
}

// Validate ensures the config is complete enough to host a daemon.
func (c Config) Validate() error {
	if !c.Enabled {
		return nil
	}
	if c.GPUGroup == "" {
		return fmt.Errorf("KAI_MPS_ENABLED is set but GPU_GROUP is empty")
	}
	return nil
}

// PipeDir / LogDir are this reservation's per-GPU directories, computed with the
// SAME shared contract the binder plugin uses to set the client env vars.
func (c Config) PipeDir() string { return commonmps.PipeDir(c.BaseDir, c.GPUGroup) }
func (c Config) LogDir() string  { return commonmps.LogDir(c.BaseDir, c.GPUGroup) }

// daemonEnv is the environment for the control daemon subprocess only. We do
// NOT export CUDA_MPS_PIPE_DIRECTORY into the reservation container at large,
// so the reservation's own NVML/CUDA use never routes through MPS.
func (c Config) daemonEnv() []string {
	return append(os.Environ(),
		commonmps.PipeDirectoryEnvVar+"="+c.PipeDir(),
		commonmps.LogDirectoryEnvVar+"="+c.LogDir(),
	)
}

// Manager owns the lifecycle of the control daemon subprocess.
type Manager struct {
	cfg Config
	cmd *exec.Cmd
}

func NewManager(cfg Config) *Manager {
	return &Manager{cfg: cfg}
}

// Start creates the pipe/log directories and launches nvidia-cuda-mps-control
// in daemon mode. The daemon runs as the reservation pod's user (root is
// recommended so it can serve fraction clients of any UID); the pipe directory
// is world-accessible so clients can reach the control socket.
func (m *Manager) Start(ctx context.Context) error {
	logger := log.FromContext(ctx)

	if err := m.ensureDirs(); err != nil {
		return err
	}

	// -d: run the control daemon in the background; it returns immediately and
	// spawns the MPS server lazily on the first client connection.
	cmd := exec.CommandContext(ctx, controlBinary, "-d")
	cmd.Env = m.cfg.daemonEnv()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to start %s: %w", controlBinary, err)
	}
	m.cmd = cmd

	logger.Info("started MPS control daemon",
		"gpuGroup", m.cfg.GPUGroup, "pipeDir", m.cfg.PipeDir(), "logDir", m.cfg.LogDir())
	return nil
}

// ensureDirs creates the per-GPU pipe/log directories with world access so
// cross-UID clients can reach the control socket.
func (m *Manager) ensureDirs() error {
	for _, dir := range []string{m.cfg.PipeDir(), m.cfg.LogDir()} {
		if err := os.MkdirAll(dir, dirPerm); err != nil {
			return fmt.Errorf("failed to create MPS dir %q: %w", dir, err)
		}
		// MkdirAll honors umask; force world-access explicitly.
		if err := os.Chmod(dir, dirPerm); err != nil {
			return fmt.Errorf("failed to chmod MPS dir %q: %w", dir, err)
		}
	}
	return nil
}

// Stop asks the control daemon to quit, releasing the GPU cleanly. Best-effort:
// the daemon also dies when the reservation pod (and hostPath) go away.
func (m *Manager) Stop(ctx context.Context) error {
	logger := log.FromContext(ctx)
	cmd := exec.Command(controlBinary)
	cmd.Env = m.cfg.daemonEnv()
	cmd.Stdin = quitReader()
	if err := cmd.Run(); err != nil {
		logger.Error(err, "failed to quit MPS control daemon", "gpuGroup", m.cfg.GPUGroup)
		return err
	}
	logger.Info("stopped MPS control daemon", "gpuGroup", m.cfg.GPUGroup)
	return nil
}

func quitReader() *os.File {
	r, w, err := os.Pipe()
	if err != nil {
		return nil
	}
	go func() {
		defer func() { _ = w.Close() }()
		_, _ = w.WriteString("quit\n")
	}()
	return r
}
