// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package mps holds the shared contract for MPS-based VRAM enforcement of
// fractional GPU pods. Three components depend on these definitions and MUST
// agree on them:
//
//   - the admission plugin (pkg/admission/webhook/v1alpha2/mps): injects the
//     CUDA_MPS_* env var references and the hostPath pipe-dir mount into the
//     fraction container at pod-creation time;
//   - the binder plugin (pkg/binder/plugins/mps): at bind time computes the
//     per-fraction values (pipe dir keyed by gpu-group, pinned-memory limit =
//     fraction x node VRAM) and writes them into the shared-gpu ConfigMap;
//   - the resource-reservation entrypoint (pkg/resourcereservation/mps): runs
//     the per-GPU nvidia-cuda-mps-control daemon and creates the pipe/log dirs
//     under the SAME base path, keyed by the SAME gpu-group.
//
// Only WORKLOAD (fraction) pods ever get the CUDA_MPS_* env vars, so only they
// route through the MPS server. Reservation pods and Slurm pods carry no such
// env, so they use the physical device directly (see design notes).
package mps

import (
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// DefaultBaseDir is the node-local base directory (shared via hostPath) that
	// holds one subdirectory per fractional GPU. Both the MPS control daemon
	// (reservation pod) and the MPS clients (fraction pods) mount this base at
	// the same in-container path, so the pipe/log paths resolve identically.
	DefaultBaseDir = "/var/run/kai-scheduler/mps"

	// VolumeName is the name of the hostPath volume mounted into both the
	// reservation pod (daemon) and the fraction pods (clients).
	VolumeName = "kai-mps-pipe"

	// PipeDirectoryEnvVar tells a CUDA client where to find the MPS control
	// daemon's named pipes / UNIX domain sockets. Setting it is what makes a
	// process an MPS client; leaving it unset means the process uses the GPU
	// directly. This is the opt-in switch that keeps reservation/slurm pods off
	// MPS.
	PipeDirectoryEnvVar = "CUDA_MPS_PIPE_DIRECTORY"

	// LogDirectoryEnvVar is where the MPS control daemon and server write logs.
	LogDirectoryEnvVar = "CUDA_MPS_LOG_DIRECTORY"

	// PinnedDeviceMemLimitEnvVar caps the device memory an MPS client may
	// allocate. Enforced by the MPS server (CUDA 11.5+); allocations past the
	// cap fail with out-of-memory. This is the VRAM enforcement mechanism.
	PinnedDeviceMemLimitEnvVar = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"

	// ActiveThreadPercentageEnvVar caps the fraction of GPU SMs an MPS client
	// may use on Volta+ hardware. Optional compute-dimension enforcement.
	ActiveThreadPercentageEnvVar = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

	// pipeSubdir / logSubdir are the per-group leaf directories.
	pipeSubdir = "pipe"
	logSubdir  = "log"
)

// GroupDir returns the per-GPU directory for a gpu-group under base. The
// gpu-group is a 1:1 stand-in for the physical GPU (it maps to exactly one
// reservation pod, which holds exactly one physical GPU), so distinct GPUs get
// distinct directories and never collide, even on the same node.
func GroupDir(base, gpuGroup string) string {
	return filepath.Join(base, sanitizeSegment(gpuGroup))
}

// PipeDir is the CUDA_MPS_PIPE_DIRECTORY value for a gpu-group.
func PipeDir(base, gpuGroup string) string {
	return filepath.Join(GroupDir(base, gpuGroup), pipeSubdir)
}

// LogDir is the CUDA_MPS_LOG_DIRECTORY value for a gpu-group.
func LogDir(base, gpuGroup string) string {
	return filepath.Join(GroupDir(base, gpuGroup), logSubdir)
}

// PinnedDeviceMemLimitValue formats a per-client pinned-memory cap in MiB.
// The client always sees exactly one visible device (NVIDIA_VISIBLE_DEVICES is
// a single GPU UUID), so from the client's point of view the device ordinal is
// always 0.
func PinnedDeviceMemLimitValue(mib int64) string {
	return fmt.Sprintf("0=%dM", mib)
}

// ActiveThreadPercentageValue converts a GPU fraction (0,1) into an integer
// percentage string clamped to [1,100].
func ActiveThreadPercentageValue(portion float64) string {
	pct := int(math.Round(portion * 100))
	return ActiveThreadPercentageValueFromPercent(pct)
}

// ActiveThreadPercentageValueFromPercent formats an explicit SM percentage as
// the CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value, clamped to [1,100] to mirror
// ActiveThreadPercentageValue. Used for the per-pod gpu-sm-percent path where
// the SM% is decoupled from the VRAM fraction.
func ActiveThreadPercentageValueFromPercent(pct int) string {
	if pct < 1 {
		pct = 1
	}
	if pct > 100 {
		pct = 100
	}
	return fmt.Sprintf("%d", pct)
}

// ParseActiveThreadPercentage parses a per-pod gpu-sm-percent annotation value:
// a base-10 integer that must fall in [1,100]. It returns an error for a
// non-integer or out-of-range value so an explicit but malformed request fails
// loudly rather than being silently ignored.
func ParseActiveThreadPercentage(s string) (int, error) {
	pct, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("active thread percentage %q must be an integer", s)
	}
	if pct < 1 || pct > 100 {
		return 0, fmt.Errorf("active thread percentage %d must be in [1,100]", pct)
	}
	return pct, nil
}

// sanitizeSegment defends the path against a gpu-group value that (against the
// Kubernetes label grammar) contains a path separator or traversal.
func sanitizeSegment(s string) string {
	s = strings.ReplaceAll(s, "/", "-")
	s = strings.ReplaceAll(s, "..", "-")
	return s
}
