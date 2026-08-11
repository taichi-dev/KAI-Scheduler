// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package mps

import "testing"

func TestPipeAndLogDir(t *testing.T) {
	base := "/var/run/kai-scheduler/mps"
	group := "abc-123"

	if got, want := GroupDir(base, group), "/var/run/kai-scheduler/mps/abc-123"; got != want {
		t.Errorf("GroupDir = %q, want %q", got, want)
	}
	if got, want := PipeDir(base, group), "/var/run/kai-scheduler/mps/abc-123/pipe"; got != want {
		t.Errorf("PipeDir = %q, want %q", got, want)
	}
	if got, want := LogDir(base, group), "/var/run/kai-scheduler/mps/abc-123/log"; got != want {
		t.Errorf("LogDir = %q, want %q", got, want)
	}
}

func TestDistinctGroupsGetDistinctDirs(t *testing.T) {
	base := "/mps"
	if PipeDir(base, "group-a") == PipeDir(base, "group-b") {
		t.Errorf("distinct gpu-groups must map to distinct pipe dirs")
	}
}

func TestSanitizeSegmentBlocksTraversal(t *testing.T) {
	base := "/mps"
	// A hostile/malformed group must not escape the base dir.
	got := GroupDir(base, "../../etc")
	if got != "/mps/------etc" && got != "/mps/--/--/etc" {
		// The exact form is not important; the invariant is that the result stays
		// under base and contains no ".." traversal segment.
		if containsDotDot(got) || !hasPrefix(got, "/mps/") {
			t.Errorf("GroupDir(%q) = %q escaped base or kept traversal", "../../etc", got)
		}
	}
}

func TestPinnedDeviceMemLimitValue(t *testing.T) {
	// 0.5 x 15360 MiB (T4) = 7680 MiB, device ordinal 0 (single visible device).
	if got, want := PinnedDeviceMemLimitValue(7680), "0=7680M"; got != want {
		t.Errorf("PinnedDeviceMemLimitValue = %q, want %q", got, want)
	}
}

func TestActiveThreadPercentageValue(t *testing.T) {
	cases := map[float64]string{
		0.5:   "50",
		0.25:  "25",
		0.999: "100",
		0.001: "1", // clamp floor
		1.5:   "100",
	}
	for portion, want := range cases {
		if got := ActiveThreadPercentageValue(portion); got != want {
			t.Errorf("ActiveThreadPercentageValue(%v) = %q, want %q", portion, got, want)
		}
	}
}

func containsDotDot(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '.' && s[i+1] == '.' {
			return true
		}
	}
	return false
}

func hasPrefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }
