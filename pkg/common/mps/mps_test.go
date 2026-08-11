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

func TestActiveThreadPercentageValueFromPercent(t *testing.T) {
	cases := map[int]string{
		80:  "80",
		25:  "25",
		1:   "1",
		100: "100",
		0:   "1",   // clamp floor
		-5:  "1",   // clamp floor
		150: "100", // clamp ceiling
	}
	for pct, want := range cases {
		if got := ActiveThreadPercentageValueFromPercent(pct); got != want {
			t.Errorf("ActiveThreadPercentageValueFromPercent(%d) = %q, want %q", pct, got, want)
		}
	}
}

func TestParseActiveThreadPercentage(t *testing.T) {
	valid := map[string]int{
		"1":    1,
		"80":   80,
		"100":  100,
		" 50 ": 50, // surrounding whitespace tolerated
	}
	for in, want := range valid {
		got, err := ParseActiveThreadPercentage(in)
		if err != nil {
			t.Errorf("ParseActiveThreadPercentage(%q) unexpected error: %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("ParseActiveThreadPercentage(%q) = %d, want %d", in, got, want)
		}
	}

	for _, in := range []string{"", "abc", "0", "-1", "101", "150", "50.5", "50%"} {
		if _, err := ParseActiveThreadPercentage(in); err == nil {
			t.Errorf("ParseActiveThreadPercentage(%q) expected error, got nil", in)
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
