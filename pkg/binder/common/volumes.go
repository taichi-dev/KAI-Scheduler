// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package common

import (
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
)

func SetConfigMapVolume(pod *v1.Pod, configMapName string) {
	volumeName := GetConfigVolumeName(configMapName)
	addConfigMapVolume(&pod.Spec, volumeName, configMapName)
}

func addConfigMapVolume(podSpec *v1.PodSpec, volumeName string, configMapName string) {
	if podSpec.Volumes == nil {
		podSpec.Volumes = make([]v1.Volume, 0)
	}

	updatedVolumes := make([]v1.Volume, 0)
	for _, volume := range podSpec.Volumes {
		if volume.Name != volumeName {
			updatedVolumes = append(updatedVolumes, volume)
		}
	}
	podSpec.Volumes = updatedVolumes

	volume := v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{
					Name: configMapName,
				},
			},
		},
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

// SetHostPathVolume idempotently adds a hostPath volume to the pod (replacing
// any existing volume of the same name) and mounts it into the given container
// at mountPath. The host directory is created if missing (DirectoryOrCreate).
// Used to share the MPS pipe/log directory between the reservation pod (which
// hosts the MPS control daemon) and the fraction pods (MPS clients).
func SetHostPathVolume(pod *v1.Pod, container *v1.Container, volumeName, hostPath, mountPath string) {
	if pod.Spec.Volumes == nil {
		pod.Spec.Volumes = make([]v1.Volume, 0)
	}

	updatedVolumes := make([]v1.Volume, 0, len(pod.Spec.Volumes))
	for _, volume := range pod.Spec.Volumes {
		if volume.Name != volumeName {
			updatedVolumes = append(updatedVolumes, volume)
		}
	}
	hostPathType := v1.HostPathDirectoryOrCreate
	pod.Spec.Volumes = append(updatedVolumes, v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: hostPath,
				Type: &hostPathType,
			},
		},
	})

	AddVolumeMountToContainer(container, volumeName, mountPath)
}

// AddVolumeMountToContainer idempotently adds a volume mount (by name) to the
// container.
func AddVolumeMountToContainer(container *v1.Container, volumeName, mountPath string) {
	mounts := make([]v1.VolumeMount, 0, len(container.VolumeMounts))
	for _, m := range container.VolumeMounts {
		if m.Name != volumeName {
			mounts = append(mounts, m)
		}
	}
	container.VolumeMounts = append(mounts, v1.VolumeMount{
		Name:      volumeName,
		MountPath: mountPath,
	})
}

var invalidDNSLabelChars = regexp.MustCompile(`[^a-z0-9-]+`)

func GetConfigVolumeName(configMapName string) string {
	// ConfigMap names may be DNS subdomains, but volume names must be DNS labels.
	volumeName := strings.ToLower(configMapName + "-vol")
	volumeName = invalidDNSLabelChars.ReplaceAllString(volumeName, "-")
	return strings.Trim(volumeName, "-")
}
