from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import AugmentationConfig
from .geometry import axis_angle_to_rotation, make_local_mirror_matrix


def sample_sequence_mirror(rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    mirror_x = bool(rng.random() < config.mirror_probability)
    mirror_y = bool(rng.random() < config.mirror_probability)
    return make_local_mirror_matrix(mirror_x, mirror_y)


def apply_position_noise(points: np.ndarray, rng: np.random.Generator, magnitude: float) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    noise = rng.uniform(-magnitude, magnitude, size=points.shape).astype(np.float32)
    return points + noise


def apply_random_tilt(points: np.ndarray, rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if max_degrees <= 0.0:
        return points.astype(np.float32, copy=False)
    angle = math.radians(rng.uniform(-max_degrees, max_degrees))
    axis_xy = rng.normal(size=2).astype(np.float32)
    axis = np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float32)
    rotation = axis_angle_to_rotation(axis, angle)
    return (points @ rotation.T).astype(np.float32, copy=False)


def _sample_patch_mask(
    points: np.ndarray,
    rng: np.random.Generator,
    patch_radius_range: Tuple[float, float],
    patch_count_range: Tuple[int, int],
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    mask = np.zeros(points.shape[0], dtype=bool)
    patch_count = int(rng.integers(patch_count_range[0], patch_count_range[1] + 1))
    for _ in range(patch_count):
        center = points[int(rng.integers(0, points.shape[0])), :2]
        radius = float(rng.uniform(*patch_radius_range))
        distances = np.linalg.norm(points[:, :2] - center[None, :], axis=1)
        mask |= distances <= radius
    return mask


def apply_height_noise(points: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    perturbed = points.copy()
    mask = _sample_patch_mask(
        perturbed,
        rng,
        config.height_patch_radius_range,
        config.height_patch_count,
    )
    if np.any(mask):
        delta = rng.uniform(-config.height_noise, config.height_noise, size=np.count_nonzero(mask))
        perturbed[mask, 2] += delta.astype(np.float32)
    return perturbed


def apply_pruning(points: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    mask = _sample_patch_mask(
        points,
        rng,
        config.prune_patch_radius_range,
        config.prune_patch_count,
    )
    return points[~mask].astype(np.float32, copy=False)


def add_outlier_clusters(points: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    clusters = [points]
    cluster_count = int(rng.integers(config.outlier_cluster_count[0], config.outlier_cluster_count[1] + 1))
    for _ in range(cluster_count):
        anchor = points[int(rng.integers(0, points.shape[0]))]
        radius = float(rng.uniform(*config.outlier_cluster_radius_range))
        num_points = int(
            rng.integers(
                config.outlier_points_per_cluster[0],
                config.outlier_points_per_cluster[1] + 1,
            )
        )
        offsets = rng.normal(size=(num_points, 3)).astype(np.float32)
        offsets /= np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-6)
        lengths = rng.uniform(0.0, radius, size=(num_points, 1)).astype(np.float32)
        clusters.append(anchor[None, :] + offsets * lengths)
    return np.concatenate(clusters, axis=0).astype(np.float32, copy=False)


def apply_measurement_augmentations(points: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    augmented = apply_position_noise(points, rng, config.position_noise)
    augmented = apply_random_tilt(augmented, rng, config.tilt_degrees)
    augmented = apply_height_noise(augmented, rng, config)
    augmented = apply_pruning(augmented, rng, config)
    augmented = add_outlier_clusters(augmented, rng, config)
    return augmented.astype(np.float32, copy=False)


def sample_pose_translation_noise(rng: np.random.Generator, magnitude: float) -> np.ndarray:
    noise = rng.uniform(-magnitude, magnitude, size=3).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = noise
    return transform



