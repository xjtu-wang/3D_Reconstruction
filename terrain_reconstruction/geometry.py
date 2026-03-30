from __future__ import annotations

import math

import numpy as np


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotated = points @ transform[:3, :3].T
    translated = rotated + transform[:3, 3]
    return translated.astype(np.float32, copy=False)


def relative_transform(previous_pose_world_from_base: np.ndarray, current_pose_world_from_base: np.ndarray) -> np.ndarray:
    return invert_transform(current_pose_world_from_base) @ previous_pose_world_from_base


def make_local_mirror_matrix(mirror_x: bool, mirror_y: bool) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = -1.0 if mirror_x else 1.0
    matrix[1, 1] = -1.0 if mirror_y else 1.0
    return matrix


def apply_local_mirror(points: np.ndarray, mirror_matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    mirrored = points.copy()
    mirrored[:, 0] *= mirror_matrix[0, 0]
    mirrored[:, 1] *= mirror_matrix[1, 1]
    return mirrored


def mirror_relative_transform(transform: np.ndarray, mirror_matrix: np.ndarray) -> np.ndarray:
    return mirror_matrix @ transform @ mirror_matrix


def axis_angle_to_rotation(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = axis.astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float32,
    )
