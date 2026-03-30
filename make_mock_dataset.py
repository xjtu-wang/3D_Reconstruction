from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


VOXEL_EXTENT = 1.6


def terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    stairs = 0.18 * (x > 0.4) + 0.12 * (x > 1.0)
    waves = 0.08 * np.sin(1.5 * x) + 0.05 * np.cos(2.0 * y)
    bump = 0.10 * np.exp(-((x - 0.6) ** 2 + (y + 0.25) ** 2) / 0.08)
    return (stairs + waves + bump).astype(np.float32)


def yaw_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_pose(timestep: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = yaw_matrix(0.03 * np.sin(0.15 * timestep))
    pose[:3, 3] = np.array([0.07 * timestep, 0.1 * np.sin(0.08 * timestep), 0.55], dtype=np.float32)
    return pose


def sample_ground_truth_local(pose: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    local_xy = rng.uniform(-VOXEL_EXTENT, VOXEL_EXTENT, size=(num_points, 2)).astype(np.float32)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    world_xy = local_xy @ rotation[:2, :2].T + translation[:2]
    world_z = terrain_height(world_xy[:, 0], world_xy[:, 1])
    world_points = np.column_stack((world_xy, world_z)).astype(np.float32)

    local_points = (world_points - translation) @ rotation
    return local_points.astype(np.float32, copy=False)


def sample_measurement_from_ground_truth(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    keep_mask = rng.random(points.shape[0]) > 0.35
    measured = points[keep_mask].copy()
    if measured.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    occlusion_band = measured[:, 0] > rng.uniform(0.2, 0.8)
    measured = measured[~occlusion_band]
    if measured.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    measured += rng.normal(scale=np.array([0.01, 0.01, 0.015], dtype=np.float32), size=measured.shape).astype(np.float32)
    return measured.astype(np.float32, copy=False)


def pack_clouds(clouds: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    splits = [0]
    packed = []
    for cloud in clouds:
        packed.append(cloud.astype(np.float32, copy=False))
        splits.append(splits[-1] + cloud.shape[0])
    packed_points = np.concatenate(packed, axis=0) if packed else np.empty((0, 3), dtype=np.float32)
    return packed_points, np.asarray(splits, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy trajectories for the terrain reconstruction pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("mock_dataset"))
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--gt-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for trajectory_index in range(args.num_trajectories):
        measurements = []
        ground_truth = []
        poses = []
        for timestep in range(args.timesteps):
            pose = make_pose(trajectory_index * args.timesteps + timestep)
            gt_local = sample_ground_truth_local(pose, args.gt_points, rng)
            measurement_local = sample_measurement_from_ground_truth(gt_local, rng)
            poses.append(pose)
            ground_truth.append(gt_local)
            measurements.append(measurement_local)

        measurement_points, measurement_splits = pack_clouds(measurements)
        ground_truth_points, ground_truth_splits = pack_clouds(ground_truth)
        np.savez_compressed(
            args.output_dir / f"trajectory_{trajectory_index:03d}.npz",
            poses=np.stack(poses, axis=0),
            measurement_points=measurement_points,
            measurement_splits=measurement_splits,
            ground_truth_points=ground_truth_points,
            ground_truth_splits=ground_truth_splits,
        )


if __name__ == "__main__":
    main()
