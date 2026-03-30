from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


DEFAULT_BOUNDS_MIN = np.array([-1.6, -1.6, -1.6], dtype=np.float32)
DEFAULT_BOUNDS_MAX = np.array([1.6, 1.6, 1.6], dtype=np.float32)
MEASUREMENT_COLOR = np.array([0.96, 0.55, 0.18], dtype=np.float64)
GROUND_TRUTH_COLOR = np.array([0.12, 0.62, 0.98], dtype=np.float64)
POSE_PATH_COLOR = np.array([0.10, 0.80, 0.25], dtype=np.float64)
BOUNDS_COLOR = np.array([0.85, 0.20, 0.20], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize packed Isaac Sim trajectory point clouds with Open3D."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=Path("data/trajectory_000.npz"),
        help="Packed trajectory .npz file.",
    )
    parser.add_argument(
        "--timestep",
        default="0",
        help="Timestep index to visualize, or 'all' to accumulate all timesteps.",
    )
    parser.add_argument(
        "--space",
        choices=("local", "world"),
        default="local",
        help="Visualize clouds in robot-local or world coordinates.",
    )
    parser.add_argument(
        "--hide-measurement",
        action="store_true",
        help="Do not show measurement point clouds.",
    )
    parser.add_argument(
        "--hide-ground-truth",
        action="store_true",
        help="Do not show ground-truth point clouds.",
    )
    parser.add_argument(
        "--measurement-max-points",
        type=int,
        default=200000,
        help="Maximum measurement points to send to Open3D after random downsampling.",
    )
    parser.add_argument(
        "--ground-truth-max-points",
        type=int,
        default=200000,
        help="Maximum ground-truth points to send to Open3D after random downsampling.",
    )
    parser.add_argument(
        "--show-poses",
        action="store_true",
        help="Show the robot base path in world space.",
    )
    parser.add_argument(
        "--show-bounds",
        action="store_true",
        help="Show the 3.2m x 3.2m x 3.2m crop bounds.",
    )
    parser.add_argument(
        "--bounds-min",
        nargs=3,
        type=float,
        default=DEFAULT_BOUNDS_MIN.tolist(),
        metavar=("X", "Y", "Z"),
        help="Minimum crop bounds for the local reconstruction window.",
    )
    parser.add_argument(
        "--bounds-max",
        nargs=3,
        type=float,
        default=DEFAULT_BOUNDS_MAX.tolist(),
        metavar=("X", "Y", "Z"),
        help="Maximum crop bounds for the local reconstruction window.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D point size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for downsampling.",
    )
    return parser.parse_args()


def load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "open3d is not installed. Install it with 'pip install open3d' first."
        ) from exc
    return o3d


def parse_timestep(value: str, num_steps: int) -> List[int]:
    if value.lower() == "all":
        return list(range(num_steps))
    timestep = int(value)
    if timestep < 0:
        timestep += num_steps
    if timestep < 0 or timestep >= num_steps:
        raise ValueError(f"timestep {value} is out of range for sequence length {num_steps}")
    return [timestep]


def unpack_cloud(points: np.ndarray, splits: np.ndarray, timestep: int) -> np.ndarray:
    start = int(splits[timestep])
    end = int(splits[timestep + 1])
    return points[start:end].astype(np.float32, copy=False)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotated = points @ transform[:3, :3].T
    translated = rotated + transform[:3, 3]
    return translated.astype(np.float32, copy=False)


def downsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices].astype(np.float32, copy=False)


def make_point_cloud(o3d, points: np.ndarray, color: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.paint_uniform_color(color.tolist())
    return cloud


def make_axis_aligned_bounds(o3d, bounds_min: np.ndarray, bounds_max: np.ndarray):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bounds_min.astype(np.float64, copy=False),
        max_bound=bounds_max.astype(np.float64, copy=False),
    )
    bbox.color = BOUNDS_COLOR.tolist()
    return bbox


def make_oriented_bounds_lineset(o3d, bounds_min: np.ndarray, bounds_max: np.ndarray, pose: np.ndarray):
    corners_local = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float32,
    )
    corners_world = transform_points(corners_local, pose)
    lines = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    colors = np.repeat(BOUNDS_COLOR[None, :], lines.shape[0], axis=0)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world.astype(np.float64, copy=False))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def make_pose_path(o3d, poses: np.ndarray):
    if poses.shape[0] == 0:
        return None
    positions = poses[:, :3, 3].astype(np.float64, copy=False)
    path = o3d.geometry.PointCloud()
    path.points = o3d.utility.Vector3dVector(positions)
    path.paint_uniform_color(POSE_PATH_COLOR.tolist())
    if poses.shape[0] < 2:
        return path

    lines = np.array([[i, i + 1] for i in range(poses.shape[0] - 1)], dtype=np.int32)
    colors = np.repeat(POSE_PATH_COLOR[None, :], lines.shape[0], axis=0)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return [path, line_set]


def flatten_geometries(items: Iterable):
    geometries = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            geometries.extend(item)
        else:
            geometries.append(item)
    return geometries


def main() -> None:
    args = parse_args()
    o3d = load_open3d()

    if args.hide_measurement and args.hide_ground_truth:
        raise SystemExit("Nothing to visualize: both measurement and ground truth are hidden.")

    trajectory_path = args.trajectory.expanduser().resolve()
    if not trajectory_path.exists():
        raise SystemExit(f"Trajectory file does not exist: {trajectory_path}")

    data = np.load(trajectory_path)
    poses = data["poses"].astype(np.float32, copy=False)
    measurement_points = data["measurement_points"].astype(np.float32, copy=False)
    measurement_splits = data["measurement_splits"].astype(np.int32, copy=False)
    ground_truth_points = data["ground_truth_points"].astype(np.float32, copy=False)
    ground_truth_splits = data["ground_truth_splits"].astype(np.int32, copy=False)

    num_steps = int(poses.shape[0])
    selected_steps = parse_timestep(args.timestep, num_steps)
    if len(selected_steps) > 1 and args.space == "local":
        raise SystemExit("Use --space world with --timestep all. Stacking multiple local frames is not meaningful.")

    rng = np.random.default_rng(args.seed)
    bounds_min = np.asarray(args.bounds_min, dtype=np.float32)
    bounds_max = np.asarray(args.bounds_max, dtype=np.float32)

    measurement_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    for timestep in selected_steps:
        measurement_local = unpack_cloud(measurement_points, measurement_splits, timestep)
        gt_local = unpack_cloud(ground_truth_points, ground_truth_splits, timestep)
        if args.space == "world":
            measurement_chunks.append(transform_points(measurement_local, poses[timestep]))
            gt_chunks.append(transform_points(gt_local, poses[timestep]))
        else:
            measurement_chunks.append(measurement_local)
            gt_chunks.append(gt_local)

    measurement_cloud = (
        np.concatenate(measurement_chunks, axis=0) if measurement_chunks else np.empty((0, 3), dtype=np.float32)
    )
    gt_cloud = np.concatenate(gt_chunks, axis=0) if gt_chunks else np.empty((0, 3), dtype=np.float32)

    measurement_cloud = downsample_points(measurement_cloud, args.measurement_max_points, rng)
    gt_cloud = downsample_points(gt_cloud, args.ground_truth_max_points, rng)

    geometries = []
    if not args.hide_measurement and measurement_cloud.size > 0:
        geometries.append(make_point_cloud(o3d, measurement_cloud, MEASUREMENT_COLOR))
    if not args.hide_ground_truth and gt_cloud.size > 0:
        geometries.append(make_point_cloud(o3d, gt_cloud, GROUND_TRUTH_COLOR))

    if args.show_bounds:
        if args.space == "local":
            geometries.append(make_axis_aligned_bounds(o3d, bounds_min, bounds_max))
        else:
            for timestep in selected_steps:
                geometries.append(make_oriented_bounds_lineset(o3d, bounds_min, bounds_max, poses[timestep]))

    if args.show_poses and args.space == "world":
        geometries.extend(flatten_geometries([make_pose_path(o3d, poses[selected_steps])]))

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    print(f"trajectory: {trajectory_path}")
    print(f"timesteps: {num_steps}, selected: {selected_steps}")
    print(f"space: {args.space}")
    print(f"measurement points shown: {measurement_cloud.shape[0]}")
    print(f"ground-truth points shown: {gt_cloud.shape[0]}")

    vis = o3d.visualization.Visualizer()
    window_name = f"Open3D Trajectory Viewer - {trajectory_path.name}"
    vis.create_window(window_name=window_name, width=1600, height=1000)
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        render_option = vis.get_render_option()
        render_option.point_size = float(args.point_size)
        render_option.background_color = np.array([0.04, 0.04, 0.04], dtype=np.float64)
        vis.run()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
