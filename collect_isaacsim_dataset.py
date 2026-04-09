from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


GRID_BOUNDS_MIN = np.array([-1.6, -1.6, -1.6], dtype=np.float32)
GRID_BOUNDS_MAX = np.array([1.6, 1.6, 1.6], dtype=np.float32)
DEFAULT_WALL_HEIGHT = 1.0
DEFAULT_WALL_THICKNESS = 0.12
DEFAULT_POLE_HEIGHT = 1.0
DEFAULT_BODY_HEIGHT = 0.55


@dataclass(frozen=True)
class CameraMount:
    name: str
    translation_in_base: np.ndarray
    forward_in_base: np.ndarray


@dataclass(frozen=True)
class BoxPrimitive:
    name: str
    center: np.ndarray
    size: np.ndarray
    yaw_radians: float
    top_only: bool = False


@dataclass(frozen=True)
class SceneDescription:
    scene_yaw_radians: float
    path_length: float
    corridor_width: float
    stair_start_x: float
    stair_tread: float
    stair_rise: float
    stair_steps: int
    stair_width: float
    landing_length: float
    ground_size: np.ndarray
    boxes_local: Tuple[BoxPrimitive, ...]

    @property
    def stair_end_x(self) -> float:
        return self.stair_start_x + self.stair_steps * self.stair_tread

    def support_height(self, x_local: float, y_local: float) -> float:
        if abs(y_local) > 0.5 * self.stair_width:
            return 0.0
        for step_index in range(self.stair_steps):
            step_start = self.stair_start_x + step_index * self.stair_tread
            step_end = step_start + self.stair_tread
            if step_start <= x_local < step_end:
                return float((step_index + 1) * self.stair_rise)
        landing_end = self.stair_end_x + self.landing_length
        if self.stair_end_x <= x_local < landing_end:
            return float(self.stair_steps * self.stair_rise)
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Isaac Sim 4.5 terrain reconstruction trajectories with an ANYmal C "
            "robot and four downward-facing depth cameras."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("isaacsim_dataset"))
    parser.add_argument("--num-trajectories", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-ui", action="store_true", help="Run Isaac Sim with a visible UI.")
    parser.add_argument("--show-robot", action="store_true", help="Keep the robot visible to the cameras.")
    parser.add_argument("--asset-root", type=str, default=None, help="Override Isaac Sim assets root.")
    parser.add_argument("--robot-usd", type=str, default=None, help="Explicit ANYmal C USD path.")
    parser.add_argument(
        "--motion-source",
        type=str,
        choices=("policy_rollout", "velocity_fallback"),
        default="policy_rollout",
        help=(
            "Robot motion source. 'policy_rollout' uses an Isaac Lab ANYmal rough-terrain policy on a USD terrain scene. "
            "'velocity_fallback' keeps the legacy kinematic reachable-trajectory baseline."
        ),
    )
    parser.add_argument(
        "--policy-task",
        type=str,
        default="Isaac-Velocity-Rough-Anymal-C-v0",
        help="Isaac Lab task identifier for metadata and validation when --motion-source policy_rollout is used.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an exported TorchScript/JIT policy checkpoint for Isaac Lab policy rollout. "
            "Required when --motion-source policy_rollout is used."
        ),
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="cuda:0",
        help="Torch / Isaac Lab device used for policy inference, for example 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--policy-heading-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
        metavar=("MIN", "MAX"),
        help=(
            "Absolute heading range in degrees for sampled high-level policy commands. "
            "Used only when --motion-source policy_rollout is active."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=800)
    parser.add_argument(
        "--camera-pixel-stride",
        type=int,
        default=4,
        help=(
            "Stride used when sampling pixels from the camera depth image before back-projection. "
            "Using a stride reduces redundant neighboring pixels that collapse into the same local voxels."
        ),
    )
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--capture-dt", type=float, default=0.10)
    parser.add_argument("--path-length", type=float, default=8.0)
    parser.add_argument("--nominal-base-height", type=float, default=DEFAULT_BODY_HEIGHT)
    parser.add_argument("--ground-truth-spacing", type=float, default=0.05)
    parser.add_argument("--max-measurement-points", type=int, default=20000)
    parser.add_argument(
        "--max-camera-points",
        type=int,
        default=65536,
        help=(
            "Maximum number of depth pixels to back-project per camera after applying --camera-pixel-stride. "
            "This is separate from the final fused measurement cap."
        ),
    )
    parser.add_argument("--max-ground-truth-points", type=int, default=40000)
    parser.add_argument(
        "--measurement-voxel-size",
        type=float,
        default=0.02,
        help=(
            "Robot-local voxel size for centroid downsampling of the fused measurement cloud before saving. "
            "This removes heavy pixel-level duplication while preserving geometric coverage."
        ),
    )
    parser.add_argument(
        "--raw-camera-voxel-size",
        type=float,
        default=0.02,
        help=(
            "Robot-local voxel size for centroid downsampling of each individual camera cloud when "
            "--save-raw-camera-clouds is enabled."
        ),
    )
    parser.add_argument("--camera-min-range", type=float, default=0.10)
    parser.add_argument("--camera-max-range", type=float, default=6.0)
    parser.add_argument(
        "--sensor-warmup-steps",
        type=int,
        default=24,
        help="Extra render steps before the first capture so camera annotators have valid frames.",
    )
    parser.add_argument(
        "--sensor-retry-steps",
        type=int,
        default=12,
        help="Extra render steps to wait when a camera frame is not ready yet.",
    )
    parser.add_argument("--wall-height", type=float, default=DEFAULT_WALL_HEIGHT)
    parser.add_argument("--wall-thickness", type=float, default=DEFAULT_WALL_THICKNESS)
    parser.add_argument("--camera-mount-height", type=float, default=0.12)
    parser.add_argument("--camera-mount-x", type=float, default=0.34)
    parser.add_argument("--camera-mount-y", type=float, default=0.18)
    parser.add_argument(
        "--camera-down-tilt-deg",
        type=float,
        default=30.0,
        help="Paper-faithful default: 30 degrees downward tilt.",
    )
    parser.add_argument(
        "--command-resample-steps",
        type=int,
        default=8,
        help=(
            "Saved-timestep interval between resampling high-level motion commands. "
            "Used by both policy_rollout command injection and velocity_fallback trajectory generation."
        ),
    )
    parser.add_argument(
        "--command-lin-vel-x-range",
        type=float,
        nargs=2,
        default=(0.30, 1.00),
        metavar=("MIN", "MAX"),
        help="Sampled forward-velocity range in m/s for motion-command generation.",
    )
    parser.add_argument(
        "--command-lin-vel-y-range",
        type=float,
        nargs=2,
        default=(-0.20, 0.20),
        metavar=("MIN", "MAX"),
        help="Sampled lateral-velocity range in m/s for motion-command generation.",
    )
    parser.add_argument(
        "--command-yaw-rate-range-deg",
        type=float,
        nargs=2,
        default=(-20.0, 20.0),
        metavar=("MIN", "MAX"),
        help="Sampled yaw-rate range in deg/s for velocity_fallback trajectory generation.",
    )
    parser.add_argument(
        "--save-raw-camera-clouds",
        action="store_true",
        help=(
            "Store each camera's robot-local point cloud in the trajectory .npz alongside the fused "
            "measurement_points arrays expected by train.py."
        ),
    )
    parser.add_argument(
        "--debug-camera",
        action="store_true",
        help="Write detailed per-camera diagnostics for each timestep to trajectory_XXX.debug.json.",
    )
    parser.add_argument(
        "--trajectory-mode",
        type=str,
        choices=("sinusoid", "velocity_command"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-layout",
        type=str,
        choices=("paper_cross", "leg_proxy"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ground-truth-mode",
        type=str,
        choices=("raycast", "analytic_surface"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--camera-leg-z", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ground-truth-raycast-start-z", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ground-truth-raycast-length", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.trajectory_mode is not None:
        if args.trajectory_mode != "velocity_command":
            raise SystemExit(
                "Legacy --trajectory-mode sinusoid has been removed. "
                "Use the paper-faithful default --motion-source policy_rollout or the explicit --motion-source velocity_fallback."
            )
        args.motion_source = "velocity_fallback"

    if args.camera_layout not in (None, "paper_cross"):
        raise SystemExit(
            "Legacy --camera-layout leg_proxy has been removed from the paper-faithful collector. "
            "Use the fixed front/back/left/right ANYmal C camera rig."
        )

    if args.camera_leg_z is not None:
        raise SystemExit("Legacy --camera-leg-z is no longer supported by the paper-faithful collector.")

    if args.ground_truth_mode is not None:
        raise SystemExit(
            "Legacy --ground-truth-mode is no longer supported. "
            "Ground-truth point clouds are always sampled densely from the simulator scene mesh."
        )

    if args.ground_truth_raycast_start_z is not None or args.ground_truth_raycast_length is not None:
        raise SystemExit("Legacy raycast ground-truth options are no longer supported.")

    if args.motion_source == "policy_rollout" and not args.policy_checkpoint:
        raise SystemExit(
            "--policy-checkpoint is required when --motion-source policy_rollout is used. "
            "Pass an exported Isaac Lab TorchScript/JIT policy checkpoint."
        )

    if not math.isclose(float(args.camera_down_tilt_deg), 30.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(
            "The paper-faithful collector fixes the camera tilt at 30 degrees. "
            "Remove --camera-down-tilt-deg or set it to 30."
        )

    if "Anymal-C" not in args.policy_task and "Anymal_C" not in args.policy_task and "Anymal" not in args.policy_task:
        raise SystemExit(
            "--policy-task must target the ANYmal C rough-terrain locomotion policy for this collector."
        )

    return args


def sample_axis(extent: float, spacing: float) -> np.ndarray:
    count = max(int(math.floor((2.0 * extent) / spacing)) + 1, 2)
    return np.linspace(-extent, extent, count, dtype=np.float32)


def sample_axis_from_bounds(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    count = max(int(math.floor((max_value - min_value) / spacing)) + 1, 2)
    return np.linspace(min_value, max_value, count, dtype=np.float32)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero-length vector cannot be normalized.")
    return (vector / norm).astype(np.float32)


def sorted_pair(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"Expected a pair of values, received {values}")
    first = float(values[0])
    second = float(values[1])
    return (first, second) if first <= second else (second, first)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=np.float32)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(component) for component in quaternion]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_from_forward(forward: np.ndarray) -> np.ndarray:
    x_axis = normalize(forward)
    left_axis = normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), x_axis))
    z_axis = normalize(np.cross(x_axis, left_axis))
    return np.column_stack((x_axis, left_axis, z_axis)).astype(np.float32)


def yaw_rotation_matrix(yaw_radians: float) -> np.ndarray:
    c = math.cos(yaw_radians)
    s = math.sin(yaw_radians)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def pose_matrix(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float32)
    return transform


def pose_matrix_from_yaw(position: Sequence[float], yaw_radians: float) -> np.ndarray:
    return pose_matrix(position, yaw_rotation_matrix(yaw_radians))


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


def crop_local_points(
    points_local: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if points_local.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    finite_mask = np.isfinite(points_local).all(axis=1)
    cropped = points_local[finite_mask]
    if cropped.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    mask = np.all(cropped >= min_bounds[None, :], axis=1)
    mask &= np.all(cropped <= max_bounds[None, :], axis=1)
    cropped = cropped[mask]
    if cropped.shape[0] > max_points:
        indices = rng.choice(cropped.shape[0], size=max_points, replace=False)
        cropped = cropped[indices]
    return cropped.astype(np.float32, copy=False)


def pack_clouds(clouds: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    packed: List[np.ndarray] = []
    splits = [0]
    for cloud in clouds:
        points = cloud.astype(np.float32, copy=False)
        packed.append(points)
        splits.append(splits[-1] + points.shape[0])
    packed_points = np.concatenate(packed, axis=0) if packed else np.empty((0, 3), dtype=np.float32)
    return packed_points, np.asarray(splits, dtype=np.int32)


def _unique_row_view(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1])))


def build_ground_truth_grid_local_xy(spacing: float) -> np.ndarray:
    xs = sample_axis_from_bounds(float(GRID_BOUNDS_MIN[0]), float(GRID_BOUNDS_MAX[0]), spacing)
    ys = sample_axis_from_bounds(float(GRID_BOUNDS_MIN[1]), float(GRID_BOUNDS_MAX[1]), spacing)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32, copy=False)


def voxel_downsample_centroids(
    points: np.ndarray,
    voxel_size: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if voxel_size <= 0.0:
        if points.shape[0] > max_points:
            indices = rng.choice(points.shape[0], size=max_points, replace=False)
            return points[indices].astype(np.float32, copy=False)
        return points.astype(np.float32, copy=False)

    voxel_indices = np.floor(points.astype(np.float32) / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    centroids = np.zeros((unique_voxels.shape[0], 3), dtype=np.float32)
    counts = np.zeros((unique_voxels.shape[0], 1), dtype=np.float32)
    np.add.at(centroids, inverse_indices, points.astype(np.float32, copy=False))
    np.add.at(counts, inverse_indices, 1.0)
    centroids /= np.maximum(counts, 1.0)

    if centroids.shape[0] > max_points:
        indices = rng.choice(centroids.shape[0], size=max_points, replace=False)
        centroids = centroids[indices]
    return centroids.astype(np.float32, copy=False)


def estimate_visible_ground_truth_fraction(
    ground_truth_local: np.ndarray,
    measurement_local: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> float:
    if ground_truth_local.size == 0:
        return 0.0
    if measurement_local.size == 0:
        return 0.0

    gt_voxels = np.floor((ground_truth_local.astype(np.float32) - origin[None, :]) / voxel_size).astype(np.int32)
    measurement_voxels = np.floor((measurement_local.astype(np.float32) - origin[None, :]) / voxel_size).astype(np.int32)
    if gt_voxels.size == 0 or measurement_voxels.size == 0:
        return 0.0

    gt_view = _unique_row_view(gt_voxels).reshape(-1)
    measurement_view = np.unique(_unique_row_view(measurement_voxels).reshape(-1))
    visible_mask = np.isin(gt_view, measurement_view, assume_unique=False)
    return float(np.mean(visible_mask)) if visible_mask.size > 0 else 0.0


def build_camera_mounts(args: argparse.Namespace) -> Tuple[CameraMount, ...]:
    tilt = math.radians(args.camera_down_tilt_deg)
    forward_component = math.cos(tilt)
    down_component = math.sin(tilt)
    return (
        CameraMount(
            name="front",
            translation_in_base=np.array([args.camera_mount_x, 0.0, args.camera_mount_height], dtype=np.float32),
            forward_in_base=np.array([forward_component, 0.0, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="back",
            translation_in_base=np.array([-args.camera_mount_x, 0.0, args.camera_mount_height], dtype=np.float32),
            forward_in_base=np.array([-forward_component, 0.0, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="left",
            translation_in_base=np.array([0.0, args.camera_mount_y, args.camera_mount_height], dtype=np.float32),
            forward_in_base=np.array([0.0, forward_component, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="right",
            translation_in_base=np.array([0.0, -args.camera_mount_y, args.camera_mount_height], dtype=np.float32),
            forward_in_base=np.array([0.0, -forward_component, -down_component], dtype=np.float32),
        ),
    )


def build_random_scene(rng: np.random.Generator, args: argparse.Namespace) -> SceneDescription:
    corridor_width = float(rng.uniform(2.0, 6.0))
    path_length = float(args.path_length * rng.uniform(0.8, 1.2))
    stair_tread = float(rng.uniform(0.2, 0.5))
    stair_rise = float(rng.uniform(0.08, 0.25))
    stair_steps = int(rng.integers(4, 7))
    stair_width = float(max(1.0, min(corridor_width - 0.4, corridor_width * rng.uniform(0.65, 0.95))))
    landing_length = float(rng.uniform(0.8, 1.4))
    ground_length = max(path_length + 4.0, stair_tread * stair_steps + landing_length + 5.0)
    ground_width = corridor_width + 4.0
    scene_yaw_radians = float(rng.uniform(-math.pi, math.pi))
    stair_start_x = float(-0.2 * path_length)

    boxes_local: List[BoxPrimitive] = []
    boxes_local.append(
        BoxPrimitive(
            name="ground",
            center=np.array([0.0, 0.0, -0.01], dtype=np.float32),
            size=np.array([ground_length, ground_width, 0.02], dtype=np.float32),
            yaw_radians=0.0,
            top_only=True,
        )
    )

    half_wall_y = 0.5 * corridor_width + 0.5 * args.wall_thickness
    for side, sign in (("left_wall", 1.0), ("right_wall", -1.0)):
        boxes_local.append(
            BoxPrimitive(
                name=side,
                center=np.array([0.0, sign * half_wall_y, 0.5 * args.wall_height], dtype=np.float32),
                size=np.array([ground_length, args.wall_thickness, args.wall_height], dtype=np.float32),
                yaw_radians=0.0,
            )
        )

    for step_index in range(stair_steps):
        step_height = (step_index + 1) * stair_rise
        step_center_x = stair_start_x + (step_index + 0.5) * stair_tread
        boxes_local.append(
            BoxPrimitive(
                name=f"stair_{step_index:02d}",
                center=np.array([step_center_x, 0.0, 0.5 * step_height], dtype=np.float32),
                size=np.array([stair_tread, stair_width, step_height], dtype=np.float32),
                yaw_radians=0.0,
            )
        )

    landing_height = stair_steps * stair_rise
    boxes_local.append(
        BoxPrimitive(
            name="stair_landing",
            center=np.array(
                [stair_start_x + stair_steps * stair_tread + 0.5 * landing_length, 0.0, 0.5 * landing_height],
                dtype=np.float32,
            ),
            size=np.array([landing_length, stair_width, landing_height], dtype=np.float32),
            yaw_radians=0.0,
        )
    )

    def sample_side_center(half_length_x: float, half_length_y: float) -> Optional[np.ndarray]:
        attempts = 0
        while attempts < 32:
            x = float(rng.uniform(-0.45 * ground_length, 0.45 * ground_length))
            lateral_limit = 0.5 * corridor_width - half_length_y - 0.15
            min_abs_y = max(0.55, half_length_y + 0.10)
            if lateral_limit <= min_abs_y:
                return None
            y = float(rng.choice([-1.0, 1.0]) * rng.uniform(min_abs_y, lateral_limit))
            near_stairs = stair_start_x - half_length_x <= x <= (
                stair_start_x + stair_steps * stair_tread + landing_length + half_length_x
            )
            if near_stairs and abs(y) < 0.5 * stair_width + half_length_y + 0.15:
                attempts += 1
                continue
            return np.array([x, y], dtype=np.float32)
        return None

    for index in range(int(rng.integers(3, 7))):
        size_x = float(rng.uniform(0.2, 2.0))
        size_y = float(rng.uniform(0.2, 2.0))
        size_z = float(rng.uniform(0.08, 0.25))
        center_xy = sample_side_center(0.5 * size_x, 0.5 * size_y)
        if center_xy is None:
            continue
        boxes_local.append(
            BoxPrimitive(
                name=f"box_{index:02d}",
                center=np.array([center_xy[0], center_xy[1], 0.5 * size_z], dtype=np.float32),
                size=np.array([size_x, size_y, size_z], dtype=np.float32),
                yaw_radians=float(rng.uniform(-0.6, 0.6)),
            )
        )

    for index in range(int(rng.integers(2, 5))):
        side_size = float(rng.uniform(0.08, 0.18))
        center_xy = sample_side_center(0.5 * side_size, 0.5 * side_size)
        if center_xy is None:
            continue
        boxes_local.append(
            BoxPrimitive(
                name=f"pole_{index:02d}",
                center=np.array([center_xy[0], center_xy[1], 0.5 * DEFAULT_POLE_HEIGHT], dtype=np.float32),
                size=np.array([side_size, side_size, DEFAULT_POLE_HEIGHT], dtype=np.float32),
                yaw_radians=0.0,
            )
        )

    return SceneDescription(
        scene_yaw_radians=scene_yaw_radians,
        path_length=path_length,
        corridor_width=corridor_width,
        stair_start_x=stair_start_x,
        stair_tread=stair_tread,
        stair_rise=stair_rise,
        stair_steps=stair_steps,
        stair_width=stair_width,
        landing_length=landing_length,
        ground_size=np.array([ground_length, ground_width, 0.02], dtype=np.float32),
        boxes_local=tuple(boxes_local),
    )


def world_box_from_local(local_box: BoxPrimitive, scene_yaw_radians: float) -> BoxPrimitive:
    scene_transform = pose_matrix_from_yaw([0.0, 0.0, 0.0], scene_yaw_radians)
    center_world = transform_points(local_box.center[None, :], scene_transform)[0]
    return BoxPrimitive(
        name=local_box.name,
        center=center_world,
        size=local_box.size.astype(np.float32, copy=False),
        yaw_radians=float(scene_yaw_radians + local_box.yaw_radians),
        top_only=local_box.top_only,
    )


def sample_box_surface_points(box: BoxPrimitive, spacing: float) -> np.ndarray:
    half_size = 0.5 * box.size.astype(np.float32)
    xs = sample_axis(float(half_size[0]), spacing)
    ys = sample_axis(float(half_size[1]), spacing)
    zs = sample_axis(float(half_size[2]), spacing)

    top = np.stack(
        np.meshgrid(xs, ys, np.array([half_size[2]], dtype=np.float32), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    points_local = [top]

    if not box.top_only:
        x_pos = np.stack(np.meshgrid(np.array([half_size[0]], dtype=np.float32), ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        x_neg = np.stack(np.meshgrid(np.array([-half_size[0]], dtype=np.float32), ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        y_pos = np.stack(np.meshgrid(xs, np.array([half_size[1]], dtype=np.float32), zs, indexing="ij"), axis=-1).reshape(-1, 3)
        y_neg = np.stack(np.meshgrid(xs, np.array([-half_size[1]], dtype=np.float32), zs, indexing="ij"), axis=-1).reshape(-1, 3)
        points_local.extend((x_pos, x_neg, y_pos, y_neg))

    rotation = yaw_rotation_matrix(box.yaw_radians)
    transform = pose_matrix(box.center, rotation)
    points = transform_points(np.concatenate(points_local, axis=0).astype(np.float32, copy=False), transform)
    return points.astype(np.float32, copy=False)


def build_scene_ground_truth(scene: SceneDescription, spacing: float) -> np.ndarray:
    world_boxes = [world_box_from_local(box, scene.scene_yaw_radians) for box in scene.boxes_local]
    samples = [sample_box_surface_points(box, spacing) for box in world_boxes]
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(samples, axis=0).astype(np.float32, copy=False)


def generate_sinusoid_robot_poses(
    scene: SceneDescription,
    timesteps: int,
    nominal_base_height: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x_positions = np.linspace(-0.5 * scene.path_length, 0.5 * scene.path_length, timesteps, dtype=np.float32)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    amplitude = float(min(0.18, max(0.05, 0.12 * scene.corridor_width)))
    y_positions = amplitude * np.sin(np.linspace(0.0, 2.0 * math.pi, timesteps, dtype=np.float32) + phase)
    yaw_bias = math.radians(float(rng.uniform(-20.0, 20.0)))

    positions_local = np.zeros((timesteps, 3), dtype=np.float32)
    yaw_local = np.zeros((timesteps,), dtype=np.float32)

    for index, (x_local, y_local) in enumerate(zip(x_positions, y_positions)):
        positions_local[index, 0] = x_local
        positions_local[index, 1] = y_local
        positions_local[index, 2] = scene.support_height(float(x_local), float(y_local)) + nominal_base_height

    tangents = np.gradient(positions_local[:, :2], axis=0)
    yaw_local[:] = np.arctan2(tangents[:, 1], tangents[:, 0]).astype(np.float32) + yaw_bias

    scene_rotation = yaw_rotation_matrix(scene.scene_yaw_radians)
    poses = np.zeros((timesteps, 4, 4), dtype=np.float32)

    for index in range(timesteps):
        world_position = positions_local[index] @ scene_rotation.T
        world_yaw = float(scene.scene_yaw_radians + yaw_local[index])
        poses[index] = pose_matrix_from_yaw(world_position, world_yaw)

    return poses.astype(np.float32, copy=False)


def generate_velocity_command_robot_poses(
    scene: SceneDescription,
    timesteps: int,
    nominal_base_height: float,
    sample_dt: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    lin_vel_x_range = sorted_pair(args.command_lin_vel_x_range)
    lin_vel_y_range = sorted_pair(args.command_lin_vel_y_range)
    yaw_rate_range_deg = sorted_pair(args.command_yaw_rate_range_deg)
    yaw_rate_range = tuple(math.radians(value) for value in yaw_rate_range_deg)
    resample_steps = max(1, int(args.command_resample_steps))

    x_limit = max(0.5 * scene.path_length - 0.25, 0.25)
    y_limit = max(0.5 * scene.corridor_width - 0.20, 0.0)

    positions_local = np.zeros((timesteps, 3), dtype=np.float32)
    yaw_local = np.zeros((timesteps,), dtype=np.float32)

    positions_local[0, 0] = -min(0.4 * scene.path_length, x_limit)
    positions_local[0, 1] = float(rng.uniform(-0.10, 0.10)) if y_limit > 0.0 else 0.0
    yaw_local[0] = float(rng.uniform(-math.radians(15.0), math.radians(15.0)))

    body_velocity_command = np.array(
        [
            rng.uniform(*lin_vel_x_range),
            rng.uniform(*lin_vel_y_range),
        ],
        dtype=np.float32,
    )
    yaw_rate = float(rng.uniform(*yaw_rate_range))

    for index in range(timesteps):
        x_local = float(positions_local[index, 0])
        y_local = float(positions_local[index, 1])
        positions_local[index, 2] = scene.support_height(x_local, y_local) + nominal_base_height
        if index == timesteps - 1:
            continue

        if index % resample_steps == 0:
            body_velocity_command = np.array(
                [
                    rng.uniform(*lin_vel_x_range),
                    rng.uniform(*lin_vel_y_range),
                ],
                dtype=np.float32,
            )
            yaw_rate = float(rng.uniform(*yaw_rate_range))

        yaw_local[index + 1] = yaw_local[index] + yaw_rate * sample_dt
        planar_rotation = yaw_rotation_matrix(float(yaw_local[index]))[:2, :2]
        delta_xy = planar_rotation @ body_velocity_command * sample_dt
        next_xy = positions_local[index, :2] + delta_xy.astype(np.float32, copy=False)
        next_xy[0] = float(np.clip(next_xy[0], -x_limit, x_limit))
        next_xy[1] = float(np.clip(next_xy[1], -y_limit, y_limit))
        positions_local[index + 1, :2] = next_xy

    scene_rotation = yaw_rotation_matrix(scene.scene_yaw_radians)
    poses = np.zeros((timesteps, 4, 4), dtype=np.float32)
    for index in range(timesteps):
        world_position = positions_local[index] @ scene_rotation.T
        world_yaw = float(scene.scene_yaw_radians + yaw_local[index])
        poses[index] = pose_matrix_from_yaw(world_position, world_yaw)

    return poses.astype(np.float32, copy=False)


def generate_robot_poses(
    scene: SceneDescription,
    timesteps: int,
    nominal_base_height: float,
    sample_dt: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    return generate_velocity_command_robot_poses(
        scene=scene,
        timesteps=timesteps,
        nominal_base_height=nominal_base_height,
        sample_dt=sample_dt,
        rng=rng,
        args=args,
    )


def save_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    robot_usd_path: str,
    trajectories: List[Dict[str, object]],
    camera_mounts: Sequence[CameraMount],
) -> None:
    manifest = {
        "paper_defaults": {
            "grid_shape": [64, 64, 64],
            "voxel_size_m": 0.05,
            "grid_bounds_min_m": GRID_BOUNDS_MIN.tolist(),
            "grid_bounds_max_m": GRID_BOUNDS_MAX.tolist(),
            "environment": {
                "stairs_tread_m": [0.2, 0.5],
                "stairs_rise_m": [0.08, 0.25],
                "box_width_length_m": [0.2, 2.0],
                "box_height_m": [0.08, 0.25],
                "corridor_width_m": [2.0, 6.0],
            },
            "sensor": {
                "cameras": [mount.name for mount in camera_mounts],
                "layout": "paper_cross",
                "tilt_deg": args.camera_down_tilt_deg,
                "resolution": [args.camera_width, args.camera_height],
            },
            "motion": {
                "source": args.motion_source,
                "policy_task": args.policy_task,
            },
            "ground_truth": {
                "source": "sim_mesh_dense_sampling",
            },
        },
        "collector_args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key
            not in {
                "trajectory_mode",
                "camera_layout",
                "ground_truth_mode",
                "camera_leg_z",
                "ground_truth_raycast_start_z",
                "ground_truth_raycast_length",
            }
        },
        "robot_usd": robot_usd_path,
        "trajectories": trajectories,
    }
    (output_dir / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_collection(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "collection_status.json"
    error_path = args.output_dir / "collection_error.txt"

    def write_status(state: str, **extra: object) -> None:
        payload = {
            "state": state,
            "output_dir": str(args.output_dir),
            "show_ui": bool(args.show_ui),
            "debug_camera": bool(args.debug_camera),
            "num_trajectories": int(args.num_trajectories),
            "timesteps": int(args.timesteps),
            "motion_source": args.motion_source,
        }
        payload.update(extra)
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if error_path.exists():
        error_path.unlink()

    write_status("starting", stage="before_simulation_app")
    print(f"[collector] starting output_dir={args.output_dir}", flush=True)

    simulation_app = None
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": not args.show_ui})
        import io
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import create_prim, delete_prim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        import torch

        rng = np.random.default_rng(args.seed)
        write_status("running", stage="simulation_app_started")
        print("[collector] simulation app started", flush=True)

        def path_exists(path: str) -> bool:
            if "://" not in path:
                return Path(path).expanduser().exists()
            result, _ = omni.client.stat(path)
            return result == omni.client.Result.OK

        def resolve_robot_usd() -> str:
            if args.robot_usd:
                if path_exists(args.robot_usd):
                    return args.robot_usd
                raise FileNotFoundError(f"--robot-usd does not exist or is not reachable: {args.robot_usd}")

            assets_root = args.asset_root or get_assets_root_path()
            if not assets_root:
                raise RuntimeError(
                    "Could not resolve Isaac Sim asset root. Pass --asset-root or configure persistent.isaac.asset_root.default."
                )

            candidates = (
                f"{assets_root}/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd",
                f"{assets_root}/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
                f"{assets_root}/Isaac/IsaacLab/Robots/ANYbotics/anymal_c/anymal_c.usd",
            )
            for candidate in candidates:
                if path_exists(candidate):
                    return candidate
            raise FileNotFoundError("Failed to find ANYmal C. Checked:\n" + "\n".join(candidates))

        def delete_stage_roots(stage_paths: Sequence[str]) -> None:
            stage = omni.usd.get_context().get_stage()
            for prim_path in stage_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    delete_prim(prim_path)

        def render_frames(
            step_count: int,
            *,
            world: Optional[World] = None,
            render_fn=None,
        ) -> None:
            for _ in range(max(0, step_count)):
                if world is not None:
                    world.step(render=True)
                elif render_fn is not None:
                    render_fn()
                elif hasattr(simulation_app, "update"):
                    simulation_app.update()

        def camera_frame_dict(camera: Camera) -> Dict[str, object]:
            try:
                frame = camera.get_current_frame(clone=False)
            except TypeError:
                frame = camera.get_current_frame()
            return frame if isinstance(frame, dict) else {}

        def camera_transform_world(camera: Camera) -> np.ndarray:
            camera_position, camera_quaternion = camera.get_world_pose(camera_axes="world")
            return pose_matrix(camera_position, quaternion_to_rotation_matrix(camera_quaternion))

        def payload_overview(payload: object) -> Dict[str, object]:
            summary: Dict[str, object] = {"python_type": type(payload).__name__}
            if payload is None:
                summary["is_none"] = True
                return summary
            if isinstance(payload, dict):
                summary["keys"] = sorted(str(key) for key in payload.keys())
                return summary
            try:
                array = np.asarray(payload)
            except Exception:
                return summary
            summary["shape"] = list(array.shape)
            summary["dtype"] = str(array.dtype)
            return summary

        def depth_to_world_points(camera: Camera, depth_image: object, debug: Optional[Dict[str, object]] = None) -> np.ndarray:
            if depth_image is None:
                if debug is not None:
                    debug["depth_present"] = False
                return np.empty((0, 3), dtype=np.float32)

            depth = np.asarray(depth_image, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.ndim != 2:
                if debug is not None:
                    debug["depth_present"] = True
                    debug["depth_shape"] = list(depth.shape)
                    debug["depth_valid_pixel_count"] = 0
                return np.empty((0, 3), dtype=np.float32)

            valid_mask = np.isfinite(depth)
            valid_mask &= depth >= args.camera_min_range
            valid_mask &= depth <= args.camera_max_range
            full_valid_pixels = np.argwhere(valid_mask)
            if debug is not None:
                debug["depth_present"] = True
                debug["depth_shape"] = list(depth.shape)
                debug["depth_valid_pixel_count"] = int(full_valid_pixels.shape[0])
            if full_valid_pixels.size == 0:
                return np.empty((0, 3), dtype=np.float32)

            pixel_stride = max(1, int(args.camera_pixel_stride))
            if pixel_stride > 1:
                strided_valid_mask = valid_mask[::pixel_stride, ::pixel_stride]
                strided_pixels = np.argwhere(strided_valid_mask)
                valid_pixels = (strided_pixels * pixel_stride).astype(np.int32, copy=False)
            else:
                valid_pixels = full_valid_pixels
            if debug is not None:
                debug["camera_pixel_stride"] = int(pixel_stride)
                debug["depth_valid_pixel_count_after_stride"] = int(valid_pixels.shape[0])
            if valid_pixels.size == 0:
                return np.empty((0, 3), dtype=np.float32)

            max_camera_points = max(1, int(args.max_camera_points))
            if valid_pixels.shape[0] > max_camera_points:
                indices = np.linspace(0, valid_pixels.shape[0] - 1, max_camera_points, dtype=np.int32)
                valid_pixels = valid_pixels[indices]
            if debug is not None:
                debug["max_camera_points"] = max_camera_points
                debug["depth_sampled_pixel_count"] = int(valid_pixels.shape[0])

            pixel_coords = np.column_stack((valid_pixels[:, 1], valid_pixels[:, 0])).astype(np.float32, copy=False)
            depth_values = depth[valid_pixels[:, 0], valid_pixels[:, 1]].astype(np.float32, copy=False)
            points_world = camera.get_world_points_from_image_coords(pixel_coords, depth_values)
            return np.asarray(points_world, dtype=np.float32)

        def coerce_pointcloud_array(payload: object, debug: Optional[Dict[str, object]] = None, label: str = "") -> np.ndarray:
            candidate = payload
            if isinstance(payload, dict):
                if debug is not None:
                    debug[f"{label}_payload"] = payload_overview(payload)
                for key in ("data", "points", "pointcloud", "xyz", "positions"):
                    if key in payload:
                        candidate = payload[key]
                        if debug is not None:
                            debug[f"{label}_selected_key"] = key
                        break
                else:
                    return np.empty((0, 3), dtype=np.float32)
            elif debug is not None:
                debug[f"{label}_payload"] = payload_overview(payload)

            try:
                points = np.asarray(candidate, dtype=np.float32)
            except (TypeError, ValueError):
                return np.empty((0, 3), dtype=np.float32)

            if points.ndim != 2 or points.shape[1] < 3:
                if debug is not None:
                    debug[f"{label}_array_shape"] = list(points.shape)
                return np.empty((0, 3), dtype=np.float32)
            if debug is not None:
                debug[f"{label}_array_shape"] = list(points.shape)
            return points[:, :3].astype(np.float32, copy=False)

        def optical_camera_points_to_world(points_camera: np.ndarray, camera: Camera) -> np.ndarray:
            if points_camera.size == 0:
                return np.empty((0, 3), dtype=np.float32)
            points_world_axes_local = np.column_stack(
                (
                    points_camera[:, 2],
                    -points_camera[:, 0],
                    -points_camera[:, 1],
                )
            ).astype(np.float32, copy=False)
            return transform_points(points_world_axes_local, camera_transform_world(camera))

        def read_camera_points_world(camera: Camera, debug: Optional[Dict[str, object]] = None) -> np.ndarray:
            frame = camera_frame_dict(camera)
            if debug is not None:
                debug["frame_keys"] = sorted(str(key) for key in frame.keys())

            depth_image = frame.get("distance_to_image_plane")
            depth_source = "frame.distance_to_image_plane"
            if depth_image is None:
                try:
                    depth_image = camera.get_depth()
                    depth_source = "camera.get_depth"
                except Exception as exc:
                    if debug is not None:
                        debug["get_depth_error"] = f"{type(exc).__name__}: {exc}"
                    depth_image = None
                    depth_source = "missing"
            if debug is not None:
                debug["depth_source"] = depth_source

            points_world = depth_to_world_points(camera, depth_image, debug=debug)
            if points_world.size > 0:
                if debug is not None:
                    debug["selected_source"] = "depth"
                    debug["selected_point_count"] = int(points_world.shape[0])
                return points_world.astype(np.float32, copy=False)

            try:
                points_camera = camera.get_pointcloud()
            except Exception as exc:
                if debug is not None:
                    debug["get_pointcloud_error"] = f"{type(exc).__name__}: {exc}"
                points_camera = None

            points_camera = coerce_pointcloud_array(points_camera, debug=debug, label="get_pointcloud")
            if points_camera.size > 0:
                if debug is not None:
                    debug["selected_source"] = "get_pointcloud"
                    debug["selected_point_count"] = int(points_camera.shape[0])
                return optical_camera_points_to_world(points_camera, camera)

            frame_pointcloud = coerce_pointcloud_array(frame.get("pointcloud"), debug=debug, label="frame_pointcloud")
            if frame_pointcloud.size > 0:
                if debug is not None:
                    debug["selected_source"] = "frame_pointcloud"
                    debug["selected_point_count"] = int(frame_pointcloud.shape[0])
                return optical_camera_points_to_world(frame_pointcloud, camera)

            return np.empty((0, 3), dtype=np.float32)

        def capture_camera_points_world(camera: Camera, *, render_retry_fn, debug: Optional[Dict[str, object]] = None) -> np.ndarray:
            attempt_summaries: List[Dict[str, object]] = []
            for attempt_index in range(args.sensor_retry_steps + 1):
                attempt_debug: Dict[str, object] = {"attempt_index": int(attempt_index)}
                points_world = read_camera_points_world(camera, debug=attempt_debug)
                attempt_debug["returned_point_count"] = int(points_world.shape[0])
                attempt_summaries.append(attempt_debug)
                if points_world.size > 0:
                    if debug is not None:
                        debug["attempts"] = attempt_summaries
                        debug["selected_attempt"] = int(attempt_index)
                    return points_world.astype(np.float32, copy=False)
                render_retry_fn(1)
            if debug is not None:
                debug["attempts"] = attempt_summaries
            return np.empty((0, 3), dtype=np.float32)

        def create_cameras(*, capture_dt_effective: float, camera_mounts: Sequence[CameraMount]) -> Tuple[Dict[str, Camera], int]:
            camera_frequency_hz = max(1, int(round(1.0 / capture_dt_effective)))
            cameras: Dict[str, Camera] = {}
            for mount in camera_mounts:
                camera = Camera(
                    prim_path=f"/World/Sensors/{mount.name}",
                    name=f"{mount.name}_depth",
                    frequency=camera_frequency_hz,
                    resolution=(args.camera_width, args.camera_height),
                    position=np.array([0.0, 0.0, 2.0], dtype=np.float32),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                camera.initialize()
                dt_configured = False
                if hasattr(camera, "set_dt"):
                    try:
                        camera.set_dt(float(capture_dt_effective))
                        dt_configured = True
                    except Exception as exc:
                        print(
                            f"[collector] camera.set_dt({capture_dt_effective:.9f}) failed for "
                            f"{camera.prim_path}: {exc}; falling back to set_frequency({camera_frequency_hz})",
                            flush=True,
                        )
                if (not dt_configured) and hasattr(camera, "set_frequency"):
                    camera.set_frequency(camera_frequency_hz)
                if hasattr(camera, "set_clipping_range"):
                    camera.set_clipping_range(args.camera_min_range, args.camera_max_range)
                camera.add_distance_to_image_plane_to_frame()
                camera.add_pointcloud_to_frame(include_unlabelled=True)
                if hasattr(camera, "resume"):
                    camera.resume()
                cameras[mount.name] = camera
            return cameras, camera_frequency_hz

        def apply_robot_and_camera_pose(
            robot_pose: np.ndarray,
            robot: Optional[SingleArticulation],
            cameras: Dict[str, Camera],
            camera_mounts: Sequence[CameraMount],
        ) -> None:
            robot_position = robot_pose[:3, 3].astype(np.float32, copy=False)
            robot_orientation = rotation_matrix_to_quaternion(robot_pose[:3, :3])
            if robot is not None:
                robot.set_world_pose(position=robot_position, orientation=robot_orientation)
                if hasattr(robot, "set_world_velocity"):
                    robot.set_world_velocity(np.zeros(6, dtype=np.float32))

            base_rotation = robot_pose[:3, :3]
            for mount in camera_mounts:
                camera_world_position = mount.translation_in_base @ base_rotation.T + robot_position
                camera_forward_world = mount.forward_in_base @ base_rotation.T
                camera_rotation = rotation_matrix_from_forward(camera_forward_world)
                cameras[mount.name].set_world_pose(
                    position=camera_world_position,
                    orientation=rotation_matrix_to_quaternion(camera_rotation),
                    camera_axes="world",
                )

        def sanitize_prim_name(name: str) -> str:
            sanitized = "".join(character if (character.isalnum() or character == "_") else "_" for character in name)
            return sanitized or "prim"

        def author_scene_usd(scene: SceneDescription, usd_path: Path) -> None:
            usd_path.parent.mkdir(parents=True, exist_ok=True)
            stage = Usd.Stage.CreateNew(str(usd_path))
            if stage is None:
                raise RuntimeError(f"Failed to create USD stage: {usd_path}")
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            root = UsdGeom.Xform.Define(stage, "/Scene")
            stage.SetDefaultPrim(root.GetPrim())
            world_boxes = [world_box_from_local(box, scene.scene_yaw_radians) for box in scene.boxes_local]
            for box in world_boxes:
                cube = UsdGeom.Cube.Define(stage, f"/Scene/{sanitize_prim_name(box.name)}")
                cube.CreateSizeAttr(2.0)
                translate_op = cube.AddTranslateOp()
                orient_op = cube.AddOrientOp()
                scale_op = cube.AddScaleOp()
                translate_op.Set(Gf.Vec3d(float(box.center[0]), float(box.center[1]), float(box.center[2])))
                box_quaternion = rotation_matrix_to_quaternion(yaw_rotation_matrix(box.yaw_radians))
                orient_op.Set(
                    Gf.Quatf(
                        float(box_quaternion[0]),
                        Gf.Vec3f(float(box_quaternion[1]), float(box_quaternion[2]), float(box_quaternion[3])),
                    )
                )
                scale_op.Set(Gf.Vec3f(float(box.size[0] * 0.5), float(box.size[1] * 0.5), float(box.size[2] * 0.5)))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.45, 0.45, 0.45)])
            stage.GetRootLayer().Save()

        def triangulate_scene_mesh(root_prim_path: str) -> np.ndarray:
            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(root_prim_path)
            if not root_prim.IsValid():
                raise RuntimeError(f"Ground-truth mesh root prim does not exist: {root_prim_path}")

            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            triangles: List[np.ndarray] = []

            def transform_point(matrix: Gf.Matrix4d, point: Sequence[float]) -> np.ndarray:
                transformed = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                return np.array([float(transformed[0]), float(transformed[1]), float(transformed[2])], dtype=np.float32)

            def cube_triangles(cube_prim: Usd.Prim) -> List[np.ndarray]:
                cube = UsdGeom.Cube(cube_prim)
                size_attr = cube.GetSizeAttr().Get()
                edge = float(size_attr) if size_attr is not None else 2.0
                half = 0.5 * edge
                local_vertices = np.array(
                    [
                        [-half, -half, -half],
                        [half, -half, -half],
                        [half, half, -half],
                        [-half, half, -half],
                        [-half, -half, half],
                        [half, -half, half],
                        [half, half, half],
                        [-half, half, half],
                    ],
                    dtype=np.float32,
                )
                matrix = xform_cache.GetLocalToWorldTransform(cube_prim)
                world_vertices = np.stack([transform_point(matrix, vertex) for vertex in local_vertices], axis=0)
                triangle_indices = (
                    (0, 1, 2), (0, 2, 3),
                    (4, 6, 5), (4, 7, 6),
                    (0, 4, 5), (0, 5, 1),
                    (1, 5, 6), (1, 6, 2),
                    (2, 6, 7), (2, 7, 3),
                    (3, 7, 4), (3, 4, 0),
                )
                return [world_vertices[np.asarray(indices, dtype=np.int32)] for indices in triangle_indices]

            def mesh_triangles(mesh_prim: Usd.Prim) -> List[np.ndarray]:
                mesh = UsdGeom.Mesh(mesh_prim)
                points_attr = mesh.GetPointsAttr().Get()
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_indices = mesh.GetFaceVertexIndicesAttr().Get()
                if points_attr is None or face_counts is None or face_indices is None:
                    raise RuntimeError(f"Mesh prim is missing topology data: {mesh_prim.GetPath()}")
                matrix = xform_cache.GetLocalToWorldTransform(mesh_prim)
                world_vertices = np.stack([transform_point(matrix, point) for point in points_attr], axis=0)
                triangles_local: List[np.ndarray] = []
                cursor = 0
                for face_vertex_count in face_counts:
                    count = int(face_vertex_count)
                    face = [int(index) for index in face_indices[cursor : cursor + count]]
                    cursor += count
                    if count < 3:
                        continue
                    for index in range(1, count - 1):
                        triangles_local.append(
                            world_vertices[np.asarray([face[0], face[index], face[index + 1]], dtype=np.int32)].astype(np.float32, copy=False)
                        )
                return triangles_local

            for prim in Usd.PrimRange(root_prim):
                if not prim.IsValid() or not prim.IsActive():
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable.GetPrim().IsValid():
                    visibility = imageable.ComputeVisibility(Usd.TimeCode.Default())
                    if visibility == UsdGeom.Tokens.invisible:
                        continue

                if prim.IsA(UsdGeom.Cube):
                    triangles.extend(cube_triangles(prim))
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    triangles.extend(mesh_triangles(prim))
                    continue
                if prim.IsA(UsdGeom.Gprim):
                    raise RuntimeError(
                        f"Unsupported visible geometry for mesh ground truth: {prim.GetPath()} ({prim.GetTypeName()})"
                    )

            if not triangles:
                raise RuntimeError(f"No triangulated geometry found below {root_prim_path}")
            return np.stack(triangles, axis=0).astype(np.float32, copy=False)

        def build_scene_mesh_ground_truth_world(root_prim_path: str) -> np.ndarray:
            triangles = triangulate_scene_mesh(root_prim_path)
            spacing = float(args.ground_truth_spacing)
            area_per_sample = max(0.5 * spacing * spacing, 1e-6)
            dense_samples: List[np.ndarray] = []
            for triangle in triangles:
                v0, v1, v2 = triangle
                area = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
                sample_count = max(1, int(math.ceil(area / area_per_sample)))
                random_u = rng.random(sample_count, dtype=np.float32)
                random_v = rng.random(sample_count, dtype=np.float32)
                sqrt_u = np.sqrt(random_u)
                bary_a = 1.0 - sqrt_u
                bary_b = sqrt_u * (1.0 - random_v)
                bary_c = sqrt_u * random_v
                sampled = (
                    bary_a[:, None] * v0[None, :]
                    + bary_b[:, None] * v1[None, :]
                    + bary_c[:, None] * v2[None, :]
                ).astype(np.float32, copy=False)
                dense_samples.append(
                    np.concatenate(
                        (
                            sampled,
                            np.stack((v0, v1, v2, (v0 + v1 + v2) / 3.0), axis=0).astype(np.float32, copy=False),
                        ),
                        axis=0,
                    )
                )

            dense_world = np.concatenate(dense_samples, axis=0) if dense_samples else np.empty((0, 3), dtype=np.float32)
            return voxel_downsample_centroids(
                dense_world,
                voxel_size=args.ground_truth_spacing * 0.5,
                max_points=max(dense_world.shape[0], args.max_ground_truth_points),
                rng=rng,
            )

        def load_torchscript_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
            if "://" in policy_path:
                file_content = omni.client.read_file(policy_path)[2]
                file_object = io.BytesIO(memoryview(file_content).tobytes())
                policy = torch.jit.load(file_object, map_location=device)
            else:
                policy = torch.jit.load(str(Path(policy_path).expanduser().resolve()), map_location=device)
            policy.eval()
            return policy

        def sample_policy_command_sequence(timesteps: int) -> np.ndarray:
            lin_vel_x_range = sorted_pair(args.command_lin_vel_x_range)
            lin_vel_y_range = sorted_pair(args.command_lin_vel_y_range)
            heading_range_deg = sorted_pair(args.policy_heading_range_deg)
            heading_range = tuple(math.radians(value) for value in heading_range_deg)
            commands = np.zeros((timesteps, 4), dtype=np.float32)
            current_command = np.zeros((4,), dtype=np.float32)
            resample_steps = max(1, int(args.command_resample_steps))
            for index in range(timesteps):
                if index % resample_steps == 0:
                    current_command[0] = float(rng.uniform(*lin_vel_x_range))
                    current_command[1] = float(rng.uniform(*lin_vel_y_range))
                    current_command[2] = 0.0
                    current_command[3] = float(rng.uniform(*heading_range))
                commands[index] = current_command
            return commands.astype(np.float32, copy=False)

        def policy_env_robot_pose(policy_env: object) -> np.ndarray:
            robot_asset = policy_env.scene["robot"]
            position = robot_asset.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            quaternion = robot_asset.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
            return pose_matrix(position, quaternion_to_rotation_matrix(quaternion))

        def capture_timestep_observation(
            *,
            step_index: int,
            robot_pose: np.ndarray,
            cameras: Dict[str, Camera],
            gt_world_points: np.ndarray,
            max_points_per_camera: int,
            render_retry_fn,
        ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, np.ndarray], Optional[Dict[str, object]]]:
            measurement_world_chunks: List[np.ndarray] = []
            world_to_robot = invert_transform(robot_pose)
            timestep_camera_locals: Dict[str, np.ndarray] = {}
            timestep_debug: Optional[Dict[str, object]] = None
            if args.debug_camera:
                timestep_debug = {
                    "timestep_index": int(step_index),
                    "motion_source": args.motion_source,
                    "robot_position_world": robot_pose[:3, 3].astype(np.float32).tolist(),
                    "robot_rotation_world": robot_pose[:3, :3].astype(np.float32).tolist(),
                    "ground_truth_source": "sim_mesh_dense_sampling",
                    "cameras": [],
                }

            for camera_name, camera in cameras.items():
                camera_debug: Optional[Dict[str, object]] = None
                if timestep_debug is not None:
                    camera_pose_world = camera_transform_world(camera)
                    camera_debug = {
                        "name": camera_name,
                        "prim_path": str(camera.prim_path),
                        "position_world": camera_pose_world[:3, 3].astype(np.float32).tolist(),
                        "forward_world": camera_pose_world[:3, 0].astype(np.float32).tolist(),
                    }
                    timestep_debug["cameras"].append(camera_debug)

                points_world = capture_camera_points_world(camera, render_retry_fn=render_retry_fn, debug=camera_debug)
                if camera_debug is not None:
                    camera_debug["points_before_finite_filter"] = int(points_world.shape[0])
                if points_world.size == 0:
                    continue

                finite_mask = np.isfinite(points_world).all(axis=1)
                points_world = points_world[finite_mask]
                if camera_debug is not None:
                    camera_debug["points_after_finite_filter"] = int(points_world.shape[0])
                if points_world.size == 0:
                    continue

                camera_position, _ = camera.get_world_pose(camera_axes="world")
                distances = np.linalg.norm(points_world - np.asarray(camera_position, dtype=np.float32), axis=1)
                range_mask = distances >= args.camera_min_range
                range_mask &= distances <= args.camera_max_range
                if camera_debug is not None and distances.size > 0:
                    camera_debug["distance_stats_m"] = {
                        "min": float(np.min(distances)),
                        "max": float(np.max(distances)),
                    }
                points_world = points_world[range_mask]
                if camera_debug is not None:
                    camera_debug["points_after_range_filter"] = int(points_world.shape[0])
                if points_world.size == 0:
                    continue

                points_world = points_world.astype(np.float32, copy=False)
                measurement_world_chunks.append(points_world)
                if args.save_raw_camera_clouds:
                    camera_points_local = crop_local_points(
                        transform_points(points_world, world_to_robot),
                        GRID_BOUNDS_MIN,
                        GRID_BOUNDS_MAX,
                        max(args.max_camera_points, max_points_per_camera),
                        rng,
                    )
                    timestep_camera_locals[camera_name] = voxel_downsample_centroids(
                        camera_points_local,
                        voxel_size=args.raw_camera_voxel_size,
                        max_points=max_points_per_camera,
                        rng=rng,
                    )

            measurement_world = (
                np.concatenate(measurement_world_chunks, axis=0)
                if measurement_world_chunks
                else np.empty((0, 3), dtype=np.float32)
            )
            measurement_local = crop_local_points(
                transform_points(measurement_world, world_to_robot),
                GRID_BOUNDS_MIN,
                GRID_BOUNDS_MAX,
                args.max_measurement_points,
                rng,
            )
            measurement_local = voxel_downsample_centroids(
                measurement_local,
                voxel_size=args.measurement_voxel_size,
                max_points=args.max_measurement_points,
                rng=rng,
            )

            ground_truth_step_local = crop_local_points(
                transform_points(gt_world_points, world_to_robot),
                GRID_BOUNDS_MIN,
                GRID_BOUNDS_MAX,
                args.max_ground_truth_points,
                rng,
            )
            ground_truth_step_local = voxel_downsample_centroids(
                ground_truth_step_local,
                voxel_size=args.ground_truth_spacing,
                max_points=args.max_ground_truth_points,
                rng=rng,
            )

            visible_ground_truth_fraction = estimate_visible_ground_truth_fraction(
                ground_truth_step_local,
                measurement_local,
                voxel_size=args.ground_truth_spacing,
                origin=GRID_BOUNDS_MIN,
            )

            if timestep_debug is not None:
                timestep_debug["measurement_world_point_count"] = int(measurement_world.shape[0])
                timestep_debug["measurement_local_point_count"] = int(measurement_local.shape[0])
                timestep_debug["ground_truth_local_point_count"] = int(ground_truth_step_local.shape[0])
                timestep_debug["visible_ground_truth_fraction"] = float(visible_ground_truth_fraction)

            return (
                measurement_local,
                ground_truth_step_local,
                visible_ground_truth_fraction,
                timestep_camera_locals,
                timestep_debug,
            )

        def save_trajectory_artifacts(
            *,
            trajectory_index: int,
            trajectory_poses: np.ndarray,
            scene: SceneDescription,
            scene_usd_path: Path,
            capture_step_count: int,
            capture_dt_effective: float,
            camera_frequency_hz: int,
            measurements_local: List[np.ndarray],
            ground_truth_local: List[np.ndarray],
            visible_ground_truth_fractions: List[float],
            raw_camera_clouds_local: Optional[Dict[str, List[np.ndarray]]],
            partial: bool,
        ) -> Dict[str, object]:
            measurement_points, measurement_splits = pack_clouds(measurements_local)
            ground_truth_points, ground_truth_splits = pack_clouds(ground_truth_local)
            extra_arrays: Dict[str, np.ndarray] = {}
            raw_camera_meta: Dict[str, int] = {}
            if raw_camera_clouds_local:
                for camera_name, camera_clouds in sorted(raw_camera_clouds_local.items()):
                    camera_points, camera_splits = pack_clouds(camera_clouds)
                    extra_arrays[f"camera_{camera_name}_points_local"] = camera_points
                    extra_arrays[f"camera_{camera_name}_splits"] = camera_splits
                    raw_camera_meta[camera_name] = int(camera_points.shape[0])

            trajectory_name = f"trajectory_{trajectory_index:03d}"
            suffix = ".partial" if partial else ""
            np.savez_compressed(
                args.output_dir / f"{trajectory_name}{suffix}.npz",
                poses=trajectory_poses.astype(np.float32, copy=False),
                measurement_points=measurement_points,
                measurement_splits=measurement_splits,
                ground_truth_points=ground_truth_points,
                ground_truth_splits=ground_truth_splits,
                **extra_arrays,
            )

            trajectory_meta = {
                "name": trajectory_name,
                "partial": bool(partial),
                "debug_camera": bool(args.debug_camera),
                "motion_source": args.motion_source,
                "policy_task": args.policy_task if args.motion_source == "policy_rollout" else None,
                "ground_truth_source": "sim_mesh_dense_sampling",
                "camera_layout": "paper_cross",
                "save_raw_camera_clouds": bool(args.save_raw_camera_clouds),
                "debug_file": f"{trajectory_name}{suffix}.debug.json" if args.debug_camera else None,
                "scene_usd": str(scene_usd_path),
                "timesteps": int(len(measurements_local)),
                "completed_timesteps": int(len(measurements_local)),
                "requested_timesteps": int(args.timesteps),
                "capture_step_count": int(capture_step_count),
                "capture_dt_requested_s": float(args.capture_dt),
                "capture_dt_effective_s": float(capture_dt_effective),
                "camera_frequency_hz": int(camera_frequency_hz),
                "path_length_m": scene.path_length,
                "corridor_width_m": scene.corridor_width,
                "scene_yaw_deg": math.degrees(scene.scene_yaw_radians),
                "stairs": {
                    "start_x_m": scene.stair_start_x,
                    "tread_m": scene.stair_tread,
                    "rise_m": scene.stair_rise,
                    "steps": scene.stair_steps,
                    "width_m": scene.stair_width,
                    "landing_length_m": scene.landing_length,
                },
                "measurement_points": int(measurement_points.shape[0]),
                "ground_truth_points": int(ground_truth_points.shape[0]),
                "measurement_points_per_timestep": [int(points.shape[0]) for points in measurements_local],
                "ground_truth_points_per_timestep": [int(points.shape[0]) for points in ground_truth_local],
                "visible_ground_truth_fraction_per_timestep": [float(value) for value in visible_ground_truth_fractions],
                "visible_ground_truth_fraction_mean": (
                    float(np.mean(visible_ground_truth_fractions)) if visible_ground_truth_fractions else 0.0
                ),
                "visibility_metric": {
                    "type": "voxel_overlap_approximation",
                    "voxel_size_m": float(args.ground_truth_spacing),
                },
                "all_measurements_empty": bool(measurement_points.shape[0] == 0),
                "raw_camera_points": raw_camera_meta,
            }
            (args.output_dir / f"{trajectory_name}{suffix}.json").write_text(
                json.dumps(trajectory_meta, indent=2),
                encoding="utf-8",
            )
            return trajectory_meta

        def save_debug_artifacts(
            *,
            trajectory_index: int,
            scene: SceneDescription,
            capture_step_count: int,
            capture_dt_effective: float,
            camera_frequency_hz: int,
            debug_steps: List[Dict[str, object]],
            partial: bool,
        ) -> None:
            if not args.debug_camera:
                return
            trajectory_name = f"trajectory_{trajectory_index:03d}"
            suffix = ".partial" if partial else ""
            debug_payload = {
                "name": trajectory_name,
                "partial": bool(partial),
                "requested_timesteps": int(args.timesteps),
                "completed_timesteps": int(len(debug_steps)),
                "camera_debug_enabled": True,
                "motion_source": args.motion_source,
                "ground_truth_source": "sim_mesh_dense_sampling",
                "physics_dt": float(args.physics_dt),
                "capture_dt_requested": float(args.capture_dt),
                "capture_substeps": int(capture_step_count),
                "capture_dt_effective": float(capture_dt_effective),
                "camera_frequency_hz": int(camera_frequency_hz),
                "sensor_warmup_steps": int(args.sensor_warmup_steps),
                "sensor_retry_steps": int(args.sensor_retry_steps),
                "scene": {
                    "path_length_m": scene.path_length,
                    "corridor_width_m": scene.corridor_width,
                    "scene_yaw_deg": math.degrees(scene.scene_yaw_radians),
                    "stairs": {
                        "start_x_m": scene.stair_start_x,
                        "tread_m": scene.stair_tread,
                        "rise_m": scene.stair_rise,
                        "steps": scene.stair_steps,
                        "width_m": scene.stair_width,
                        "landing_length_m": scene.landing_length,
                    },
                },
                "timesteps": debug_steps,
            }
            (args.output_dir / f"{trajectory_name}{suffix}.debug.json").write_text(
                json.dumps(debug_payload, indent=2),
                encoding="utf-8",
            )

        def clear_world_instance(active_world: Optional[World], trajectory_index: int) -> None:
            if active_world is None:
                return
            print(
                f"[collector] clearing world instance before trajectory {trajectory_index + 1}/{args.num_trajectories}",
                flush=True,
            )
            for method_name in ("stop", "pause"):
                method = getattr(active_world, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        print(f"[collector] world.{method_name}() failed: {exc}", flush=True)
                    break

            instance = None
            instance_getter = getattr(World, "instance", None)
            if callable(instance_getter):
                try:
                    instance = instance_getter()
                except Exception as exc:
                    print(f"[collector] World.instance() failed: {exc}", flush=True)

            clear_instance = None
            if instance is not None:
                clear_instance = getattr(instance, "clear_instance", None)
            if clear_instance is None:
                clear_instance = getattr(World, "clear_instance", None)
            if callable(clear_instance):
                try:
                    clear_instance()
                except Exception as exc:
                    print(f"[collector] clear_instance() failed: {exc}", flush=True)

        robot_usd_path = resolve_robot_usd() if args.motion_source == "velocity_fallback" else f"isaaclab::{args.policy_task}"
        write_status("running", stage="assets_resolved", robot_usd=robot_usd_path)
        print(f"[collector] resolved robot source: {robot_usd_path}", flush=True)

        camera_mounts = build_camera_mounts(args)
        max_points_per_camera = max(1, int(math.ceil(args.max_measurement_points / max(1, len(camera_mounts)))))
        fallback_capture_steps = max(1, int(round(args.capture_dt / args.physics_dt)))
        fallback_capture_dt_effective = fallback_capture_steps * args.physics_dt
        if not math.isclose(fallback_capture_dt_effective, args.capture_dt, rel_tol=0.0, abs_tol=1e-9):
            print(
                f"[collector] adjusted capture_dt from {args.capture_dt:.9f}s "
                f"to {fallback_capture_dt_effective:.9f}s to align with physics_dt={args.physics_dt:.9f}s",
                flush=True,
            )

        scene_usd_dir = args.output_dir / "_scene_usd"
        scene_usd_dir.mkdir(parents=True, exist_ok=True)
        policy_command_sequence = sample_policy_command_sequence(args.timesteps)
        trajectories_meta: List[Dict[str, object]] = []
        world: Optional[World] = None
        policy_env = None

        for trajectory_index in range(args.num_trajectories):
            write_status("running", stage="trajectory_setup", trajectory_index=int(trajectory_index))
            print(
                f"[collector] setting up trajectory {trajectory_index + 1}/{args.num_trajectories}",
                flush=True,
            )
            clear_world_instance(world, trajectory_index)
            world = None
            if policy_env is not None:
                close_method = getattr(policy_env, "close", None)
                if callable(close_method):
                    close_method()
                policy_env = None
            delete_stage_roots(("/World/Scene", "/World/Sensors", "/World/Robot", "/World/envs", "/World/ground"))

            scene = build_random_scene(rng, args)
            scene_usd_path = scene_usd_dir / f"scene_{trajectory_index:03d}.usd"
            author_scene_usd(scene, scene_usd_path)

            measurements_local: List[np.ndarray] = []
            ground_truth_local: List[np.ndarray] = []
            visible_ground_truth_fractions: List[float] = []
            raw_camera_clouds_local: Optional[Dict[str, List[np.ndarray]]] = (
                {mount.name: [] for mount in camera_mounts} if args.save_raw_camera_clouds else None
            )
            trajectory_debug_steps: List[Dict[str, object]] = []
            collected_poses: List[np.ndarray] = []
            capture_step_count = fallback_capture_steps
            capture_dt_effective = fallback_capture_dt_effective
            camera_frequency_hz = max(1, int(round(1.0 / capture_dt_effective)))
            ui_interrupted = False
            rollout_interrupted = False

            if args.motion_source == "velocity_fallback":
                world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
                print(f"[collector] recreated fallback world for trajectory {trajectory_index:03d}", flush=True)
                create_prim("/World/Scene", "Xform")
                create_prim("/World/Sensors", "Xform")
                add_reference_to_stage(str(scene_usd_path), "/World/Scene")
                add_reference_to_stage(robot_usd_path, "/World/Robot")
                robot = SingleArticulation(prim_path="/World/Robot", name=f"anymal_c_{trajectory_index:03d}")
                cameras, camera_frequency_hz = create_cameras(
                    capture_dt_effective=capture_dt_effective,
                    camera_mounts=camera_mounts,
                )

                world.reset()
                if hasattr(world, "play"):
                    world.play()
                robot.initialize()
                if hasattr(robot, "disable_gravity"):
                    robot.disable_gravity()
                if (not args.show_robot) and (not args.show_ui) and hasattr(robot, "set_visibility"):
                    robot.set_visibility(False)
                gt_world_points = build_scene_mesh_ground_truth_world("/World/Scene")
                trajectory_poses = generate_robot_poses(
                    scene=scene,
                    timesteps=args.timesteps,
                    nominal_base_height=args.nominal_base_height,
                    sample_dt=capture_dt_effective,
                    rng=rng,
                    args=args,
                )

                for step_index in range(args.timesteps):
                    if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                        ui_interrupted = True
                        break
                    write_status(
                        "running",
                        stage="collecting",
                        trajectory_index=int(trajectory_index),
                        timestep_index=int(step_index),
                    )
                    robot_pose = trajectory_poses[step_index]
                    collected_poses.append(robot_pose.astype(np.float32, copy=True))
                    apply_robot_and_camera_pose(robot_pose, robot, cameras, camera_mounts)
                    render_frames(
                        capture_step_count + (args.sensor_warmup_steps if step_index == 0 else 0),
                        world=world,
                    )
                    measurement_local, ground_truth_step_local, visible_fraction, timestep_camera_locals, timestep_debug = capture_timestep_observation(
                        step_index=step_index,
                        robot_pose=robot_pose,
                        cameras=cameras,
                        gt_world_points=gt_world_points,
                        max_points_per_camera=max_points_per_camera,
                        render_retry_fn=lambda count: render_frames(count, world=world),
                    )
                    measurements_local.append(measurement_local)
                    ground_truth_local.append(ground_truth_step_local)
                    visible_ground_truth_fractions.append(visible_fraction)
                    if raw_camera_clouds_local is not None:
                        empty_local = np.empty((0, 3), dtype=np.float32)
                        for camera_name in raw_camera_clouds_local:
                            raw_camera_clouds_local[camera_name].append(timestep_camera_locals.get(camera_name, empty_local))
                    if timestep_debug is not None:
                        trajectory_debug_steps.append(timestep_debug)
            else:
                from isaaclab.envs import ManagerBasedRLEnv
                from isaaclab.terrains import TerrainImporterCfg
                from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg_PLAY

                env_cfg = AnymalCRoughEnvCfg_PLAY()
                env_cfg.scene.num_envs = 1
                env_cfg.scene.env_spacing = 2.5
                env_cfg.episode_length_s = max(60.0, float(args.timesteps) * float(args.capture_dt) * 4.0)
                env_cfg.curriculum = None
                env_cfg.scene.terrain = TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="usd",
                    usd_path=str(scene_usd_path.resolve()),
                )
                env_cfg.sim.device = args.policy_device
                if str(args.policy_device).startswith("cpu"):
                    env_cfg.sim.use_fabric = False

                policy_env = ManagerBasedRLEnv(cfg=env_cfg)
                policy_step_dt = float(policy_env.step_dt)
                capture_step_count = max(1, int(round(args.capture_dt / policy_step_dt)))
                capture_dt_effective = capture_step_count * policy_step_dt
                create_prim("/World/Sensors", "Xform")
                cameras, camera_frequency_hz = create_cameras(
                    capture_dt_effective=capture_dt_effective,
                    camera_mounts=camera_mounts,
                )
                policy = load_torchscript_policy(args.policy_checkpoint, args.policy_device)
                obs, _ = policy_env.reset()
                gt_world_points = build_scene_mesh_ground_truth_world("/World/ground")
                policy_render = getattr(policy_env.sim, "render", None)
                if not callable(policy_render):
                    policy_render = lambda: simulation_app.update()

                for step_index in range(args.timesteps):
                    if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                        ui_interrupted = True
                        break
                    write_status(
                        "running",
                        stage="collecting",
                        trajectory_index=int(trajectory_index),
                        timestep_index=int(step_index),
                    )
                    command_tensor = torch.as_tensor(
                        policy_command_sequence[step_index],
                        dtype=obs["policy"].dtype,
                        device=obs["policy"].device,
                    )
                    for _ in range(capture_step_count):
                        obs["policy"][:, 9:13] = command_tensor.unsqueeze(0)
                        with torch.inference_mode():
                            action = policy(obs["policy"])
                        step_result = policy_env.step(action)
                        if len(step_result) == 5:
                            obs, _, terminated, truncated, _ = step_result
                            terminated_now = bool(torch.any(terminated).item()) or bool(torch.any(truncated).item())
                        else:
                            obs, _, terminated, _ = step_result
                            terminated_now = bool(torch.any(terminated).item()) if torch.is_tensor(terminated) else bool(terminated)
                        if terminated_now:
                            rollout_interrupted = True
                            break

                    robot_pose = policy_env_robot_pose(policy_env)
                    collected_poses.append(robot_pose.astype(np.float32, copy=True))
                    apply_robot_and_camera_pose(robot_pose, None, cameras, camera_mounts)
                    render_frames(
                        args.sensor_warmup_steps if step_index == 0 else 1,
                        render_fn=policy_render,
                    )
                    measurement_local, ground_truth_step_local, visible_fraction, timestep_camera_locals, timestep_debug = capture_timestep_observation(
                        step_index=step_index,
                        robot_pose=robot_pose,
                        cameras=cameras,
                        gt_world_points=gt_world_points,
                        max_points_per_camera=max_points_per_camera,
                        render_retry_fn=lambda count: render_frames(count, render_fn=policy_render),
                    )
                    measurements_local.append(measurement_local)
                    ground_truth_local.append(ground_truth_step_local)
                    visible_ground_truth_fractions.append(visible_fraction)
                    if raw_camera_clouds_local is not None:
                        empty_local = np.empty((0, 3), dtype=np.float32)
                        for camera_name in raw_camera_clouds_local:
                            raw_camera_clouds_local[camera_name].append(timestep_camera_locals.get(camera_name, empty_local))
                    if timestep_debug is not None:
                        timestep_debug["policy_command"] = policy_command_sequence[step_index].astype(np.float32).tolist()
                        trajectory_debug_steps.append(timestep_debug)
                    if rollout_interrupted:
                        break

            if ui_interrupted:
                print(
                    f"[collector] ui requested shutdown after {len(measurements_local)} timesteps of trajectory {trajectory_index:03d}",
                    flush=True,
                )
                write_status(
                    "interrupted",
                    stage="ui_shutdown",
                    trajectory_index=int(trajectory_index),
                    completed_timesteps=int(len(measurements_local)),
                )
                if measurements_local or ground_truth_local:
                    partial_meta = save_trajectory_artifacts(
                        trajectory_index=trajectory_index,
                        trajectory_poses=np.asarray(collected_poses, dtype=np.float32),
                        scene=scene,
                        scene_usd_path=scene_usd_path,
                        capture_step_count=capture_step_count,
                        capture_dt_effective=capture_dt_effective,
                        camera_frequency_hz=camera_frequency_hz,
                        measurements_local=measurements_local,
                        ground_truth_local=ground_truth_local,
                        visible_ground_truth_fractions=visible_ground_truth_fractions,
                        raw_camera_clouds_local=raw_camera_clouds_local,
                        partial=True,
                    )
                    trajectories_meta.append(partial_meta)
                save_debug_artifacts(
                    trajectory_index=trajectory_index,
                    scene=scene,
                    capture_step_count=capture_step_count,
                    capture_dt_effective=capture_dt_effective,
                    camera_frequency_hz=camera_frequency_hz,
                    debug_steps=trajectory_debug_steps,
                    partial=True,
                )
                save_manifest(args.output_dir, args, robot_usd_path, trajectories_meta, camera_mounts)
                return

            if rollout_interrupted:
                print(
                    f"[collector] rollout terminated early for trajectory {trajectory_index:03d}; saving partial trajectory",
                    flush=True,
                )
                partial_meta = save_trajectory_artifacts(
                    trajectory_index=trajectory_index,
                    trajectory_poses=np.asarray(collected_poses, dtype=np.float32),
                    scene=scene,
                    scene_usd_path=scene_usd_path,
                    capture_step_count=capture_step_count,
                    capture_dt_effective=capture_dt_effective,
                    camera_frequency_hz=camera_frequency_hz,
                    measurements_local=measurements_local,
                    ground_truth_local=ground_truth_local,
                    visible_ground_truth_fractions=visible_ground_truth_fractions,
                    raw_camera_clouds_local=raw_camera_clouds_local,
                    partial=True,
                )
                save_debug_artifacts(
                    trajectory_index=trajectory_index,
                    scene=scene,
                    capture_step_count=capture_step_count,
                    capture_dt_effective=capture_dt_effective,
                    camera_frequency_hz=camera_frequency_hz,
                    debug_steps=trajectory_debug_steps,
                    partial=True,
                )
                trajectories_meta.append(partial_meta)
                continue

            write_status("running", stage="saving", trajectory_index=int(trajectory_index))
            trajectory_meta = save_trajectory_artifacts(
                trajectory_index=trajectory_index,
                trajectory_poses=np.asarray(collected_poses, dtype=np.float32),
                scene=scene,
                scene_usd_path=scene_usd_path,
                capture_step_count=capture_step_count,
                capture_dt_effective=capture_dt_effective,
                camera_frequency_hz=camera_frequency_hz,
                measurements_local=measurements_local,
                ground_truth_local=ground_truth_local,
                visible_ground_truth_fractions=visible_ground_truth_fractions,
                raw_camera_clouds_local=raw_camera_clouds_local,
                partial=False,
            )
            save_debug_artifacts(
                trajectory_index=trajectory_index,
                scene=scene,
                capture_step_count=capture_step_count,
                capture_dt_effective=capture_dt_effective,
                camera_frequency_hz=camera_frequency_hz,
                debug_steps=trajectory_debug_steps,
                partial=False,
            )
            trajectories_meta.append(trajectory_meta)
            if int(trajectory_meta["measurement_points"]) == 0:
                print(
                    f"warning: {trajectory_meta['name']} has zero measurement points across all timesteps",
                    flush=True,
                )
            print(
                f"[{trajectory_index + 1}/{args.num_trajectories}] "
                f"saved {trajectory_meta['name']}.npz "
                f"(measurement={trajectory_meta['measurement_points']}, "
                f"gt={trajectory_meta['ground_truth_points']}, "
                f"visible_gt_mean={trajectory_meta['visible_ground_truth_fraction_mean']:.3f})",
                flush=True,
            )

        save_manifest(args.output_dir, args, robot_usd_path, trajectories_meta, camera_mounts)
        write_status("completed", stage="done", trajectories_saved=len(trajectories_meta))
        print(f"Dataset collection completed: {args.output_dir}", flush=True)
    except BaseException as exc:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_status(
            "failed",
            stage="exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_log=str(error_path),
        )
        print(f"[collector] failed with {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def main() -> None:
    args = parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
