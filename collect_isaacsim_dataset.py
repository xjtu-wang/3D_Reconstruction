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
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=800)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--capture-dt", type=float, default=0.10)
    parser.add_argument("--path-length", type=float, default=8.0)
    parser.add_argument("--nominal-base-height", type=float, default=DEFAULT_BODY_HEIGHT)
    parser.add_argument("--ground-truth-spacing", type=float, default=0.05)
    parser.add_argument("--max-measurement-points", type=int, default=20000)
    parser.add_argument("--max-ground-truth-points", type=int, default=40000)
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
        "--debug-camera",
        action="store_true",
        help="Write detailed per-camera diagnostics for each timestep to trajectory_XXX.debug.json.",
    )
    return parser.parse_args()


def sample_axis(extent: float, spacing: float) -> np.ndarray:
    count = max(int(math.floor((2.0 * extent) / spacing)) + 1, 2)
    return np.linspace(-extent, extent, count, dtype=np.float32)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero-length vector cannot be normalized.")
    return (vector / norm).astype(np.float32)


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


def generate_robot_poses(
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


def save_manifest(output_dir: Path, args: argparse.Namespace, robot_usd_path: str, trajectories: List[Dict[str, object]]) -> None:
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
                "cameras": ["front", "back", "left", "right"],
                "tilt_deg": args.camera_down_tilt_deg,
                "resolution": [args.camera_width, args.camera_height],
            },
        },
        "collector_args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
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
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import create_prim, delete_prim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path

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

        def step_render_frames(step_count: int) -> None:
            for _ in range(max(0, step_count)):
                world.step(render=True)

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
            valid_pixels = np.argwhere(valid_mask)
            if debug is not None:
                debug["depth_present"] = True
                debug["depth_shape"] = list(depth.shape)
                debug["depth_valid_pixel_count"] = int(valid_pixels.shape[0])
            if valid_pixels.size == 0:
                return np.empty((0, 3), dtype=np.float32)

            if valid_pixels.shape[0] > args.max_measurement_points:
                indices = rng.choice(valid_pixels.shape[0], size=args.max_measurement_points, replace=False)
                valid_pixels = valid_pixels[indices]
            if debug is not None:
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
            # Isaac Sim get_pointcloud() returns optical camera coordinates:
            # +X right, +Y down, +Z forward. get_world_pose(camera_axes="world")
            # uses +X forward, +Y left, +Z up. Convert before applying the pose.
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

        def capture_camera_points_world(camera: Camera, debug: Optional[Dict[str, object]] = None) -> np.ndarray:
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
                step_render_frames(1)
            if debug is not None:
                debug["attempts"] = attempt_summaries
            return np.empty((0, 3), dtype=np.float32)

        def apply_robot_and_camera_pose(robot_pose: np.ndarray, robot: SingleArticulation, cameras: Dict[str, Camera]) -> None:
            robot_position = robot_pose[:3, 3].astype(np.float32, copy=False)
            robot_orientation = rotation_matrix_to_quaternion(robot_pose[:3, :3])
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

        def save_trajectory_artifacts(
            *,
            trajectory_index: int,
            trajectory_poses: np.ndarray,
            scene: SceneDescription,
            measurements_local: List[np.ndarray],
            ground_truth_local: List[np.ndarray],
            partial: bool,
        ) -> Dict[str, object]:
            measurement_points, measurement_splits = pack_clouds(measurements_local)
            ground_truth_points, ground_truth_splits = pack_clouds(ground_truth_local)

            trajectory_name = f"trajectory_{trajectory_index:03d}"
            suffix = ".partial" if partial else ""
            np.savez_compressed(
                args.output_dir / f"{trajectory_name}{suffix}.npz",
                poses=trajectory_poses.astype(np.float32, copy=False),
                measurement_points=measurement_points,
                measurement_splits=measurement_splits,
                ground_truth_points=ground_truth_points,
                ground_truth_splits=ground_truth_splits,
            )

            trajectory_meta = {
                "name": trajectory_name,
                "partial": bool(partial),
                "debug_camera": bool(args.debug_camera),
                "debug_file": f"{trajectory_name}{suffix}.debug.json" if args.debug_camera else None,
                "timesteps": int(len(measurements_local)),
                "completed_timesteps": int(len(measurements_local)),
                "requested_timesteps": int(args.timesteps),
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
                "all_measurements_empty": bool(measurement_points.shape[0] == 0),
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
                "physics_dt": float(args.physics_dt),
                "capture_dt_requested": float(args.capture_dt),
                "capture_substeps": int(capture_substeps),
                "capture_dt_effective": float(effective_capture_dt),
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

        robot_usd_path = resolve_robot_usd()
        write_status("running", stage="assets_resolved", robot_usd=robot_usd_path)
        print(f"[collector] resolved robot usd: {robot_usd_path}", flush=True)
        world: Optional[World] = None
        stage = omni.usd.get_context().get_stage()

        camera_mounts = build_camera_mounts(args)
        capture_substeps = max(1, int(round(args.capture_dt / args.physics_dt)))
        effective_capture_dt = capture_substeps * args.physics_dt
        camera_frequency_hz = max(1, int(round(1.0 / effective_capture_dt)))
        if not math.isclose(effective_capture_dt, args.capture_dt, rel_tol=0.0, abs_tol=1e-9):
            print(
                f"[collector] adjusted capture_dt from {args.capture_dt:.9f}s "
                f"to {effective_capture_dt:.9f}s to align with physics_dt={args.physics_dt:.9f}s",
                flush=True,
            )
        trajectories_meta: List[Dict[str, object]] = []

        for trajectory_index in range(args.num_trajectories):
            write_status("running", stage="trajectory_setup", trajectory_index=int(trajectory_index))
            print(
                f"[collector] setting up trajectory {trajectory_index + 1}/{args.num_trajectories}",
                flush=True,
            )
            clear_world_instance(world, trajectory_index)
            world = None
            stage = omni.usd.get_context().get_stage()
            for prim_path in ("/World/Scene", "/World/Sensors", "/World/Robot"):
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    delete_prim(prim_path)

            world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
            print(f"[collector] recreated world for trajectory {trajectory_index:03d}", flush=True)
            stage = omni.usd.get_context().get_stage()
            create_prim("/World/Scene", "Xform")
            create_prim("/World/Sensors", "Xform")

            scene = build_random_scene(rng, args)
            gt_world_points = build_scene_ground_truth(scene, args.ground_truth_spacing)
            world_boxes = [world_box_from_local(box, scene.scene_yaw_radians) for box in scene.boxes_local]

            for box in world_boxes:
                FixedCuboid(
                    prim_path=f"/World/Scene/{box.name}",
                    name=box.name,
                    position=box.center.astype(np.float32),
                    orientation=rotation_matrix_to_quaternion(yaw_rotation_matrix(box.yaw_radians)),
                    scale=box.size.astype(np.float32),
                    size=1.0,
                    color=np.array([0.45, 0.45, 0.45], dtype=np.float32),
                )

            add_reference_to_stage(robot_usd_path, "/World/Robot")
            robot = SingleArticulation(prim_path="/World/Robot", name=f"anymal_c_{trajectory_index:03d}")

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
                cameras[mount.name] = camera

            world.reset()
            if hasattr(world, "play"):
                world.play()
            print(f"[collector] world reset complete for trajectory {trajectory_index:03d}", flush=True)
            robot.initialize()
            if hasattr(robot, "disable_gravity"):
                robot.disable_gravity()
            if (not args.show_robot) and (not args.show_ui) and hasattr(robot, "set_visibility"):
                robot.set_visibility(False)
            for camera in cameras.values():
                camera.initialize()
                dt_configured = False
                if hasattr(camera, "set_dt"):
                    try:
                        camera.set_dt(effective_capture_dt)
                        dt_configured = True
                    except Exception as exc:
                        print(
                            f"[collector] camera.set_dt({effective_capture_dt:.9f}) failed for "
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
            print(f"[collector] cameras initialized for trajectory {trajectory_index:03d}", flush=True)

            trajectory_poses = generate_robot_poses(
                scene=scene,
                timesteps=args.timesteps,
                nominal_base_height=args.nominal_base_height,
                rng=rng,
            )

            measurements_local: List[np.ndarray] = []
            ground_truth_local: List[np.ndarray] = []
            trajectory_debug_steps: List[Dict[str, object]] = []

            for step_index in range(args.timesteps):
                if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                    print(
                        f"[collector] ui requested shutdown before timestep {step_index} "
                        f"of trajectory {trajectory_index:03d}",
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
                            trajectory_poses=trajectory_poses,
                            scene=scene,
                            measurements_local=measurements_local,
                            ground_truth_local=ground_truth_local,
                            partial=True,
                        )
                        trajectories_meta.append(partial_meta)
                        print(
                            f"[collector] saved partial trajectory_{trajectory_index:03d}.partial.npz",
                            flush=True,
                        )
                    save_debug_artifacts(
                        trajectory_index=trajectory_index,
                        scene=scene,
                        debug_steps=trajectory_debug_steps,
                        partial=True,
                    )
                    save_manifest(args.output_dir, args, robot_usd_path, trajectories_meta)
                    return

                write_status(
                    "running",
                    stage="collecting",
                    trajectory_index=int(trajectory_index),
                    timestep_index=int(step_index),
                )
                robot_pose = trajectory_poses[step_index]
                # Replace this teleport with the original locomotion policy rollout if you
                # have the checkpoint. The environment and camera rig stay unchanged.
                apply_robot_and_camera_pose(robot_pose, robot, cameras)
                step_render_frames(capture_substeps + (args.sensor_warmup_steps if step_index == 0 else 0))

                measurement_world_chunks: List[np.ndarray] = []
                timestep_debug: Optional[Dict[str, object]] = None
                if args.debug_camera:
                    timestep_debug = {
                        "timestep_index": int(step_index),
                        "robot_position_world": robot_pose[:3, 3].astype(np.float32).tolist(),
                        "robot_rotation_world": robot_pose[:3, :3].astype(np.float32).tolist(),
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

                    points_world = capture_camera_points_world(camera, debug=camera_debug)
                    if camera_debug is not None:
                        camera_debug["points_before_finite_filter"] = int(points_world.shape[0])
                    if points_world.size == 0:
                        print(f"camera frame is empty after retries: {camera.prim_path}", flush=True)
                        if camera_debug is not None:
                            attempts = camera_debug.get("attempts", [])
                            last_attempt = attempts[-1] if attempts else {}
                            print(
                                f"[debug-camera] {camera_name} empty "
                                f"source={last_attempt.get('selected_source', 'unknown')} "
                                f"depth_valid={last_attempt.get('depth_valid_pixel_count', 0)} "
                                f"returned={last_attempt.get('returned_point_count', 0)}",
                                flush=True,
                            )
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
                    measurement_world_chunks.append(points_world.astype(np.float32, copy=False))

                measurement_world = (
                    np.concatenate(measurement_world_chunks, axis=0)
                    if measurement_world_chunks
                    else np.empty((0, 3), dtype=np.float32)
                )
                world_to_robot = invert_transform(robot_pose)
                measurement_local = crop_local_points(
                    transform_points(measurement_world, world_to_robot),
                    GRID_BOUNDS_MIN,
                    GRID_BOUNDS_MAX,
                    args.max_measurement_points,
                    rng,
                )
                ground_truth_step_local = crop_local_points(
                    transform_points(gt_world_points, world_to_robot),
                    GRID_BOUNDS_MIN,
                    GRID_BOUNDS_MAX,
                    args.max_ground_truth_points,
                    rng,
                )
                measurements_local.append(measurement_local)
                ground_truth_local.append(ground_truth_step_local)
                if timestep_debug is not None:
                    timestep_debug["measurement_world_point_count"] = int(measurement_world.shape[0])
                    timestep_debug["measurement_local_point_count"] = int(measurement_local.shape[0])
                    timestep_debug["ground_truth_local_point_count"] = int(ground_truth_step_local.shape[0])
                    trajectory_debug_steps.append(timestep_debug)

            write_status("running", stage="saving", trajectory_index=int(trajectory_index))
            trajectory_meta = save_trajectory_artifacts(
                trajectory_index=trajectory_index,
                trajectory_poses=trajectory_poses,
                scene=scene,
                measurements_local=measurements_local,
                ground_truth_local=ground_truth_local,
                partial=False,
            )
            save_debug_artifacts(
                trajectory_index=trajectory_index,
                scene=scene,
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
                f"gt={trajectory_meta['ground_truth_points']})",
                flush=True,
            )

        save_manifest(args.output_dir, args, robot_usd_path, trajectories_meta)
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
