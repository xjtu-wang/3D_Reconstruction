from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCENE_LENGTH = 16.0
WALL_THICKNESS = 0.1
WALL_HEIGHT = 3.0
START_WALL_ENABLED = False
GOAL_WALL_ENABLED = True
SIDE_WALL_MODE = "offset"
SIDE_WALL_OUTWARD_OFFSET = 1.2

STAIR_START_X = 2.2
STAIR_STEP_DEPTH = 0.3
STAIR_STEP_HEIGHT = 0.2
STAIR_NUM_STEPS = 10

BOX_COUNT = 10
POLE_COUNT = 16
POLE_RADIUS = 0.4
POLE_HEIGHT = 3.0
BOX_SIZE_X_RANGE = (1.5, 2.0)
BOX_SIZE_Y_RANGE = (1.5, 2.0)
BOX_SIZE_Z_RANGE = (0.22, 0.28)
BOX_SLOT_X_MARGIN = 0.08
BOX_LANE_Y_MARGIN = 0.05
BOX_CENTER_INWARD_SHIFT = 0.18
BOX_STAIR_CLEARANCE_X = STAIR_START_X + 0.45
CENTER_CLEAR_WIDTH = 1.2
DEFAULT_CENTER_CLEAR_WIDTH_MIN = 0.55
DEFAULT_CENTER_CLEAR_WIDTH_RATIO = 0.30
LAYOUT_EDGE_X_MARGIN = 0.1
POLE_LAYOUT_Y_MARGIN = 0.1

START_BASE_X = -SCENE_LENGTH / 2.0 + 1.0
START_BASE_Y = 0.0
START_BASE_YAW = 0.0
DEFAULT_BASE_Z = 0.60

COMMAND_LIN_VEL_X_RANGE = (0.6, 1.2)
COMMAND_LIN_VEL_Y_RANGE = (-0.15, 0.15)
COMMAND_HEADING_RANGE_DEG = (-6.0, 6.0)

RECORD_CAMERA_WIDTH = 1280
RECORD_CAMERA_HEIGHT = 720
RECORD_CAMERA_WARMUP_FRAMES = 0
RECORD_CAMERA_FOCAL_LENGTH = 12.0
RECORD_CAMERA_HORIZONTAL_APERTURE = 20.955

FLOOR_THICKNESS = 0.10
FLOOR_EXTRA_LENGTH = 2.0
FLOOR_EXTRA_WIDTH = 2.0
FLOOR_SIDE_MARGIN = 0.4
CYLINDER_MESH_SEGMENTS = 20

GROUND_TRUTH_SAMPLE_SPACING = 0.025
GRID_BOUNDS_MIN = np.array([-1.6, -1.6, -1.6], dtype=np.float32)
GRID_BOUNDS_MAX = np.array([1.6, 1.6, 1.6], dtype=np.float32)
GRID_SHAPE = (64, 64, 64)
GRID_VOXEL_SIZE = 0.05
VISIBILITY_TARGET = 0.43

DEPTH_CAMERA_WIDTH = 424
DEPTH_CAMERA_HEIGHT = 240
DEPTH_CAMERA_HFOV_DEG = 86.0
DEPTH_CAMERA_VFOV_DEG = 57.0
DEPTH_CAMERA_FOCAL_LENGTH_MM = 1.93
DEPTH_CAMERA_NEAR = 0.15
DEPTH_CAMERA_FAR = 3.5
DEPTH_CAMERA_PIXEL_STRIDE = 2
DEPTH_CAMERA_MAX_POINTS = 32768
CAMERA_DOWN_TILT_DEG = 30.0

RAYCAST_TERRAIN_GROUP_PRIM_PATH = "/World/RaycastTerrain"
RAYCAST_TERRAIN_MESH_PRIM_PATH = f"{RAYCAST_TERRAIN_GROUP_PRIM_PATH}/TerrainMesh"
IMPORTED_RAYCAST_TERRAIN_PRIM_PATH = "/World/ground/terrain/RaycastTerrain"


@dataclass(frozen=True)
class CuboidSpec:
    name: str
    prim_path: str
    center: np.ndarray
    size: np.ndarray
    color: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prim_path": self.prim_path,
            "center": vector_to_list(self.center),
            "size": vector_to_list(self.size),
            "color": [float(value) for value in self.color],
        }


@dataclass(frozen=True)
class CylinderSpec:
    name: str
    prim_path: str
    center: np.ndarray
    radius: float
    height: float
    color: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prim_path": self.prim_path,
            "center": vector_to_list(self.center),
            "radius": float(self.radius),
            "height": float(self.height),
            "color": [float(value) for value in self.color],
        }


@dataclass(frozen=True)
class CameraMount:
    name: str
    translation_in_base: np.ndarray
    forward_in_base: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "translation_in_base": vector_to_list(self.translation_in_base),
            "forward_in_base": vector_to_list(self.forward_in_base),
        }


@dataclass(frozen=True)
class DepthCameraSpec:
    name: str
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    fx: float
    fy: float
    cx: float
    cy: float
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    near_m: float
    far_m: float

    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": int(self.width),
            "height": int(self.height),
            "hfov_deg": float(self.hfov_deg),
            "vfov_deg": float(self.vfov_deg),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "focal_length_mm": float(self.focal_length_mm),
            "horizontal_aperture_mm": float(self.horizontal_aperture_mm),
            "vertical_aperture_mm": float(self.vertical_aperture_mm),
            "near_m": float(self.near_m),
            "far_m": float(self.far_m),
            "pixel_stride": int(DEPTH_CAMERA_PIXEL_STRIDE),
            "max_points_per_camera": int(DEPTH_CAMERA_MAX_POINTS),
        }


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    corridor_width: float
    stair_width: float
    center_clear_width: float
    side_lane_width: float
    box_size: np.ndarray
    floor: CuboidSpec
    walls: tuple[CuboidSpec, ...]
    stair_positive: tuple[CuboidSpec, ...]
    stair_negative: tuple[CuboidSpec, ...]
    boxes: tuple[CuboidSpec, ...]
    poles: tuple[CylinderSpec, ...]

    @property
    def all_cuboids(self) -> tuple[CuboidSpec, ...]:
        return (self.floor,) + self.walls + self.stair_positive + self.stair_negative + self.boxes

    @property
    def stair_zone(self) -> dict[str, float]:
        return {
            "x_min": float(-STAIR_START_X),
            "x_max": float(STAIR_START_X),
            "abs_y_max": float(self.stair_width * 0.5),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "length": float(SCENE_LENGTH),
            "corridor_width": float(self.corridor_width),
            "corridor_width_semantics": "obstacle_layout_width",
            "center_clear_width": float(self.center_clear_width),
            "side_lane_width": float(self.side_lane_width),
            "wall_thickness": float(WALL_THICKNESS),
            "wall_height": float(WALL_HEIGHT),
            "floor_thickness": float(FLOOR_THICKNESS),
            "box_size": vector_to_list(self.box_size),
            "box_layout": {
                "mode": "fixed_uniform_side_slots",
                "center_clear_width": float(self.center_clear_width),
                "side_lane_width": float(self.side_lane_width),
                "center_inward_shift": float(BOX_CENTER_INWARD_SHIFT),
                "slot_count": BOX_COUNT // 2,
                "stair_clearance_x": float(BOX_STAIR_CLEARANCE_X),
                "slot_spacing_x": float(
                    (2.0 * ((SCENE_LENGTH / 2.0 - 1.0) - BOX_STAIR_CLEARANCE_X)) / float(BOX_COUNT // 2)
                ),
                "guaranteed_non_overlap": True,
            },
            "stair": {
                "start_x_positive": float(STAIR_START_X),
                "start_x_negative": float(-STAIR_START_X),
                "step_depth": float(STAIR_STEP_DEPTH),
                "step_height": float(STAIR_STEP_HEIGHT),
                "num_steps": int(STAIR_NUM_STEPS),
                "stair_width": float(self.stair_width),
            },
            "wall_layout": {
                "start_wall_enabled": bool(START_WALL_ENABLED),
                "goal_wall_enabled": bool(GOAL_WALL_ENABLED),
                "side_wall_mode": SIDE_WALL_MODE,
                "side_wall_outward_offset": float(SIDE_WALL_OUTWARD_OFFSET),
                "side_wall_half_span": resolve_side_wall_half_span(self.corridor_width),
            },
            "pole": {
                "radius": float(POLE_RADIUS),
                "height": float(POLE_HEIGHT),
            },
            "counts": {
                "walls": int(len(self.walls)),
                "stair_groups": 2,
                "stair_steps_total": int(len(self.stair_positive) + len(self.stair_negative)),
                "boxes": int(len(self.boxes)),
                "poles": int(len(self.poles)),
            },
            "stair_zone": self.stair_zone,
            "floor": self.floor.to_dict(),
            "walls": [wall.to_dict() for wall in self.walls],
            "stairs": {
                "positive": [step.to_dict() for step in self.stair_positive],
                "negative": [step.to_dict() for step in self.stair_negative],
            },
            "boxes": [box.to_dict() for box in self.boxes],
            "poles": [pole.to_dict() for pole in self.poles],
        }


def add_rollout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--policy-checkpoint", type=Path, required=True, help="TorchScript/JIT rough-terrain policy checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory that receives summary.json and trajectory.npz, plus optional debug artifacts.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for scene generation and command sampling.")
    parser.add_argument("--steps", type=int, default=250, help="Number of rollout steps to run.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim headless. Use --no-headless to keep the UI visible.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="IsaacLab simulation and policy device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--export-usd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the authored scene USD into output-dir. If disabled, the script uses a temporary /tmp USD only for runtime import.",
    )
    parser.add_argument(
        "--record-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export debug PNG frames from the record camera.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If imageio is available, also stream record-camera frames into rollout.mp4.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one DataCollect-style corridor scene, export it as USD, "
            "run an IsaacLab ANYmal C rollout, and save real depth-camera measurements."
        )
    )
    add_rollout_args(parser)
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    return args


def vector_to_list(values: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1).tolist()]


def ensure_output_dir(output_dir: Path) -> Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {exc}") from exc
    return output_dir.resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def log_progress(message: str) -> None:
    print(f"[rollout] {message}", flush=True)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32)


def yaw_to_quaternion_wxyz(yaw_radians: float) -> np.ndarray:
    half = 0.5 * float(yaw_radians)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quaternion_to_yaw(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def quaternion_conjugate_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=np.float32)


def quaternion_multiply_wxyz(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = [float(value) for value in lhs_wxyz]
    rw, rx, ry, rz = [float(value) for value in rhs_wxyz]
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def quaternion_rotate_vector_wxyz(quaternion_wxyz: np.ndarray, vector_xyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-8)
    pure_quaternion = np.array([0.0, vector_xyz[0], vector_xyz[1], vector_xyz[2]], dtype=np.float32)
    rotated = quaternion_multiply_wxyz(
        quaternion_multiply_wxyz(quaternion, pure_quaternion),
        quaternion_conjugate_wxyz(quaternion),
    )
    return rotated[1:].astype(np.float32, copy=False)


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


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


def rotation_from_forward(forward: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32) if up_hint is None else normalize(up_hint)
    x_axis = normalize(forward)
    if abs(float(np.dot(x_axis, up_hint))) > 0.995:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y_axis = normalize(np.cross(up_hint, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def angle_between_vectors_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_unit = normalize(lhs)
    rhs_unit = normalize(rhs)
    dot_value = float(np.clip(np.dot(lhs_unit, rhs_unit), -1.0, 1.0))
    return float(math.degrees(math.acos(dot_value)))


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(position, dtype=np.float32)
    return transform


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


def pack_clouds(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    packed: list[np.ndarray] = []
    splits = [0]
    for cloud in clouds:
        points = np.asarray(cloud, dtype=np.float32)
        packed.append(points)
        splits.append(splits[-1] + points.shape[0])
    packed_points = np.concatenate(packed, axis=0) if packed else np.empty((0, 3), dtype=np.float32)
    return packed_points, np.asarray(splits, dtype=np.int32)


def write_packed_trajectory_npz(
    path: Path,
    *,
    poses: list[np.ndarray],
    measurements_local: list[np.ndarray],
    ground_truth_local: list[np.ndarray],
    step_indices: list[int],
    timestamps_s: list[float],
    visibility_fractions: list[float],
) -> None:
    if poses:
        poses_array = np.stack([np.asarray(pose, dtype=np.float32) for pose in poses], axis=0)
    else:
        poses_array = np.empty((0, 4, 4), dtype=np.float32)
    measurement_points, measurement_splits = pack_clouds(measurements_local)
    ground_truth_points, ground_truth_splits = pack_clouds(ground_truth_local)
    np.savez_compressed(
        path,
        poses=poses_array,
        measurement_points=measurement_points.astype(np.float32, copy=False),
        measurement_splits=measurement_splits.astype(np.int32, copy=False),
        ground_truth_points=ground_truth_points.astype(np.float32, copy=False),
        ground_truth_splits=ground_truth_splits.astype(np.int32, copy=False),
        step_indices=np.asarray(step_indices, dtype=np.int32),
        timestamps_s=np.asarray(timestamps_s, dtype=np.float32),
        visibility_fractions=np.asarray(visibility_fractions, dtype=np.float32),
    )


def voxel_downsample_centroids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if voxel_size <= 0.0:
        return np.asarray(points, dtype=np.float32)
    voxel_indices = np.floor(np.asarray(points, dtype=np.float32) / voxel_size).astype(np.int32)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    centroids = np.zeros((int(np.max(inverse_indices)) + 1, 3), dtype=np.float32)
    counts = np.zeros((centroids.shape[0], 1), dtype=np.float32)
    np.add.at(centroids, inverse_indices, np.asarray(points, dtype=np.float32))
    np.add.at(counts, inverse_indices, 1.0)
    centroids /= np.maximum(counts, 1.0)
    return centroids.astype(np.float32, copy=False)


def crop_local_points(points_local: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
    if points_local.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    finite_mask = np.isfinite(points_local).all(axis=1)
    cropped = np.asarray(points_local, dtype=np.float32)[finite_mask]
    if cropped.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    mask = np.all(cropped >= min_bounds[None, :], axis=1)
    mask &= np.all(cropped <= max_bounds[None, :], axis=1)
    return cropped[mask].astype(np.float32, copy=False)


def voxelize_points(points_local: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=np.uint8)
    if points_local.size == 0:
        return grid
    span = (max_bounds - min_bounds).astype(np.float32)
    voxel_size = span / np.asarray(grid_shape, dtype=np.float32)
    indices = np.floor((points_local.astype(np.float32) - min_bounds[None, :]) / voxel_size[None, :]).astype(np.int32)
    valid_mask = np.all(indices >= 0, axis=1)
    valid_mask &= np.all(indices < np.asarray(grid_shape, dtype=np.int32)[None, :], axis=1)
    indices = indices[valid_mask]
    if indices.size > 0:
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return grid


def visible_fraction_from_voxels(ground_truth_voxels: np.ndarray, measurement_voxels: np.ndarray) -> float:
    gt_count = int(np.count_nonzero(ground_truth_voxels))
    if gt_count == 0:
        return 0.0
    overlap = int(np.count_nonzero((ground_truth_voxels > 0) & (measurement_voxels > 0)))
    return float(overlap / gt_count)


def write_chunk(handle, chunk_type: bytes, data: bytes) -> None:
    handle.write(struct.pack(">I", len(data)))
    handle.write(chunk_type)
    handle.write(data)
    handle.write(struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF))


def write_png(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"PNG writer expects HxWx3 or HxWx4 image, received shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.asarray(array, dtype=np.float32)
        scale = 255.0 if array.size > 0 and float(np.nanmax(array)) <= 1.0 + 1e-6 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    color_type = 2 if array.shape[2] == 3 else 6
    height, width, channels = array.shape
    row_bytes = width * channels
    raw = bytearray((row_bytes + 1) * height)
    for row_index in range(height):
        row_start = row_index * (row_bytes + 1)
        raw[row_start] = 0
        row_slice = array[row_index].reshape(-1)
        raw[row_start + 1 : row_start + 1 + row_bytes] = row_slice.tobytes()
    compressed = zlib.compress(bytes(raw), level=6)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        write_chunk(handle, b"IHDR", ihdr)
        write_chunk(handle, b"IDAT", compressed)
        write_chunk(handle, b"IEND", b"")


def coerce_image_to_rgb_u8(payload: Any) -> np.ndarray:
    array = np.asarray(payload)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise RuntimeError(f"Unsupported RGB frame payload shape: {array.shape}")
    if array.dtype != np.uint8:
        array = np.asarray(array, dtype=np.float32)
        scale = 255.0 if array.size > 0 and float(np.nanmax(array)) <= 1.0 + 1e-6 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def import_imageio_writer_if_available(video_path: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[rollout] imageio unavailable, skipping MP4 export: {exc}", file=sys.stderr, flush=True)
        return None
    try:
        return imageio.get_writer(str(video_path), fps=max(1, int(fps)))
    except Exception as exc:
        print(f"[rollout] failed to initialize MP4 writer {video_path}: {exc}", file=sys.stderr, flush=True)
        return None


def make_cuboid(
    name: str,
    prim_path: str,
    center: list[float] | tuple[float, ...],
    size: list[float] | tuple[float, ...],
    color: tuple[float, float, float],
) -> CuboidSpec:
    return CuboidSpec(
        name=name,
        prim_path=prim_path,
        center=np.asarray(center, dtype=np.float32),
        size=np.asarray(size, dtype=np.float32),
        color=color,
    )


def make_cylinder(
    name: str,
    prim_path: str,
    center: list[float] | tuple[float, ...],
    radius: float,
    height: float,
    color: tuple[float, float, float],
) -> CylinderSpec:
    return CylinderSpec(
        name=name,
        prim_path=prim_path,
        center=np.asarray(center, dtype=np.float32),
        radius=float(radius),
        height=float(height),
        color=color,
    )


def build_depth_camera_spec(width: int = DEPTH_CAMERA_WIDTH, height: int = DEPTH_CAMERA_HEIGHT) -> DepthCameraSpec:
    hfov_radians = math.radians(DEPTH_CAMERA_HFOV_DEG)
    vfov_radians = math.radians(DEPTH_CAMERA_VFOV_DEG)
    fx = float(width / (2.0 * math.tan(0.5 * hfov_radians)))
    fy = float(height / (2.0 * math.tan(0.5 * vfov_radians)))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    horizontal_aperture_mm = 2.0 * DEPTH_CAMERA_FOCAL_LENGTH_MM * math.tan(0.5 * hfov_radians)
    vertical_aperture_mm = 2.0 * DEPTH_CAMERA_FOCAL_LENGTH_MM * math.tan(0.5 * vfov_radians)
    return DepthCameraSpec(
        name="realsense_depth",
        width=int(width),
        height=int(height),
        hfov_deg=float(DEPTH_CAMERA_HFOV_DEG),
        vfov_deg=float(DEPTH_CAMERA_VFOV_DEG),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        focal_length_mm=float(DEPTH_CAMERA_FOCAL_LENGTH_MM),
        horizontal_aperture_mm=float(horizontal_aperture_mm),
        vertical_aperture_mm=float(vertical_aperture_mm),
        near_m=float(DEPTH_CAMERA_NEAR),
        far_m=float(DEPTH_CAMERA_FAR),
    )


def build_camera_mounts() -> tuple[CameraMount, ...]:
    tilt = math.radians(CAMERA_DOWN_TILT_DEG)
    forward_component = math.cos(tilt)
    down_component = math.sin(tilt)
    return (
        CameraMount(
            name="front",
            translation_in_base=np.array([0.42, 0.00, 0.18], dtype=np.float32),
            forward_in_base=np.array([forward_component, 0.0, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="back",
            translation_in_base=np.array([-0.42, 0.00, 0.18], dtype=np.float32),
            forward_in_base=np.array([-forward_component, 0.0, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="left",
            translation_in_base=np.array([0.00, 0.24, 0.18], dtype=np.float32),
            forward_in_base=np.array([0.0, forward_component, -down_component], dtype=np.float32),
        ),
        CameraMount(
            name="right",
            translation_in_base=np.array([0.00, -0.24, 0.18], dtype=np.float32),
            forward_in_base=np.array([0.0, -forward_component, -down_component], dtype=np.float32),
        ),
    )


def depth_camera_sensor_name(mount_name: str) -> str:
    return f"{mount_name}_depth_camera"


def compressed_axis_to_scene_x(distance: float, interval_length: float, x_max: float, x_clear: float) -> float:
    if distance < interval_length:
        return float(-x_max + distance)
    return float(x_clear + (distance - interval_length))


def boxes_overlap_aabb(center_a: np.ndarray, size_a: np.ndarray, center_b: np.ndarray, size_b: np.ndarray, margin: float) -> bool:
    delta = np.abs(center_a[:3] - center_b[:3])
    limits = 0.5 * (size_a[:3] + size_b[:3]) + margin
    return bool(np.all(delta < limits))


def build_uniform_boxes(
    rng: np.random.Generator,
    center_clear_width: float,
    side_lane_width: float,
) -> tuple[np.ndarray, tuple[CuboidSpec, ...]]:
    x_max = SCENE_LENGTH / 2.0 - 1.0
    interval_length = x_max - BOX_STAIR_CLEARANCE_X
    slot_count = BOX_COUNT // 2
    compressed_length = 2.0 * interval_length
    slot_spacing = compressed_length / float(slot_count)

    box_size_x_max = min(BOX_SIZE_X_RANGE[1], slot_spacing - BOX_SLOT_X_MARGIN)
    box_size_y_max = min(BOX_SIZE_Y_RANGE[1], side_lane_width - 2.0 * BOX_LANE_Y_MARGIN)
    if box_size_x_max < BOX_SIZE_X_RANGE[0]:
        raise RuntimeError(
            "Box slot spacing is too tight for the requested fixed-slot box layout: "
            f"slot_spacing={slot_spacing:.3f}, max_box_x={box_size_x_max:.3f}"
        )
    if box_size_y_max < BOX_SIZE_Y_RANGE[0]:
        raise RuntimeError(
            "Corridor side lane is too narrow for the requested fixed-slot box layout: "
            f"side_lane_width={side_lane_width:.3f}, max_box_y={box_size_y_max:.3f}"
        )

    box_size = np.array(
        [
            float(rng.uniform(BOX_SIZE_X_RANGE[0], box_size_x_max)),
            float(rng.uniform(BOX_SIZE_Y_RANGE[0], box_size_y_max)),
            float(rng.uniform(*BOX_SIZE_Z_RANGE)),
        ],
        dtype=np.float32,
    )

    lane_center_offset = max(
        0.5 * float(box_size[1]),
        0.5 * (center_clear_width + side_lane_width) - BOX_CENTER_INWARD_SHIFT,
    )
    z_center = float(box_size[2] * 0.5)

    boxes: list[CuboidSpec] = []
    for slot_index in range(slot_count):
        compressed_center = (slot_index + 0.5) * slot_spacing
        base_x = compressed_axis_to_scene_x(compressed_center, interval_length, x_max, BOX_STAIR_CLEARANCE_X)
        for lane_name, lane_sign in (("left", 1.0), ("right", -1.0)):
            name = f"box_{lane_name}_{slot_index:02d}"
            prim_path = f"/World/Boxes/{name}"
            center = np.array(
                [
                    float(base_x),
                    float(lane_sign * lane_center_offset),
                    z_center,
                ],
                dtype=np.float32,
            )
            boxes.append(
                make_cuboid(
                    name=name,
                    prim_path=prim_path,
                    center=center.tolist(),
                    size=box_size.tolist(),
                    color=(0.20, 0.50, 0.82),
                )
            )
    return box_size, tuple(boxes)


def build_uniform_poles(corridor_width: float) -> tuple[CylinderSpec, ...]:
    if POLE_COUNT % 2 != 0:
        raise RuntimeError(f"POLE_COUNT must be even for the mirrored fixed pole layout, received {POLE_COUNT}.")
    x_min = -SCENE_LENGTH / 2.0 + LAYOUT_EDGE_X_MARGIN + POLE_RADIUS
    x_max = SCENE_LENGTH / 2.0 - LAYOUT_EDGE_X_MARGIN - POLE_RADIUS
    x_positions = np.linspace(x_min, x_max, POLE_COUNT // 2, dtype=np.float32)
    y_negative = -corridor_width / 2.0 + POLE_LAYOUT_Y_MARGIN + POLE_RADIUS
    y_positive = corridor_width / 2.0 - POLE_LAYOUT_Y_MARGIN - POLE_RADIUS

    poles: list[CylinderSpec] = []
    for lane_index, y_position in enumerate((y_negative, y_positive)):
        for offset, x_position in enumerate(x_positions):
            pole_index = lane_index * (POLE_COUNT // 2) + offset
            poles.append(
                make_cylinder(
                    name=f"pole_{pole_index:02d}",
                    prim_path=f"/World/Poles/Pole_{pole_index:02d}",
                    center=[float(x_position), float(y_position), POLE_HEIGHT / 2.0],
                    radius=POLE_RADIUS,
                    height=POLE_HEIGHT,
                    color=(0.30, 0.30, 0.30),
                )
            )
    return tuple(poles)


def compute_default_center_clear_width(corridor_width: float) -> float:
    return float(max(DEFAULT_CENTER_CLEAR_WIDTH_MIN, DEFAULT_CENTER_CLEAR_WIDTH_RATIO * corridor_width))


def resolve_center_clear_width(corridor_width: float) -> float:
    if CENTER_CLEAR_WIDTH is None:
        return compute_default_center_clear_width(corridor_width)
    if CENTER_CLEAR_WIDTH <= 0.0:
        raise RuntimeError(f"CENTER_CLEAR_WIDTH must be positive when set, received {CENTER_CLEAR_WIDTH}.")
    return float(CENTER_CLEAR_WIDTH)


def compute_side_lane_width(corridor_width: float, center_clear_width: float) -> float:
    return float(0.5 * (corridor_width - center_clear_width))


def resolve_side_wall_half_span(corridor_width: float) -> float | None:
    if SIDE_WALL_MODE == "disabled":
        return None
    if SIDE_WALL_MODE == "aligned":
        return float(0.5 * corridor_width)
    if SIDE_WALL_MODE == "offset":
        if SIDE_WALL_OUTWARD_OFFSET < 0.0:
            raise RuntimeError(
                f"SIDE_WALL_OUTWARD_OFFSET must be non-negative, received {SIDE_WALL_OUTWARD_OFFSET}."
            )
        return float(0.5 * corridor_width + SIDE_WALL_OUTWARD_OFFSET)
    raise RuntimeError(f"Unsupported SIDE_WALL_MODE={SIDE_WALL_MODE!r}. Expected one of disabled/aligned/offset.")


def expected_wall_count() -> int:
    count = int(START_WALL_ENABLED) + int(GOAL_WALL_ENABLED)
    if SIDE_WALL_MODE != "disabled":
        count += 2
    return count


def resolve_floor_width(corridor_width: float) -> float:
    floor_width = float(corridor_width + FLOOR_EXTRA_WIDTH)
    side_wall_half_span = resolve_side_wall_half_span(corridor_width)
    if side_wall_half_span is None:
        return floor_width
    wall_span_width = 2.0 * (side_wall_half_span + 0.5 * WALL_THICKNESS)
    return float(max(floor_width, wall_span_width + FLOOR_SIDE_MARGIN))


def build_walls(corridor_width: float) -> tuple[CuboidSpec, ...]:
    walls: list[CuboidSpec] = []
    if START_WALL_ENABLED:
        walls.append(
            make_cuboid(
                name="wall_start_x",
                prim_path="/World/Walls/WallStartX",
                center=[-SCENE_LENGTH / 2.0, 0.0, WALL_HEIGHT / 2.0],
                size=[WALL_THICKNESS, corridor_width, WALL_HEIGHT],
                color=(0.72, 0.72, 0.72),
            )
        )
    if GOAL_WALL_ENABLED:
        walls.append(
            make_cuboid(
                name="wall_goal_x",
                prim_path="/World/Walls/WallGoalX",
                center=[SCENE_LENGTH / 2.0, 0.0, WALL_HEIGHT / 2.0],
                size=[WALL_THICKNESS, corridor_width, WALL_HEIGHT],
                color=(0.72, 0.72, 0.72),
            )
        )

    side_wall_half_span = resolve_side_wall_half_span(corridor_width)
    if side_wall_half_span is not None:
        walls.append(
            make_cuboid(
                name="wall_negative_y",
                prim_path="/World/Walls/WallNegativeY",
                center=[0.0, -side_wall_half_span, WALL_HEIGHT / 2.0],
                size=[SCENE_LENGTH, WALL_THICKNESS, WALL_HEIGHT],
                color=(0.72, 0.72, 0.72),
            )
        )
        walls.append(
            make_cuboid(
                name="wall_positive_y",
                prim_path="/World/Walls/WallPositiveY",
                center=[0.0, side_wall_half_span, WALL_HEIGHT / 2.0],
                size=[SCENE_LENGTH, WALL_THICKNESS, WALL_HEIGHT],
                color=(0.72, 0.72, 0.72),
            )
        )
    return tuple(walls)


def resolve_corridor_width_sampling_range() -> tuple[float, float]:
    corridor_width_min = 2.0
    corridor_width_max = 6.0

    required_side_lane_width = BOX_SIZE_Y_RANGE[0] + 2.0 * BOX_LANE_Y_MARGIN
    required_corridor_half_width = POLE_LAYOUT_Y_MARGIN + POLE_RADIUS
    required_corridor_width = max(
        corridor_width_min,
        2.0 * required_corridor_half_width,
    )

    candidate_widths = np.linspace(required_corridor_width, corridor_width_max, 4001, dtype=np.float32)
    valid_widths = [
        float(width)
        for width in candidate_widths
        if compute_side_lane_width(
            float(width),
            resolve_center_clear_width(float(width)),
        )
        >= required_side_lane_width
    ]
    if not valid_widths:
        mode_description = (
            f"script CENTER_CLEAR_WIDTH={CENTER_CLEAR_WIDTH:.3f} m"
            if CENTER_CLEAR_WIDTH is not None
            else (
                "default center-clear rule max("
                f"{DEFAULT_CENTER_CLEAR_WIDTH_MIN:.2f}, {DEFAULT_CENTER_CLEAR_WIDTH_RATIO:.2f} * corridor_width)"
            )
        )
        raise RuntimeError(
            "The requested obstacle dimensions cannot fit inside the DataCollect corridor width range [2.0, 6.0]. "
            f"Need side_lane_width >= {required_side_lane_width:.3f} m for BOX_SIZE_Y_RANGE={BOX_SIZE_Y_RANGE}, "
            f"but {mode_description} only allows up to "
            f"{compute_side_lane_width(corridor_width_max, resolve_center_clear_width(corridor_width_max)):.3f} m."
        )
    return valid_widths[0], corridor_width_max


def build_scene(rng: np.random.Generator, seed: int) -> SceneSpec:
    if BOX_COUNT % 2 != 0:
        raise RuntimeError(f"BOX_COUNT must be even for the mirrored lane layout, received {BOX_COUNT}.")

    corridor_width_range = resolve_corridor_width_sampling_range()
    corridor_width = float(rng.uniform(*corridor_width_range))
    stair_width = float(corridor_width - 0.5)
    if stair_width <= 0.0:
        raise RuntimeError(f"Invalid stair width generated: {stair_width}")

    center_clear_width = resolve_center_clear_width(corridor_width)
    side_lane_width = compute_side_lane_width(corridor_width, center_clear_width)
    if side_lane_width <= 0.0:
        raise RuntimeError(f"Invalid side-lane width generated: {side_lane_width}")
    if center_clear_width >= corridor_width:
        raise RuntimeError(
            f"Requested center clear width {center_clear_width:.3f} m leaves no side lanes inside layout width {corridor_width:.3f} m."
        )

    floor = make_cuboid(
        name="floor",
        prim_path="/World/Floor",
        center=[0.0, 0.0, -0.5 * FLOOR_THICKNESS],
        size=[SCENE_LENGTH + FLOOR_EXTRA_LENGTH, resolve_floor_width(corridor_width), FLOOR_THICKNESS],
        color=(0.55, 0.55, 0.55),
    )

    walls = build_walls(corridor_width)

    stair_positive: list[CuboidSpec] = []
    stair_negative: list[CuboidSpec] = []
    for step_index in range(STAIR_NUM_STEPS):
        size = np.array([STAIR_STEP_DEPTH, stair_width, STAIR_STEP_HEIGHT * (step_index + 1)], dtype=np.float32)
        z_center = STAIR_STEP_HEIGHT / 2.0 + step_index * STAIR_STEP_HEIGHT
        stair_positive.append(
            make_cuboid(
                name=f"stair_positive_step_{step_index:02d}",
                prim_path=f"/World/Stairs/StairPositive/Step_{step_index:02d}",
                center=[STAIR_START_X - step_index * STAIR_STEP_DEPTH, 0.0, z_center],
                size=size.tolist(),
                color=(0.85, 0.67, 0.25),
            )
        )
        stair_negative.append(
            make_cuboid(
                name=f"stair_negative_step_{step_index:02d}",
                prim_path=f"/World/Stairs/StairNegative/Step_{step_index:02d}",
                center=[-STAIR_START_X + step_index * STAIR_STEP_DEPTH, 0.0, z_center],
                size=size.tolist(),
                color=(0.85, 0.67, 0.25),
            )
        )

    box_size, boxes = build_uniform_boxes(rng, center_clear_width, side_lane_width)

    poles = build_uniform_poles(corridor_width)

    return SceneSpec(
        seed=int(seed),
        corridor_width=corridor_width,
        stair_width=stair_width,
        center_clear_width=center_clear_width,
        side_lane_width=side_lane_width,
        box_size=box_size,
        floor=floor,
        walls=walls,
        stair_positive=tuple(stair_positive),
        stair_negative=tuple(stair_negative),
        boxes=boxes,
        poles=poles,
    )


def validate_scene_spec(scene: SceneSpec) -> None:
    expected_walls = expected_wall_count()
    if len(scene.walls) != expected_walls:
        raise RuntimeError(f"Expected {expected_walls} walls, found {len(scene.walls)}.")
    if len(scene.stair_positive) != STAIR_NUM_STEPS or len(scene.stair_negative) != STAIR_NUM_STEPS:
        raise RuntimeError("Scene stairs do not match the requested 2 x 10 step layout.")
    if len(scene.boxes) != BOX_COUNT:
        raise RuntimeError(f"Expected {BOX_COUNT} boxes, found {len(scene.boxes)}.")
    if len(scene.poles) != POLE_COUNT:
        raise RuntimeError(f"Expected {POLE_COUNT} poles, found {len(scene.poles)}.")


def sample_command(rng: np.random.Generator) -> dict[str, float]:
    heading_deg = float(rng.uniform(*COMMAND_HEADING_RANGE_DEG))
    heading_rad = float(math.radians(heading_deg))
    return {
        "lin_vel_x": float(rng.uniform(*COMMAND_LIN_VEL_X_RANGE)),
        "lin_vel_y": float(rng.uniform(*COMMAND_LIN_VEL_Y_RANGE)),
        "ang_vel_z": 0.0,
        "heading_rad": heading_rad,
        "heading_deg": heading_deg,
    }


def entered_stair_zone(scene: SceneSpec, position_scene: np.ndarray) -> bool:
    x = float(position_scene[0])
    y = float(position_scene[1])
    return (-STAIR_START_X <= x <= STAIR_START_X) and (abs(y) <= scene.stair_width * 0.5)


def build_cuboid_mesh(spec: CuboidSpec, top_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * spec.size.astype(np.float32)
    cx, cy, cz = [float(value) for value in spec.center]
    hx, hy, hz = [float(value) for value in half]
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    if top_only:
        faces = np.array([[4, 6, 5], [4, 7, 6]], dtype=np.int32)
    else:
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )
    return vertices, faces


def build_cylinder_mesh(spec: CylinderSpec, segments: int = CYLINDER_MESH_SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False, dtype=np.float32)
    ring_xy = np.stack((np.cos(angles), np.sin(angles)), axis=1) * float(spec.radius)
    center = spec.center.astype(np.float32)
    bottom_z = float(center[2] - 0.5 * spec.height)
    top_z = float(center[2] + 0.5 * spec.height)
    bottom_center = np.array([[float(center[0]), float(center[1]), bottom_z]], dtype=np.float32)
    top_center = np.array([[float(center[0]), float(center[1]), top_z]], dtype=np.float32)
    bottom_ring = np.column_stack(
        (
            ring_xy[:, 0] + float(center[0]),
            ring_xy[:, 1] + float(center[1]),
            np.full(int(segments), bottom_z, dtype=np.float32),
        )
    )
    top_ring = np.column_stack(
        (
            ring_xy[:, 0] + float(center[0]),
            ring_xy[:, 1] + float(center[1]),
            np.full(int(segments), top_z, dtype=np.float32),
        )
    )
    vertices = np.vstack((bottom_center, top_center, bottom_ring, top_ring)).astype(np.float32)
    faces: list[list[int]] = []
    bottom_center_index = 0
    top_center_index = 1
    bottom_ring_start = 2
    top_ring_start = bottom_ring_start + int(segments)
    for segment_index in range(int(segments)):
        next_index = (segment_index + 1) % int(segments)
        bottom_i = bottom_ring_start + segment_index
        bottom_j = bottom_ring_start + next_index
        top_i = top_ring_start + segment_index
        top_j = top_ring_start + next_index
        faces.append([bottom_center_index, bottom_j, bottom_i])
        faces.append([top_center_index, top_i, top_j])
        faces.append([bottom_i, bottom_j, top_j])
        faces.append([bottom_i, top_j, top_i])
    return vertices, np.asarray(faces, dtype=np.int32)


def triangulate_scene(scene: SceneSpec) -> tuple[np.ndarray, np.ndarray]:
    mesh_vertices: list[np.ndarray] = []
    mesh_faces: list[np.ndarray] = []
    vertex_offset = 0
    for cuboid in scene.all_cuboids:
        vertices, faces = build_cuboid_mesh(cuboid, top_only=(cuboid.name == "floor"))
        mesh_vertices.append(vertices)
        mesh_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    for pole in scene.poles:
        vertices, faces = build_cylinder_mesh(pole)
        mesh_vertices.append(vertices)
        mesh_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    if not mesh_vertices:
        raise RuntimeError("Scene geometry is empty.")
    return np.vstack(mesh_vertices).astype(np.float32), np.vstack(mesh_faces).astype(np.int32)


def sample_scene_surface_points(scene: SceneSpec, spacing: float, rng: np.random.Generator) -> np.ndarray:
    vertices, faces = triangulate_scene(scene)
    area_per_sample = max(0.5 * spacing * spacing, 1e-6)
    dense_samples: list[np.ndarray] = []
    for face in faces:
        triangle = vertices[face]
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
    return voxel_downsample_centroids(dense_world, voxel_size=spacing * 0.5)


def build_raycast_terrain_mesh(scene: SceneSpec) -> tuple[np.ndarray, np.ndarray]:
    return triangulate_scene(scene)


def compute_grid_env_origins(num_envs: int, env_spacing: float) -> np.ndarray:
    if num_envs <= 0:
        return np.empty((0, 3), dtype=np.float32)
    num_rows = int(math.ceil(num_envs / int(math.sqrt(num_envs))))
    num_cols = int(math.ceil(num_envs / num_rows))
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    env_index = 0
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            if env_index >= num_envs:
                break
            env_origins[env_index, 0] = -((row_index - (num_rows - 1) * 0.5) * float(env_spacing))
            env_origins[env_index, 1] = (col_index - (num_cols - 1) * 0.5) * float(env_spacing)
            env_index += 1
    return env_origins


def imported_terrain_prim_path(local_prim_path: str, import_root_prim_path: str = "/World/ground/terrain") -> str:
    if not local_prim_path.startswith("/World"):
        raise ValueError(f"Expected a local scene prim path under /World, received: {local_prim_path}")
    suffix = local_prim_path[len("/World") :]
    return f"{import_root_prim_path}{suffix}"


def author_scene_usd(scene: SceneSpec, usd_path: Path, Usd: Any, UsdGeom: Any, UsdPhysics: Any, Gf: Any) -> None:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to create USD stage: {usd_path}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Walls")
    UsdGeom.Xform.Define(stage, "/World/Stairs")
    UsdGeom.Xform.Define(stage, "/World/Stairs/StairPositive")
    UsdGeom.Xform.Define(stage, "/World/Stairs/StairNegative")
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    UsdGeom.Xform.Define(stage, "/World/Poles")

    def define_cuboid(spec: CuboidSpec) -> None:
        cube = UsdGeom.Cube.Define(stage, spec.prim_path)
        cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(float(spec.center[0]), float(spec.center[1]), float(spec.center[2])))
        cube.AddScaleOp().Set(
            Gf.Vec3f(float(spec.size[0] * 0.5), float(spec.size[1] * 0.5), float(spec.size[2] * 0.5))
        )
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in spec.color])])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    def define_cylinder(spec: CylinderSpec) -> None:
        cylinder = UsdGeom.Cylinder.Define(stage, spec.prim_path)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(float(spec.radius))
        cylinder.CreateHeightAttr(float(spec.height))
        cylinder.AddTranslateOp().Set(Gf.Vec3d(float(spec.center[0]), float(spec.center[1]), float(spec.center[2])))
        cylinder.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in spec.color])])
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    for cuboid in scene.all_cuboids:
        define_cuboid(cuboid)
    for pole in scene.poles:
        define_cylinder(pole)

    raycast_vertices, raycast_faces = build_raycast_terrain_mesh(scene)
    UsdGeom.Xform.Define(stage, RAYCAST_TERRAIN_GROUP_PRIM_PATH)
    raycast_mesh = UsdGeom.Mesh.Define(stage, RAYCAST_TERRAIN_MESH_PRIM_PATH)
    raycast_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in raycast_vertices])
    raycast_mesh.CreateFaceVertexCountsAttr([3] * int(len(raycast_faces)))
    raycast_mesh.CreateFaceVertexIndicesAttr([int(index) for face in raycast_faces for index in face.tolist()])
    raycast_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    raycast_mesh.CreateDoubleSidedAttr().Set(True)
    raycast_mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()


def author_scene_instances_usd(
    scene: SceneSpec,
    usd_path: Path,
    env_origins: np.ndarray,
    *,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    Gf: Any,
) -> list[str]:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to create USD stage: {usd_path}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/SceneCopies")

    raycast_vertices, raycast_faces = build_raycast_terrain_mesh(scene)
    combined_raycast_vertices: list[np.ndarray] = []
    combined_raycast_faces: list[np.ndarray] = []
    raycast_vertex_offset = 0

    def prefixed_prim_path(base_prim_path: str, scene_root_prim_path: str) -> str:
        if not base_prim_path.startswith("/World"):
            raise ValueError(f"Expected prim path under /World, received: {base_prim_path}")
        return scene_root_prim_path + base_prim_path[len("/World") :]

    def define_cuboid(spec: CuboidSpec, *, scene_root_prim_path: str, translation: np.ndarray) -> None:
        cube = UsdGeom.Cube.Define(stage, prefixed_prim_path(spec.prim_path, scene_root_prim_path))
        cube.CreateSizeAttr(2.0)
        translated_center = spec.center.astype(np.float32) + translation.astype(np.float32)
        cube.AddTranslateOp().Set(
            Gf.Vec3d(float(translated_center[0]), float(translated_center[1]), float(translated_center[2]))
        )
        cube.AddScaleOp().Set(
            Gf.Vec3f(float(spec.size[0] * 0.5), float(spec.size[1] * 0.5), float(spec.size[2] * 0.5))
        )
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in spec.color])])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    def define_cylinder(spec: CylinderSpec, *, scene_root_prim_path: str, translation: np.ndarray) -> None:
        cylinder = UsdGeom.Cylinder.Define(stage, prefixed_prim_path(spec.prim_path, scene_root_prim_path))
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(float(spec.radius))
        cylinder.CreateHeightAttr(float(spec.height))
        translated_center = spec.center.astype(np.float32) + translation.astype(np.float32)
        cylinder.AddTranslateOp().Set(
            Gf.Vec3d(float(translated_center[0]), float(translated_center[1]), float(translated_center[2]))
        )
        cylinder.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in spec.color])])
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    for env_index, env_origin in enumerate(np.asarray(env_origins, dtype=np.float32)):
        scene_root_prim_path = f"/World/SceneCopies/SceneCopy_{env_index:03d}"
        UsdGeom.Xform.Define(stage, scene_root_prim_path)
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Walls")
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Stairs")
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Stairs/StairPositive")
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Stairs/StairNegative")
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Boxes")
        UsdGeom.Xform.Define(stage, f"{scene_root_prim_path}/Poles")

        for cuboid in scene.all_cuboids:
            define_cuboid(cuboid, scene_root_prim_path=scene_root_prim_path, translation=env_origin)
        for pole in scene.poles:
            define_cylinder(pole, scene_root_prim_path=scene_root_prim_path, translation=env_origin)
        translated_vertices = raycast_vertices + env_origin[None, :]
        combined_raycast_vertices.append(translated_vertices)
        combined_raycast_faces.append(raycast_faces + raycast_vertex_offset)
        raycast_vertex_offset += translated_vertices.shape[0]

    combined_vertices = (
        np.concatenate(combined_raycast_vertices, axis=0)
        if combined_raycast_vertices
        else np.empty((0, 3), dtype=np.float32)
    )
    combined_faces = (
        np.concatenate(combined_raycast_faces, axis=0)
        if combined_raycast_faces
        else np.empty((0, 3), dtype=np.int32)
    )
    UsdGeom.Xform.Define(stage, RAYCAST_TERRAIN_GROUP_PRIM_PATH)
    raycast_mesh = UsdGeom.Mesh.Define(stage, RAYCAST_TERRAIN_MESH_PRIM_PATH)
    raycast_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in combined_vertices])
    raycast_mesh.CreateFaceVertexCountsAttr([3] * int(len(combined_faces)))
    raycast_mesh.CreateFaceVertexIndicesAttr([int(index) for face in combined_faces for index in face.tolist()])
    raycast_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    raycast_mesh.CreateDoubleSidedAttr().Set(True)
    raycast_mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    stage.GetRootLayer().Save()
    return [imported_terrain_prim_path(RAYCAST_TERRAIN_GROUP_PRIM_PATH)]


def configure_isaac_asset_root() -> str:
    import carb
    from isaacsim.storage.native import get_assets_root_path

    settings = carb.settings.get_settings()
    default_key = "/persistent/isaac/asset_root/default"
    cloud_key = "/persistent/isaac/asset_root/cloud"
    default_root = settings.get(default_key)
    cloud_root = settings.get(cloud_key)
    resolved_root = default_root if isinstance(default_root, str) and default_root else get_assets_root_path()
    if not isinstance(resolved_root, str) or not resolved_root:
        raise RuntimeError(
            "Unable to resolve Isaac Sim asset root. "
            "This IsaacLab terminal is missing a usable /persistent/isaac/asset_root/default setting."
        )
    if default_root != resolved_root:
        settings.set(default_key, resolved_root)
    if cloud_root != resolved_root:
        settings.set(cloud_key, resolved_root)
    return resolved_root


def configure_robot_spawn_if_possible(env_cfg: Any) -> None:
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    init_state = getattr(robot_cfg, "init_state", None)
    if init_state is None:
        return
    init_state.pos = (float(START_BASE_X), float(START_BASE_Y), float(DEFAULT_BASE_Z))
    init_state.rot = tuple(float(value) for value in yaw_to_quaternion_wxyz(START_BASE_YAW))


def build_record_camera_pose(scene: SceneSpec) -> tuple[np.ndarray, np.ndarray]:
    eye = np.array(
        [
            START_BASE_X - 3.2,
            -0.12 * scene.corridor_width,
            6.0,
        ],
        dtype=np.float32,
    )
    target = np.array([0.6, 0.0, 0.70], dtype=np.float32)
    return eye, target


def configure_base_velocity_command(env_cfg: Any, command: dict[str, float] | None = None) -> None:
    commands_cfg = getattr(env_cfg, "commands", None)
    base_velocity_cfg = getattr(commands_cfg, "base_velocity", None)
    if base_velocity_cfg is None:
        raise RuntimeError("The environment configuration does not expose commands.base_velocity.")

    base_velocity_cfg.resampling_time_range = (1.0e6, 1.0e6)
    base_velocity_cfg.rel_standing_envs = 0.0
    base_velocity_cfg.rel_heading_envs = 1.0
    if hasattr(base_velocity_cfg, "debug_vis"):
        base_velocity_cfg.debug_vis = False
    if command is None:
        base_velocity_cfg.ranges.lin_vel_x = (float(COMMAND_LIN_VEL_X_RANGE[0]), float(COMMAND_LIN_VEL_X_RANGE[1]))
        base_velocity_cfg.ranges.lin_vel_y = (float(COMMAND_LIN_VEL_Y_RANGE[0]), float(COMMAND_LIN_VEL_Y_RANGE[1]))
        if getattr(base_velocity_cfg, "heading_command", False):
            heading_min = math.radians(float(COMMAND_HEADING_RANGE_DEG[0]))
            heading_max = math.radians(float(COMMAND_HEADING_RANGE_DEG[1]))
            base_velocity_cfg.ranges.heading = (heading_min, heading_max)
            base_velocity_cfg.ranges.ang_vel_z = (-1.0, 1.0)
        else:
            base_velocity_cfg.ranges.ang_vel_z = (-1.0, 1.0)
        return

    base_velocity_cfg.ranges.lin_vel_x = (float(command["lin_vel_x"]), float(command["lin_vel_x"]))
    base_velocity_cfg.ranges.lin_vel_y = (float(command["lin_vel_y"]), float(command["lin_vel_y"]))
    if getattr(base_velocity_cfg, "heading_command", False):
        base_velocity_cfg.ranges.heading = (float(command["heading_rad"]), float(command["heading_rad"]))
        base_velocity_cfg.ranges.ang_vel_z = (-1.0, 1.0)
    else:
        base_velocity_cfg.ranges.ang_vel_z = (float(command["ang_vel_z"]), float(command["ang_vel_z"]))


def configure_fixed_base_velocity_command(env_cfg: Any, command: dict[str, float]) -> None:
    configure_base_velocity_command(env_cfg, command=command)


def disable_base_velocity_debug_vis(policy_env: Any) -> None:
    command_manager = getattr(policy_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return
    base_velocity_term = command_manager.get_term("base_velocity")
    if hasattr(base_velocity_term, "set_debug_vis"):
        base_velocity_term.set_debug_vis(False)


def world_quaternion_to_ros(
    quaternion_world: np.ndarray,
    *,
    torch_module: Any,
    convert_camera_frame_orientation_convention: Any,
) -> tuple[float, float, float, float]:
    quaternion_world_tensor = torch_module.as_tensor(quaternion_world, dtype=torch_module.float32).unsqueeze(0)
    quaternion_ros = convert_camera_frame_orientation_convention(
        quaternion_world_tensor,
        origin="world",
        target="ros",
    )[0]
    return tuple(float(value) for value in quaternion_ros.detach().cpu().tolist())


def configure_scene_cameras(
    scene: SceneSpec,
    env_cfg: Any,
    *,
    sim_utils: Any,
    CameraCfg: Any,
    torch_module: Any,
    convert_camera_frame_orientation_convention: Any,
    include_record_camera: bool = True,
) -> tuple[DepthCameraSpec, tuple[CameraMount, ...], np.ndarray | None, np.ndarray | None]:
    depth_spec = build_depth_camera_spec()
    camera_mounts = build_camera_mounts()

    for mount in camera_mounts:
        mount_quaternion_base = rotation_matrix_to_quaternion(rotation_from_forward(mount.forward_in_base))
        setattr(
            env_cfg.scene,
            depth_camera_sensor_name(mount.name),
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base/{mount.name}_depth_camera",
                update_period=0.0,
                update_latest_camera_pose=True,
                height=depth_spec.height,
                width=depth_spec.width,
                data_types=["distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=depth_spec.focal_length_mm,
                    focus_distance=400.0,
                    horizontal_aperture=depth_spec.horizontal_aperture_mm,
                    clipping_range=(depth_spec.near_m, depth_spec.far_m),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(value) for value in mount.translation_in_base.tolist()),
                    rot=tuple(float(value) for value in mount_quaternion_base.tolist()),
                    convention="world",
                ),
            ),
        )

    record_eye = None
    record_target = None
    if include_record_camera:
        record_eye, record_target = build_record_camera_pose(scene)
        record_rotation_world = rotation_from_forward(record_target - record_eye)
        record_quaternion_world = rotation_matrix_to_quaternion(record_rotation_world)
        record_quaternion_ros = world_quaternion_to_ros(
            record_quaternion_world,
            torch_module=torch_module,
            convert_camera_frame_orientation_convention=convert_camera_frame_orientation_convention,
        )
        env_cfg.scene.record_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/record_camera",
            update_period=0.0,
            update_latest_camera_pose=True,
            height=RECORD_CAMERA_HEIGHT,
            width=RECORD_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=RECORD_CAMERA_FOCAL_LENGTH,
                focus_distance=400.0,
                horizontal_aperture=RECORD_CAMERA_HORIZONTAL_APERTURE,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(value) for value in record_eye.tolist()),
                rot=record_quaternion_ros,
                convention="ros",
            ),
        )

        viewer_cfg = getattr(env_cfg, "viewer", None)
        if viewer_cfg is not None:
            viewer_cfg.eye = tuple(float(value) for value in record_eye)
            viewer_cfg.lookat = tuple(float(value) for value in record_target)

    return depth_spec, camera_mounts, record_eye, record_target


def depth_camera_intrinsic_matrix_tensor(depth_spec: DepthCameraSpec, torch_module: Any, device: Any) -> Any:
    return torch_module.tensor(depth_spec.intrinsic_matrix(), dtype=torch_module.float32, device=device)


def configure_depth_camera_intrinsics(
    policy_env: Any,
    depth_spec: DepthCameraSpec,
    camera_mounts: tuple[CameraMount, ...],
    torch_module: Any,
) -> None:
    camera_names = [depth_camera_sensor_name(mount.name) for mount in camera_mounts]
    first_camera = policy_env.scene[camera_names[0]]
    device = first_camera.data.intrinsic_matrices.device
    intrinsic = depth_camera_intrinsic_matrix_tensor(depth_spec, torch_module, device)
    intrinsic_batch = intrinsic.unsqueeze(0).repeat(first_camera.data.intrinsic_matrices.shape[0], 1, 1)
    for camera_name in camera_names:
        policy_env.scene[camera_name].set_intrinsic_matrices(intrinsic_batch)


def load_policy_checkpoint(policy_checkpoint: Path, device: str, torch_module: Any) -> Any:
    policy_path = policy_checkpoint.expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {policy_path}")
    policy = torch_module.jit.load(str(policy_path.resolve()), map_location=device)
    policy.eval()
    return policy


def extract_pose(robot_asset: Any) -> tuple[np.ndarray, np.ndarray]:
    position = robot_asset.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
    quaternion = robot_asset.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
    return position, quaternion


def resolve_base_body_index(robot_asset: Any) -> int:
    if not hasattr(robot_asset, "find_bodies"):
        raise RuntimeError("Robot asset does not expose find_bodies(), cannot resolve base body.")
    body_ids, _ = robot_asset.find_bodies("base")
    if len(body_ids) == 0:
        raise RuntimeError("Unable to resolve robot base body with find_bodies('base').")
    return int(body_ids[0])


def extract_base_body_pose(robot_asset: Any, base_body_index: int, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    position = robot_asset.data.body_pos_w[env_index, base_body_index].detach().cpu().numpy().astype(np.float32)
    quaternion = robot_asset.data.body_quat_w[env_index, base_body_index].detach().cpu().numpy().astype(np.float32)
    return position, quaternion


def maybe_place_robot_at_start(policy_env: Any, torch_module: Any) -> dict[str, list[float]]:
    robot_asset = policy_env.scene["robot"]
    env_origins = getattr(policy_env.scene, "env_origins", None)
    if env_origins is None:
        env_origin = torch_module.zeros((1, 3), device=robot_asset.data.root_pos_w.device, dtype=robot_asset.data.root_pos_w.dtype)
    else:
        env_origin = env_origins.to(device=robot_asset.data.root_pos_w.device, dtype=robot_asset.data.root_pos_w.dtype)

    if hasattr(robot_asset.data, "default_root_state"):
        root_state = robot_asset.data.default_root_state.clone()
    elif hasattr(robot_asset.data, "root_state_w"):
        root_state = robot_asset.data.root_state_w.clone()
    else:
        world_position, _ = extract_pose(robot_asset)
        return {
            "start_position_world": vector_to_list(world_position),
            "start_position_scene": vector_to_list(world_position),
        }

    start_world = root_state.clone()
    start_world[:, 0] = env_origin[:, 0] + float(START_BASE_X)
    start_world[:, 1] = env_origin[:, 1] + float(START_BASE_Y)
    start_world[:, 2] = env_origin[:, 2] + float(DEFAULT_BASE_Z)
    start_world[:, 3:7] = torch_module.as_tensor(
        yaw_to_quaternion_wxyz(START_BASE_YAW),
        dtype=start_world.dtype,
        device=start_world.device,
    )
    if start_world.shape[1] >= 13:
        start_world[:, 7:13] = 0.0

    if hasattr(robot_asset, "write_root_state_to_sim"):
        robot_asset.write_root_state_to_sim(start_world)
    else:
        robot_asset.write_root_pose_to_sim(start_world[:, :7])
        if start_world.shape[1] >= 13:
            robot_asset.write_root_velocity_to_sim(start_world[:, 7:13])

    if (
        hasattr(robot_asset, "write_joint_state_to_sim")
        and hasattr(robot_asset.data, "default_joint_pos")
        and hasattr(robot_asset.data, "default_joint_vel")
    ):
        robot_asset.write_joint_state_to_sim(
            robot_asset.data.default_joint_pos.clone(),
            robot_asset.data.default_joint_vel.clone(),
        )

    if hasattr(policy_env.scene, "write_data_to_sim"):
        policy_env.scene.write_data_to_sim()
    if hasattr(policy_env.sim, "forward"):
        policy_env.sim.forward()
    if hasattr(policy_env.scene, "update"):
        policy_env.scene.update(policy_env.step_dt)

    world_position, _ = extract_pose(robot_asset)
    scene_position = world_position - env_origin[0].detach().cpu().numpy().astype(np.float32)
    return {
        "start_position_world": vector_to_list(world_position),
        "start_position_scene": vector_to_list(scene_position),
    }


def refresh_policy_observations(policy_env: Any, fallback_obs: Any | None = None) -> Any:
    for manager_name in ("observation_manager", "obs_manager"):
        manager = getattr(policy_env, manager_name, None)
        if manager is not None and hasattr(manager, "compute"):
            observations = manager.compute()
            if isinstance(observations, dict) and "policy" in observations:
                return observations
    if fallback_obs is not None:
        return fallback_obs
    raise RuntimeError("Unable to refresh policy observations after manual robot placement.")


def subsample_depth_for_stride(depth_image: Any, intrinsic_matrix: Any, torch_module: Any) -> tuple[Any, Any]:
    if DEPTH_CAMERA_PIXEL_STRIDE <= 1:
        return depth_image, intrinsic_matrix
    stride = int(DEPTH_CAMERA_PIXEL_STRIDE)
    depth_sub = depth_image[::stride, ::stride]
    intrinsic_sub = intrinsic_matrix.clone()
    intrinsic_sub[0, 0] = intrinsic_sub[0, 0] / stride
    intrinsic_sub[1, 1] = intrinsic_sub[1, 1] / stride
    intrinsic_sub[0, 2] = intrinsic_sub[0, 2] / stride
    intrinsic_sub[1, 2] = intrinsic_sub[1, 2] / stride
    return depth_sub, intrinsic_sub


def summarize_depth_image(depth_image: Any, torch_module: Any) -> dict[str, Any]:
    finite_mask = torch_module.isfinite(depth_image)
    positive_mask = finite_mask & (depth_image > 0.0)
    valid_mask = positive_mask & (depth_image >= DEPTH_CAMERA_NEAR) & (depth_image <= DEPTH_CAMERA_FAR)

    finite_count = int(finite_mask.sum().item())
    positive_count = int(positive_mask.sum().item())
    valid_count = int(valid_mask.sum().item())
    if valid_count > 0:
        valid_values = depth_image[valid_mask]
        valid_min = float(valid_values.min().item())
        valid_max = float(valid_values.max().item())
        valid_mean = float(valid_values.mean().item())
    else:
        valid_min = None
        valid_max = None
        valid_mean = None
    return {
        "image_height": int(depth_image.shape[0]),
        "image_width": int(depth_image.shape[1]),
        "finite_pixel_count": finite_count,
        "positive_pixel_count": positive_count,
        "valid_pixel_count": valid_count,
        "valid_pixel_fraction": float(valid_count / max(int(depth_image.numel()), 1)),
        "valid_depth_min_m": valid_min,
        "valid_depth_max_m": valid_max,
        "valid_depth_mean_m": valid_mean,
    }


def capture_depth_camera_points_world(
    camera_sensor: Any,
    *,
    camera_position_world: np.ndarray,
    camera_quaternion_ros: tuple[float, float, float, float],
    create_pointcloud_from_depth: Any,
    torch_module: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_image = camera_sensor.data.output["distance_to_image_plane"][0, ..., 0]
    intrinsic_matrix = camera_sensor.data.intrinsic_matrices[0]
    depth_image, intrinsic_matrix = subsample_depth_for_stride(depth_image, intrinsic_matrix, torch_module)
    depth_stats = summarize_depth_image(depth_image, torch_module)
    points_world = create_pointcloud_from_depth(
        intrinsic_matrix=intrinsic_matrix,
        depth=depth_image,
        position=torch_module.as_tensor(camera_position_world, dtype=torch_module.float32, device=depth_image.device),
        orientation=torch_module.as_tensor(camera_quaternion_ros, dtype=torch_module.float32, device=depth_image.device),
        device=depth_image.device,
    )
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32), {
            **depth_stats,
            "raw_world_point_count": 0,
            "finite_world_point_count": 0,
            "returned_world_point_count": 0,
        }
    raw_world_point_count = int(points_world.shape[0])
    finite_mask = torch_module.isfinite(points_world).all(dim=1)
    points_world = points_world[finite_mask]
    finite_world_point_count = int(points_world.shape[0])
    if points_world.shape[0] > DEPTH_CAMERA_MAX_POINTS:
        keep_indices = torch_module.linspace(
            0,
            points_world.shape[0] - 1,
            DEPTH_CAMERA_MAX_POINTS,
            device=points_world.device,
        ).long()
        points_world = points_world[keep_indices]
    points_world_np = points_world.detach().cpu().numpy().astype(np.float32, copy=False)
    return points_world_np, {
        **depth_stats,
        "raw_world_point_count": raw_world_point_count,
        "finite_world_point_count": finite_world_point_count,
        "returned_world_point_count": int(points_world_np.shape[0]),
    }


def capture_record_camera_frame(record_camera_sensor: Any, env_index: int = 0) -> np.ndarray:
    rgb = record_camera_sensor.data.output["rgb"][env_index]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    return coerce_image_to_rgb_u8(rgb)


def collect_sensor_step(
    *,
    policy_env: Any,
    robot_pose_world: np.ndarray,
    camera_mounts: tuple[CameraMount, ...],
    gt_world_points: np.ndarray,
    create_pointcloud_from_depth: Any,
    torch_module: Any,
    env_index: int = 0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    world_to_robot = invert_transform(robot_pose_world)
    robot_rotation_world = robot_pose_world[:3, :3]
    measurement_world_chunks: list[np.ndarray] = []
    camera_point_counts: dict[str, int] = {}
    camera_raw_point_counts: dict[str, int] = {}
    camera_pre_crop_point_counts: dict[str, int] = {}
    camera_depth_diagnostics: dict[str, dict[str, Any]] = {}
    camera_mount_diagnostics: dict[str, dict[str, Any]] = {}

    for mount in camera_mounts:
        camera_sensor = policy_env.scene[depth_camera_sensor_name(mount.name)]
        camera_position_world = camera_sensor.data.pos_w[env_index].detach().cpu().numpy().astype(np.float32, copy=False)
        camera_quaternion_world = camera_sensor.data.quat_w_world[env_index].detach().cpu().numpy().astype(np.float32, copy=False)
        camera_quaternion_ros = camera_sensor.data.quat_w_ros[env_index].detach().cpu().numpy().astype(np.float32, copy=False)
        points_world, depth_diagnostics = capture_depth_camera_points_world(
            camera_sensor,
            camera_position_world=camera_position_world,
            camera_quaternion_ros=camera_quaternion_ros,
            create_pointcloud_from_depth=create_pointcloud_from_depth,
            torch_module=torch_module,
        )
        measurement_world_chunks.append(points_world)
        camera_points_local_unclipped = transform_points(points_world, world_to_robot)
        camera_points_local = crop_local_points(camera_points_local_unclipped, GRID_BOUNDS_MIN, GRID_BOUNDS_MAX)
        camera_points_local = voxel_downsample_centroids(camera_points_local, voxel_size=GRID_VOXEL_SIZE)
        camera_point_counts[mount.name] = int(camera_points_local.shape[0])
        camera_raw_point_counts[mount.name] = int(depth_diagnostics["returned_world_point_count"])
        camera_pre_crop_point_counts[mount.name] = int(camera_points_local_unclipped.shape[0])
        camera_depth_diagnostics[mount.name] = depth_diagnostics

        camera_position_robot = transform_points(camera_position_world[None, :], world_to_robot)[0]
        camera_forward_world = quaternion_rotate_vector_wxyz(
            camera_quaternion_world,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        camera_forward_robot = normalize(robot_rotation_world.T @ camera_forward_world)
        expected_forward_robot = normalize(mount.forward_in_base)
        camera_mount_diagnostics[mount.name] = {
            "position_robot_frame": vector_to_list(camera_position_robot),
            "position_error_m": float(np.linalg.norm(camera_position_robot - mount.translation_in_base)),
            "forward_robot_frame": vector_to_list(camera_forward_robot),
            "forward_error_deg": angle_between_vectors_deg(camera_forward_robot, expected_forward_robot),
        }

    measurement_world = (
        np.concatenate(measurement_world_chunks, axis=0)
        if measurement_world_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    measurement_local = crop_local_points(transform_points(measurement_world, world_to_robot), GRID_BOUNDS_MIN, GRID_BOUNDS_MAX)
    measurement_local = voxel_downsample_centroids(measurement_local, voxel_size=GRID_VOXEL_SIZE)

    ground_truth_local = crop_local_points(transform_points(gt_world_points, world_to_robot), GRID_BOUNDS_MIN, GRID_BOUNDS_MAX)
    ground_truth_local = voxel_downsample_centroids(ground_truth_local, voxel_size=GRID_VOXEL_SIZE)

    measurement_voxels = voxelize_points(measurement_local, GRID_BOUNDS_MIN, GRID_BOUNDS_MAX, GRID_SHAPE)
    ground_truth_voxels = voxelize_points(ground_truth_local, GRID_BOUNDS_MIN, GRID_BOUNDS_MAX, GRID_SHAPE)
    visible_fraction = visible_fraction_from_voxels(ground_truth_voxels, measurement_voxels)

    sensor_meta = {
        "measurement_point_count": int(measurement_local.shape[0]),
        "ground_truth_point_count": int(ground_truth_local.shape[0]),
        "measurement_occupied_voxel_count": int(np.count_nonzero(measurement_voxels)),
        "ground_truth_occupied_voxel_count": int(np.count_nonzero(ground_truth_voxels)),
        "visible_gt_fraction": float(visible_fraction),
        "camera_point_counts": camera_point_counts,
        "camera_raw_point_counts": camera_raw_point_counts,
        "camera_pre_crop_point_counts": camera_pre_crop_point_counts,
        "camera_depth_diagnostics": camera_depth_diagnostics,
        "camera_mount_diagnostics": camera_mount_diagnostics,
        "all_cameras_zero": bool(all(count == 0 for count in camera_point_counts.values())),
    }
    return sensor_meta, measurement_local, ground_truth_local


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = ensure_output_dir(args.output_dir)
    debug_record_camera_enabled = bool(args.record_frames or args.save_video)
    frames_dir = output_dir / "frames" if args.record_frames else None
    packed_trajectory_path = output_dir / "trajectory.npz"
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    gt_rng = np.random.default_rng(args.seed + 1000)
    scene = build_scene(rng, seed=args.seed)
    validate_scene_spec(scene)
    command = sample_command(rng)
    scene_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.export_usd:
        scene_usd_path = output_dir / f"scene_seed_{args.seed}.usd"
    else:
        scene_temp_dir = tempfile.TemporaryDirectory(prefix=f"isaaclab_datacollect_scene_seed_{args.seed}_", dir="/tmp")
        scene_usd_path = Path(scene_temp_dir.name) / f"scene_seed_{args.seed}.usd"

    simulation_app = None
    policy_env = None
    video_writer = None
    video_enabled = False
    video_failed = False
    video_path = output_dir / "rollout.mp4"

    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(
            {
                "headless": bool(args.headless),
                "device": str(args.device),
                "enable_cameras": True,
            }
        )
        simulation_app = app_launcher.app

        import torch
        import isaaclab.sim as sim_utils
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors import CameraCfg
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
        from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg_PLAY

        asset_root = configure_isaac_asset_root()
        log_progress(f"resolved Isaac asset root: {asset_root}")
        author_scene_usd(scene, scene_usd_path, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf)
        if args.export_usd:
            log_progress(f"scene USD written to {scene_usd_path}")
        else:
            log_progress(f"scene USD prepared at temporary path {scene_usd_path}")

        env_cfg = AnymalCRoughEnvCfg_PLAY()
        env_cfg.seed = int(args.seed)
        env_cfg.scene.num_envs = 1
        env_cfg.scene.env_spacing = 2.5
        env_cfg.curriculum = None
        configure_robot_spawn_if_possible(env_cfg)
        configure_fixed_base_velocity_command(env_cfg, command)
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=str(scene_usd_path.resolve()),
        )
        if getattr(env_cfg.scene, "height_scanner", None) is not None and hasattr(env_cfg.scene.height_scanner, "mesh_prim_paths"):
            env_cfg.scene.height_scanner.mesh_prim_paths = [IMPORTED_RAYCAST_TERRAIN_PRIM_PATH]
        env_cfg.sim.device = str(args.device)
        if str(args.device).startswith("cpu"):
            env_cfg.sim.use_fabric = False
        if hasattr(env_cfg, "episode_length_s"):
            env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 0.0)), float(args.steps) * 0.1 + 10.0)

        depth_spec, camera_mounts, requested_record_eye, requested_record_target = configure_scene_cameras(
            scene,
            env_cfg,
            sim_utils=sim_utils,
            CameraCfg=CameraCfg,
            torch_module=torch,
            convert_camera_frame_orientation_convention=convert_camera_frame_orientation_convention,
            include_record_camera=debug_record_camera_enabled,
        )

        log_progress("creating ManagerBasedRLEnv with real CameraCfg sensors")
        policy_env = ManagerBasedRLEnv(cfg=env_cfg)
        log_progress("environment created")
        disable_base_velocity_debug_vis(policy_env)

        policy = load_policy_checkpoint(args.policy_checkpoint, str(args.device), torch)
        log_progress("policy loaded")

        reset_result = policy_env.reset()
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        configure_depth_camera_intrinsics(policy_env, depth_spec, camera_mounts, torch)
        placement_meta = maybe_place_robot_at_start(policy_env, torch)
        policy_env.command_manager.compute(0.0)
        observations = refresh_policy_observations(policy_env, fallback_obs=observations)

        if not isinstance(observations, dict) or "policy" not in observations:
            raise RuntimeError("Policy environment reset did not produce obs['policy'].")

        robot_asset = policy_env.scene["robot"]
        base_body_index = resolve_base_body_index(robot_asset)
        velocity_command_term = policy_env.command_manager.get_term("base_velocity")
        initial_command_base = policy_env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy().astype(np.float32)
        initial_heading_target = (
            float(velocity_command_term.heading_target[0].detach().cpu().item())
            if hasattr(velocity_command_term, "heading_target")
            else None
        )
        env_origin = getattr(policy_env.scene, "env_origins", None)
        if env_origin is None:
            env_origin_np = np.zeros(3, dtype=np.float32)
        else:
            env_origin_np = env_origin[0].detach().cpu().numpy().astype(np.float32)

        initial_world_position, initial_quaternion = extract_base_body_pose(robot_asset, base_body_index)
        initial_scene_position = initial_world_position - env_origin_np
        step_dt = float(policy_env.step_dt)
        gt_points_env = sample_scene_surface_points(scene, spacing=GROUND_TRUTH_SAMPLE_SPACING, rng=gt_rng)
        gt_world_points = gt_points_env + env_origin_np[None, :]
        log_progress(f"prepared ground-truth cloud with {gt_world_points.shape[0]} points")

        if args.save_video:
            video_writer = import_imageio_writer_if_available(
                video_path,
                fps=max(1, int(round(1.0 / max(step_dt, 1e-6)))),
            )
            video_enabled = video_writer is not None

        trajectory_steps: list[dict[str, Any]] = []
        packed_poses: list[np.ndarray] = []
        packed_measurements_local: list[np.ndarray] = []
        packed_ground_truth_local: list[np.ndarray] = []
        packed_step_indices: list[int] = []
        packed_timestamps_s: list[float] = []
        visibility_values: list[float] = []
        entered_stair = entered_stair_zone(scene, initial_scene_position)
        entered_step = 0 if entered_stair else None
        first_all_cameras_zero_step = None
        terminated_early = False
        frames_written = 0

        record_camera_sensor = policy_env.scene["record_camera"] if debug_record_camera_enabled else None
        log_progress("starting rollout loop")
        for step_index in range(args.steps):
            if step_index % 50 == 0 or step_index == args.steps - 1:
                log_progress(f"step {step_index + 1}/{args.steps}")

            observations = refresh_policy_observations(policy_env, fallback_obs=observations)
            with torch.inference_mode():
                actions = policy(observations["policy"])
            step_result = policy_env.step(actions)

            if len(step_result) == 5:
                observations, _, terminated, truncated, _ = step_result
                terminated_now = bool(torch.any(terminated).item()) or bool(torch.any(truncated).item())
            elif len(step_result) == 4:
                observations, _, terminated, _ = step_result
                terminated_now = bool(torch.any(terminated).item()) if torch.is_tensor(terminated) else bool(terminated)
            else:
                raise RuntimeError(f"Unexpected env.step return format with {len(step_result)} values.")

            world_position, quaternion_wxyz = extract_base_body_pose(robot_asset, base_body_index)
            scene_position = world_position - env_origin_np
            yaw = quaternion_to_yaw(quaternion_wxyz)
            robot_pose_world = pose_matrix(world_position, quaternion_to_rotation_matrix(quaternion_wxyz))
            in_stair_zone = entered_stair_zone(scene, scene_position)
            if in_stair_zone and entered_step is None:
                entered_step = step_index + 1
            entered_stair = entered_stair or in_stair_zone

            if record_camera_sensor is not None:
                frame_rgb = capture_record_camera_frame(record_camera_sensor)
                if frames_dir is not None:
                    frame_path = frames_dir / f"frame_{step_index:04d}.png"
                    write_png(frame_path, frame_rgb)
                    frames_written += 1
                if video_writer is not None:
                    try:
                        video_writer.append_data(frame_rgb)
                    except Exception as exc:
                        print(f"[rollout] video writer append failed, disabling MP4 export: {exc}", file=sys.stderr, flush=True)
                        try:
                            video_writer.close()
                        except Exception:
                            pass
                        video_writer = None
                        video_failed = True

            sensor_meta, measurement_local, ground_truth_local = collect_sensor_step(
                policy_env=policy_env,
                robot_pose_world=robot_pose_world,
                camera_mounts=camera_mounts,
                gt_world_points=gt_world_points,
                create_pointcloud_from_depth=create_pointcloud_from_depth,
                torch_module=torch,
            )
            packed_poses.append(robot_pose_world.astype(np.float32, copy=True))
            packed_measurements_local.append(measurement_local.astype(np.float32, copy=False))
            packed_ground_truth_local.append(ground_truth_local.astype(np.float32, copy=False))
            packed_step_indices.append(int(step_index))
            packed_timestamps_s.append(float((step_index + 1) * step_dt))
            visibility_values.append(float(sensor_meta["visible_gt_fraction"]))
            if sensor_meta["all_cameras_zero"] and first_all_cameras_zero_step is None:
                first_all_cameras_zero_step = int(step_index + 1)
                log_progress(f"all four depth cameras returned zero cropped points at step {first_all_cameras_zero_step}")

            trajectory_steps.append(
                {
                    "step": int(step_index + 1),
                    "time_s": float((step_index + 1) * step_dt),
                    "position_world": vector_to_list(world_position),
                    "position_scene": vector_to_list(scene_position),
                    "orientation_wxyz": vector_to_list(quaternion_wxyz),
                    "yaw_rad": float(yaw),
                    "yaw_deg": float(math.degrees(yaw)),
                    "entered_stair_zone": bool(in_stair_zone),
                    "measurement_point_count": int(sensor_meta["measurement_point_count"]),
                    "ground_truth_point_count": int(sensor_meta["ground_truth_point_count"]),
                    "measurement_occupied_voxel_count": int(sensor_meta["measurement_occupied_voxel_count"]),
                    "ground_truth_occupied_voxel_count": int(sensor_meta["ground_truth_occupied_voxel_count"]),
                    "visible_gt_fraction": float(sensor_meta["visible_gt_fraction"]),
                    "camera_point_counts": sensor_meta["camera_point_counts"],
                    "camera_raw_point_counts": sensor_meta["camera_raw_point_counts"],
                    "camera_pre_crop_point_counts": sensor_meta["camera_pre_crop_point_counts"],
                    "camera_depth_diagnostics": sensor_meta["camera_depth_diagnostics"],
                    "camera_mount_diagnostics": sensor_meta["camera_mount_diagnostics"],
                    "all_cameras_zero": bool(sensor_meta["all_cameras_zero"]),
                }
            )

            if terminated_now:
                terminated_early = True
                print(f"[rollout] environment terminated at step {step_index + 1}", file=sys.stderr, flush=True)
                break

        if record_camera_sensor is not None:
            record_camera_position_world = record_camera_sensor.data.pos_w[0].detach().cpu().numpy().astype(np.float32)
            record_camera_quaternion_ros = record_camera_sensor.data.quat_w_ros[0].detach().cpu().numpy().astype(np.float32)
        else:
            record_camera_position_world = None
            record_camera_quaternion_ros = None
        front_camera_intrinsics = (
            policy_env.scene[depth_camera_sensor_name(camera_mounts[0].name)]
            .data.intrinsic_matrices[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        final_command_base = policy_env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy().astype(np.float32)
        final_heading_target = (
            float(velocity_command_term.heading_target[0].detach().cpu().item())
            if hasattr(velocity_command_term, "heading_target")
            else None
        )

        x_values = [float(initial_scene_position[0])] + [float(step["position_scene"][0]) for step in trajectory_steps]
        y_values = [float(initial_scene_position[1])] + [float(step["position_scene"][1]) for step in trajectory_steps]
        visibility_mean = float(np.mean(visibility_values)) if visibility_values else 0.0
        visibility_min = float(np.min(visibility_values)) if visibility_values else 0.0
        visibility_max = float(np.max(visibility_values)) if visibility_values else 0.0
        camera_names = [mount.name for mount in camera_mounts]
        camera_mount_position_error_max_m = {
            camera_name: float(
                max(
                    (step["camera_mount_diagnostics"][camera_name]["position_error_m"] for step in trajectory_steps),
                    default=0.0,
                )
            )
            for camera_name in camera_names
        }
        camera_mount_forward_error_max_deg = {
            camera_name: float(
                max(
                    (step["camera_mount_diagnostics"][camera_name]["forward_error_deg"] for step in trajectory_steps),
                    default=0.0,
                )
            )
            for camera_name in camera_names
        }
        camera_valid_pixel_min = {
            camera_name: int(
                min(
                    (step["camera_depth_diagnostics"][camera_name]["valid_pixel_count"] for step in trajectory_steps),
                    default=0,
                )
            )
            for camera_name in camera_names
        }
        camera_valid_pixel_max = {
            camera_name: int(
                max(
                    (step["camera_depth_diagnostics"][camera_name]["valid_pixel_count"] for step in trajectory_steps),
                    default=0,
                )
            )
            for camera_name in camera_names
        }
        camera_returned_point_count_min = {
            camera_name: int(
                min(
                    (step["camera_raw_point_counts"][camera_name] for step in trajectory_steps),
                    default=0,
                )
            )
            for camera_name in camera_names
        }
        camera_returned_point_count_max = {
            camera_name: int(
                max(
                    (step["camera_raw_point_counts"][camera_name] for step in trajectory_steps),
                    default=0,
                )
            )
            for camera_name in camera_names
        }
        all_cameras_zero_step_count = int(sum(1 for step in trajectory_steps if step["all_cameras_zero"]))
        all_cameras_zero_started_in_stair_zone = bool(
            first_all_cameras_zero_step is not None
            and trajectory_steps[first_all_cameras_zero_step - 1]["entered_stair_zone"]
        )

        summary = {
            "seed": int(args.seed),
            "steps_requested": int(args.steps),
            "steps_completed": int(len(trajectory_steps)),
            "headless": bool(args.headless),
            "device": str(args.device),
            "policy_checkpoint": str(args.policy_checkpoint.expanduser().resolve()),
            "artifacts": {
                "export_usd": bool(args.export_usd),
                "record_frames": bool(args.record_frames),
                "save_video": bool(args.save_video),
            },
            "scene_generation": {
                "center_clear_width_config": float(CENTER_CLEAR_WIDTH) if CENTER_CLEAR_WIDTH is not None else None,
                "center_clear_width_rule": (
                    "script_constant"
                    if CENTER_CLEAR_WIDTH is not None
                    else f"default_max({DEFAULT_CENTER_CLEAR_WIDTH_MIN:.2f},{DEFAULT_CENTER_CLEAR_WIDTH_RATIO:.2f}*corridor_width)"
                ),
                "box_size_z_range": [float(BOX_SIZE_Z_RANGE[0]), float(BOX_SIZE_Z_RANGE[1])],
                "box_center_inward_shift": float(BOX_CENTER_INWARD_SHIFT),
                "corridor_width_semantics": "obstacle_layout_width",
                "wall_layout": {
                    "start_wall_enabled": bool(START_WALL_ENABLED),
                    "goal_wall_enabled": bool(GOAL_WALL_ENABLED),
                    "side_wall_mode": SIDE_WALL_MODE,
                    "side_wall_outward_offset": float(SIDE_WALL_OUTWARD_OFFSET),
                },
            },
            "scene_usd": scene_usd_path.name if args.export_usd else None,
            "scene_usd_exported": bool(args.export_usd),
            "scene": scene.to_dict(),
            "command": {
                "lin_vel_x": float(command["lin_vel_x"]),
                "lin_vel_y": float(command["lin_vel_y"]),
                "ang_vel_z": float(command["ang_vel_z"]),
                "heading_rad": float(command["heading_rad"]),
                "heading_deg": float(command["heading_deg"]),
                "source": "command_manager_base_velocity",
                "observation_override": False,
            },
            "command_term_initial_base_frame": vector_to_list(initial_command_base),
            "command_term_final_base_frame": vector_to_list(final_command_base),
            "heading_target_initial_rad": initial_heading_target,
            "heading_target_final_rad": final_heading_target,
            "robot_start": {
                "reference_body": "base",
                "requested_position_scene": [float(START_BASE_X), float(START_BASE_Y), float(DEFAULT_BASE_Z)],
                "requested_yaw_deg": float(math.degrees(START_BASE_YAW)),
                "actual_position_world": vector_to_list(initial_world_position),
                "actual_position_scene": vector_to_list(initial_scene_position),
                "actual_orientation_wxyz": vector_to_list(initial_quaternion),
                "manual_placement": placement_meta,
            },
            "record_camera": (
                {
                    "source": "isaaclab_camera_cfg_rgb",
                    "resolution": [int(RECORD_CAMERA_WIDTH), int(RECORD_CAMERA_HEIGHT)],
                    "warmup_frames": int(RECORD_CAMERA_WARMUP_FRAMES),
                    "requested_position_env": vector_to_list(requested_record_eye) if requested_record_eye is not None else None,
                    "requested_target_env": vector_to_list(requested_record_target) if requested_record_target is not None else None,
                    "actual_position_world": vector_to_list(record_camera_position_world) if record_camera_position_world is not None else None,
                    "actual_orientation_ros_wxyz": vector_to_list(record_camera_quaternion_ros) if record_camera_quaternion_ros is not None else None,
                }
                if debug_record_camera_enabled
                else None
            ),
            "sensor_config": {
                "measurement_source": "isaaclab_camera_cfg_depth",
                "camera_pose_source": "camera.data.pos_w_quat_w_ros",
                "camera_mounting_mode": "camera_prim_attached_under_robot_base",
                "update_latest_camera_pose": True,
                "camera_profile": depth_spec.to_dict(),
                "mounts": [mount.to_dict() for mount in camera_mounts],
                "camera_count": 4,
                "sensor_names": [depth_camera_sensor_name(mount.name) for mount in camera_mounts],
                "local_grid_reference_body": "base",
                "applied_intrinsic_matrix": front_camera_intrinsics.tolist(),
                "grid_bounds_min_m": vector_to_list(GRID_BOUNDS_MIN),
                "grid_bounds_max_m": vector_to_list(GRID_BOUNDS_MAX),
                "grid_shape": [int(value) for value in GRID_SHAPE],
                "grid_voxel_size_m": float(GRID_VOXEL_SIZE),
                "ground_truth_sample_spacing_m": float(GROUND_TRUTH_SAMPLE_SPACING),
                "visibility_fraction_target": float(VISIBILITY_TARGET),
            },
            "outputs": {
                "frames_dir": "frames" if frames_dir is not None else None,
                "frame_count": int(frames_written),
                "video_path": video_path.name if video_enabled and not video_failed else None,
                "trajectory_npz": packed_trajectory_path.name,
            },
            "policy_step_dt_s": float(step_dt),
            "terminated_early": bool(terminated_early),
            "entered_stair_zone": bool(entered_stair),
            "entered_stair_zone_step": int(entered_step) if entered_step is not None else None,
            "stair_zone": scene.stair_zone,
            "visibility_fraction_mean": float(visibility_mean),
            "visibility_fraction_min": float(visibility_min),
            "visibility_fraction_max": float(visibility_max),
            "visibility_fraction_target": float(VISIBILITY_TARGET),
            "visibility_fraction_target_gap_mean": float(visibility_mean - VISIBILITY_TARGET),
            "visibility_fraction_target_met_mean": bool(visibility_mean >= VISIBILITY_TARGET),
            "visibility_fraction_target_met_any_step": bool(any(value >= VISIBILITY_TARGET for value in visibility_values)),
            "camera_diagnostics": {
                "first_all_cameras_zero_step": int(first_all_cameras_zero_step) if first_all_cameras_zero_step is not None else None,
                "all_cameras_zero_step_count": all_cameras_zero_step_count,
                "all_cameras_zero_started_in_stair_zone": all_cameras_zero_started_in_stair_zone,
                "mount_position_error_max_m": camera_mount_position_error_max_m,
                "mount_forward_error_max_deg": camera_mount_forward_error_max_deg,
                "valid_pixel_count_min": camera_valid_pixel_min,
                "valid_pixel_count_max": camera_valid_pixel_max,
                "returned_world_point_count_min": camera_returned_point_count_min,
                "returned_world_point_count_max": camera_returned_point_count_max,
            },
            "x_progress_scene": {
                "start": float(x_values[0]),
                "end": float(x_values[-1]),
                "min": float(min(x_values)),
                "max": float(max(x_values)),
            },
            "y_progress_scene": {
                "start": float(y_values[0]),
                "end": float(y_values[-1]),
                "min": float(min(y_values)),
                "max": float(max(y_values)),
            },
        }

        trajectory = {
            "seed": int(args.seed),
            "policy_step_dt_s": float(step_dt),
            "command": summary["command"],
            "scene_usd": scene_usd_path.name if args.export_usd else None,
            "stair_zone": scene.stair_zone,
            "initial_pose": {
                "position_world": vector_to_list(initial_world_position),
                "position_scene": vector_to_list(initial_scene_position),
                "orientation_wxyz": vector_to_list(initial_quaternion),
                "yaw_rad": float(quaternion_to_yaw(initial_quaternion)),
                "yaw_deg": float(math.degrees(quaternion_to_yaw(initial_quaternion))),
                "entered_stair_zone": bool(entered_stair_zone(scene, initial_scene_position)),
            },
            "steps": trajectory_steps,
        }
        write_packed_trajectory_npz(
            packed_trajectory_path,
            poses=packed_poses,
            measurements_local=packed_measurements_local,
            ground_truth_local=packed_ground_truth_local,
            step_indices=packed_step_indices,
            timestamps_s=packed_timestamps_s,
            visibility_fractions=visibility_values,
        )
        write_json(output_dir / "summary.json", summary)
        print(f"[rollout] packed trajectory written to {packed_trajectory_path}", flush=True)
        print(f"[rollout] summary written to {output_dir / 'summary.json'}", flush=True)
        return summary, trajectory
    finally:
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
        if policy_env is not None:
            close_method = getattr(policy_env, "close", None)
            if callable(close_method):
                close_method()
        if simulation_app is not None:
            simulation_app.close()
        if scene_temp_dir is not None:
            scene_temp_dir.cleanup()


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    summary_path = output_dir / "summary.json"
    try:
        run_rollout(args)
    except Exception as exc:
        error_summary = {
            "error": f"{type(exc).__name__}: {exc}",
            "seed": int(args.seed),
            "output_dir": str(output_dir),
        }
        try:
            write_json(summary_path, error_summary)
        except Exception:
            pass
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
