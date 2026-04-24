from __future__ import annotations

import argparse
import json
import re
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import isaaclab_datacollect_anymal_rollout as rollout


DEFAULT_TRAJECTORIES_PER_SCENE = 8
TRAJECTORY_FILE_RE = re.compile(r"^trajectory_(\d+)\.npz$")


@dataclass
class EpisodeBuffer:
    poses: list[np.ndarray] = field(default_factory=list)
    measurements_local: list[np.ndarray] = field(default_factory=list)
    ground_truth_local: list[np.ndarray] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    visibility_fractions: list[float] = field(default_factory=list)
    x_positions_scene: list[float] = field(default_factory=list)
    y_positions_scene: list[float] = field(default_factory=list)
    steps_completed: int = 0
    terminated_early: bool = False
    entered_stair_zone: bool = False


@dataclass
class VisibilityAccumulator:
    count: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @classmethod
    def from_summary(cls, *, count: int, mean: float, min_value: float, max_value: float) -> "VisibilityAccumulator":
        if count <= 0:
            return cls()
        return cls(
            count=int(count),
            total=float(mean) * int(count),
            min_value=float(min_value),
            max_value=float(max_value),
        )

    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return float(self.total / self.count)

    def min(self) -> float:
        return float(self.min_value) if self.min_value is not None else 0.0

    def max(self) -> float:
        return float(self.max_value) if self.max_value is not None else 0.0


@dataclass
class DatasetState:
    dataset_dir: Path
    manifest_path: Path
    batch_summary_path: Path
    scenes_dir: Path
    started_at: str
    trajectories_saved: int
    next_trajectory_id: int
    scenes_completed: int
    scene_index_start: int
    incomplete_trajectory_count: int
    entered_stair_saved_count: int
    visibility_accumulator: VisibilityAccumulator
    last_scene_seed: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect DataCollect-style ANYmal C trajectories in a single Isaac Sim process. "
            "One scene is created, then the same single-env policy is reset multiple times before moving on."
        )
    )
    parser.add_argument("--policy-checkpoint", type=Path, required=True, help="TorchScript/JIT rough-terrain policy checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Output directory for trajectory_*.npz, manifest.jsonl, and batch_summary.json.")
    parser.add_argument("--target-trajectories", type=int, default=800, help="Number of complete trajectories to save.")
    parser.add_argument("--steps", type=int, default=250, help="Number of policy steps per trajectory.")
    parser.add_argument("--seed", type=int, default=0, help="Base scene seed. Scene index i uses seed + i.")
    parser.add_argument(
        "--trajectories-per-scene",
        type=int,
        default=DEFAULT_TRAJECTORIES_PER_SCENE,
        help="How many reset-based trajectories to collect from the same scene before rebuilding the environment.",
    )
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
        help="Export one scene USD per scene under dataset-dir/scenes.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume into an existing dataset directory without overwriting saved trajectories.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional trajectory id to start writing from. Useful when resuming or manually offsetting file numbering.",
    )
    args = parser.parse_args()
    if args.target_trajectories <= 0:
        raise SystemExit("--target-trajectories must be positive.")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.trajectories_per_scene <= 0:
        raise SystemExit("--trajectories-per-scene must be positive.")
    if args.start_index is not None and args.start_index < 0:
        raise SystemExit("--start-index must be non-negative.")
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_existing_trajectory_ids(dataset_dir: Path) -> list[int]:
    trajectory_ids: list[int] = []
    for path in sorted(dataset_dir.glob("trajectory_*.npz")):
        match = TRAJECTORY_FILE_RE.match(path.name)
        if match is None:
            continue
        trajectory_ids.append(int(match.group(1)))
    return trajectory_ids


def read_manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                entry = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse {manifest_path} line {line_number}: {exc}") from exc
            if not isinstance(entry, dict):
                raise RuntimeError(f"Expected dict payload in {manifest_path} line {line_number}, received {type(entry).__name__}.")
            entries.append(entry)
    return entries


def read_summary_payload(batch_summary_path: Path) -> dict[str, Any] | None:
    if not batch_summary_path.exists():
        return None
    payload = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {batch_summary_path}, received {type(payload).__name__}.")
    return payload


def build_visibility_accumulator(
    *,
    manifest_entries: list[dict[str, Any]],
    summary_payload: dict[str, Any] | None,
    trajectories_saved: int,
) -> VisibilityAccumulator:
    if summary_payload is not None and int(summary_payload.get("saved_trajectories", 0)) == trajectories_saved:
        return VisibilityAccumulator.from_summary(
            count=trajectories_saved,
            mean=float(summary_payload.get("saved_visibility_fraction_mean", 0.0)),
            min_value=float(summary_payload.get("saved_visibility_fraction_min", 0.0)),
            max_value=float(summary_payload.get("saved_visibility_fraction_max", 0.0)),
        )
    accumulator = VisibilityAccumulator()
    for entry in manifest_entries:
        accumulator.add(float(entry.get("visibility_fraction_mean", 0.0)))
    return accumulator


def initialize_dataset_state(args: argparse.Namespace) -> DatasetState:
    dataset_dir = rollout.ensure_output_dir(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.jsonl"
    batch_summary_path = dataset_dir / "batch_summary.json"
    scenes_dir = dataset_dir / "scenes"

    if args.export_usd:
        scenes_dir.mkdir(parents=True, exist_ok=True)

    trajectory_ids = list_existing_trajectory_ids(dataset_dir)
    summary_payload = read_summary_payload(batch_summary_path)
    manifest_entries = read_manifest_entries(manifest_path)

    if not args.resume:
        if trajectory_ids:
            raise RuntimeError(
                f"Dataset directory already contains trajectory files. Use --resume or a new --dataset-dir. First file id: {trajectory_ids[0]}"
            )
        for reserved_path in (manifest_path, batch_summary_path):
            if reserved_path.exists():
                raise RuntimeError(f"Dataset directory already contains {reserved_path}. Use --resume or a new --dataset-dir.")
        return DatasetState(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            batch_summary_path=batch_summary_path,
            scenes_dir=scenes_dir,
            started_at=utc_now_iso(),
            trajectories_saved=0,
            next_trajectory_id=int(args.start_index) if args.start_index is not None else 0,
            scenes_completed=0,
            scene_index_start=0,
            incomplete_trajectory_count=0,
            entered_stair_saved_count=0,
            visibility_accumulator=VisibilityAccumulator(),
            last_scene_seed=None,
        )

    next_trajectory_id = (max(trajectory_ids) + 1) if trajectory_ids else 0
    if args.start_index is not None:
        if int(args.start_index) < next_trajectory_id:
            raise RuntimeError(
                f"--start-index={args.start_index} is smaller than the next available trajectory id {next_trajectory_id}."
            )
        next_trajectory_id = int(args.start_index)

    trajectories_saved = len(trajectory_ids)
    if summary_payload is not None:
        started_at = str(summary_payload.get("started_at_utc", utc_now_iso()))
        scenes_completed = int(summary_payload.get("scenes_completed", 0))
        incomplete_trajectory_count = int(summary_payload.get("counts", {}).get("incomplete_trajectory_count", 0))
        entered_stair_saved_count = int(summary_payload.get("counts", {}).get("entered_stair_zone_saved", 0))
        last_scene_seed = summary_payload.get("last_scene_seed")
        last_scene_seed = int(last_scene_seed) if last_scene_seed is not None else None
    else:
        started_at = utc_now_iso()
        scenes_completed = 0
        incomplete_trajectory_count = 0
        entered_stair_saved_count = 0
        last_scene_seed = None

    if manifest_entries:
        scene_index_start = max(int(entry.get("scene_index", -1)) for entry in manifest_entries) + 1
        if summary_payload is None:
            entered_stair_saved_count = sum(bool(entry.get("entered_stair_zone", False)) for entry in manifest_entries)
            last_scene_seed = max(int(entry.get("scene_seed", 0)) for entry in manifest_entries)
    else:
        scene_index_start = scenes_completed

    if manifest_entries and len(manifest_entries) != trajectories_saved:
        rollout.log_progress(
            "resume warning: manifest entry count does not match saved trajectory files; "
            "visibility and stair-zone aggregates may rely on batch_summary.json where available"
        )

    return DatasetState(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        batch_summary_path=batch_summary_path,
        scenes_dir=scenes_dir,
        started_at=started_at,
        trajectories_saved=trajectories_saved,
        next_trajectory_id=next_trajectory_id,
        scenes_completed=scenes_completed,
        scene_index_start=scene_index_start,
        incomplete_trajectory_count=incomplete_trajectory_count,
        entered_stair_saved_count=entered_stair_saved_count,
        visibility_accumulator=build_visibility_accumulator(
            manifest_entries=manifest_entries,
            summary_payload=summary_payload,
            trajectories_saved=trajectories_saved,
        ),
        last_scene_seed=last_scene_seed,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def parse_done_mask(step_result: Any, torch_module: Any) -> bool:
    if len(step_result) == 5:
        _, _, terminated, truncated, _ = step_result
        done = torch_module.logical_or(terminated, truncated)
    elif len(step_result) == 4:
        _, _, terminated, _ = step_result
        done = terminated if torch_module.is_tensor(terminated) else torch_module.as_tensor(terminated)
    else:
        raise RuntimeError(f"Unexpected env.step return format with {len(step_result)} values.")
    done_tensor = done if torch_module.is_tensor(done) else torch_module.as_tensor(done)
    return bool(done_tensor.detach().cpu().numpy().astype(bool, copy=False)[0])


def make_command_record(command_vector: np.ndarray, heading_target_rad: float | None) -> dict[str, Any]:
    return {
        "lin_vel_x": float(command_vector[0]),
        "lin_vel_y": float(command_vector[1]),
        "ang_vel_z": float(command_vector[2]),
        "heading_target_rad": float(heading_target_rad) if heading_target_rad is not None else None,
        "heading_target_deg": float(np.degrees(heading_target_rad)) if heading_target_rad is not None else None,
        "source": "command_manager_base_velocity",
    }


def visibility_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    values_np = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
    }


def build_manifest_entry(
    *,
    trajectory_id: int,
    file_name: str,
    scene_index: int,
    scene_seed: int,
    trajectory_in_scene: int,
    command_record: dict[str, Any],
    buffer: EpisodeBuffer,
) -> dict[str, Any]:
    visibility = visibility_stats(buffer.visibility_fractions)
    x_values = buffer.x_positions_scene
    y_values = buffer.y_positions_scene
    return {
        "trajectory_id": int(trajectory_id),
        "file_name": file_name,
        "scene_index": int(scene_index),
        "scene_seed": int(scene_seed),
        "trajectory_in_scene": int(trajectory_in_scene),
        "steps_completed": int(buffer.steps_completed),
        "terminated_early": bool(buffer.terminated_early),
        "entered_stair_zone": bool(buffer.entered_stair_zone),
        "visibility_fraction_mean": float(visibility["mean"]),
        "visibility_fraction_min": float(visibility["min"]),
        "visibility_fraction_max": float(visibility["max"]),
        "command": command_record,
        "x_progress_scene": {
            "start": float(x_values[0]) if x_values else None,
            "end": float(x_values[-1]) if x_values else None,
            "min": float(min(x_values)) if x_values else None,
            "max": float(max(x_values)) if x_values else None,
        },
        "y_progress_scene": {
            "start": float(y_values[0]) if y_values else None,
            "end": float(y_values[-1]) if y_values else None,
            "min": float(min(y_values)) if y_values else None,
            "max": float(max(y_values)) if y_values else None,
        },
    }


def build_batch_summary(
    *,
    args: argparse.Namespace,
    started_at: str,
    dataset_dir: Path,
    manifest_path: Path,
    scenes_completed: int,
    trajectories_saved: int,
    incomplete_trajectory_count: int,
    visibility_accumulator: VisibilityAccumulator,
    entered_stair_saved_count: int,
    last_scene_seed: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "started_at_utc": started_at,
        "updated_at_utc": utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "manifest_path": manifest_path.name,
        "target_trajectories": int(args.target_trajectories),
        "saved_trajectories": int(trajectories_saved),
        "remaining_trajectories": int(max(args.target_trajectories - trajectories_saved, 0)),
        "scenes_completed": int(scenes_completed),
        "trajectories_per_scene": int(args.trajectories_per_scene),
        "steps_per_rollout": int(args.steps),
        "seed": int(args.seed),
        "last_scene_seed": int(last_scene_seed) if last_scene_seed is not None else None,
        "device": str(args.device),
        "headless": bool(args.headless),
        "collection_mode": "single_process_single_env_multi_reset",
        "acceptance_rule": "save if and only if steps_completed == steps_per_rollout and terminated_early == false",
        "artifacts": {
            "export_usd": bool(args.export_usd),
        },
        "counts": {
            "entered_stair_zone_saved": int(entered_stair_saved_count),
            "incomplete_trajectory_count": int(incomplete_trajectory_count),
        },
        "saved_visibility_fraction_mean": visibility_accumulator.mean(),
        "saved_visibility_fraction_min": visibility_accumulator.min(),
        "saved_visibility_fraction_max": visibility_accumulator.max(),
        "error": error,
    }


def reset_episode(
    *,
    policy_env: Any,
    robot_asset: Any,
    base_body_index: int,
    env_origin_np: np.ndarray,
    torch_module: Any,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float | None]:
    reset_result = policy_env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    placement_meta = rollout.maybe_place_robot_at_start(policy_env, torch_module)
    policy_env.command_manager.compute(0.0)
    observations = rollout.refresh_policy_observations(policy_env, fallback_obs=observations)
    if not isinstance(observations, dict) or "policy" not in observations:
        raise RuntimeError("Policy environment reset did not produce obs['policy'].")

    velocity_command_term = policy_env.command_manager.get_term("base_velocity")
    initial_command_base = policy_env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy().astype(np.float32)
    initial_heading_target = (
        float(velocity_command_term.heading_target[0].detach().cpu().item())
        if hasattr(velocity_command_term, "heading_target")
        else None
    )
    initial_world_position, initial_quaternion = rollout.extract_base_body_pose(robot_asset, base_body_index)
    initial_scene_position = initial_world_position - env_origin_np
    return (
        observations,
        placement_meta,
        initial_command_base,
        initial_world_position,
        initial_scene_position,
        initial_heading_target,
    )


def collect_episode(
    *,
    args: argparse.Namespace,
    policy_env: Any,
    policy: Any,
    robot_asset: Any,
    base_body_index: int,
    scene: rollout.SceneSpec,
    gt_world_points: np.ndarray,
    camera_mounts: tuple[rollout.CameraMount, ...],
    create_pointcloud_from_depth: Any,
    torch_module: Any,
    env_origin_np: np.ndarray,
    step_dt: float,
    observations: Any,
) -> EpisodeBuffer:
    buffer = EpisodeBuffer()
    entered_stair = False

    for step_index in range(args.steps):
        if step_index % 50 == 0 or step_index == args.steps - 1:
            rollout.log_progress(f"step {step_index + 1}/{args.steps}")

        observations = rollout.refresh_policy_observations(policy_env, fallback_obs=observations)
        with torch_module.inference_mode():
            actions = policy(observations["policy"])
        step_result = policy_env.step(actions)
        observations = step_result[0]
        terminated_now = parse_done_mask(step_result, torch_module)

        world_position, quaternion_wxyz = rollout.extract_base_body_pose(robot_asset, base_body_index)
        scene_position = world_position - env_origin_np
        robot_pose_world = rollout.pose_matrix(world_position, rollout.quaternion_to_rotation_matrix(quaternion_wxyz))
        sensor_meta, measurement_local, ground_truth_local = rollout.collect_sensor_step(
            policy_env=policy_env,
            robot_pose_world=robot_pose_world,
            camera_mounts=camera_mounts,
            gt_world_points=gt_world_points,
            create_pointcloud_from_depth=create_pointcloud_from_depth,
            torch_module=torch_module,
        )

        buffer.poses.append(robot_pose_world.astype(np.float32, copy=True))
        buffer.measurements_local.append(measurement_local.astype(np.float32, copy=False))
        buffer.ground_truth_local.append(ground_truth_local.astype(np.float32, copy=False))
        buffer.step_indices.append(int(step_index))
        buffer.timestamps_s.append(float((step_index + 1) * step_dt))
        buffer.visibility_fractions.append(float(sensor_meta["visible_gt_fraction"]))
        buffer.x_positions_scene.append(float(scene_position[0]))
        buffer.y_positions_scene.append(float(scene_position[1]))
        buffer.steps_completed = int(step_index + 1)
        entered_stair = bool(entered_stair or rollout.entered_stair_zone(scene, scene_position))
        buffer.entered_stair_zone = entered_stair

        if terminated_now and step_index < args.steps - 1:
            buffer.terminated_early = True
            break

    return buffer


def main() -> int:
    args = parse_args()
    dataset_state = initialize_dataset_state(args)
    dataset_dir = dataset_state.dataset_dir
    manifest_path = dataset_state.manifest_path
    batch_summary_path = dataset_state.batch_summary_path
    scenes_dir = dataset_state.scenes_dir

    started_at = dataset_state.started_at
    trajectories_saved = dataset_state.trajectories_saved
    next_trajectory_id = dataset_state.next_trajectory_id
    scenes_completed = dataset_state.scenes_completed
    incomplete_trajectory_count = dataset_state.incomplete_trajectory_count
    entered_stair_saved_count = dataset_state.entered_stair_saved_count
    visibility_accumulator = dataset_state.visibility_accumulator
    last_scene_seed = dataset_state.last_scene_seed

    simulation_app = None
    policy = None
    torch = None
    sim_utils = None
    ManagerBasedRLEnv = None
    CameraCfg = None
    create_pointcloud_from_depth = None
    TerrainImporterCfg = None
    convert_camera_frame_orientation_convention = None
    AnymalCRoughEnvCfg_PLAY = None

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

        import torch as torch_module
        import isaaclab.sim as sim_utils_module
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRLEnvType
        from isaaclab.sensors import CameraCfg as CameraCfgType
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth as create_pointcloud_from_depth_fn
        from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgType
        from isaaclab.utils.math import convert_camera_frame_orientation_convention as convert_camera_frame_orientation_convention_fn
        from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
            AnymalCRoughEnvCfg_PLAY as AnymalCRoughEnvCfgPlayType,
        )

        torch = torch_module
        sim_utils = sim_utils_module
        ManagerBasedRLEnv = ManagerBasedRLEnvType
        CameraCfg = CameraCfgType
        create_pointcloud_from_depth = create_pointcloud_from_depth_fn
        TerrainImporterCfg = TerrainImporterCfgType
        convert_camera_frame_orientation_convention = convert_camera_frame_orientation_convention_fn
        AnymalCRoughEnvCfg_PLAY = AnymalCRoughEnvCfgPlayType

        asset_root = rollout.configure_isaac_asset_root()
        rollout.log_progress(f"resolved Isaac asset root: {asset_root}")
        policy = rollout.load_policy_checkpoint(args.policy_checkpoint, str(args.device), torch)
        rollout.log_progress("policy loaded for sequential collection")

        scene_index = dataset_state.scene_index_start
        while trajectories_saved < args.target_trajectories:
            scene_seed = int(args.seed + scene_index)
            last_scene_seed = scene_seed
            rollout.log_progress(
                f"starting scene {scene_index + 1}: scene_seed={scene_seed}, saved={trajectories_saved}/{args.target_trajectories}"
            )

            scene_rng = np.random.default_rng(scene_seed)
            gt_rng = np.random.default_rng(scene_seed + 1000)
            scene = rollout.build_scene(scene_rng, seed=scene_seed)
            rollout.validate_scene_spec(scene)
            gt_points_env = rollout.sample_scene_surface_points(scene, spacing=rollout.GROUND_TRUTH_SAMPLE_SPACING, rng=gt_rng)

            scene_temp_dir: tempfile.TemporaryDirectory[str] | None = None
            if args.export_usd:
                scene_usd_path = scenes_dir / f"scene_{scene_index:06d}_seed_{scene_seed}.usd"
            else:
                scene_temp_dir = tempfile.TemporaryDirectory(prefix=f"isaaclab_sequential_scene_{scene_index:06d}_", dir="/tmp")
                scene_usd_path = Path(scene_temp_dir.name) / f"scene_{scene_index:06d}_seed_{scene_seed}.usd"

            policy_env = None
            try:
                rollout.author_scene_usd(scene, scene_usd_path, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf)

                env_cfg = AnymalCRoughEnvCfg_PLAY()
                env_cfg.seed = scene_seed
                env_cfg.scene.num_envs = 1
                env_cfg.scene.env_spacing = 2.5
                env_cfg.curriculum = None
                rollout.configure_robot_spawn_if_possible(env_cfg)
                rollout.configure_base_velocity_command(env_cfg, command=None)
                env_cfg.scene.terrain = TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="usd",
                    usd_path=str(scene_usd_path.resolve()),
                )
                if getattr(env_cfg.scene, "height_scanner", None) is not None and hasattr(env_cfg.scene.height_scanner, "mesh_prim_paths"):
                    env_cfg.scene.height_scanner.mesh_prim_paths = [rollout.IMPORTED_RAYCAST_TERRAIN_PRIM_PATH]
                env_cfg.sim.device = str(args.device)
                if str(args.device).startswith("cpu"):
                    env_cfg.sim.use_fabric = False
                if hasattr(env_cfg, "episode_length_s"):
                    env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 0.0)), float(args.steps) * 0.1 + 10.0)

                depth_spec, camera_mounts, _, _ = rollout.configure_scene_cameras(
                    scene,
                    env_cfg,
                    sim_utils=sim_utils,
                    CameraCfg=CameraCfg,
                    torch_module=torch,
                    convert_camera_frame_orientation_convention=convert_camera_frame_orientation_convention,
                    include_record_camera=False,
                )

                rollout.log_progress(f"creating ManagerBasedRLEnv for scene {scene_index + 1}")
                policy_env = ManagerBasedRLEnv(cfg=env_cfg)
                rollout.disable_base_velocity_debug_vis(policy_env)
                rollout.configure_depth_camera_intrinsics(policy_env, depth_spec, camera_mounts, torch)

                robot_asset = policy_env.scene["robot"]
                base_body_index = rollout.resolve_base_body_index(robot_asset)
                env_origin_tensor = getattr(policy_env.scene, "env_origins", None)
                if env_origin_tensor is None:
                    env_origin_np = np.zeros(3, dtype=np.float32)
                else:
                    env_origin_np = env_origin_tensor[0].detach().cpu().numpy().astype(np.float32)
                gt_world_points = gt_points_env + env_origin_np[None, :]
                step_dt = float(policy_env.step_dt)

                scene_trajectory_count = 0
                while (
                    scene_trajectory_count < args.trajectories_per_scene
                    and trajectories_saved < args.target_trajectories
                ):
                    rollout.log_progress(
                        f"scene {scene_index + 1} trajectory {scene_trajectory_count + 1}/{args.trajectories_per_scene} "
                        f"(saved={trajectories_saved}/{args.target_trajectories})"
                    )

                    (
                        observations,
                        _placement_meta,
                        initial_command_base,
                        initial_world_position,
                        initial_scene_position,
                        initial_heading_target,
                    ) = reset_episode(
                        policy_env=policy_env,
                        robot_asset=robot_asset,
                        base_body_index=base_body_index,
                        env_origin_np=env_origin_np,
                        torch_module=torch,
                    )

                    buffer = EpisodeBuffer()
                    buffer.x_positions_scene.append(float(initial_scene_position[0]))
                    buffer.y_positions_scene.append(float(initial_scene_position[1]))
                    buffer.entered_stair_zone = bool(rollout.entered_stair_zone(scene, initial_scene_position))
                    episode_buffer = collect_episode(
                        args=args,
                        policy_env=policy_env,
                        policy=policy,
                        robot_asset=robot_asset,
                        base_body_index=base_body_index,
                        scene=scene,
                        gt_world_points=gt_world_points,
                        camera_mounts=camera_mounts,
                        create_pointcloud_from_depth=create_pointcloud_from_depth,
                        torch_module=torch,
                        env_origin_np=env_origin_np,
                        step_dt=step_dt,
                        observations=observations,
                    )
                    episode_buffer.x_positions_scene = [float(initial_scene_position[0])] + episode_buffer.x_positions_scene
                    episode_buffer.y_positions_scene = [float(initial_scene_position[1])] + episode_buffer.y_positions_scene
                    episode_buffer.entered_stair_zone = bool(
                        buffer.entered_stair_zone or episode_buffer.entered_stair_zone
                    )

                    complete_trajectory = (not episode_buffer.terminated_early) and (episode_buffer.steps_completed == args.steps)
                    if complete_trajectory:
                        file_name = f"trajectory_{next_trajectory_id:06d}.npz"
                        trajectory_path = dataset_dir / file_name
                        rollout.write_packed_trajectory_npz(
                            trajectory_path,
                            poses=episode_buffer.poses,
                            measurements_local=episode_buffer.measurements_local,
                            ground_truth_local=episode_buffer.ground_truth_local,
                            step_indices=episode_buffer.step_indices,
                            timestamps_s=episode_buffer.timestamps_s,
                            visibility_fractions=episode_buffer.visibility_fractions,
                        )
                        command_record = make_command_record(initial_command_base, initial_heading_target)
                        append_jsonl(
                            manifest_path,
                            build_manifest_entry(
                                trajectory_id=next_trajectory_id,
                                file_name=file_name,
                                scene_index=scene_index,
                                scene_seed=scene_seed,
                                trajectory_in_scene=scene_trajectory_count,
                                command_record=command_record,
                                buffer=episode_buffer,
                            ),
                        )
                        next_trajectory_id += 1
                        trajectories_saved += 1
                        if episode_buffer.entered_stair_zone:
                            entered_stair_saved_count += 1
                        visibility_accumulator.add(float(np.mean(np.asarray(episode_buffer.visibility_fractions, dtype=np.float32))))
                    else:
                        incomplete_trajectory_count += 1

                    scene_trajectory_count += 1
                    rollout.write_json(
                        batch_summary_path,
                        build_batch_summary(
                            args=args,
                            started_at=started_at,
                            dataset_dir=dataset_dir,
                            manifest_path=manifest_path,
                            scenes_completed=scenes_completed,
                            trajectories_saved=trajectories_saved,
                            incomplete_trajectory_count=incomplete_trajectory_count,
                            visibility_accumulator=visibility_accumulator,
                            entered_stair_saved_count=entered_stair_saved_count,
                            last_scene_seed=last_scene_seed,
                        ),
                    )
            finally:
                if policy_env is not None:
                    close_method = getattr(policy_env, "close", None)
                    if callable(close_method):
                        close_method()
                if sim_utils is not None:
                    sim_utils.clear_stage()
                if scene_temp_dir is not None:
                    scene_temp_dir.cleanup()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            scenes_completed = scene_index + 1
            rollout.write_json(
                batch_summary_path,
                build_batch_summary(
                    args=args,
                    started_at=started_at,
                    dataset_dir=dataset_dir,
                    manifest_path=manifest_path,
                    scenes_completed=scenes_completed,
                    trajectories_saved=trajectories_saved,
                    incomplete_trajectory_count=incomplete_trajectory_count,
                    visibility_accumulator=visibility_accumulator,
                    entered_stair_saved_count=entered_stair_saved_count,
                    last_scene_seed=last_scene_seed,
                ),
            )
            rollout.log_progress(
                f"finished scene {scene_index + 1}: saved={trajectories_saved}/{args.target_trajectories}, "
                f"incomplete={incomplete_trajectory_count}"
            )
            scene_index += 1

        rollout.write_json(
            batch_summary_path,
            build_batch_summary(
                args=args,
                started_at=started_at,
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                scenes_completed=scenes_completed,
                trajectories_saved=trajectories_saved,
                incomplete_trajectory_count=incomplete_trajectory_count,
                visibility_accumulator=visibility_accumulator,
                entered_stair_saved_count=entered_stair_saved_count,
                last_scene_seed=last_scene_seed,
            ),
        )
        return 0
    except Exception as exc:
        rollout.write_json(
            batch_summary_path,
            build_batch_summary(
                args=args,
                started_at=started_at,
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                scenes_completed=scenes_completed,
                trajectories_saved=trajectories_saved,
                incomplete_trajectory_count=incomplete_trajectory_count,
                visibility_accumulator=visibility_accumulator,
                entered_stair_saved_count=entered_stair_saved_count,
                last_scene_seed=last_scene_seed,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        traceback.print_exc()
        return 1
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
