from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors
    from matplotlib import pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "matplotlib is required to export report figures. Install it with 'pip install matplotlib'."
    ) from exc

try:
    import MinkowskiEngine as ME
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "MinkowskiEngine is required to run prediction export. Install it before running this script."
    ) from exc

from src.config import GridConfig, ModelConfig
from src.data import slice_packed_points
from src.geometry import relative_transform, transform_points
from src.model import TerrainReconstructionModel
from src.preprocess import (
    build_network_input,
    decode_predictions_to_point_clouds,
    make_batched_coordinate_tensor,
    make_batched_feature_tensor,
    voxelize_points,
)


ROW_LABELS = ("Measurement", "Reconstruction", "Ground truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-style terrain reconstruction figures from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("data/trajectory_000.npz"),
        help="Packed trajectory .npz file to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prediction_report.png"),
        help="Path to the exported PNG figure.",
    )
    parser.add_argument(
        "--save-rollout",
        type=Path,
        default=None,
        help="Optional .npz path for saving the full predicted rollout.",
    )
    parser.add_argument(
        "--timesteps",
        nargs="*",
        default=["auto"],
        help="Timesteps to display, for example: --timesteps 0 30. Default: evenly spaced columns.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of columns when --timesteps auto is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=None,
        help="Override the checkpoint pruning threshold.",
    )
    parser.add_argument(
        "--disable-feedback",
        action="store_true",
        help="Disable autoregressive feedback and only use the current measurement.",
    )
    parser.add_argument(
        "--measurement-max-points",
        type=int,
        default=16000,
        help="Maximum occupied measurement voxels rendered per panel.",
    )
    parser.add_argument(
        "--prediction-max-points",
        type=int,
        default=20000,
        help="Maximum occupied prediction voxels rendered per panel.",
    )
    parser.add_argument(
        "--ground-truth-max-points",
        type=int,
        default=20000,
        help="Maximum occupied ground-truth voxels rendered per panel.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.4,
        help="Voxel edge width scale kept for backward compatibility.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=22.0,
        help="3D camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D camera azimuth in degrees.",
    )
    parser.add_argument(
        "--bounds-min",
        nargs=3,
        type=float,
        default=GridConfig().bounds_min,
        metavar=("X", "Y", "Z"),
        help="Minimum plot bounds.",
    )
    parser.add_argument(
        "--bounds-max",
        nargs=3,
        type=float,
        default=GridConfig().bounds_max,
        metavar=("X", "Y", "Z"),
        help="Maximum plot bounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for point downsampling.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def load_trajectory(trajectory_path: Path) -> Dict[str, object]:
    with np.load(trajectory_path) as data:
        poses = data["poses"].astype(np.float32, copy=False)
        measurement_points = data["measurement_points"].astype(np.float32, copy=False)
        measurement_splits = data["measurement_splits"].astype(np.int32, copy=False)
        ground_truth_points = data["ground_truth_points"].astype(np.float32, copy=False)
        ground_truth_splits = data["ground_truth_splits"].astype(np.int32, copy=False)

    num_steps = int(poses.shape[0])
    measurements = [
        slice_packed_points(measurement_points, measurement_splits, timestep)
        for timestep in range(num_steps)
    ]
    ground_truth = [
        slice_packed_points(ground_truth_points, ground_truth_splits, timestep)
        for timestep in range(num_steps)
    ]
    return {
        "poses": poses,
        "measurements": measurements,
        "ground_truth": ground_truth,
    }


def parse_timestep_values(values: Sequence[str], num_steps: int, columns: int) -> List[int]:
    if not values or (len(values) == 1 and values[0].lower() == "auto"):
        if num_steps <= 1:
            return [0]
        columns = max(1, min(columns, num_steps))
        return sorted(
            {
                int(round(index * (num_steps - 1) / max(columns - 1, 1)))
                for index in range(columns)
            }
        )

    selected: List[int] = []
    for value in values:
        timestep = int(value)
        if timestep < 0:
            timestep += num_steps
        if timestep < 0 or timestep >= num_steps:
            raise ValueError(f"timestep {value} is out of range for sequence length {num_steps}")
        if timestep not in selected:
            selected.append(timestep)
    return selected


def make_sparse_tensor(
    coordinate_batches: Sequence[np.ndarray],
    feature_batches: Sequence[np.ndarray],
    device: torch.device,
) -> ME.SparseTensor:
    coordinates = make_batched_coordinate_tensor(coordinate_batches, device)
    features = make_batched_feature_tensor(feature_batches, device)
    return ME.SparseTensor(coordinates=coordinates, features=features)


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    pruning_threshold: Optional[float],
) -> Tuple[TerrainReconstructionModel, Dict[str, object]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model_config = ModelConfig()
    if pruning_threshold is not None:
        model_config = ModelConfig(pruning_threshold=pruning_threshold)

    model = TerrainReconstructionModel(model_config).to(device)
    model.load_state_dict(normalize_state_dict(state_dict))
    model.eval()
    return model, metadata


@torch.no_grad()
def rollout_predictions(
    model: TerrainReconstructionModel,
    poses: np.ndarray,
    measurements: Sequence[np.ndarray],
    grid: GridConfig,
    device: torch.device,
    disable_feedback: bool,
    pruning_threshold: Optional[float],
) -> List[np.ndarray]:
    predictions: List[np.ndarray] = []
    previous_prediction = np.empty((0, 3), dtype=np.float32)

    for step, measurement_local in enumerate(measurements):
        previous_prediction_local = np.empty((0, 3), dtype=np.float32)
        if not disable_feedback and step > 0 and previous_prediction.size > 0:
            transform_prev_to_current = relative_transform(poses[step - 1], poses[step])
            previous_prediction_local = transform_points(previous_prediction, transform_prev_to_current)

        model_input = build_network_input(
            current_measurement=measurement_local.astype(np.float32, copy=False),
            previous_prediction=previous_prediction_local,
            grid=grid,
        )
        if model_input.coordinates.size == 0:
            previous_prediction = np.empty((0, 3), dtype=np.float32)
            predictions.append(previous_prediction)
            continue

        sparse_input = make_sparse_tensor([model_input.coordinates], [model_input.features], device)
        output = model(sparse_input, pruning_threshold=pruning_threshold)
        current_prediction = decode_predictions_to_point_clouds(
            output.offsets.C,
            output.offsets.F,
            grid=grid,
            batch_size=1,
            time_index=grid.current_time_index,
        )[0]
        previous_prediction = current_prediction.astype(np.float32, copy=False)
        predictions.append(previous_prediction)

    return predictions


def pack_point_cloud_sequence(clouds: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    splits = [0]
    packed_chunks: List[np.ndarray] = []
    total = 0
    for cloud in clouds:
        packed_chunks.append(cloud.astype(np.float32, copy=False))
        total += int(cloud.shape[0])
        splits.append(total)
    packed = (
        np.concatenate(packed_chunks, axis=0)
        if packed_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    return packed, np.asarray(splits, dtype=np.int32)


def maybe_save_rollout(
    output_path: Optional[Path],
    poses: np.ndarray,
    predictions: Sequence[np.ndarray],
) -> Optional[Path]:
    if output_path is None:
        return None
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed_points, splits = pack_point_cloud_sequence(predictions)
    np.savez_compressed(
        output_path,
        poses=poses.astype(np.float32, copy=False),
        predicted_points=packed_points,
        predicted_splits=splits,
    )
    return output_path


def sequence_point_counts(clouds: Sequence[np.ndarray]) -> List[int]:
    return [int(cloud.shape[0]) for cloud in clouds]


def voxelize_cloud(points: np.ndarray, grid: GridConfig) -> np.ndarray:
    voxelized = voxelize_points(points, grid, time_index=grid.current_time_index)
    if voxelized.coordinates.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return voxelized.coordinates[:, :3].astype(np.int32, copy=False)


def voxelize_sequence(clouds: Sequence[np.ndarray], grid: GridConfig) -> List[np.ndarray]:
    return [voxelize_cloud(cloud, grid) for cloud in clouds]


def sequence_voxel_counts(voxel_clouds: Sequence[np.ndarray]) -> List[int]:
    return [int(cloud.shape[0]) for cloud in voxel_clouds]


def _row_view(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    if contiguous.size == 0:
        return np.empty((0,), dtype=np.dtype((np.void, 0)))
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)


def occupancy_metrics(query_voxels: np.ndarray, target_voxels: np.ndarray) -> Dict[str, float]:
    query_unique = np.unique(_row_view(query_voxels))
    target_unique = np.unique(_row_view(target_voxels))
    if query_unique.size == 0 and target_unique.size == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    if query_unique.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    if target_unique.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    overlap = np.intersect1d(query_unique, target_unique, assume_unique=True).size
    precision = float(overlap / query_unique.size)
    recall = float(overlap / target_unique.size)
    f1 = float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0)
    union = query_unique.size + target_unique.size - overlap
    iou = float(overlap / union) if union > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def sequence_occupancy_metrics(query_sequence: Sequence[np.ndarray], target_sequence: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": [],
    }
    for query_voxels, target_voxels in zip(query_sequence, target_sequence):
        timestep_metrics = occupancy_metrics(query_voxels, target_voxels)
        for key, value in timestep_metrics.items():
            metrics[key].append(float(value))
    return metrics


def summarize_metric_sequence(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    values_array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(values_array.min()),
        "max": float(values_array.max()),
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
    }


def summarize_metric_dict(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, Optional[float]]]:
    return {key: summarize_metric_sequence(values) for key, values in metrics.items()}


def summarize_counts(counts: Sequence[int]) -> Dict[str, Optional[float]]:
    if not counts:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    counts_array = np.asarray(counts, dtype=np.int32)
    return {
        "min": float(counts_array.min()),
        "max": float(counts_array.max()),
        "mean": float(counts_array.mean()),
        "median": float(np.median(counts_array)),
    }


def downsample_rows(rows: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or rows.shape[0] <= max_rows:
        return rows
    indices = rng.choice(rows.shape[0], size=max_rows, replace=False)
    return rows[indices]


def style_axis(ax, bounds_min: np.ndarray, bounds_max: np.ndarray, elev: float, azim: float) -> None:
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_box_aspect((bounds_max - bounds_min).tolist())
    ax.view_init(elev=elev, azim=azim)
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        if hasattr(axis, "line"):
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))


def render_voxel_cloud(
    ax,
    voxel_coordinates: np.ndarray,
    grid: GridConfig,
    max_voxels: int,
    edge_width_scale: float,
    rng: np.random.Generator,
    cmap,
    norm: colors.Normalize,
) -> int:
    sampled = downsample_rows(voxel_coordinates, max_voxels, rng).astype(np.int32, copy=False)
    if sampled.size == 0:
        ax.text2D(
            0.5,
            0.5,
            "empty",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )
        return 0

    bounds_min = np.asarray(grid.bounds_min, dtype=np.float32)
    spatial_shape = np.asarray(grid.spatial_shape, dtype=np.int32)
    occupancy = np.zeros(tuple(spatial_shape.tolist()), dtype=bool)
    occupancy[sampled[:, 0], sampled[:, 1], sampled[:, 2]] = True

    facecolors = np.zeros(tuple(spatial_shape.tolist()) + (4,), dtype=np.float32)
    voxel_centers_z = bounds_min[2] + grid.voxel_size * (sampled[:, 2].astype(np.float32) + 0.5)
    facecolors[sampled[:, 0], sampled[:, 1], sampled[:, 2]] = cmap(norm(voxel_centers_z))

    grid_x, grid_y, grid_z = np.indices(spatial_shape + 1, dtype=np.float32)
    grid_x = bounds_min[0] + grid_x * grid.voxel_size
    grid_y = bounds_min[1] + grid_y * grid.voxel_size
    grid_z = bounds_min[2] + grid_z * grid.voxel_size
    ax.voxels(
        grid_x,
        grid_y,
        grid_z,
        occupancy,
        facecolors=facecolors,
        edgecolor=(1.0, 1.0, 1.0, 0.06),
        linewidth=max(0.02, 0.02 * edge_width_scale),
    )
    return int(sampled.shape[0])


def render_report_figure(
    output_path: Path,
    trajectory_name: str,
    timesteps: Sequence[int],
    measurement_voxels: Sequence[np.ndarray],
    prediction_voxels: Sequence[np.ndarray],
    ground_truth_voxels: Sequence[np.ndarray],
    grid: GridConfig,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    point_size: float,
    elev: float,
    azim: float,
    measurement_max_points: int,
    prediction_max_points: int,
    ground_truth_max_points: int,
    seed: int,
    title: Optional[str],
    disable_feedback: bool,
    dpi: int,
) -> Dict[str, List[int]]:
    column_count = len(timesteps)
    figure_width = max(1, column_count) * 4.0
    figure_height = 10.5
    fig, axes = plt.subplots(
        nrows=3,
        ncols=column_count,
        figsize=(figure_width, figure_height),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    if column_count == 1:
        axes = np.asarray(axes).reshape(3, 1)

    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap("turbo")
    norm = colors.Normalize(vmin=float(bounds_min[2]), vmax=float(bounds_max[2]))
    rendered_counts = {
        "measurement": [],
        "prediction": [],
        "ground_truth": [],
    }

    for column_index, timestep in enumerate(timesteps):
        panels = (
            measurement_voxels[timestep],
            prediction_voxels[timestep],
            ground_truth_voxels[timestep],
        )
        limits = (
            measurement_max_points,
            prediction_max_points,
            ground_truth_max_points,
        )
        keys = ("measurement", "prediction", "ground_truth")

        for row_index, (points, max_points, key) in enumerate(zip(panels, limits, keys)):
            ax = axes[row_index, column_index]
            style_axis(ax, bounds_min, bounds_max, elev=elev, azim=azim)
            count = render_voxel_cloud(
                ax=ax,
                voxel_coordinates=points,
                grid=grid,
                max_voxels=max_points,
                edge_width_scale=point_size,
                rng=rng,
                cmap=cmap,
                norm=norm,
            )
            rendered_counts[key].append(count)

            if column_index == 0:
                ax.text2D(
                    -0.12,
                    0.5,
                    ROW_LABELS[row_index],
                    rotation=90,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="semibold",
                )

            if row_index == 0:
                ax.set_title(f"Step {timestep}", fontsize=12, pad=12.0)

    mode_label = "current measurement only" if disable_feedback else "autoregressive rollout"
    figure_title = title or f"{trajectory_name} | {mode_label} | occupied voxels"
    fig.suptitle(figure_title, fontsize=15, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return rendered_counts


def write_metadata(
    figure_path: Path,
    *,
    checkpoint_path: Path,
    trajectory_path: Path,
    timesteps: Sequence[int],
    rendered_counts: Dict[str, List[int]],
    raw_counts: Dict[str, List[int]],
    raw_count_summary: Dict[str, Dict[str, Optional[float]]],
    voxel_counts: Dict[str, List[int]],
    voxel_count_summary: Dict[str, Dict[str, Optional[float]]],
    voxel_metrics: Dict[str, Dict[str, List[float]]],
    voxel_metric_summary: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    voxel_size_m: float,
    disable_feedback: bool,
    checkpoint_metadata: Dict[str, object],
) -> Path:
    metadata_path = figure_path.with_suffix(".json")
    payload = {
        "figure": str(figure_path),
        "checkpoint": str(checkpoint_path),
        "trajectory": str(trajectory_path),
        "timesteps": list(timesteps),
        "mode": "current_measurement_only" if disable_feedback else "autoregressive_rollout",
        "rendered_counts": rendered_counts,
        "raw_counts": raw_counts,
        "raw_count_summary": raw_count_summary,
        "voxel_size_m": float(voxel_size_m),
        "voxel_counts": voxel_counts,
        "voxel_count_summary": voxel_count_summary,
        "voxel_metrics": voxel_metrics,
        "voxel_metric_summary": voxel_metric_summary,
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_global_step": checkpoint_metadata.get("global_step"),
        "train_metrics": checkpoint_metadata.get("train_metrics"),
        "validation_metrics": checkpoint_metadata.get("validation_metrics"),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    save_rollout_path = (
        args.save_rollout.expanduser().resolve()
        if args.save_rollout is not None
        else None
    )

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")
    if not trajectory_path.exists():
        raise SystemExit(f"Trajectory does not exist: {trajectory_path}")

    grid = GridConfig()
    bounds_min = np.asarray(args.bounds_min, dtype=np.float32)
    bounds_max = np.asarray(args.bounds_max, dtype=np.float32)
    device = select_device(args.device)

    trajectory = load_trajectory(trajectory_path)
    poses = trajectory["poses"]  # type: ignore[assignment]
    measurements = trajectory["measurements"]  # type: ignore[assignment]
    ground_truth = trajectory["ground_truth"]  # type: ignore[assignment]
    timesteps = parse_timestep_values(args.timesteps, len(measurements), args.columns)

    model, checkpoint_metadata = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        pruning_threshold=args.pruning_threshold,
    )
    predictions = rollout_predictions(
        model=model,
        poses=poses,
        measurements=measurements,
        grid=grid,
        device=device,
        disable_feedback=args.disable_feedback,
        pruning_threshold=args.pruning_threshold,
    )
    measurement_counts = sequence_point_counts(measurements)
    prediction_counts = sequence_point_counts(predictions)
    ground_truth_counts = sequence_point_counts(ground_truth)
    raw_counts = {
        "measurement": measurement_counts,
        "prediction": prediction_counts,
        "ground_truth": ground_truth_counts,
    }
    raw_count_summary = {
        "measurement": summarize_counts(measurement_counts),
        "prediction": summarize_counts(prediction_counts),
        "ground_truth": summarize_counts(ground_truth_counts),
    }
    measurement_voxels = voxelize_sequence(measurements, grid)
    prediction_voxels = voxelize_sequence(predictions, grid)
    ground_truth_voxels = voxelize_sequence(ground_truth, grid)
    voxel_counts = {
        "measurement": sequence_voxel_counts(measurement_voxels),
        "prediction": sequence_voxel_counts(prediction_voxels),
        "ground_truth": sequence_voxel_counts(ground_truth_voxels),
    }
    voxel_count_summary = {
        "measurement": summarize_counts(voxel_counts["measurement"]),
        "prediction": summarize_counts(voxel_counts["prediction"]),
        "ground_truth": summarize_counts(voxel_counts["ground_truth"]),
    }
    voxel_metrics = {
        "measurement_vs_ground_truth": sequence_occupancy_metrics(measurement_voxels, ground_truth_voxels),
        "prediction_vs_ground_truth": sequence_occupancy_metrics(prediction_voxels, ground_truth_voxels),
    }
    voxel_metric_summary = {
        key: summarize_metric_dict(value)
        for key, value in voxel_metrics.items()
    }
    rollout_path = maybe_save_rollout(save_rollout_path, poses, predictions)
    rendered_counts = render_report_figure(
        output_path=output_path,
        trajectory_name=trajectory_path.name,
        timesteps=timesteps,
        measurement_voxels=measurement_voxels,
        prediction_voxels=prediction_voxels,
        ground_truth_voxels=ground_truth_voxels,
        grid=grid,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        measurement_max_points=args.measurement_max_points,
        prediction_max_points=args.prediction_max_points,
        ground_truth_max_points=args.ground_truth_max_points,
        seed=args.seed,
        title=args.title,
        disable_feedback=args.disable_feedback,
        dpi=args.dpi,
    )
    metadata_path = write_metadata(
        figure_path=output_path,
        checkpoint_path=checkpoint_path,
        trajectory_path=trajectory_path,
        timesteps=timesteps,
        rendered_counts=rendered_counts,
        raw_counts=raw_counts,
        raw_count_summary=raw_count_summary,
        voxel_counts=voxel_counts,
        voxel_count_summary=voxel_count_summary,
        voxel_metrics=voxel_metrics,
        voxel_metric_summary=voxel_metric_summary,
        voxel_size_m=grid.voxel_size,
        disable_feedback=args.disable_feedback,
        checkpoint_metadata=checkpoint_metadata,
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"trajectory: {trajectory_path}")
    print(f"device: {device}")
    print(f"timesteps shown: {timesteps}")
    print(f"prediction point-count summary: {raw_count_summary['prediction']}")
    print(f"prediction voxel-count summary: {voxel_count_summary['prediction']}")
    print(
        "prediction voxel metric summary vs gt: "
        f"{voxel_metric_summary['prediction_vs_ground_truth']}"
    )
    print(
        "selected prediction counts: {}".format(
            {int(step): prediction_counts[step] for step in timesteps}
        )
    )
    print(
        "selected prediction voxel recall: {}".format(
            {
                int(step): voxel_metrics["prediction_vs_ground_truth"]["recall"][step]
                for step in timesteps
            }
        )
    )
    print(f"saved figure: {output_path}")
    print(f"saved metadata: {metadata_path}")
    if rollout_path is not None:
        print(f"saved rollout: {rollout_path}")


if __name__ == "__main__":
    main()
