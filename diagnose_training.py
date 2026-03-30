from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.augmentations import (
    apply_measurement_augmentations,
    sample_pose_translation_noise,
    sample_sequence_mirror,
)
from src.config import AugmentationConfig, GridConfig, ModelConfig
from src.data import TrajectoryDataset, collate_trajectory_batch
from src.geometry import apply_local_mirror, mirror_relative_transform, relative_transform, transform_points
from src.losses import reconstruction_loss
from src.model import TerrainReconstructionModel
from src.preprocess import (
    build_network_input,
    decode_predictions_to_point_clouds,
    downsample_batched_coordinates,
    make_batched_coordinate_tensor,
    make_batched_feature_tensor,
    voxelize_points,
)

try:
    import MinkowskiEngine as ME
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError(
        "diagnose_training.py requires MinkowskiEngine. Install it before running diagnostics."
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose terrain reconstruction training on tiny subsets.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of packed .npz trajectories.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=2, help="Number of dataset windows to use.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("batch", "overfit", "all"), default="all")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional model checkpoint to inspect.")
    parser.add_argument("--overfit-epochs", type=int, default=100)
    parser.add_argument("--overfit-learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--occupancy-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sparse_tensor(
    coordinate_batches: List[np.ndarray],
    feature_batches: List[np.ndarray],
    device: torch.device,
) -> ME.SparseTensor:
    coordinates = make_batched_coordinate_tensor(coordinate_batches, device)
    features = make_batched_feature_tensor(feature_batches, device)
    return ME.SparseTensor(coordinates=coordinates, features=features)


class LayerRecorder:
    def __init__(self, model: TerrainReconstructionModel) -> None:
        self.latest: Dict[str, int] = {}
        self._handles: List[Any] = []
        layer_names = (
            "stem",
            "enc1",
            "enc2",
            "enc3",
            "enc4",
            "dec3",
            "dec2",
            "dec1",
            "dec0",
            "occ3",
            "occ2",
            "occ1",
            "occ0",
            "offset_head",
        )
        for name in layer_names:
            module = getattr(model, name, None)
            if module is None:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            coordinates = getattr(output, "C", None)
            if coordinates is None:
                return
            self.latest[name] = int(coordinates.shape[0])

        return hook

    def clear(self) -> None:
        self.latest = {}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _coordinate_set(coordinates: torch.Tensor) -> Set[Tuple[int, ...]]:
    if coordinates.numel() == 0:
        return set()
    coordinates_cpu = coordinates.detach().cpu()
    return {tuple(int(value) for value in row.tolist()) for row in coordinates_cpu}


def _feature_stats(features: torch.Tensor) -> Dict[str, Any]:
    if features.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    features_cpu = features.detach().float().cpu()
    return {
        "count": int(features_cpu.shape[0]),
        "min": [float(value) for value in features_cpu.min(dim=0).values.tolist()],
        "max": [float(value) for value in features_cpu.max(dim=0).values.tolist()],
        "mean": [float(value) for value in features_cpu.mean(dim=0).tolist()],
    }


def _spatial_range(coordinates: torch.Tensor) -> Dict[str, Optional[List[int]]]:
    if coordinates.numel() == 0:
        return {
            "spatial_min": None,
            "spatial_max": None,
            "time_min": None,
            "time_max": None,
        }
    coordinates_cpu = coordinates.detach().cpu().to(dtype=torch.int32)
    spatial = coordinates_cpu[:, 1:4]
    time_column = coordinates_cpu[:, -1]
    return {
        "spatial_min": [int(value) for value in spatial.min(dim=0).values.tolist()],
        "spatial_max": [int(value) for value in spatial.max(dim=0).values.tolist()],
        "time_min": int(time_column.min().item()),
        "time_max": int(time_column.max().item()),
    }


def _occupancy_head_stats(model_output: Any, target_coordinates: torch.Tensor) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for occupancy_prediction in model_output.occupancy_predictions:
        query_coordinates = occupancy_prediction.logits.C.to(dtype=torch.int32)
        target_at_stride = downsample_batched_coordinates(target_coordinates, occupancy_prediction.stride)
        target_lookup = _coordinate_set(target_at_stride)

        positives = 0
        matched_target_coordinates: Set[Tuple[int, ...]] = set()
        for row in query_coordinates.detach().cpu():
            key = tuple(int(value) for value in row.tolist())
            if key in target_lookup:
                positives += 1
                matched_target_coordinates.add(key)

        query_count = int(query_coordinates.shape[0])
        target_count = int(target_at_stride.shape[0])
        query_range = _spatial_range(query_coordinates)
        target_range = _spatial_range(target_at_stride)
        stats.append(
            {
                "stride": int(occupancy_prediction.stride),
                "query_count": query_count,
                "target_count": target_count,
                "positive_queries": positives,
                "positive_ratio": float(positives / query_count) if query_count > 0 else None,
                "target_coverage": float(len(matched_target_coordinates) / target_count) if target_count > 0 else None,
                "query_range": query_range,
                "target_range": target_range,
            }
        )
    return stats


def _regression_stats(
    prediction_coordinates: torch.Tensor,
    prediction_features: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_features: torch.Tensor,
) -> Dict[str, Any]:
    target_lookup = {
        tuple(int(value) for value in coordinate.tolist()): index
        for index, coordinate in enumerate(target_coordinates.detach().cpu())
    }

    matched_prediction_indices: List[int] = []
    matched_target_indices: List[int] = []
    for prediction_index, coordinate in enumerate(prediction_coordinates.detach().cpu()):
        key = tuple(int(value) for value in coordinate.tolist())
        target_index = target_lookup.get(key)
        if target_index is None:
            continue
        matched_prediction_indices.append(prediction_index)
        matched_target_indices.append(target_index)

    matched_prediction = prediction_features[matched_prediction_indices] if matched_prediction_indices else prediction_features[:0]
    matched_target = target_features[matched_target_indices] if matched_target_indices else target_features[:0]

    if matched_prediction_indices:
        matched_target = matched_target.to(
            device=matched_prediction.device,
            dtype=matched_prediction.dtype,
        )
        matched_error = torch.linalg.norm(matched_prediction - matched_target, dim=1)
        matched_error_mean = float(matched_error.mean().item())
        matched_error_max = float(matched_error.max().item())
    else:
        matched_error_mean = None
        matched_error_max = None

    prediction_count = int(prediction_coordinates.shape[0])
    target_count = int(target_coordinates.shape[0])
    matched_count = len(matched_prediction_indices)
    return {
        "prediction_count": prediction_count,
        "target_count": target_count,
        "matched_count": matched_count,
        "prediction_match_ratio": float(matched_count / prediction_count) if prediction_count > 0 else None,
        "target_match_ratio": float(matched_count / target_count) if target_count > 0 else None,
        "prediction_feature_stats": _feature_stats(prediction_features),
        "target_feature_stats": _feature_stats(target_features),
        "matched_prediction_feature_stats": _feature_stats(matched_prediction),
        "matched_target_feature_stats": _feature_stats(matched_target),
        "matched_l2_mean": matched_error_mean,
        "matched_l2_max": matched_error_max,
    }


def rollout_batch(
    model: TerrainReconstructionModel,
    batch: Dict[str, object],
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    device: torch.device,
    rng: np.random.Generator,
    use_augmentation: bool,
    occupancy_weight: float,
    regression_weight: float,
    collect_diagnostics: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    poses_batch: List[np.ndarray] = batch["poses"]  # type: ignore[assignment]
    measurements_batch: List[List[np.ndarray]] = batch["measurements"]  # type: ignore[assignment]
    ground_truth_batch: List[List[np.ndarray]] = batch["ground_truth"]  # type: ignore[assignment]

    batch_size = len(poses_batch)
    sequence_length = len(measurements_batch[0])
    mirror_matrices = [
        sample_sequence_mirror(rng, augmentation_config) if use_augmentation else np.eye(4, dtype=np.float32)
        for _ in range(batch_size)
    ]
    previous_predictions = [np.empty((0, 3), dtype=np.float32) for _ in range(batch_size)]

    diagnostics: Optional[Dict[str, Any]] = None
    layer_recorder: Optional[LayerRecorder] = None
    if collect_diagnostics:
        diagnostics = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "steps": [],
        }
        layer_recorder = LayerRecorder(model)

    accumulated_total = None
    accumulated_occupancy = None
    accumulated_regression = None
    valid_steps = 0

    for step in range(sequence_length):
        input_coordinate_batches: List[np.ndarray] = []
        input_feature_batches: List[np.ndarray] = []
        target_coordinate_batches: List[np.ndarray] = []
        target_feature_batches: List[np.ndarray] = []
        step_sample_stats: List[Dict[str, int]] = []

        for batch_index in range(batch_size):
            current_pose = poses_batch[batch_index][step]
            measurement_local = measurements_batch[batch_index][step].astype(np.float32, copy=False)
            target_local = ground_truth_batch[batch_index][step].astype(np.float32, copy=False)

            mirror_matrix = mirror_matrices[batch_index]
            measurement_local = apply_local_mirror(measurement_local, mirror_matrix)
            target_local = apply_local_mirror(target_local, mirror_matrix)

            if use_augmentation:
                measurement_local = apply_measurement_augmentations(
                    measurement_local,
                    rng,
                    augmentation_config,
                )

            previous_prediction_local = previous_predictions[batch_index]
            previous_prediction_count = int(previous_prediction_local.shape[0])
            if step > 0 and previous_prediction_local.size > 0:
                prev_pose = poses_batch[batch_index][step - 1]
                transform_prev_to_current = relative_transform(prev_pose, current_pose)
                transform_prev_to_current = mirror_relative_transform(
                    transform_prev_to_current,
                    mirror_matrix,
                )
                if use_augmentation:
                    transform_prev_to_current = (
                        sample_pose_translation_noise(
                            rng,
                            augmentation_config.pose_translation_noise,
                        )
                        @ transform_prev_to_current
                    )
                previous_prediction_local = transform_points(
                    previous_prediction_local,
                    transform_prev_to_current,
                )

            model_input = build_network_input(
                current_measurement=measurement_local,
                previous_prediction=previous_prediction_local,
                grid=grid_config,
            )
            target = voxelize_points(
                target_local,
                grid_config,
                time_index=grid_config.current_time_index,
            )

            input_coordinate_batches.append(model_input.coordinates)
            input_feature_batches.append(model_input.features)
            target_coordinate_batches.append(target.coordinates)
            target_feature_batches.append(target.features)
            step_sample_stats.append(
                {
                    "measurement_points": int(measurement_local.shape[0]),
                    "previous_prediction_points": previous_prediction_count,
                    "input_voxels": int(model_input.coordinates.shape[0]),
                    "target_points": int(target_local.shape[0]),
                    "target_voxels": int(target.coordinates.shape[0]),
                }
            )

        if not any(coordinates.size > 0 for coordinates in input_coordinate_batches):
            if diagnostics is not None:
                diagnostics["steps"].append(
                    {
                        "step": step,
                        "skipped": True,
                        "sample_stats": step_sample_stats,
                    }
                )
            continue

        sparse_input = make_sparse_tensor(
            input_coordinate_batches,
            input_feature_batches,
            device,
        )
        if layer_recorder is not None:
            layer_recorder.clear()
        output = model(sparse_input)

        target_coordinates = make_batched_coordinate_tensor(target_coordinate_batches, device)
        target_features = make_batched_feature_tensor(target_feature_batches, device)
        losses = reconstruction_loss(
            output,
            target_coordinates,
            target_features,
            occupancy_weight=occupancy_weight,
            regression_weight=regression_weight,
        )

        accumulated_total = losses["total"] if accumulated_total is None else accumulated_total + losses["total"]
        accumulated_occupancy = (
            losses["occupancy"]
            if accumulated_occupancy is None
            else accumulated_occupancy + losses["occupancy"]
        )
        accumulated_regression = (
            losses["regression"]
            if accumulated_regression is None
            else accumulated_regression + losses["regression"]
        )
        valid_steps += 1

        if diagnostics is not None:
            diagnostics["steps"].append(
                {
                    "step": step,
                    "skipped": False,
                    "sample_stats": step_sample_stats,
                    "input_voxels_total": int(sum(coords.shape[0] for coords in input_coordinate_batches)),
                    "target_voxels_total": int(sum(coords.shape[0] for coords in target_coordinate_batches)),
                    "loss": float(losses["total"].item()),
                    "occupancy_loss": float(losses["occupancy"].item()),
                    "regression_loss": float(losses["regression"].item()),
                    "layer_points": dict(layer_recorder.latest) if layer_recorder is not None else {},
                    "occupancy_heads": _occupancy_head_stats(output, target_coordinates),
                    "regression": _regression_stats(
                        output.offsets.C.to(dtype=torch.int32),
                        output.offsets.F,
                        target_coordinates,
                        target_features,
                    ),
                }
            )

        previous_predictions = decode_predictions_to_point_clouds(
            output.offsets.C,
            output.offsets.F,
            grid=grid_config,
            batch_size=batch_size,
            time_index=grid_config.current_time_index,
        )

    if layer_recorder is not None:
        layer_recorder.close()

    if valid_steps == 0:
        zero = torch.tensor(0.0, device=device)
        losses = {
            "total": zero,
            "occupancy": zero,
            "regression": zero,
        }
        if diagnostics is not None:
            diagnostics["valid_steps"] = 0
        return losses, diagnostics

    assert accumulated_total is not None
    assert accumulated_occupancy is not None
    assert accumulated_regression is not None

    divisor = float(valid_steps)
    losses = {
        "total": accumulated_total / divisor,
        "occupancy": accumulated_occupancy / divisor,
        "regression": accumulated_regression / divisor,
    }
    if diagnostics is not None:
        diagnostics["valid_steps"] = valid_steps
        diagnostics["mean_loss"] = float(losses["total"].item())
        diagnostics["mean_occupancy_loss"] = float(losses["occupancy"].item())
        diagnostics["mean_regression_loss"] = float(losses["regression"].item())
    return losses, diagnostics


def load_checkpoint_if_available(model: TerrainReconstructionModel, checkpoint_path: Optional[Path], device: torch.device) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def build_subset_loader(
    data_root: Path,
    sequence_length: int,
    sequence_stride: int,
    batch_size: int,
    num_workers: int,
    sample_limit: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TrajectoryDataset(
        root=data_root,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )
    sample_count = min(sample_limit, len(dataset))
    if sample_count < 1:
        raise ValueError("No samples available for diagnostics. Increase --sample-limit or check the dataset.")
    subset = Subset(dataset, list(range(sample_count)))
    effective_batch_size = min(batch_size, sample_count)
    return DataLoader(
        subset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
    )


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return "{:.6f}".format(value)


def print_batch_diagnostics(diagnostics: Dict[str, Any]) -> None:
    print(
        "batch_summary "
        "batch_size={} sequence_length={} valid_steps={} mean_loss={} mean_occ={} mean_reg={}".format(
            diagnostics["batch_size"],
            diagnostics["sequence_length"],
            diagnostics.get("valid_steps", 0),
            format_float(diagnostics.get("mean_loss")),
            format_float(diagnostics.get("mean_occupancy_loss")),
            format_float(diagnostics.get("mean_regression_loss")),
        )
    )
    for step_info in diagnostics["steps"]:
        if step_info["skipped"]:
            print("step={} skipped input_voxels=0".format(step_info["step"]))
            continue

        sample_stats = step_info["sample_stats"]
        measurement_counts = [stats["measurement_points"] for stats in sample_stats]
        previous_prediction_counts = [stats["previous_prediction_points"] for stats in sample_stats]
        input_voxel_counts = [stats["input_voxels"] for stats in sample_stats]
        target_counts = [stats["target_points"] for stats in sample_stats]
        target_voxel_counts = [stats["target_voxels"] for stats in sample_stats]
        print(
            "step={} loss={} occ={} reg={} input_voxels_total={} target_voxels_total={}".format(
                step_info["step"],
                format_float(step_info["loss"]),
                format_float(step_info["occupancy_loss"]),
                format_float(step_info["regression_loss"]),
                step_info["input_voxels_total"],
                step_info["target_voxels_total"],
            )
        )
        print(
            "  sample_counts measurements={} prev_predictions={} input_voxels={} target_points={} target_voxels={}".format(
                measurement_counts,
                previous_prediction_counts,
                input_voxel_counts,
                target_counts,
                target_voxel_counts,
            )
        )
        print("  layer_points {}".format(step_info["layer_points"]))
        for head in step_info["occupancy_heads"]:
            print(
                "  occ_head stride={} query_count={} target_count={} positive_queries={} positive_ratio={} target_coverage={} query_range={} target_range={}".format(
                    head["stride"],
                    head["query_count"],
                    head["target_count"],
                    head["positive_queries"],
                    format_float(head["positive_ratio"]),
                    format_float(head["target_coverage"]),
                    head["query_range"],
                    head["target_range"],
                )
            )
        regression = step_info["regression"]
        print(
            "  reg_match prediction_count={} target_count={} matched_count={} prediction_match_ratio={} target_match_ratio={} matched_l2_mean={} matched_l2_max={}".format(
                regression["prediction_count"],
                regression["target_count"],
                regression["matched_count"],
                format_float(regression["prediction_match_ratio"]),
                format_float(regression["target_match_ratio"]),
                format_float(regression["matched_l2_mean"]),
                format_float(regression["matched_l2_max"]),
            )
        )
        print("  reg_target_stats {}".format(regression["target_feature_stats"]))
        print("  reg_prediction_stats {}".format(regression["prediction_feature_stats"]))
        print("  reg_matched_target_stats {}".format(regression["matched_target_feature_stats"]))
        print("  reg_matched_prediction_stats {}".format(regression["matched_prediction_feature_stats"]))


def run_batch_mode(
    args: argparse.Namespace,
    device: torch.device,
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    model_config: ModelConfig,
) -> None:
    print("running batch diagnostics")
    loader = build_subset_loader(
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_limit=args.sample_limit,
        shuffle=False,
    )
    batch = next(iter(loader))
    model = TerrainReconstructionModel(model_config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        _losses, diagnostics = rollout_batch(
            model=model,
            batch=batch,
            grid_config=grid_config,
            augmentation_config=augmentation_config,
            device=device,
            rng=np.random.default_rng(args.seed),
            use_augmentation=args.use_augmentation,
            occupancy_weight=args.occupancy_weight,
            regression_weight=args.regression_weight,
            collect_diagnostics=True,
        )
    assert diagnostics is not None
    print_batch_diagnostics(diagnostics)


def run_overfit_mode(
    args: argparse.Namespace,
    device: torch.device,
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    model_config: ModelConfig,
) -> None:
    print("running overfit experiment")
    train_loader = build_subset_loader(
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_limit=args.sample_limit,
        shuffle=True,
    )
    eval_loader = build_subset_loader(
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_limit=args.sample_limit,
        shuffle=False,
    )

    model = TerrainReconstructionModel(model_config).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.overfit_learning_rate,
        weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed)
    global_step = 0
    for epoch in range(1, args.overfit_epochs + 1):
        epoch_totals: List[float] = []
        epoch_occ: List[float] = []
        epoch_reg: List[float] = []
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses, _diagnostics = rollout_batch(
                model=model,
                batch=batch,
                grid_config=grid_config,
                augmentation_config=augmentation_config,
                device=device,
                rng=rng,
                use_augmentation=args.use_augmentation,
                occupancy_weight=args.occupancy_weight,
                regression_weight=args.regression_weight,
                collect_diagnostics=False,
            )
            losses["total"].backward()
            optimizer.step()

            epoch_totals.append(float(losses["total"].item()))
            epoch_occ.append(float(losses["occupancy"].item()))
            epoch_reg.append(float(losses["regression"].item()))
            global_step += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.overfit_epochs:
            print(
                "overfit_epoch={} steps={} loss={} occ={} reg={}".format(
                    epoch,
                    global_step,
                    format_float(float(np.mean(epoch_totals))),
                    format_float(float(np.mean(epoch_occ))),
                    format_float(float(np.mean(epoch_reg))),
                )
            )

    model.eval()
    eval_batch = next(iter(eval_loader))
    with torch.no_grad():
        losses, diagnostics = rollout_batch(
            model=model,
            batch=eval_batch,
            grid_config=grid_config,
            augmentation_config=augmentation_config,
            device=device,
            rng=np.random.default_rng(args.seed),
            use_augmentation=False,
            occupancy_weight=args.occupancy_weight,
            regression_weight=args.regression_weight,
            collect_diagnostics=True,
        )
    print(
        "overfit_final loss={} occ={} reg={}".format(
            format_float(float(losses["total"].item())),
            format_float(float(losses["occupancy"].item())),
            format_float(float(losses["regression"].item())),
        )
    )
    assert diagnostics is not None
    print_batch_diagnostics(diagnostics)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    grid_config = GridConfig()
    augmentation_config = AugmentationConfig()
    model_config = ModelConfig()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode in ("batch", "all"):
        run_batch_mode(args, device, grid_config, augmentation_config, model_config)
    if args.mode in ("overfit", "all"):
        run_overfit_mode(args, device, grid_config, augmentation_config, model_config)


if __name__ == "__main__":
    main()
