from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.augmentations import (
    apply_measurement_augmentations,
    sample_pose_translation_noise,
    sample_sequence_mirror,
)
from src.config import AugmentationConfig, GridConfig, ModelConfig, TrainConfig
from src.data import TrajectoryDataset, collate_trajectory_batch
from src.geometry import apply_local_mirror, mirror_relative_transform, relative_transform, transform_points
from src.losses import reconstruction_loss
from src.model import TerrainReconstructionModel
from src.preprocess import (
    build_network_input,
    decode_predictions_to_point_clouds,
    make_batched_coordinate_tensor,
    make_batched_feature_tensor,
    voxelize_points,
)

try:
    import MinkowskiEngine as ME
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError(
        "train.py requires MinkowskiEngine. Install it before running training."
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the terrain reconstruction model from Hoeller et al. (2022)."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of packed .npz trajectories.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/nsr_terrain"))
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--val-root", type=Path, default=None, help="Optional separate directory for validation .npz trajectories.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction when splitting --data-root locally. Set to 0 to disable.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm after backward. Set <= 0 to disable.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a checkpoint file.")
    parser.add_argument("--cache-dataset", action="store_true", help="Keep decoded .npz trajectories in memory.")
    parser.add_argument("--disable-pin-memory", action="store_true", help="Disable DataLoader pin_memory on CUDA.")
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers when num_workers > 0.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched by each worker.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--print-timing", action="store_true", help="Log data wait time and step time.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(device: torch.device, enabled: bool) -> object:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def mean_loss_dict(loss_sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {key: 0.0 for key in loss_sums}
    divisor = float(count)
    return {key: value / divisor for key, value in loss_sums.items()}


def make_data_loader(
    dataset: object,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_cuda: bool,
    disable_pin_memory: bool,
    disable_persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    pin_memory = use_cuda and not disable_pin_memory
    persistent_workers = num_workers > 0 and not disable_persistent_workers
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_trajectory_batch,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def split_train_validation_dataset(
    dataset: TrajectoryDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[object, Optional[object]]:
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, None

    val_size = int(round(len(dataset) * val_fraction))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    permutation = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@torch.no_grad()
def evaluate_model(
    model: TerrainReconstructionModel,
    loader: Optional[DataLoader],
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    device: torch.device,
    use_amp: bool,
    autocast_dtype: torch.dtype,
    occupancy_weight: float,
    regression_weight: float,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None

    model.eval()
    eval_rng = np.random.default_rng(0)
    loss_sums = {"total": 0.0, "occupancy": 0.0, "regression": 0.0}
    batch_count = 0

    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            losses = rollout_batch(
                model=model,
                batch=batch,
                grid_config=grid_config,
                augmentation_config=augmentation_config,
                device=device,
                rng=eval_rng,
                apply_augmentation=False,
                occupancy_weight=occupancy_weight,
                regression_weight=regression_weight,
            )
        for key in loss_sums:
            loss_sums[key] += float(losses[key].item())
        batch_count += 1

    return mean_loss_dict(loss_sums, batch_count)


def checkpoint_payload(
    *,
    model: TerrainReconstructionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR,
    scaler: object,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    args: argparse.Namespace,
    augmentation_rng: np.random.Generator,
    train_metrics: Dict[str, float],
    validation_metrics: Optional[Dict[str, float]],
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "augmentation_rng_state": augmentation_rng.bit_generator.state,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }


def save_checkpoint(checkpoint: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_checkpoint_if_available(
    resume_path: Optional[Path],
    *,
    model: TerrainReconstructionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR,
    scaler: object,
    device: torch.device,
    augmentation_rng: np.random.Generator,
) -> Tuple[int, int, float]:
    if resume_path is None:
        return 1, 0, float("inf")

    try:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resume_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {resume_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("torch_cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state_all"])
    if "augmentation_rng_state" in checkpoint:
        augmentation_rng.bit_generator.state = checkpoint["augmentation_rng_state"]

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
    return start_epoch, global_step, best_validation_loss


def make_sparse_tensor(
    coordinate_batches: List[np.ndarray],
    feature_batches: List[np.ndarray],
    device: torch.device,
) -> ME.SparseTensor:
    coordinates = make_batched_coordinate_tensor(coordinate_batches, device)
    features = make_batched_feature_tensor(feature_batches, device)
    return ME.SparseTensor(coordinates=coordinates, features=features)


def prepare_rollout_step(
    *,
    poses_batch: List[np.ndarray],
    measurements_batch: List[List[np.ndarray]],
    ground_truth_batch: List[List[np.ndarray]],
    step: int,
    previous_predictions: List[np.ndarray],
    mirror_matrices: List[np.ndarray],
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    device: torch.device,
    rng: np.random.Generator,
    apply_augmentation: bool,
) -> Optional[Tuple[ME.SparseTensor, torch.Tensor, torch.Tensor]]:
    input_coordinate_batches: List[np.ndarray] = []
    input_feature_batches: List[np.ndarray] = []
    target_coordinate_batches: List[np.ndarray] = []
    target_feature_batches: List[np.ndarray] = []

    for batch_index in range(len(poses_batch)):
        current_pose = poses_batch[batch_index][step]
        measurement_local = measurements_batch[batch_index][step].astype(np.float32, copy=False)
        target_local = ground_truth_batch[batch_index][step].astype(np.float32, copy=False)

        mirror_matrix = mirror_matrices[batch_index]
        measurement_local = apply_local_mirror(measurement_local, mirror_matrix)
        target_local = apply_local_mirror(target_local, mirror_matrix)

        if apply_augmentation:
            measurement_local = apply_measurement_augmentations(
                measurement_local,
                rng,
                augmentation_config,
            )

        previous_prediction_local = previous_predictions[batch_index]
        if step > 0 and previous_prediction_local.size > 0:
            prev_pose = poses_batch[batch_index][step - 1]
            transform_prev_to_current = relative_transform(prev_pose, current_pose)
            transform_prev_to_current = mirror_relative_transform(
                transform_prev_to_current,
                mirror_matrix,
            )
            if apply_augmentation:
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

    if not any(coordinates.size > 0 for coordinates in input_coordinate_batches):
        return None

    sparse_input = make_sparse_tensor(
        input_coordinate_batches,
        input_feature_batches,
        device,
    )
    target_coordinates = make_batched_coordinate_tensor(target_coordinate_batches, device)
    target_features = make_batched_feature_tensor(target_feature_batches, device)
    return sparse_input, target_coordinates, target_features


def forward_rollout_step(
    *,
    model: TerrainReconstructionModel,
    sparse_input: ME.SparseTensor,
    target_coordinates: torch.Tensor,
    target_features: torch.Tensor,
    grid_config: GridConfig,
    batch_size: int,
    occupancy_weight: float,
    regression_weight: float,
) -> Tuple[Dict[str, torch.Tensor], List[np.ndarray]]:
    output = model(sparse_input)
    losses = reconstruction_loss(
        output,
        target_coordinates,
        target_features,
        occupancy_weight=occupancy_weight,
        regression_weight=regression_weight,
    )
    predictions = decode_predictions_to_point_clouds(
        output.offsets.C,
        output.offsets.F,
        grid=grid_config,
        batch_size=batch_size,
        time_index=grid_config.current_time_index,
    )
    return losses, predictions


def rollout_batch(
    model: TerrainReconstructionModel,
    batch: Dict[str, object],
    grid_config: GridConfig,
    augmentation_config: AugmentationConfig,
    device: torch.device,
    rng: np.random.Generator,
    apply_augmentation: bool,
    occupancy_weight: float,
    regression_weight: float,
) -> Dict[str, torch.Tensor]:
    poses_batch: List[np.ndarray] = batch["poses"]  # type: ignore[assignment]
    measurements_batch: List[List[np.ndarray]] = batch["measurements"]  # type: ignore[assignment]
    ground_truth_batch: List[List[np.ndarray]] = batch["ground_truth"]  # type: ignore[assignment]

    batch_size = len(poses_batch)
    sequence_length = len(measurements_batch[0])
    mirror_matrices = [
        sample_sequence_mirror(rng, augmentation_config) if apply_augmentation else np.eye(4, dtype=np.float32)
        for _ in range(batch_size)
    ]
    previous_predictions = [np.empty((0, 3), dtype=np.float32) for _ in range(batch_size)]

    accumulated_total = None
    accumulated_occupancy = None
    accumulated_regression = None
    valid_steps = 0

    for step in range(sequence_length):
        step_tensors = prepare_rollout_step(
            poses_batch=poses_batch,
            measurements_batch=measurements_batch,
            ground_truth_batch=ground_truth_batch,
            step=step,
            previous_predictions=previous_predictions,
            mirror_matrices=mirror_matrices,
            grid_config=grid_config,
            augmentation_config=augmentation_config,
            device=device,
            rng=rng,
            apply_augmentation=apply_augmentation,
        )
        if step_tensors is None:
            continue
        sparse_input, target_coordinates, target_features = step_tensors
        losses, previous_predictions = forward_rollout_step(
            model=model,
            sparse_input=sparse_input,
            target_coordinates=target_coordinates,
            target_features=target_features,
            grid_config=grid_config,
            batch_size=batch_size,
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

    if valid_steps == 0:
        zero = torch.tensor(0.0, device=device)
        return {
            "total": zero,
            "occupancy": zero,
            "regression": zero,
        }

    assert accumulated_total is not None
    assert accumulated_occupancy is not None
    assert accumulated_regression is not None

    divisor = float(valid_steps)
    return {
        "total": accumulated_total / divisor,
        "occupancy": accumulated_occupancy / divisor,
        "regression": accumulated_regression / divisor,
    }

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    grid_config = GridConfig()
    model_config = ModelConfig()
    augmentation_config = AugmentationConfig()
    train_config = TrainConfig(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
        output_dir=args.output_dir,
    )

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = bool(args.amp and use_cuda)
    autocast_dtype = torch.float16 if use_cuda else torch.bfloat16
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_dataset = TrajectoryDataset(
        root=args.data_root,
        sequence_length=train_config.sequence_length,
        sequence_stride=args.sequence_stride,
        cache_data=args.cache_dataset,
    )
    if args.val_root is not None:
        val_dataset: Optional[object] = TrajectoryDataset(
            root=args.val_root,
            sequence_length=train_config.sequence_length,
            sequence_stride=args.sequence_stride,
            cache_data=args.cache_dataset,
        )
    else:
        train_dataset, val_dataset = split_train_validation_dataset(
            train_dataset,
            args.val_fraction,
            args.seed,
        )

    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        use_cuda=use_cuda,
        disable_pin_memory=args.disable_pin_memory,
        disable_persistent_workers=args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = (
        make_data_loader(
            dataset=val_dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            shuffle=False,
            use_cuda=use_cuda,
            disable_pin_memory=args.disable_pin_memory,
            disable_persistent_workers=args.disable_persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if val_dataset is not None
        else None
    )
    print(
        f"train_windows={len(train_dataset)} val_windows={len(val_dataset) if val_dataset is not None else 0}",
    )

    model = TerrainReconstructionModel(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    total_steps = max(len(train_loader) * train_config.sequence_length * train_config.epochs, 1)
    gamma = math.exp(
        math.log(train_config.min_learning_rate / train_config.learning_rate) / total_steps
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scaler = make_grad_scaler(device, use_amp)

    rng = np.random.default_rng(args.seed)
    start_epoch, global_step, best_validation_loss = load_checkpoint_if_available(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        augmentation_rng=rng,
    )
    if args.resume is not None:
        print(
            f"resumed checkpoint={args.resume} start_epoch={start_epoch} global_step={global_step} "
            f"best_val={best_validation_loss:.4f}",
        )
    last_step_end = time.perf_counter()

    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        train_loss_sums = {"total": 0.0, "occupancy": 0.0, "regression": 0.0}
        train_step_count = 0

        for batch in train_loader:
            poses_batch: List[np.ndarray] = batch["poses"]  # type: ignore[assignment]
            measurements_batch: List[List[np.ndarray]] = batch["measurements"]  # type: ignore[assignment]
            ground_truth_batch: List[List[np.ndarray]] = batch["ground_truth"]  # type: ignore[assignment]
            batch_size = len(poses_batch)
            sequence_length = len(measurements_batch[0])
            mirror_matrices = [
                sample_sequence_mirror(rng, augmentation_config)
                if not args.disable_augmentation
                else np.eye(4, dtype=np.float32)
                for _ in range(batch_size)
            ]
            previous_predictions = [np.empty((0, 3), dtype=np.float32) for _ in range(batch_size)]
            batch_ready_time = time.perf_counter()
            data_wait_time = batch_ready_time - last_step_end
            batch_step_count = 0

            for step in range(sequence_length):
                step_tensors = prepare_rollout_step(
                    poses_batch=poses_batch,
                    measurements_batch=measurements_batch,
                    ground_truth_batch=ground_truth_batch,
                    step=step,
                    previous_predictions=previous_predictions,
                    mirror_matrices=mirror_matrices,
                    grid_config=grid_config,
                    augmentation_config=augmentation_config,
                    device=device,
                    rng=rng,
                    apply_augmentation=not args.disable_augmentation,
                )
                if step_tensors is None:
                    continue
                sparse_input, target_coordinates, target_features = step_tensors

                step_start_time = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                    losses, previous_predictions = forward_rollout_step(
                        model=model,
                        sparse_input=sparse_input,
                        target_coordinates=target_coordinates,
                        target_features=target_features,
                        grid_config=grid_config,
                        batch_size=batch_size,
                        occupancy_weight=train_config.occupancy_loss_weight,
                        regression_weight=train_config.regression_loss_weight,
                    )
                if use_amp:
                    scaler.scale(losses["total"]).backward()
                    if args.gradient_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if args.gradient_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)
                    optimizer.step()
                scheduler.step()
                if use_cuda and args.print_timing:
                    torch.cuda.synchronize(device)
                step_end_time = time.perf_counter()
                step_time = step_end_time - step_start_time
                step_data_wait_time = data_wait_time if batch_step_count == 0 else 0.0
                for key in train_loss_sums:
                    train_loss_sums[key] += float(losses[key].item())
                train_step_count += 1
                batch_step_count += 1

                if global_step % train_config.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    message = (
                        f"epoch={epoch} step={global_step} "
                        f"loss={losses['total'].item():.4f} "
                        f"occ={losses['occupancy'].item():.4f} "
                        f"reg={losses['regression'].item():.4f} "
                        f"lr={lr:.6f}"
                    )
                    if args.print_timing:
                        message += f" data={step_data_wait_time:.4f}s step={step_time:.4f}s"
                    print(message)
                global_step += 1
                last_step_end = step_end_time
            if batch_step_count == 0:
                last_step_end = time.perf_counter()

        train_metrics = mean_loss_dict(train_loss_sums, train_step_count)
        validation_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            grid_config=grid_config,
            augmentation_config=augmentation_config,
            device=device,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            occupancy_weight=train_config.occupancy_loss_weight,
            regression_weight=train_config.regression_loss_weight,
        )
        is_best = False
        if validation_metrics is not None and validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]
            is_best = True

        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            args=args,
            augmentation_rng=rng,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )
        if epoch % train_config.save_every == 0:
            save_checkpoint(checkpoint, train_config.output_dir / f"epoch_{epoch:04d}.pt")
        save_checkpoint(checkpoint, train_config.output_dir / "last.pt")
        if is_best:
            save_checkpoint(checkpoint, train_config.output_dir / "best.pt")

        summary = (
            f"epoch={epoch} train_loss={train_metrics['total']:.4f} "
            f"train_occ={train_metrics['occupancy']:.4f} "
            f"train_reg={train_metrics['regression']:.4f}"
        )
        if validation_metrics is not None:
            summary += (
                f" val_loss={validation_metrics['total']:.4f}"
                f" val_occ={validation_metrics['occupancy']:.4f}"
                f" val_reg={validation_metrics['regression']:.4f}"
                f" best_val={best_validation_loss:.4f}"
            )
            if is_best:
                summary += " best=1"
        print(summary)
        last_step_end = time.perf_counter()


if __name__ == "__main__":
    main()
