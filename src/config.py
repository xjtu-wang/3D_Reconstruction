from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class GridConfig:
    voxel_size: float = 0.05
    spatial_shape: Tuple[int, int, int] = (64, 64, 64)
    bounds_min: Tuple[float, float, float] = (-1.6, -1.6, -1.6)
    current_time_index: int = 0
    previous_time_index: int = 1

    @property
    def bounds_max(self) -> Tuple[float, float, float]:
        return tuple(
            self.bounds_min[i] + self.spatial_shape[i] * self.voxel_size
            for i in range(3)
        )


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    stem_channels: int = 32
    encoder_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 32)
    pruning_threshold: float = 0.5
    spacetime_kernel: Tuple[int, int, int, int] = (3, 3, 3, 3)
    stride_kernel: Tuple[int, int, int, int] = (2, 2, 2, 1)


@dataclass(frozen=True)
class AugmentationConfig:
    position_noise: float = 0.05
    tilt_degrees: float = 1.0
    height_noise: float = 0.05
    pose_translation_noise: float = 0.05
    prune_patch_radius_range: Tuple[float, float] = (0.10, 0.35)
    prune_patch_count: Tuple[int, int] = (1, 3)
    height_patch_radius_range: Tuple[float, float] = (0.10, 0.35)
    height_patch_count: Tuple[int, int] = (1, 3)
    outlier_cluster_radius_range: Tuple[float, float] = (0.05, 0.12)
    outlier_cluster_count: Tuple[int, int] = (1, 3)
    outlier_points_per_cluster: Tuple[int, int] = (16, 64)
    mirror_probability: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    sequence_length: int = 12
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-2
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    occupancy_loss_weight: float = 1.0
    regression_loss_weight: float = 1.0
    log_every: int = 10
    save_every: int = 50
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("runs/nsr_terrain"))
