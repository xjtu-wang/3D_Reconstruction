from __future__ import annotations

from typing import Dict, Set, Tuple

import torch
import torch.nn.functional as F

from .model import ModelOutput
from .preprocessing import downsample_batched_coordinates


def _coordinate_set(coordinates: torch.Tensor) -> Set[Tuple[int, ...]]:
    if coordinates.numel() == 0:
        return set()
    coordinates_cpu = coordinates.detach().cpu()
    return {tuple(int(value) for value in row.tolist()) for row in coordinates_cpu}


def occupancy_bce_loss(model_output: ModelOutput, target_coordinates: torch.Tensor) -> torch.Tensor:
    total_loss = None
    valid_heads = 0
    for occupancy_prediction in model_output.occupancy_predictions:
        logits = occupancy_prediction.logits
        if logits.F.numel() == 0:
            continue

        query_coordinates = logits.C.to(dtype=torch.int32)
        target_at_stride = downsample_batched_coordinates(target_coordinates, occupancy_prediction.stride)
        target_lookup = _coordinate_set(target_at_stride)

        labels = torch.tensor(
            [
                1.0 if tuple(int(value) for value in row.tolist()) in target_lookup else 0.0
                for row in query_coordinates.detach().cpu()
            ],
            dtype=logits.F.dtype,
            device=logits.F.device,
        )
        head_loss = F.binary_cross_entropy_with_logits(logits.F.squeeze(1), labels)
        total_loss = head_loss if total_loss is None else total_loss + head_loss
        valid_heads += 1

    if total_loss is None:
        reference = target_coordinates.float()
        return reference.sum() * 0.0
    return total_loss / valid_heads


def subvoxel_regression_loss(
    model_output: ModelOutput,
    target_coordinates: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    prediction_coordinates = model_output.offsets.C.to(dtype=torch.int32)
    prediction_features = model_output.offsets.F
    if prediction_features.numel() == 0 or target_features.numel() == 0:
        reference = prediction_features if prediction_features.numel() > 0 else target_features
        return reference.sum() * 0.0

    target_lookup = {
        tuple(int(value) for value in coordinate.tolist()): index
        for index, coordinate in enumerate(target_coordinates.detach().cpu())
    }
    matched_prediction_indices = []
    matched_target_indices = []
    for prediction_index, coordinate in enumerate(prediction_coordinates.detach().cpu()):
        key = tuple(int(value) for value in coordinate.tolist())
        target_index = target_lookup.get(key)
        if target_index is None:
            continue
        matched_prediction_indices.append(prediction_index)
        matched_target_indices.append(target_index)

    if not matched_prediction_indices:
        return prediction_features.sum() * 0.0

    prediction_subset = prediction_features[matched_prediction_indices]
    target_subset = target_features[matched_target_indices].to(
        device=prediction_subset.device,
        dtype=prediction_subset.dtype,
    )
    return torch.linalg.norm(prediction_subset - target_subset, dim=1).mean()


def reconstruction_loss(
    model_output: ModelOutput,
    target_coordinates: torch.Tensor,
    target_features: torch.Tensor,
    occupancy_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    occupancy = occupancy_bce_loss(model_output, target_coordinates)
    regression = subvoxel_regression_loss(model_output, target_coordinates, target_features)
    total = occupancy_weight * occupancy + regression_weight * regression
    return {
        "total": total,
        "occupancy": occupancy,
        "regression": regression,
    }

