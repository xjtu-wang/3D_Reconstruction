from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .model import ModelOutput
from .preprocess import downsample_batched_coordinates


def _coordinate_hash_params(*coordinate_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    non_empty = [coordinates.to(dtype=torch.int64) for coordinates in coordinate_tensors if coordinates.numel() > 0]
    if not non_empty:
        reference = coordinate_tensors[0]
        return (
            torch.empty((0,), dtype=torch.int64, device=reference.device),
            torch.empty((0,), dtype=torch.int64, device=reference.device),
        )

    combined = torch.cat(non_empty, dim=0) if len(non_empty) > 1 else non_empty[0]
    min_values = combined.min(dim=0).values
    max_values = combined.max(dim=0).values
    bases = (max_values - min_values) + 1
    multipliers = torch.ones_like(bases)
    for index in range(bases.numel() - 2, -1, -1):
        multipliers[index] = multipliers[index + 1] * bases[index + 1]
    return min_values, multipliers


def _encode_coordinates(
    coordinates: torch.Tensor,
    min_values: torch.Tensor,
    multipliers: torch.Tensor,
) -> torch.Tensor:
    if coordinates.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=coordinates.device)
    coordinates_int64 = coordinates.to(dtype=torch.int64)
    return ((coordinates_int64 - min_values) * multipliers).sum(dim=1)


def _match_target_indices(
    query_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
) -> torch.Tensor:
    if query_coordinates.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=query_coordinates.device)
    if target_coordinates.numel() == 0:
        return torch.full((query_coordinates.shape[0],), -1, dtype=torch.long, device=query_coordinates.device)

    min_values, multipliers = _coordinate_hash_params(query_coordinates, target_coordinates)
    query_keys = _encode_coordinates(query_coordinates, min_values, multipliers)
    target_keys = _encode_coordinates(target_coordinates, min_values, multipliers)

    sorted_target_keys, sorted_target_indices = torch.sort(target_keys)
    insertion_indices = torch.searchsorted(sorted_target_keys, query_keys)
    matched_target_indices = torch.full(
        (query_coordinates.shape[0],),
        -1,
        dtype=torch.long,
        device=query_coordinates.device,
    )

    valid_rows = insertion_indices < sorted_target_keys.shape[0]
    if not torch.any(valid_rows):
        return matched_target_indices

    candidate_rows = torch.nonzero(valid_rows, as_tuple=False).squeeze(1)
    candidate_insertions = insertion_indices[candidate_rows]
    candidate_matches = sorted_target_keys[candidate_insertions] == query_keys[candidate_rows]
    if not torch.any(candidate_matches):
        return matched_target_indices

    matched_rows = candidate_rows[candidate_matches]
    matched_insertions = candidate_insertions[candidate_matches]
    matched_target_indices[matched_rows] = sorted_target_indices[matched_insertions]
    return matched_target_indices


def occupancy_bce_loss(model_output: ModelOutput, target_coordinates: torch.Tensor) -> torch.Tensor:
    total_loss = None
    valid_heads = 0
    for occupancy_prediction in model_output.occupancy_predictions:
        logits = occupancy_prediction.logits
        if logits.F.numel() == 0:
            continue

        query_coordinates = logits.C.to(dtype=torch.int32)
        target_at_stride = downsample_batched_coordinates(target_coordinates, occupancy_prediction.stride)
        matched_target_indices = _match_target_indices(query_coordinates, target_at_stride)
        labels = matched_target_indices.ge(0).to(dtype=logits.F.dtype)
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

    matched_target_indices = _match_target_indices(prediction_coordinates, target_coordinates)
    matched_prediction_mask = matched_target_indices.ge(0)
    if not torch.any(matched_prediction_mask):
        return prediction_features.sum() * 0.0

    prediction_subset = prediction_features[matched_prediction_mask]
    target_subset = target_features[matched_target_indices[matched_prediction_mask]].to(
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
    total = occupancy * occupancy_weight + regression * regression_weight
    return {
        "total": total,
        "occupancy": occupancy,
        "regression": regression,
    }
