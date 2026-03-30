from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .config import GridConfig

@dataclass(frozen=True)
class VoxelizedPointCloud:
    """
    体素化的点云数据类，包含坐标和特征。
    坐标(N,4)包含空间坐标和时间索引，特征(N,3)包含相对于体素中心的偏移。
    """
    coordinates: np.ndarray
    features: np.ndarray

def _empty_voxelized() -> VoxelizedPointCloud:
    """"
    返回一个空的体素化点云对象，坐标和特征均为空数组。
    """
    return VoxelizedPointCloud(
        coordinates=np.empty((0,4),dtype=np.int32),
        features=np.empty((0,3),dtype=np.float32),
    )

def _aggregate_duplicates(coordinates: np.ndarray, features: np.ndarray) -> VoxelizedPointCloud:
    """
    聚合重复的体素化点云数据。新的特征为体素格内点的重心位置。
    """
    if coordinates.size == 0:
        return _empty_voxelized()
    
    unique_coordinates, inverse_indices = np.unique(
        coordinates,
        axis=0,
        return_inverse=True,
    )

    aggregated_features = np.zeros((len(unique_coordinates), features.shape[1]), dtype=np.float32)
    counts = np.zeros((len(unique_coordinates), 1), dtype=np.float32)
    np.add.at(aggregated_features, inverse_indices, features)
    np.add.at(counts, inverse_indices, 1.0)
    aggregated_features /= np.maximum(counts, 1.0)
    return VoxelizedPointCloud(unique_coordinates, aggregated_features)

def voxelize_points(points: np.ndarray, grid: GridConfig, time_index: int) -> VoxelizedPointCloud:
    """
    将点云数据体素化，返回体素化的点云对象。
    """
    if points.size == 0:
        return _empty_voxelized()
    
    bounds_min = np.asarray(grid.bounds_min, dtype=np.float32)
    spatial_shape = np.asarray(grid.spatial_shape, dtype=np.int32)
    normalized_points = (points.astype(np.float32) - bounds_min) / grid.voxel_size

    valid_mask = np.all(normalized_points >= 0.0, axis=1)
    valid_mask &= np.all(normalized_points < spatial_shape[None, :], axis=1)
    if not np.any(valid_mask):
        return _empty_voxelized()
    
    normalized_points = normalized_points[valid_mask]
    spatial_coordinates = np.floor(normalized_points).astype(np.int32)
    temporal_column = np.full((spatial_coordinates.shape[0],1),time_index,dtype=np.int32)
    coordinates = np.concatenate((spatial_coordinates, temporal_column), axis=1)
    features = normalized_points - spatial_coordinates.astype(np.float32)
    return _aggregate_duplicates(coordinates, features.astype(np.float32, copy=False))

def build_network_input(
        current_measurement: np.ndarray,
        previous_prediction: np.ndarray,
        grid: GridConfig,
) -> VoxelizedPointCloud:
    """
    构建网络输入，将当前测量量和之前的预测体素化，并合并成一个体素化点云对象。
    """
    current = voxelize_points(current_measurement, grid, grid.current_time_index)
    previous = voxelize_points(previous_prediction, grid, grid.previous_time_index)
    if current.coordinates.size == 0 and previous.coordinates.size == 0:
        return _empty_voxelized()
    if current.coordinates.size == 0:
        return previous
    if previous.coordinates.size == 0:
        return current
    
    coordinates = np.concatenate((current.coordinates, previous.coordinates), axis=0)
    features = np.concatenate((current.features, previous.features), axis=0)
    return _aggregate_duplicates(coordinates, features)

def downsample_batched_coordinates(coordinates: torch.Tensor, spatial_stride: int) -> torch.Tensor:
    """
    对批量坐标进行下采样，返回下采样后的坐标。
     - 输入坐标形状为(N,4)，其中第一列为批次索引，后面三列为空间坐标。
     - 通过除以空间步长并向下取整实现下采样，然后使用torch.unique去除重复的坐标。
     - 如果输入坐标为空，则直接返回空张量。
     - 输出坐标形状为(M,4)，其中M是下采样后唯一坐标的数量。
     - 该函数适用于点云数据的体素化处理，可以有效减少输入数据的数量，提高后续网络处理的效率。
     - 注意：下采样可能会导致信息丢失，因此在使用时需要根据具体应用场景权衡下采样率和信息保留之间的关系。
     - 示例用法：
        coordinates = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [1, 1, 2, 3]])
        spatial_stride = 2
        downsampled = downsample_batched_coordinates(coordinates, spatial_stride)
        print(downsampled)
    """
    if coordinates.numel() == 0:
        return coordinates
    downsampled = coordinates.clone()
    downsampled[:,1:4] = torch.div(
        downsampled[:,1:4],
        spatial_stride,
        rounding_mode='floor',
    )
    downsampled[:,1:4] *= spatial_stride
    return torch.unique(downsampled, dim=0)

def make_batched_coordinate_tensor(coordinate_batches: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    将批量坐标列表转换为一个批量坐标张量，返回转换后的张量。
    """
    chunks: List[torch.Tensor] = []

    for batch_index, coordinates in enumerate(coordinate_batches):
        if coordinates.size == 0:
            continue
        batch_column = np.full((coordinates.shape[0], 1), batch_index, dtype=np.int32)
        batched = np.concatenate((batch_column, coordinates), axis=1)
        chunks.append(torch.from_numpy(batched))
    if not chunks:
        return torch.empty((0,5), dtype=torch.int32, device=device)
    return torch.cat(chunks, dim=0).to(device=device, dtype=torch.int32)

def make_batched_feature_tensor(feature_batches: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    将批量特征列表转换为一个批量特征张量，返回转换后的张量。
    """
    valid = [torch.from_numpy(features) for features in feature_batches if features.size > 0]
    if not valid:
        return torch.empty((0,3), dtype=torch.float32, device=device)
    return torch.cat(valid, dim=0).to(device=device, dtype=torch.float32)

def decode_predictions_to_point_clouds(
        coordinates: torch.Tensor,
        features: torch.Tensor,
        grid: GridConfig,
        batch_size: int,
        time_index: int,
) -> List[np.ndarray]:
    """
    将网络预测的坐标和特征解码回点云列表，返回解码后的点云列表。
    """
    clouds: List[np.ndarray] = []
    if coordinates.numel() == 0:
        return [np.empty((0,3), dtype=np.float32) for _ in range(batch_size)]
    
    coordinates_cpu = coordinates.detach().cpu()
    features_cpu = features.detach().cpu()
    bounds_min = torch.tensor(grid.bounds_min, dtype=torch.float32)

    for batch_index in range(batch_size):
        mask = coordinates_cpu[:,0] == batch_index
        mask &= coordinates_cpu[:,4] == time_index
        if not torch.any(mask):
            clouds.append(np.empty((0,3), dtype=np.float32))
            continue
        spatial_coords = coordinates_cpu[mask][:,1:4].float()
        offsets = features_cpu[mask]
        points = bounds_min + grid.voxel_size * (spatial_coords + offsets)
        clouds.append(points.numpy().astype(np.float32, copy=False))
    return clouds
