import numpy as np


def cloud_to_voxel(cloud, voxel_size):
    """
    Convert a point cloud to a voxel grid.

    Parameters:
    cloud (numpy.ndarray): A point cloud of shape (N, 3).
    voxel_size (float): The size of each voxel.

    Returns:
    numpy.ndarray: A voxel grid of shape (X, Y, Z) where X, Y, Z are the number of voxels in each dimension.
    """
    # Calculate the minimum and maximum coordinates of the point cloud
    min_coords = np.min(cloud, axis=0)
    max_coords = np.max(cloud, axis=0)

    # Calculate the number of voxels in each dimension
    num_voxels = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Initialize an empty voxel grid
    voxel_grid = np.zeros(num_voxels, dtype=bool)

    # Convert point cloud to voxel indices
    indices = np.floor((cloud - min_coords) / voxel_size).astype(int)

    # Set the corresponding voxels to True
    for idx in indices:
        if np.all(idx >= 0) and np.all(idx < num_voxels):
            voxel_grid[tuple(idx)] = True

    return voxel_grid

def voxel_to_cloud(voxel_grid, voxel_size, min_coords):
    """
    Convert a voxel grid back to a point cloud.

    Parameters:
    voxel_grid (numpy.ndarray): A voxel grid of shape (X, Y, Z) where X, Y, Z are the number of voxels in each dimension.
    voxel_size (float): The size of each voxel.
    min_coords (numpy.ndarray): The minimum coordinates of the original point cloud.

    Returns:
    numpy.ndarray: A point cloud of shape (N, 3).
    """
    # Get the indices of the occupied voxels
    occupied_indices = np.argwhere(voxel_grid)

    # Convert voxel indices back to point coordinates
    cloud = occupied_indices * voxel_size + min_coords + voxel_size / 2

    return cloud