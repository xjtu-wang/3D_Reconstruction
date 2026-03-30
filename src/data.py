from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

REQUIRED_KEYS = (
    "poses",
    "measurement_points",
    "measurement_splits",
    "ground_truth_points",
    "ground_truth_splits",
)

def slice_packed_points(points: np.ndarray, splits: np.ndarray, index: int) -> np.ndarray:
    start = int(splits[index])
    end = int(splits[index + 1])
    return points[start:end].astype(np.float32, copy=False)

@dataclass(frozen=True)
class SequenceIndex:
    file_path: Path
    start_index: int


TrajectoryArrays = Dict[str, np.ndarray]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        sequence_length: int = 12,
        sequence_stride: int = 1,
        cache_data: bool = False,
    ) -> None:
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.cache_data = cache_data
        self._cache: Dict[Path, TrajectoryArrays] = {}
        self.files = sorted(
            file_path
            for file_path in self.root.glob("*.npz")
            if not file_path.name.endswith(".partial.npz")
        )
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectories found under {self.root}")
        self.indices = self._build_index()

    @staticmethod
    def _read_trajectory(file_path: Path) -> TrajectoryArrays:
        with np.load(file_path) as trajectory:
            TrajectoryDataset._validate_file(file_path, trajectory)
            return {key: trajectory[key] for key in REQUIRED_KEYS}

    def _load_trajectory(self, file_path: Path) -> TrajectoryArrays:
        if self.cache_data:
            cached = self._cache.get(file_path)
            if cached is not None:
                return cached

        trajectory = self._read_trajectory(file_path)
        if self.cache_data:
            self._cache[file_path] = trajectory
        return trajectory

    def _build_index(self) -> List[SequenceIndex]:
        indices: List[SequenceIndex] = []
        for file_path in self.files:
            trajectory = self._load_trajectory(file_path)
            trajectory_length = int(trajectory["poses"].shape[0])
            if trajectory_length < self.sequence_length:
                continue
            for start in range(0, trajectory_length - self.sequence_length + 1, self.sequence_stride):
                indices.append(SequenceIndex(file_path=file_path, start_index=start))
        if not indices:
            raise ValueError("No valid rollout windows found. Check sequence_length and trajectory lengths.")
        return indices
    
    @staticmethod
    def _validate_file(file_path: Path, trajectory: np.lib.npyio.NpzFile) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in trajectory]
        if missing:
            raise KeyError(f"{file_path} is missing required arrays: {missing}")
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.indices[index]
        end_index = entry.start_index + self.sequence_length
        trajectory = self._load_trajectory(entry.file_path)
        poses = trajectory["poses"][entry.start_index:end_index].astype(np.float32)
        measurement_points = trajectory["measurement_points"]
        measurement_splits = trajectory["measurement_splits"]
        ground_truth_points = trajectory["ground_truth_points"]
        ground_truth_splits = trajectory["ground_truth_splits"]

        measurements = [
            slice_packed_points(measurement_points, measurement_splits, timestep)
            for timestep in range(entry.start_index, end_index)
        ]
        ground_truth = [
            slice_packed_points(ground_truth_points, ground_truth_splits, timestep)
            for timestep in range(entry.start_index, end_index)
        ]
        return {
            "poses": poses,
            "measurements": measurements,
            "ground_truth": ground_truth,
            "file_path": str(entry.file_path),
            "start_index": entry.start_index,
        }
    
def collate_trajectory_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "poses": [sample["poses"] for sample in batch],
        "measurements": [sample["measurements"] for sample in batch],
        "ground_truth": [sample["ground_truth"] for sample in batch],
        "file_paths": [sample["file_path"] for sample in batch],
        "start_indices": [sample["start_index"] for sample in batch],
    }

