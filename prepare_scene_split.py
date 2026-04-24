from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    input_root: Path
    source_path: Path
    output_name: str
    trajectory_length: int
    window_count: int
    scene_seed: int | None
    scene_index: int | None
    trajectory_in_scene: int | None
    file_name: str


@dataclass(frozen=True)
class SceneGroup:
    group_id: str
    scene_seed: int | None
    records: tuple[TrajectoryRecord, ...]

    @property
    def trajectory_count(self) -> int:
        return len(self.records)

    @property
    def window_count(self) -> int:
        return sum(record.window_count for record in self.records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val directories for main.py by splitting trajectories at the scene level. "
            "Each output directory contains materialized trajectory files pointing back to the originals."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Input trajectory directory. Repeat this flag for the main dataset dir and each shard dir.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory containing train/ and val/.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of scene groups assigned to validation. Must be in (0, 1).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=12,
        help="Sequence length used by main.py. This is only used to estimate training-window counts in the summary.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=1,
        help="Sequence stride used by main.py. This is only used to estimate training-window counts in the summary.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic split seed.")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="hardlink",
        help="How train/val files should reference the original trajectories. hardlink avoids symlinks without duplicating data.",
    )
    parser.add_argument(
        "--group-by",
        choices=("scene_seed", "trajectory"),
        default="scene_seed",
        help="Grouping rule. scene_seed keeps all trajectories from one scene in the same split.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.jsonl",
        help="Manifest file name inside each input root.",
    )
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be strictly between 0 and 1.")
    if args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be positive.")
    if args.sequence_stride <= 0:
        raise SystemExit("--sequence-stride must be positive.")
    return args


def count_windows(trajectory_length: int, sequence_length: int, sequence_stride: int) -> int:
    if trajectory_length < sequence_length:
        return 0
    return 1 + (trajectory_length - sequence_length) // sequence_stride


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            entry = json.loads(payload)
            if not isinstance(entry, dict):
                raise RuntimeError(f"{manifest_path}:{line_number} is not a JSON object.")
            file_name = entry.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise RuntimeError(f"{manifest_path}:{line_number} is missing a valid file_name field.")
            entries[file_name] = entry
    return entries


def sanitize_root_token(root_name: str, root_index: int) -> str:
    cleaned = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in root_name)
    cleaned = cleaned.strip("_") or f"dataset_{root_index:02d}"
    return f"root{root_index:02d}_{cleaned}"


def build_records_for_root(
    *,
    input_root: Path,
    root_index: int,
    manifest_name: str,
    sequence_length: int,
    sequence_stride: int,
) -> tuple[list[TrajectoryRecord], int]:
    manifest_entries = load_manifest(input_root / manifest_name)
    output_prefix = sanitize_root_token(input_root.name, root_index)
    records: list[TrajectoryRecord] = []
    fallback_group_count = 0

    for source_path in sorted(input_root.glob("trajectory_*.npz")):
        with np.load(source_path) as trajectory:
            if "poses" not in trajectory:
                raise KeyError(f"{source_path} is missing the required poses array.")
            trajectory_length = int(trajectory["poses"].shape[0])
        window_count = count_windows(trajectory_length, sequence_length, sequence_stride)
        if window_count <= 0:
            continue

        manifest_entry = manifest_entries.get(source_path.name)
        scene_seed = None
        scene_index = None
        trajectory_in_scene = None
        if manifest_entry is not None:
            if manifest_entry.get("scene_seed") is not None:
                scene_seed = int(manifest_entry["scene_seed"])
            if manifest_entry.get("scene_index") is not None:
                scene_index = int(manifest_entry["scene_index"])
            if manifest_entry.get("trajectory_in_scene") is not None:
                trajectory_in_scene = int(manifest_entry["trajectory_in_scene"])
        else:
            fallback_group_count += 1

        records.append(
            TrajectoryRecord(
                input_root=input_root,
                source_path=source_path,
                output_name=f"{output_prefix}__{source_path.name}",
                trajectory_length=trajectory_length,
                window_count=window_count,
                scene_seed=scene_seed,
                scene_index=scene_index,
                trajectory_in_scene=trajectory_in_scene,
                file_name=source_path.name,
            )
        )

    return records, fallback_group_count


def make_group_id(record: TrajectoryRecord, group_by: str) -> str:
    root_id = str(record.input_root.resolve())
    if group_by == "scene_seed" and record.scene_seed is not None:
        return f"{root_id}::scene_seed::{record.scene_seed}"
    return f"{root_id}::trajectory::{record.file_name}"


def build_groups(records: list[TrajectoryRecord], group_by: str) -> list[SceneGroup]:
    grouped: dict[str, list[TrajectoryRecord]] = {}
    for record in records:
        group_id = make_group_id(record, group_by)
        grouped.setdefault(group_id, []).append(record)

    groups: list[SceneGroup] = []
    for group_id in sorted(grouped):
        group_records = tuple(sorted(grouped[group_id], key=lambda item: item.output_name))
        scene_seed = group_records[0].scene_seed
        groups.append(SceneGroup(group_id=group_id, scene_seed=scene_seed, records=group_records))
    return groups


def split_groups(groups: list[SceneGroup], val_fraction: float, seed: int) -> tuple[list[SceneGroup], list[SceneGroup]]:
    if len(groups) < 2:
        raise RuntimeError("Need at least two scene groups to create separate train and val splits.")

    shuffled = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    val_group_count = int(round(len(shuffled) * val_fraction))
    val_group_count = min(max(val_group_count, 1), len(shuffled) - 1)
    val_groups = sorted(shuffled[:val_group_count], key=lambda group: group.group_id)
    train_groups = sorted(shuffled[val_group_count:], key=lambda group: group.group_id)
    return train_groups, val_groups


def ensure_output_root(output_root: Path) -> None:
    if output_root.exists():
        existing = [path.name for path in output_root.iterdir()]
        if existing:
            raise RuntimeError(f"{output_root} already exists and is not empty: {existing[:5]}")
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_link(source_path: Path, destination_path: Path, link_mode: str) -> None:
    if destination_path.exists() or destination_path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing output file: {destination_path}")
    if link_mode == "symlink":
        destination_path.symlink_to(source_path.resolve())
    elif link_mode == "hardlink":
        os.link(source_path, destination_path)
    elif link_mode == "copy":
        shutil.copy2(source_path, destination_path)
    else:  # pragma: no cover - argparse already constrains this
        raise ValueError(f"Unsupported link mode: {link_mode}")


def summarize_groups(groups: list[SceneGroup]) -> dict[str, Any]:
    trajectories = sum(group.trajectory_count for group in groups)
    windows = sum(group.window_count for group in groups)
    seeds = sorted({int(group.scene_seed) for group in groups if group.scene_seed is not None})
    return {
        "scene_group_count": len(groups),
        "trajectory_count": trajectories,
        "window_count": windows,
        "scene_seed_min": seeds[0] if seeds else None,
        "scene_seed_max": seeds[-1] if seeds else None,
        "scene_seed_count": len(seeds),
    }


def write_split_manifest(path: Path, split_name: str, groups: list[SceneGroup]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            for record in group.records:
                payload = {
                    "split": split_name,
                    "group_id": group.group_id,
                    "scene_seed": record.scene_seed,
                    "scene_index": record.scene_index,
                    "trajectory_in_scene": record.trajectory_in_scene,
                    "trajectory_length": record.trajectory_length,
                    "window_count": record.window_count,
                    "source_root": str(record.input_root),
                    "source_path": str(record.source_path),
                    "output_name": record.output_name,
                }
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")


def main() -> int:
    args = parse_args()
    input_roots = [path.expanduser().resolve() for path in args.input_root]
    output_root = args.output_root.expanduser().resolve()

    for input_root in input_roots:
        if not input_root.is_dir():
            raise RuntimeError(f"Input root does not exist or is not a directory: {input_root}")

    all_records: list[TrajectoryRecord] = []
    manifest_fallback_count = 0
    for root_index, input_root in enumerate(input_roots):
        records, fallback_count = build_records_for_root(
            input_root=input_root,
            root_index=root_index,
            manifest_name=args.manifest_name,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
        )
        all_records.extend(records)
        manifest_fallback_count += fallback_count

    if not all_records:
        raise RuntimeError("No valid trajectory files were found after applying the sequence-length filter.")

    groups = build_groups(all_records, args.group_by)
    train_groups, val_groups = split_groups(groups, args.val_fraction, args.seed)

    ensure_output_root(output_root)
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    for split_name, split_dir, split_groups_list in (
        ("train", train_dir, train_groups),
        ("val", val_dir, val_groups),
    ):
        for group in split_groups_list:
            for record in group.records:
                materialize_link(record.source_path, split_dir / record.output_name, args.link_mode)
        write_split_manifest(output_root / f"{split_name}_manifest.jsonl", split_name, split_groups_list)

    train_summary = summarize_groups(train_groups)
    val_summary = summarize_groups(val_groups)
    total_windows = train_summary["window_count"] + val_summary["window_count"]
    total_trajectories = train_summary["trajectory_count"] + val_summary["trajectory_count"]
    summary = {
        "input_roots": [str(path) for path in input_roots],
        "output_root": str(output_root),
        "group_by": args.group_by,
        "link_mode": args.link_mode,
        "seed": int(args.seed),
        "sequence_length": int(args.sequence_length),
        "sequence_stride": int(args.sequence_stride),
        "requested_val_fraction": float(args.val_fraction),
        "manifest_name": args.manifest_name,
        "manifest_fallback_trajectory_count": int(manifest_fallback_count),
        "totals": {
            "scene_group_count": len(groups),
            "trajectory_count": int(total_trajectories),
            "window_count": int(total_windows),
        },
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
        "actual_val_fraction": {
            "scene_groups": float(len(val_groups) / len(groups)),
            "trajectories": float(val_summary["trajectory_count"] / total_trajectories),
            "windows": float(val_summary["window_count"] / total_windows),
        },
    }
    (output_root / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote split under {output_root}")
    print(
        "train: "
        f"{train_summary['scene_group_count']} scene groups, "
        f"{train_summary['trajectory_count']} trajectories, "
        f"{train_summary['window_count']} windows"
    )
    print(
        "val: "
        f"{val_summary['scene_group_count']} scene groups, "
        f"{val_summary['trajectory_count']} trajectories, "
        f"{val_summary['window_count']} windows"
    )
    if manifest_fallback_count > 0:
        print(
            "warning: "
            f"{manifest_fallback_count} trajectories were missing manifest scene metadata and fell back to per-trajectory grouping"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
