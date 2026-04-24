from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from main import evaluate_model, make_data_loader, set_seed
from src.config import AugmentationConfig, GridConfig, ModelConfig
from src.data import TrajectoryDataset
from src.model import TerrainReconstructionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved training checkpoints on a validation split and write best.pt.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing step_*.pt / epoch_*.pt.")
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=None,
        help="Checkpoint glob to evaluate. Repeatable. Defaults to step_*.pt and epoch_*.pt.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Validation trajectory directory. If omitted, try to read it from checkpoint args['val_root'].",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Validation sequence length. Set <= 0 to reuse the value stored in the checkpoint args.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=0,
        help="Validation sequence stride. Set <= 0 to reuse the value stored in the checkpoint args.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        help="Validation batch size. Set <= 0 to auto-cap it at min(train_batch_size, 64).",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp", action="store_true", help="Enable autocast during offline validation.")
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Summary JSON output path. Defaults to <checkpoint-dir>/checkpoint_validation_summary.json.",
    )
    parser.add_argument(
        "--best-path",
        type=Path,
        default=None,
        help="Where to write the best checkpoint. Defaults to <checkpoint-dir>/best.pt.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, map_location: torch.device | str) -> Dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def resolve_checkpoint_candidates(checkpoint_dir: Path, checkpoint_globs: List[str]) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()
    for pattern in checkpoint_globs:
        for path in sorted(checkpoint_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_globs = args.checkpoint_glob or ["step_*.pt", "epoch_*.pt"]
    candidates = resolve_checkpoint_candidates(checkpoint_dir, checkpoint_globs)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matched in {checkpoint_dir} with patterns: {', '.join(checkpoint_globs)}",
        )

    reference_checkpoint = load_checkpoint(candidates[0], map_location="cpu")
    checkpoint_args = reference_checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    val_root_value = args.val_root
    if val_root_value is None:
        raw_val_root = checkpoint_args.get("val_root")
        if not raw_val_root:
            raise ValueError("Validation root not provided and not found in checkpoint args['val_root'].")
        val_root_value = Path(str(raw_val_root))
    sequence_length = args.sequence_length if args.sequence_length > 0 else int(checkpoint_args.get("sequence_length", 12))
    sequence_stride = args.sequence_stride if args.sequence_stride > 0 else int(checkpoint_args.get("sequence_stride", 1))
    train_batch_size = int(checkpoint_args.get("batch_size", 64))
    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else min(train_batch_size, 64)
    summary_path = args.summary_path.resolve() if args.summary_path is not None else checkpoint_dir / "checkpoint_validation_summary.json"
    best_path = args.best_path.resolve() if args.best_path is not None else checkpoint_dir / "best.pt"

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    grid_config = GridConfig()
    model_config = ModelConfig()
    augmentation_config = AugmentationConfig()
    val_dataset = TrajectoryDataset(
        root=val_root_value,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        cache_data=args.cache_dataset,
    )
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_cuda=use_cuda,
        disable_pin_memory=args.disable_pin_memory,
        disable_persistent_workers=args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(
        f"checkpoint_count={len(candidates)} val_windows={len(val_dataset)} "
        f"val_batch_size={val_batch_size} device={device}",
    )

    model = TerrainReconstructionModel(model_config).to(device)
    results: List[Dict[str, object]] = []
    best_checkpoint_path: Optional[Path] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_validation_loss = float("inf")
    evaluation_started_at = time.time()

    for index, checkpoint_path in enumerate(candidates, start=1):
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if use_cuda:
            torch.cuda.empty_cache()
        started = time.perf_counter()
        metrics = evaluate_model(
            model=model,
            loader=val_loader,
            grid_config=grid_config,
            augmentation_config=augmentation_config,
            device=device,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            occupancy_weight=1.0,
            regression_weight=1.0,
        )
        elapsed_seconds = time.perf_counter() - started
        if metrics is None:
            raise RuntimeError(f"Validation unexpectedly returned None for checkpoint: {checkpoint_path}")
        result = {
            "checkpoint": checkpoint_path.name,
            "epoch": int(checkpoint.get("epoch", 0)),
            "completed_epoch": bool(checkpoint.get("completed_epoch", True)),
            "global_step": int(checkpoint.get("global_step", 0)),
            "validation_metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
        }
        results.append(result)
        print(
            f"[{index}/{len(candidates)}] checkpoint={checkpoint_path.name} "
            f"step={result['global_step']} val_loss={metrics['total']:.4f} "
            f"val_occ={metrics['occupancy']:.4f} val_reg={metrics['regression']:.4f} "
            f"elapsed={elapsed_seconds:.1f}s",
        )
        if metrics["total"] < best_validation_loss:
            best_validation_loss = metrics["total"]
            best_checkpoint_path = checkpoint_path
            best_metrics = metrics

    if best_checkpoint_path is None or best_metrics is None:
        raise RuntimeError("No valid checkpoint evaluation result was produced.")

    best_checkpoint = load_checkpoint(best_checkpoint_path, map_location="cpu")
    best_checkpoint["best_validation_loss"] = best_validation_loss
    best_checkpoint["validation_metrics"] = best_metrics
    best_checkpoint["offline_validation"] = {
        "summary_path": str(summary_path),
        "evaluated_at_unix": evaluation_started_at,
        "checkpoint_count": len(candidates),
    }
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, best_path)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_globs": checkpoint_globs,
        "val_root": str(val_root_value),
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "val_batch_size": val_batch_size,
        "candidate_count": len(candidates),
        "evaluated_at_unix": evaluation_started_at,
        "best_checkpoint": best_checkpoint_path.name,
        "best_path": str(best_path),
        "best_validation_metrics": best_metrics,
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"best_checkpoint={best_checkpoint_path.name} val_loss={best_metrics['total']:.4f} "
        f"best_path={best_path} summary_path={summary_path}",
    )


if __name__ == "__main__":
    main()
