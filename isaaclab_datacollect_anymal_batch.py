from __future__ import annotations

import argparse
import sys

import isaaclab_datacollect_anymal_sequential as sequential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility entrypoint for batch collection. "
            "This now runs the single-process sequential collector and reuses one Isaac Sim env "
            "for multiple reset-based trajectories per scene."
        )
    )
    parser.add_argument("--policy-checkpoint", type=sequential.Path, required=True, help="TorchScript/JIT rough-terrain policy checkpoint.")
    parser.add_argument("--dataset-dir", type=sequential.Path, required=True, help="Output directory for trajectory_*.npz, manifest.jsonl, and batch_summary.json.")
    parser.add_argument("--target-trajectories", type=int, default=800, help="Number of complete trajectories to save.")
    parser.add_argument("--steps", type=int, default=250, help="Number of policy steps per trajectory.")
    parser.add_argument("--seed", type=int, default=0, help="Base scene seed. Scene index i uses seed + i.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=sequential.DEFAULT_TRAJECTORIES_PER_SCENE,
        help=(
            "Legacy alias. In the sequential collector this means how many trajectories to collect "
            "from the same scene via repeated env.reset() before rebuilding the scene."
        ),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim headless. Use --no-headless to keep the UI visible.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="IsaacLab simulation and policy device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--export-usd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export one scene USD per scene under dataset-dir/scenes.",
    )
    parser.add_argument(
        "--record-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy flag from the vectorized collector. Sequential dataset collection keeps this disabled.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy flag from the vectorized collector. Sequential dataset collection keeps this disabled.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume into an existing dataset directory without overwriting saved trajectories.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional trajectory id to start writing from. Useful when resuming or manually offsetting file numbering.",
    )
    args = parser.parse_args()
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.record_frames:
        raise SystemExit("--record-frames is not supported by the sequential batch collector.")
    if args.save_video:
        raise SystemExit("--save-video is not supported by the sequential batch collector.")
    return args


def main() -> int:
    args = parse_args()
    forwarded_argv = [
        sys.argv[0],
        "--policy-checkpoint",
        str(args.policy_checkpoint),
        "--dataset-dir",
        str(args.dataset_dir),
        "--target-trajectories",
        str(args.target_trajectories),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--trajectories-per-scene",
        str(args.num_envs),
        "--device",
        str(args.device),
        "--headless" if args.headless else "--no-headless",
        "--export-usd" if args.export_usd else "--no-export-usd",
        "--resume" if args.resume else "--no-resume",
    ]
    if args.start_index is not None:
        forwarded_argv.extend(["--start-index", str(args.start_index)])
    original_argv = sys.argv
    try:
        sys.argv = forwarded_argv
        return sequential.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
