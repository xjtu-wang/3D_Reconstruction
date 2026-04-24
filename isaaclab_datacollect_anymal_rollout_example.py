from __future__ import annotations

import argparse

import isaaclab_datacollect_anymal_rollout as rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beginner-friendly example for the DataCollect ANYmal C rollout with real IsaacLab cameras."
    )
    rollout.add_rollout_args(parser)
    return parser.parse_args()


def main() -> int:
    # The heavy lifting lives in the main script:
    # 1. generate the corridor scene
    # 2. attach four depth cameras through CameraCfg
    # 3. run the rough-terrain policy and save summary.json + trajectory.npz
    rollout.run_rollout(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
