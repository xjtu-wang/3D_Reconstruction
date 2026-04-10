# Neural Scene Representation Reproduction

This repository now contains a paper-oriented implementation of the core pipeline from:

`Hoeller et al., "Neural Scene Representation for Locomotion on Structured Terrain", IEEE RA-L 2022`

Implemented pieces:

- Input preprocessing into a `64 x 64 x 64` robot-centric sparse voxel grid with `0.05 m` cells.
- Temporal concatenation of `current measurement` and `previous prediction` using a 4D coordinate `(x, y, z, k)`.
- A MinkowskiEngine-based 4D U-Net style encoder-decoder with pruning heads.
- A rollout training script for `12` autoregressive steps without backpropagation through time.
- Paper-style synthetic measurement augmentations.
- A toy packed-trajectory generator for local sanity checks when no real dataset is available.

## What Is Paper-Faithful Here

- Spatial grid: `3.2 m x 3.2 m x 3.2 m`, resolution `0.05 m`, shape `64 x 64 x 64`.
- Voxel feature: centroid offset inside the voxel, stored in `[0, 1)`.
- Time index: `k = 0` for current measurement, `k = 1` for previous prediction.
- Encoder downsampling: `4` spatial stride-2 stages, temporal axis preserved in stride.
- Decoder pruning: occupancy logits per decoder stage, threshold default `alpha = 0.5`.
- Training rollout: `12` steps, Adam, exponential decay from `1e-2` to `1e-4`.

## Assumptions You Should Know

- The paper body does not spell out every channel width in text, so this repo uses a reasonable 4-level sparse U-Net:
  - stem `32`
  - encoder `64, 128, 256, 256`
  - decoder `256, 128, 64, 32`
- The expected dataset here is a packed trajectory format instead of the original Isaac Gym generator.
- Measurements and ground-truth point clouds are assumed to already be expressed in the robot-local frame for each timestep. Poses are used to align the previous prediction into the current frame during rollout.

## Dataset Format

Each trajectory file under `--data-root` should be a `.npz` with:

- `poses`: shape `[T, 4, 4]`
- `measurement_points`: shape `[N_measurement, 3]`
- `measurement_splits`: shape `[T + 1]`
- `ground_truth_points`: shape `[N_ground_truth, 3]`
- `ground_truth_splits`: shape `[T + 1]`

The packed arrays are sliced as:

```text
points_t = packed_points[splits[t] : splits[t + 1]]
```

## Train

```bash
python train.py --data-root path/to/trajectories
```

Useful flags:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
  --data-root path/to/trajectories \
  --output-dir runs/test2 \
  --epochs 50 \
  --batch-size 2 \
  --sequence-length 12 \
  --learning-rate 1e-2 \
  --min-learning-rate 1e-4 \
  --num-workers 8 \
  --amp \
  --val-fraction 0 \
  --print-timing \
  --cache-dataset 2>&1 | tee runs2.log &
```

## Mock Data

If you only want to validate the data format and training entrypoint locally, generate toy trajectories first:

```bash
python make_mock_dataset.py --output-dir mock_dataset
```

Then point the training script at that folder.

## Isaac Sim 4.5 Data Collection

For the dataset collection stage there is now a standalone script:

```bash
./python.sh collect_isaacsim_dataset.py \
  --output-dir data/isaacsim_paper_style \
  --num-trajectories 64 \
  --timesteps 48 \
  --motion-source policy_rollout \
  --policy-checkpoint path/to/exported/policy.pt
```

What it does:

- Builds a [`DataCollect.py`](DataCollect.py)-aligned scene: four walls, two mirrored staircases, `30` boxes, and `10` cylindrical poles in a `20 m` corridor.
- Samples one global scene ratio per trajectory and maps it jointly into the paper ranges for corridor width, stair tread/rise, and box size.
- Uses the fixed ANYmal C `front / back / left / right` depth-camera rig with `30°` downward tilt and explicit `RealSense D435-like` pinhole intrinsics.
- Uses dense sampling from the live simulator scene mesh as ground truth.
- Writes packed `.npz` trajectories in the exact format expected by `train.py`.
- Can optionally save each camera's robot-local point cloud alongside the fused measurement cloud for later multi-view experiments.

Useful flags:

```bash
./python.sh collect_isaacsim_dataset.py \
  --output-dir data/isaacsim_paper_style \
  --motion-source policy_rollout \
  --policy-checkpoint path/to/exported/policy.pt \
  --policy-device cuda:0 \
  --save-raw-camera-clouds \
  --show-ui \
  --show-robot
```

Notes:

- `--motion-source policy_rollout` is now the paper-faithful default path. It expects an exported Isaac Lab TorchScript/JIT checkpoint.
- `--motion-source velocity_fallback` is still available when you need a format-compatible dataset without a policy checkpoint.
- In `policy_rollout` mode the collector loads the DataCollect-aligned USD scene into Isaac Lab, drives ANYmal C with the policy, and samples GT from the loaded scene mesh rather than from camera observations.
- If you use `velocity_fallback`, `--robot-usd` or `--asset-root` may still be needed to resolve the ANYmal C USD asset.
- `--path-length`, `--wall-height`, `--wall-thickness`, and camera tilt are now locked to the DataCollect / paper-aligned values and will error if changed.
- See [`docs/isaaclab_reproduction_baseline.md`](docs/isaaclab_reproduction_baseline.md) for the strict paper-alignment notes and fallback tradeoffs.

## Open3D Visualization

To inspect the collected point clouds interactively, use:

```bash
python visualize_open3d_trajectory.py data/trajectory_000.npz
```

Useful examples:

```bash
# Show timestep 0 in the robot-local frame.
python visualize_open3d_trajectory.py data/trajectory_000.npz --timestep 0 --space local --show-bounds

# Show the full trajectory accumulated in the world frame.
python visualize_open3d_trajectory.py data/trajectory_000.npz --timestep all --space world --show-poses

# Show only measurement points.
python visualize_open3d_trajectory.py data/trajectory_000.npz --hide-ground-truth
```

Notes:

- `open3d` is optional and not listed in the minimal training dependencies. Install it separately with `pip install open3d`.
- `--timestep all` should be used with `--space world`; stacking multiple local robot frames is not meaningful.

## Report Figures

To export paper-style static figures from a trained checkpoint:

```bash
python export_prediction_report.py \
  --checkpoint path/to/best_checkpoint.pt \
  --trajectory data/trajectory_000.npz \
  --timesteps 0 30 \
  --output outputs/fig4_style.png
```

Useful variants:

```bash
# Let the script pick two evenly spaced columns automatically.
python export_prediction_report.py \
  --checkpoint path/to/best_checkpoint.pt \
  --trajectory data/trajectory_000.npz \
  --output outputs/report_auto.png

# Disable autoregressive feedback to mimic a zero-shot current-frame estimate.
python export_prediction_report.py \
  --checkpoint path/to/best_checkpoint.pt \
  --trajectory data/trajectory_000.npz \
  --timesteps 12 \
  --disable-feedback \
  --output outputs/fig5_style.png

# Save the full predicted rollout for later inspection.
python export_prediction_report.py \
  --checkpoint path/to/best_checkpoint.pt \
  --trajectory data/trajectory_000.npz \
  --output outputs/report.png \
  --save-rollout outputs/trajectory_000_predictions.npz
```

Notes:

- The figure layout follows the paper-style comparison of `Measurement / Reconstruction / Ground truth`.
- `matplotlib` is required for figure export and is optional relative to the minimal training dependencies.

## Dependencies

Minimum Python packages:

```text
numpy
torch
MinkowskiEngine
```

`MinkowskiEngine` is usually easiest to install on Linux with CUDA. If you are on Windows, using WSL2 is often the practical path.

## Important Limitation

In the current environment, `torch` is available but `MinkowskiEngine` is not, so this session could only verify syntax and file structure, not execute the sparse model end-to-end.
