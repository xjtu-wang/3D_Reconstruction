# 《Neural Scene Representation for Locomotion on Structured Terrain》复现

本仓库是对以下论文的工程复现，而不是官方实现：

`Hoeller et al., "Neural Scene Representation for Locomotion on Structured Terrain", IEEE RA-L 2022`

当前仓库已经打通了下面这条主线：

- 用 IsaacLab + ANYmal C 采集机器人局部 `measurement / ground truth` 点云轨迹
- 将点云预处理为论文使用的 `64 x 64 x 64` 机器人局部稀疏体素网格
- 使用 MinkowskiEngine 训练 4D sparse U-Net 风格的地形重建模型
- 对保存的 checkpoints 做离线验证，并导出 `best.pt`
- 用 Open3D 和静态报告脚本检查轨迹与预测结果

## 复现范围

当前实现重点覆盖了论文中的核心表示、训练和数据流程：

- 局部空间范围：`3.2 m x 3.2 m x 3.2 m`
- 体素分辨率：`0.05 m`
- 体素网格大小：`64 x 64 x 64`
- 时序输入组织：当前观测和上一帧预测通过 `(x, y, z, k)` 的 4D 稀疏坐标拼接
- 模型形式：基于 MinkowskiEngine 的 4D encoder-decoder，并带 occupancy pruning 头
- 训练方式：`12` 步自回归 rollout，不做跨时间反向传播
- 数据采集：IsaacLab 中的 ANYmal C rough-terrain policy rollout，输出 robot-local measurement / ground truth 点云

和论文保持一致或尽量对齐的部分：

- 机器人局部稀疏体素表示
- 4D 稀疏卷积网络输入形式
- 逐步 rollout 训练方式
- 测量点云与上一帧预测融合的训练接口

本仓库的工程化取舍：

- 数据来源不是原论文的 Isaac Gym 生成链路，而是 IsaacLab 版本的复现采集链路
- 场景为 DataCollect 风格走廊、楼梯、箱体、圆柱障碍物
- 批量采集主线使用单进程反复 `reset`，而不是多进程 supervisor

## 仓库结构

当前维护的主线文件如下：

- `main.py`
  训练入口
- `src/`
  模型、损失、几何、预处理、数据集定义
- `isaaclab_datacollect_anymal_batch.py`
  大规模数据采集入口
- `isaaclab_datacollect_anymal_sequential.py`
  批量采集的实际实现；`batch.py` 会转发到这里
- `isaaclab_datacollect_anymal_rollout.py`
  IsaacLab 单条 rollout 调试入口，同时也是批量采集复用的共享运行时模块
- `isaaclab_datacollect_anymal_rollout_example.py`
  面向初学者的单条 rollout 薄封装示例
- `prepare_scene_split.py`
  按 `scene_seed` 做 train/val 划分
- `evaluate_saved_checkpoints.py`
  离线验证 checkpoints，并导出 `best.pt`
- `visualize_open3d_trajectory.py`
  轨迹点云可视化
- `export_prediction_report.py`
  导出论文风格的静态对比图
- `make_mock_dataset.py`
  生成 toy 数据，仅用于本地格式联调

## 环境依赖

最小 Python 依赖见 `requirements.txt`：

```text
numpy
torch
MinkowskiEngine
```

额外依赖说明：

- 采集数据需要 IsaacLab / Isaac Sim 环境
- 采集 rough-terrain 轨迹需要可直接 `torch.jit.load(...)` 的 ANYmal C policy checkpoint
- 交互式轨迹查看需要 `open3d`
- 静态报告导出需要 `matplotlib`

一般建议：

- Linux + CUDA + MinkowskiEngine
- 如果本机是 Windows，训练部分更适合在 WSL2 或 Linux 机器上运行

## 数据格式

每条轨迹保存为一个 `.npz` 文件，训练脚本默认读取如下字段：

- `poses`: `[T, 4, 4]`
- `measurement_points`: `[N_measurement, 3]`
- `measurement_splits`: `[T + 1]`
- `ground_truth_points`: `[N_ground_truth, 3]`
- `ground_truth_splits`: `[T + 1]`

打包点云的切片方式为：

```text
points_t = packed_points[splits[t] : splits[t + 1]]
```

默认约定：

- `measurement_points` 和 `ground_truth_points` 都在每个时刻对应的 robot-local 坐标系下
- `poses` 用于在 rollout 训练中把上一帧预测对齐到当前帧

## 数据采集

### 1. 批量采集主线

当前推荐的大规模采集入口是 `isaaclab_datacollect_anymal_batch.py`。

它对外提供稳定 CLI，但内部会转发到 `isaaclab_datacollect_anymal_sequential.py`，通过“单个 Isaac Sim 进程 + 反复 reset”的方式连续采轨迹。这是当前仓库实际跑通的主线。

将下面命令中的路径替换成你自己的环境：

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/3D_Reconstruction/isaaclab_datacollect_anymal_batch.py \
  --policy-checkpoint /path/to/policy.pt \
  --dataset-dir /path/to/3D_Reconstruction/data/datacollect_anymal_seq_dataset \
  --target-trajectories 800 \
  --steps 250 \
  --num-envs 8 \
  --headless
```

常用输出：

- `trajectory_*.npz`
- `manifest.jsonl`
- `batch_summary.json`
- 可选的 `scenes/` USD 导出

### 2. 单条 rollout 调试

如果你先想检查场景、相机、measurement 和 ground truth 是否正常，可以先跑单条 rollout：

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/3D_Reconstruction/isaaclab_datacollect_anymal_rollout.py \
  --policy-checkpoint /path/to/policy.pt \
  --output-dir /path/to/3D_Reconstruction/runs/debug_rollout \
  --steps 250 \
  --headless
```

说明：

- `isaaclab_datacollect_anymal_rollout.py` 不是当前大规模采集入口
- 但它仍然是当前仓库的共享运行时模块，`sequential.py` 会直接复用其中的场景生成、相机和点云导出逻辑

## 数据集划分

推荐按 `scene_seed` 做 train/val 划分，而不是按窗口随机拆分。这样可以避免同一场景的相邻轨迹同时落到训练集和验证集，减少泄漏。

基础用法如下：

```bash
python prepare_scene_split.py \
  --input-root /path/to/3D_Reconstruction/data/datacollect_anymal_seq_dataset \
  --output-root /path/to/3D_Reconstruction/data/datacollect_anymal_seq_split \
  --group-by scene_seed \
  --val-fraction 0.1 \
  --sequence-length 12 \
  --sequence-stride 1 \
  --seed 7
```

如果你的数据分散在多个目录，可以重复传入 `--input-root`。

输出目录结构：

- `train/`
- `val/`
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `split_summary.json`

默认行为：

- `prepare_scene_split.py` 默认使用 `hardlink`
- 如果当前文件系统不支持 hardlink，可以显式传 `--link-mode copy`

`split_summary.json` 会同时记录按以下口径统计的验证集比例：

- `scene_groups`
- `trajectories`
- `windows`

由于划分单位是 scene group，这三个比例不一定都恰好等于 `0.1`。

## 训练

推荐的训练流程是：

1. 主训练过程不做在线验证
2. 周期性保存 `step_*.pt`
3. 训练结束后离线验证所有保存的 checkpoints
4. 从离线验证结果中导出 `best.pt`

训练命令示例：

```bash
cd /path/to/3D_Reconstruction
CUDA_VISIBLE_DEVICES=<gpu_id> nohup python -u main.py \
  --data-root /path/to/3D_Reconstruction/data/datacollect_anymal_seq_split/train \
  --output-dir /path/to/3D_Reconstruction/runs/nsr_datacollect_scene_split \
  --epochs 5 \
  --batch-size 128 \
  --sequence-length 12 \
  --sequence-stride 1 \
  --learning-rate 1e-2 \
  --min-learning-rate 1e-4 \
  --num-workers 8 \
  --amp \
  --print-timing \
  --save-every 0 \
  --save-every-steps 1000 \
  --disable-validation-during-training \
  > /path/to/3D_Reconstruction/log/train_nsr_datacollect_scene_split.log 2>&1 &
```

训练阶段典型产物：

- `step_*.pt`
- `last.pt`
- 训练日志

## 离线验证与最佳模型导出

训练完成后，使用离线脚本统一验证保存过的 checkpoints：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python evaluate_saved_checkpoints.py \
  --checkpoint-dir /path/to/3D_Reconstruction/runs/nsr_datacollect_scene_split \
  --val-root /path/to/3D_Reconstruction/data/datacollect_anymal_seq_split/val \
  --val-batch-size 32 \
  --num-workers 8 \
  --amp
```

典型输出：

- `best.pt`
- `checkpoint_validation_summary.json`

说明：

- `main.py` 仍然支持训练过程中在线验证
- 但对于当前这条 IsaacLab 数据主线，离线验证通常更稳定，也更容易控制显存

## 可视化与结果导出

### 1. Open3D 轨迹可视化

查看单条采集轨迹：

```bash
python visualize_open3d_trajectory.py data/trajectory_000.npz
```

常见用法：

```bash
# 查看第 0 帧的 robot-local 点云
python visualize_open3d_trajectory.py data/trajectory_000.npz --timestep 0 --space local --show-bounds

# 将整条轨迹累积到 world frame 查看
python visualize_open3d_trajectory.py data/trajectory_000.npz --timestep all --space world --show-poses

# 只看 measurement
python visualize_open3d_trajectory.py data/trajectory_000.npz --hide-ground-truth
```

注意：

- `open3d` 不是最小训练依赖的一部分，需要单独安装
- `--timestep all` 应该和 `--space world` 配合使用；多个 local frame 直接堆叠没有物理意义

### 2. 论文风格静态报告图

从训练好的 checkpoint 导出静态对比图：

```bash
python export_prediction_report.py \
  --checkpoint /path/to/best.pt \
  --trajectory data/trajectory_000.npz \
  --timesteps 0 30 \
  --output outputs/report.png
```

常见变体：

```bash
# 自动挑选两个时刻
python export_prediction_report.py \
  --checkpoint /path/to/best.pt \
  --trajectory data/trajectory_000.npz \
  --output outputs/report_auto.png

# 关闭上一帧预测反馈，只看当前帧估计
python export_prediction_report.py \
  --checkpoint /path/to/best.pt \
  --trajectory data/trajectory_000.npz \
  --timesteps 12 \
  --disable-feedback \
  --output outputs/report_no_feedback.png

# 额外保存整条 rollout 的预测结果
python export_prediction_report.py \
  --checkpoint /path/to/best.pt \
  --trajectory data/trajectory_000.npz \
  --output outputs/report.png \
  --save-rollout outputs/trajectory_000_predictions.npz
```

## Toy 数据联调

如果你只是想先验证数据格式和训练入口，可以先生成 toy 数据：

```bash
python make_mock_dataset.py --output-dir mock_dataset
```

然后把训练脚本的 `--data-root` 指向该目录即可。

## 已知限制

- 当前仓库是复现工程，不是论文作者发布的官方代码
- 数据采集部分依赖外部 IsaacLab / Isaac Sim 安装，以及可导出的 ANYmal C rough-terrain policy
- 本会话环境里可以检查代码结构和语法，但如果没有 MinkowskiEngine，就无法在当前环境完整执行稀疏卷积训练链路
