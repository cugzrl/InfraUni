# TransFusion 官方代码复现记录

本轮只跑官方 TransFusion，没有接入 DeTra，也没有再用 `infrauni/models` 里的自写检测器训练。

## 仓库

- 上游：https://github.com/XuyangBai/TransFusion
- 固定 commit：`73c596f7bd3460c17cbcc58dd9bcc5a0896774a8`（2022-08-21，`Update transfusion_head.py`）
- 干净检出：`third_party/TransFusion`
- 兼容分支：`compat/rtx4090-bevfusion`，补丁提交 `09a1dd2`

`/home/kemove/devdata1/zrl/TransFusion` 不是这次使用的副本。那个工作区相对官方 commit 还有未提交修改（检测头、检测器、KITTI coder、KITTI 数据集，以及两份未跟踪的 DAIR / KITTI 配置）。`third_party/TransFusion` 从该 commit 重新检出，不包含这些改动。

## 环境

使用已有的 `bevfusion` 环境，没有新建环境，也没有改写 Sparse Encoder、SECOND、FPN 或检测头。

| 组件 | 版本 |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.10.0+cu113 |
| MMCV | 1.4.0 |
| MMDetection | 2.20.0 |
| 仓库内 MMDetection3D | 0.11.0 |
| 编译 CUDA | 11.8，`sm_86` |
| GPU | 空闲的 RTX 4090，`CUDA_VISIBLE_DEVICES=1` |

官方 README 写的是 PyTorch 1.7 和 CUDA 10.1。当前 PyTorch 1.10 不接受 `sm_89`，系统默认的 CUDA 12 又会在扩展编译时因主版本不一致直接失败，所以扩展用 CUDA 11.8 编成 `sm_86`。`sm_86` 的 cubin 可以在 RTX 4090 上运行。

另外安装了官方导入链需要、而该环境原先没有的 `lyft-dataset-sdk`、`trimesh==2.35.39` 和 `networkx`。没有按 `requirements/runtime.txt` 把 NumPy 降到 1.20 以下，避免破坏现有 PyTorch。

## 源码差异

只改了三处，都在 `third_party/TransFusion` 的 `09a1dd2`：

1. `mmdet3d/ops/voxel/src/scatter_points_cuda.cu`：`index_put_` 改为传入索引列表。CUDA 11.8 下旧的单 Tensor 重载无法编译，计算仍是把排序后的体素 id 写回 map。
2. `mmdet3d/models/builder.py`：MMDetection 2.20 去掉了模块级 `build()`，改为调用 `registry.build`。
3. `mmdet3d/models/detectors/mvx_two_stage.py`：`BaseModule.init_weights()` 不再接收 `pretrained`。图像 backbone 若签名不匹配，则退回无参数的 `init_weights()`。

模型结构、损失、数据 pipeline 和 nuScenes 评估没有改。

## 数据

原始 nuScenes-mini 仍在 `/home/kemove/devdata1/zrl/bevfusion/data/nuscenes`。`third_party/TransFusion/data/nuscenes` 里用软链接指向 `samples`、`sweeps`、`maps` 和 `v1.0-mini`，没有复制点云和图像。

infos 用官方 `nuscenes_converter.create_nuscenes_infos` 生成，写在检出目录内，没有覆盖 BEVFusion 自己的 pkl：

- train 323，val 81，`metadata.version = v1.0-mini`
- 每帧 6 个相机，含 `cam_intrinsic`
- 最多 10 个 sweep

GT database 复用同一份 mini 数据里已经生成的 `nuscenes_gt_database` 和 `nuscenes_dbinfos_train.pkl`。官方配置里 sampler 的 `data_root=None`，数据库路径相对仓库根目录，因此在检出目录下加了同名软链接。

## 权重

官方模型表里的 Google Drive 链接打开后是提交文件，不是网络权重：

- `1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN`：`transfusionL_submission.json`，约 339 MB
- `1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W`：`transfusion_submission.json`，约 339 MB

README 也写明因华为 IAS BU 的政策不发布预训练模型。本机没有 `fusion_voxel0075_R50.pth`。因此 LiDAR 模型从随机初始化做 mini smoke test，LC 没有开始训练。

## 相对官方配置的运行参数

配置文件只覆盖 batch、worker、epoch、日志和输出目录。网络、pipeline 和损失仍继承官方文件。

| 项 | 官方 `transfusion_nusc_voxel_L.py` | 本次 mini |
| --- | --- | --- |
| 数据 | 完整 nuScenes | v1.0-mini |
| GPU × batch | 8 × 2 | 1 × 1 |
| epoch | 20 | 1 |
| 学习率 | 1e-4，按 16 的总 batch 设定 | 仍为 1e-4，没有按单卡缩小 |
| 日志 | 每 50 iter，含 TensorBoard | 每 20 iter，只保留文本日志 |

LC 配置 `configs/transfusion_mini_voxel_LC.py` 保留 `load_from = checkpoints/fusion_voxel0075_R50.pth`。该文件不存在时，`load_checkpoint` 抛出 `OSError: ... is not a checkpoint file`，不会静默改成随机初始化再训练。

## TransFusion-L

```bash
cd /home/kemove/devdata1/zrl/InfraUni
bash scripts/train_transfusion_l.sh
bash scripts/test_transfusion_l.sh experiments/transfusion_voxel_L_mini/epoch_1.pth
```

等价命令：

```bash
cd /home/kemove/devdata1/zrl/InfraUni/third_party/TransFusion
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python
$PY tools/train.py /home/kemove/devdata1/zrl/InfraUni/configs/transfusion_mini_voxel_L.py --gpus 1
$PY tools/test.py /home/kemove/devdata1/zrl/InfraUni/configs/transfusion_mini_voxel_L.py \
  /home/kemove/devdata1/zrl/InfraUni/experiments/transfusion_voxel_L_mini/epoch_1.pth --eval bbox
```

一次 epoch 的实际规模：

- 原始 train infos：323
- `CBGSDataset` 之后、batch 为 1：1690 iteration
- 稳定阶段约 0.21 秒 / iteration，这一 epoch 的训练大约 6 分钟
- 验证 81 帧，推理约 7.3 帧 / 秒
- 日志中的 `memory` 约 2805 MB，这是 PyTorch `max_memory_allocated`。同一时刻 `nvidia-smi` 上该卡大约占用 18 GB
- 总损失从第 20 iter 的 635.4（heatmap 617.4）降到第 1680 iter 的 6.45（heatmap 2.08）
- checkpoint：`experiments/transfusion_voxel_L_mini/epoch_1.pth`，`latest.pth` 指向它

官方 devkit 在 `mini_val` 上，训练结束时的验证和随后 `tools/test.py` 重新加载同一权重，结果一致：

| 指标 | 数值 |
| --- | ---: |
| mAP | 0.0197 |
| NDS | 0.0734 |
| mATE | 0.860 |
| mASE | 0.895 |
| mAOE | 1.208 |
| mAVE | 0.838 |
| mAAE | 0.772 |
| car AP | 0.100 |
| pedestrian AP | 0.097 |

这是 mini 上从零训练 1 个 epoch 的 smoke test，不是论文在完整 nuScenes 上的 65.1 mAP / 70.1 NDS。

验证集第 0、10、20 帧上，分数最高的 5 个框中心到最近 LiDAR 点的距离都在 0.01 m 到 0.22 m。框落在点云上。1 个 epoch 后大多数分数仍低于 0.1，所以按 0.1 过滤后可见框很少。BEV 图在 `experiments/transfusion_voxel_L_mini/vis/`。

## TransFusion-LC

官方配置可以建出完整模型：`TransFusionDetector`，`fuse_img=True`，图像分支 175 组参数，LiDAR 分支 399 组参数。验证集第 0 帧通过官方 pipeline 读入的图像是 `(6, 3, 448, 800)`，点云是 `(34752, 5)`，`lidar2img` 有 6 个。

`checkpoints/fusion_voxel0075_R50.pth` 不存在，因此没有启动 LC 训练，也没有把随机初始化的图像 FPN 当成已经复现的融合模型。论文里的 LC 是在预训练 LiDAR backbone 上再训 6 个 epoch，并冻结 LiDAR 分量。没有这份初始化权重时，不能进行这一阶段。

## 还没做的事

- 没有公开的检测权重可做推理对照。
- LC 训练被缺少的 `load_from` 挡住。
- mini 上 1 个 epoch 只说明官方训练、保存、重载和 devkit 评估是通的。
- 自写的 `infrauni/models` 不再作为正式检测基线。
