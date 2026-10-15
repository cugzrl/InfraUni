# 第一阶段：TransFusion 检测基线

## 实现

运行环境是本机 `bevfusion`（PyTorch 1.10 + CUDA 11.3，已编译的 sparse conv）。不要用 `infrauni` 那个 OpenPCDet 环境，它没有 mmcv。

LiDAR 使用 BEVFusion 里已经编译好的模块，结构与 TransFusion-L 的 voxel 配置一致：

- 0.075 m 硬体素，体素内 5 维特征取均值，最多 10 帧 sweep
- Sparse Encoder，basic block，稀疏尺寸 `[1440, 1440, 41]`
- SECOND 与 SECONDFPN，输出 512 维 BEV

相机是 ImageNet ResNet-50 加 FPN，六路一起输入，分辨率 448×800。检测头是官方 `TransFusionHead`（`fuse_img=True`，200 个 query，热力图局部极大值，一层 LiDAR decoder，再按相机做高斯掩码融合）。

类别顺序、Hungarian 匹配、Focal、Gaussian Focal、带速度的 L1，以及 code 权重，都按 `transfusion_nusc_voxel_LC.py`。

## 和论文及本地官方代码的差异

- 磁盘上没有 `fusion_voxel0075_R50.pth`。ResNet-50 用 ImageNet 权重并冻结；FPN 从随机初始化开始训练。官方 `freeze_img` 会把 FPN 也冻住，那是因为他们加载了已经训练好的融合权重。
- BEVFusion 的 sparse conv 使用 `[X, Y, Z]`，第三段 padding 是 `[1, 1, 0]`。官方仓库写的是 Z 在前。两边下采样后的 BEV 都是 180×180。
- 本地 TransFusion 在 batch 为 1 时跳过了三维增广的逆变换。训练就是 batch 1，这里改成始终把 query 变回原始 LiDAR 系再投影。
- 本地检测头会把没投到任何相机上的 query 损失乘零。这些 query 实际保留的是 LiDAR 预测，这里仍对它们计算检测损失。
- mini 上从零训练，不追求完整 nuScenes trainval 的论文数字。

## mini 验证结果

从零训练约 12 个 epoch，batch 为 1，没有融合预训练权重。官方 devkit 在 `mini_val` 上给出 `experiments/mini_det/infer/metrics.json`：

- mAP = 0.000
- NDS = 0.016
- car 的 mATE = 1.373 m，mASE = 0.735，mAOE = 1.568，mAVE = 0.714，mAAE = 0.223

mAP 为 0，是因为召回还没进入 devkit 的积分区间（min recall 0.1 且 min precision 0.1）。car 已经有匹配上的真阳性，所以 NDS 和 mATE 不是空值。其余类别在这次评估里没有进入真阳性统计，误差项保持 devkit 的空匹配默认值 1。热力图峰值仍低于 0.1，检测分数整体低于预测协议的 0.1 阈值。这是 mini 从零训练的结果，不是完整 nuScenes 的论文数字。

## 已核对的行为

- 验证集第一帧 21 个真值框全部能投进至少一路相机。图像为 `[6, 3, 448, 800]`。
- 场景起始帧没有历史 sweep，时间通道为 0；有历史的帧时间均值约 0.23 s，最大约 0.5 s。
- 检测损失可以回传到检测头和 LiDAR 骨干。单步前向加反向可以在空闲的 RTX 4090 上完成。

## 命令

```bash
cd /home/kemove/devdata1/zrl/InfraUni
/home/kemove/anaconda3/envs/bevfusion/bin/python tools/build_infos.py
/home/kemove/anaconda3/envs/bevfusion/bin/python tools/train.py --config configs/mini.yaml --mode det --epochs 3
/home/kemove/anaconda3/envs/bevfusion/bin/python tools/infer.py --config configs/mini.yaml --checkpoint experiments/mini_det/last.pt
```

`tools/infer.py` 会调用官方 devkit，写出 mAP 和 NDS。
