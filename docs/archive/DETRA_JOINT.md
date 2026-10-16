# TransFusion-LC 与 DeTra 联合轨迹

检测网络仍是 `third_party/TransFusion` 里的官方 TransFusion-LC。轨迹修正复用 `third_party/MC-DeTra`，没有按论文重写一套注意力。本轮没有路侧知识库，也没有迁移 RESOLVE 或 V2X-Seq。

## 开源代码

MC-DeTra 是 DeTra（ECCV 2024）的第三方复现，另外带有只在训练时使用的运动一致性损失。快照来自 `https://github.com/diuzhevVlad/MC-DeTra`，树 `65a85efe0539d929b788514851e166a7edc779f2`。仓库没有公开权重，数据是 Waymo，本机没有 Waymo，因此没有跑它的完整训练。

直接复用的部分：

- `DeTraMini`：可学习的 mode / time query，初始位姿在模式和时间上复制
- `RefinementBlock`：LiDAR 可变形注意力、Time、Mode、Object 自注意力，每层后面有各自的 LayerNorm 和 FFN
- `PoseUpdate`：当前框残差，以及双向 GRU 的多模态未来轨迹。未来位置是当前中心加偏移，所以轨迹损失会带动当前框
- `wta_forecasting_loss`：按 L1 选择赢家模式，再做回归

`tools/check_detra_core.py` 在不接检测器的情况下跑了 2 个 block。未来轨迹形状是 `[1, 4, 6, 12, 2]`，六个模式在当前时刻的位置相同。`smooth_l1` 赢家损失反传后，BEV 特征、可变形采样偏移和框头都有梯度。

和论文的差异：

- MC-DeTra 的 LiDAR 采样用 `grid_sample`，不是论文里的 CUDA deformable attention
- 默认未来是 Waymo 上 5 秒、10 个点。这里按 MASAR 改成 6 秒、12 个关键帧，加当前时刻后 query 时间长度是 13
- 没有地图分支。MC-DeTra 自己也写明地图编码还没有接到 Waymo 地图
- 修正轮数用 2，论文是 3。短程验证里两轮已经能执行并回传梯度
- `bevfusion` 环境是 Python 3.8。`config.py`、`refinement.py`、`losses.py` 增加了 `from __future__ import annotations`，否则类型注解无法导入。算法没有改

## 和 TransFusion 的接口

官方检测器先正常前向，得到 200 个 query 和 BEV。BEV 来自 SECONDFPN，形状是 `[1, 512, 180, 180]`，对应 0.075 米体素、8 倍下采样。适配器把它投到 128 维，作为 DeTra 的 stride 8，并插值出 stride 4 和 stride 16，只为对齐它的多尺度采样接口。

融合 query 是图像解码器输出和被 detach 的 LiDAR query 拼在一起，通道数 256，再投到 128 维，加到 DeTra 的 mode / time query 上。论文里的 query 只靠初始位姿区分。这里多加了检测 query，是为了让相机融合特征进入修正，而不是只留下一个框中心。

两个版本共用同一个 LC checkpoint `experiments/transfusion_voxel_LC_mini/epoch_1.pth`，检测器参数冻结：

| 版本 | 做法 |
| --- | --- |
| A | 官方 TransFusion-LC，不预测轨迹 |
| B | `pred_head`。query 和当前中心都 detach，MLP 只预测 6 条未来偏移 |
| C | `joint`。`QueryDeTra` 跑 2 轮修正，`PoseUpdate` 用同一条 query 更新当前框和未来轨迹 |

未来监督由 nuScenes 实例 id 串起后续 12 个关键帧，再变回当前 LiDAR 坐标系。场景结束或该实例消失的点掩码为 0。推理不读这些 id。缓存是 `data/nuscenes_mini_future12.pkl`，404 帧里有 8970 个目标具备完整 12 步。联合训练用官方测试 pipeline，没有随机旋转和平移，避免增广之后对不上未来轨迹。

## 梯度

`tools/check_joint_grad.py` 只打开 BEV neck、图像 query decoder 和最后一层预测头，反传轨迹损失：

| 模块 | 有梯度 |
| --- | --- |
| DeTra 框头、LiDAR 注意力、BEV 投影、query 投影 | 是 |
| `pts_neck` | 是 |
| 图像 query decoder | 是 |
| TransFusion 最后一层预测 MLP | 否 |

预测 MLP 没有梯度，是因为送进 DeTra 的框来自 detach 之后的解码结果，query 也取在这个 MLP 之前。轨迹损失会更新检测特征和 DeTra 里的当前框，不会直接改官方预测 MLP。40 步训练本身保持检测器冻结，避免短训练冲掉 LC 基线。

B 的检查里，只有独立预测头有梯度，检测器没有。

## mini 上 40 步的结果

训练 batch 为 1，学习率 1e-4，轨迹损失用 `smooth_l1` 赢家损失，再加匹配框中心的平滑 L1。B 的预测损失从 3.11 降到 2.56，C 从 3.06 降到 1.84。这是训练集上抽到的帧，最后一步样本不同，所以不是严格单调。

`mini_val` 81 帧，检测走官方 devkit，预测按分数不低于 0.1 的框计算。A 是已经记录的 LC 基线。

| 版本 | mAP | NDS | EPA | minADE | minFDE | MR | matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 官方 LC | 0.530 | 0.508 | 无轨迹 | 无轨迹 | 无轨迹 | 无轨迹 | 无轨迹 |
| B 独立预测头 | 0.530 | 0.508 | -0.563 | 1.697 | 2.990 | 0.368 | 1390 |
| C DeTra 联合修正 | 0.526 | 0.501 | -0.558 | 1.638 | 2.993 | 0.367 | 1386 |

B 的检测数字和 A 一致，说明预测头没有改框。C 的 mAP / NDS 略降，minADE 和 EPA 略好，minFDE 基本不动。40 步还看不出联合修正已经优于独立预测头。EPA 为负，是因为 0.1 分以上的误检会按 0.5 的系数扣分。

BEV 图在 `experiments/joint_pred_head_mini/vis/` 和 `experiments/joint_joint_mini/vis/`。每个保留框画了 6 条未来轨迹。

## 还没解决的问题

- 检测器在正式的 40 步里是冻结的。梯度检查说明轨迹损失能够进到 BEV 和图像 query decoder，但还没有做解冻后的联合微调
- 修正只有 2 轮，没有地图，BEV 的 stride 4 和 stride 16 是插值出来的
- 训练没有用官方的随机增广和 CBGS，也没有跑完一个 epoch
- MC-DeTra 没有在 Waymo 上复现论文表格，不能把这次 mini 数字写成 DeTra 的论文结果
