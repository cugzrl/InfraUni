# 第二阶段：DeTra 式联合轨迹

DeTra 原文用 LiDAR 和 HD Map。这里保留统一轨迹、分解注意力和多轮修正，传感器换成 TransFusion 的 LiDAR BEV 与六路相机特征。没有加入地图。

## 表示

每个检测 query 展开成 6 个模式、13 个时刻。时刻 0 是检测中心，后面 12 个时刻是相对这个中心的未来偏移，2 Hz，共 6 秒。因此任意模式在当前时刻的位置都等于最终输出的检测中心。

## 每一轮 RefineBlock

1. Object、Time、Mode 三路注意力，再接前馈网络。
2. 在 SECOND 的两级 LiDAR 特征上，用可变形注意力读取当前中心附近的特征。
3. 把每个路点投影到六路相机，在 FPN 的 stride 4 和 stride 8 上采样。投影前会撤销三维增广。
4. 用更新后的 query 预测中心位移和未来偏移。上一轮位姿停止梯度，和 DeTra 的 block 间 stop-grad 一样。

默认两轮。`DetraDecoder.executed_blocks` 会记录实际执行次数。

## 三种开关

| mode | 行为 |
| --- | --- |
| det | 只训练 TransFusion |
| pred_head | 在检测 query 上接一次性 MLP。未来偏移加在被 detach 的检测中心上，预测损失不直接移动检测框 |
| joint | 两轮 refinement。未来损失通过“未来 = 当前中心 + 偏移”直接更新检测中心 |

单帧检查结果：

- joint 执行了 2 个 block
- 轨迹时刻 0 与检测中心一致
- 只反传轨迹损失时，joint 的检测中心有梯度；pred_head 没有
- joint 的检测头、LiDAR 骨干和 DeTra 参数都有梯度

这只说明耦合方向是对的。下面的数字来自同一套推理脚本，权重分别是检测基线、在该基线上再训 2 个 epoch 的预测头，以及再训 2 个 epoch 的联合解码器。

## mini 对照

正式协议使用分数不低于 0.1 的框。三个权重的最高分都低于这个阈值，所以 EPA、minADE、minFDE、MR 在协议内没有样本，`matched` 为 0。检测仍把范围内的全部框交给官方评估器。

| 模式 | mAP | NDS | 协议内 matched |
| --- | ---: | ---: | ---: |
| det | 0.000 | 0.016 | 0 |
| pred_head | 0.000 | 0.004 | 0 |
| joint | 0.000 | 0.000 | 0 |

pred_head 的 NDS 来自行人上的少量真阳性（mATE 1.342）。joint 两个 epoch 后，各类别都没有进入官方真阳性统计。

为了看到轨迹头是否学到位移，另外统计了不设分数阈值的诊断量。它不是论文表格里的数字：200 个 query 里的低分框会全部算成假阳性，EPA 会被假阳性主导。

| 模式 | 诊断 matched | minADE | minFDE | MR | EPA |
| --- | ---: | ---: | ---: | ---: | ---: |
| det，静止外推 | 113 | 2.47 | 3.78 | 0.239 | -1.58 |
| pred_head | 174 | 2.64 | 3.49 | 0.489 | -1.56 |
| joint | 30 | 3.90 | 6.60 | 0.867 | -107.86 |

预测头比静止外推匹配到更多目标，终点误差略低，但误检率更高。两轮联合修正在这个短训练里把当前中心推离了真值，匹配数从 113 降到 30，检测 NDS 也降到 0。因此目前不能把联合训练写成相对检测基线的收益。

对应文件：

- `experiments/mini_det/infer/metrics.json`
- `experiments/mini_pred_head/infer/metrics.json`
- `experiments/mini_joint/infer/metrics.json`

## 命令

检测训练结束后，用同一检测权重分别接预测头和联合解码器：

```bash
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python
$PY tools/train.py --config configs/mini.yaml --mode pred_head --epochs 2 --init experiments/mini_det/last.pt
$PY tools/train.py --config configs/mini.yaml --mode joint --epochs 2 --init experiments/mini_det/last.pt
$PY tools/infer.py --config configs/mini.yaml --checkpoint experiments/mini_pred_head/last.pt
$PY tools/infer.py --config configs/mini.yaml --checkpoint experiments/mini_joint/last.pt
```

`tools/infer.py` 从 checkpoint 读取 `mode` 来建模型。`--mode` 只在需要覆盖时使用。
