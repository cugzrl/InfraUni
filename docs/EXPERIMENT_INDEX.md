# 实验索引

有效权重都保留，不覆盖。代码版本以仓库提交为准，这一轮改动在分支 `resolve-l-baseline`。

## 当前对照

| 实验 | 权重 | 说明 |
| --- | --- | --- |
| TransFusion-LC mini，1 epoch | `experiments/transfusion_voxel_LC_mini/epoch_1.pth` | 联合模型初始化。mAP 0.530，NDS 0.508 |
| 独立预测头 B，2 epoch | `experiments/ablation_b_mini/epoch_2.pt` | 检测仍是 LC。EPA 0.272，minADE 0.972，minFDE 1.612，MR 0.269 |
| 统一 `u0`，2 epoch | `experiments/unified_u0_mini/epoch_2.pt` | 类别门匹配。最终 mAP 0.456，NDS 0.418，EPA -2.218，minADE 1.021，minFDE 1.654，MR 0.256 |
| 统一 `u_nofore`，2 epoch | `experiments/unified_nofore_mini/epoch_2.pt` | 预测损失为 0。最终 mAP 0.486，NDS 0.430 |
| 统一 `u_causal`，2 epoch | `experiments/unified_causal_mini/epoch_2.pt` | t=0 不读未来。最终 mAP 0.456，NDS 0.419，minADE 0.987 |
| 统一 `u_cost`，2 epoch | `experiments/unified_ucost_mini/epoch_2.pt` | 代价匹配。最终 mAP 0.465，NDS 0.427，EPA -1.314，minADE 1.033，minFDE 1.720，MR 0.261 |

指标文件是各目录下的 `metrics_e02.pkl`，协议 V3。

## 仍可复现的旧权重

| 实验 | 权重 | 不要使用的文件 |
| --- | --- | --- |
| 串联 C1 / C2 / C3 | `experiments/ablation_c1_mini/epoch_2.pt` 等 | — |
| D1，15 epoch | `experiments/ablation_d1_mini/epoch_15.pt` | 轮次和上面的 2 epoch 不同，不能直接比 |
| D5 | `experiments/ablation_d5_mini/epoch_2.pt` | 第 3 个 epoch 开始发散 |
| F0 | `experiments/ablation_f0_mini/epoch_5.pt` | `epoch_6.pt` 和 `epoch_15.pt` 含非有限参数 |
| F0 诊断 | `experiments/ablation_f0_diag_mini/epoch_6.pt` | 新优化器，不能当作原来的第 6 个 epoch |

更早的表格在 `docs/archive/`。

## RESOLVE

单序列 LiDAR info 在 `/home/kemove/devdata1/zrl/dataset/RESOLVE_lidar_smoke/v1.0-trainval/`。全量 info 写到 `/home/kemove/devdata1/zrl/dataset/RESOLVE_layout/v1.0-trainval/`。官方检测权重完成后记在 RESOLVE 的 `output/`，tag `infrauni_official`。场景划分 info 的文件名带 `scene`，不用于这次检测基线。
