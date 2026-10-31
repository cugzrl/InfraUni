# SparseDrive 训练中断与 loss 排查记录

机器：346-4090。仓库：`/home/kemove/devdata1/zrl/InfraUni`。日期：2026-10-28。

## 结论与实际复现

原训练于17:24在第一轮约第3590批次附近因非有限梯度主动退出，原日志只记录 `FloatingPointError: Non-finite gradient`。没有发现显存不足或机器重启记录。

加入算子级异常追踪后，在第一轮第4078批次再次捕获同类问题：

- 两帧：`rosbag2_2025_12_26-12_31_11_75`、`rosbag2_2026_02_01-11_36_35_95`。
- 当批总 loss 为17.1008008，全部模型权重仍为有限值。
- 内部候选框最大对数尺寸达到59.386，已是完全不合理的巨大尺寸；真实训练框最大长度约17.23米。
- 首个被异常追踪捕获的反向操作为 `DivBackward0`，指向官方仓库 `projects/mmdet3d_plugin/models/blocks.py` 中 `project_points` 的深度除法。

三维采样点先乘相机投影矩阵，再除以深度。巨大的候选框会产生很大的采样点坐标；深度又接近零时，求导中的除法会溢出。这些点在图像外，CUDA正向采样会忽略它们，因此总loss可以正常；反向传播仍经过投影除法，产生NaN。正向loss有限并不保证反向梯度有限。

本轮保存了真实失败批次，重放结果如下：

| 同一批次的运行路径 | 结果 |
|---|---|
| 原始采样和投影路径 | 稳定复现 `DivBackward0` NaN，退出码1 |
| 仅替换投影路径，其余恢复原路径 | 前向、反向及全部参数梯度有限，退出码0 |
| 完整数值保护 | 前向、反向及全部参数梯度有限，退出码0 |

原始日志缺少首个异常张量，不能逐位还原原运行第3590批次的故障；上表是对本轮实际捕获的4078批次进行的可重复因果验证。

## 已部署的修复

中央适配代码在 `infrauni/resolve/sparsedrive/models.py`：

1. `ResolveFeatureAggregation.project_points` 在除法前判断点是否位于相机前方及图像内，对无效点使用安全分子、分母和图像外哨兵坐标。正常可见点的坐标及梯度与原实现逐元素一致。
2. `ResolveBox3DKeyPointsGenerator` 对用于采样的对数尺寸限制到[-20,20]，避免指数溢出。这一范围约对应2纳米到4.9亿米，所有正常目标都在范围内；原始框回归输出、训练真值和回归损失仍使用原值。
3. 时序缓存中的非有限候选或尺寸超出上述极端范围的候选，重新初始化其缓存框、特征、置信度和缓存ID，阻断无意义候选逐帧传递。

这些保护会改变极端异常候选的取特征和缓存行为，属于本次RESOLVE适配运行时修复。配置生成器也已同步，之后重新生成配置会保留修复。

训练循环在 `tools/train/train_sparsedrive_resolve.py`：

- 分别检查梯度元素和梯度范数；范数以float64累计再裁剪，避免仅在范数计算中发生FP32溢出。这是额外保护，本轮明确捕获的原因是投影除法。
- 每250次优化器更新保存 `recovery.pth`，通过原子替换更新 `latest.pth`。
- 完成一轮后先保存权重，再运行验证，避免验证阶段中断造成整轮权重丢失。
- 恢复时采样器直接跳过已完成批次，不重新解码对应的图像。
- 保留分项loss、裁剪前梯度范数、帧token和采样尺寸范围；出现异常时保留失败批次供独立重放，随后报错退出。
- `avg_since_resume` 表示本次恢复后的累计均值，避免将它误认为从本轮最开始累计的均值。

## loss 为什么先升后降

第一阶段的总loss叠加6层解码器的分类、框回归、中心质量和朝向质量损失，不能直接与另一种模型的总loss比较大小。

原实现保留 `cls_threshold_to_reg=0.05`：只有分类分数超过0.05的匹配框参与回归。开始时参与回归的框很少；随着分类分数提高，更多匹配框被纳入，框回归损失会先增加。

| 原运行的记录点 | 总loss | 分类项合计 | 框回归项合计 |
|---|---:|---:|---:|
| 第1批次 | 13.745 | 13.458 | 0.251 |
| 第120批次 | 41.709 | 8.746 | 28.056 |
| 最后40个日志点均值 | 21.052 | 5.863 | 11.214 |

分类与框回归损失后续均有下降。序列内目标数量和场景变化也会产生波动，因此单批loss不必单调下降。上述窗口来自不同批次，趋势不能代替固定完整验证集的mAP/NDS。第一阶段只有检测任务，预测指标要到第二阶段才计算。

曲线保存在实验目录的 `diagnostics/loss_before_failure.png`。

## 验证和故障证据

实验根目录：`/home/kemove/devdata1/zrl/InfraUni/experiments/sparsedrive_camera_resolve_20261028`。

- `diagnostics/failing_batch_4078.pth`：实际失败时的模型、输入、时序缓存和RNG状态。
- `diagnostics/replay_legacy.log`、`replay_projection-only.log`、`replay_fixed.log`：三组对照。
- `diagnostics/numerical_regression.json`：正常可见投影等价性及极端反向传播检查。
- `diagnostics/integration_after_fix.log`、`data/resolve_sparsedrive/validation.json`：真实数据两阶段前后向、预测损失回传主干、无真值推理、异常缓存恢复检查均通过。
- `diagnostics/console_before_numerical_fix.log`：完整原始及诊断运行日志。
- `diagnostics/train_before_numerical_fix.jsonl`：原运行和诊断阶段的loss记录。

数值回归用例还验证：所有投影值及正向loss均有限时，旧除法仍可能产生非有限梯度；修复后同一输入梯度有限。

训练标签、完整序列划分、mAP/NDS/EPA/minADE/minFDE评估入口和训练轮数保持现有协议。另增加每个阶段第一轮的完整验证，之后仍每5轮及最终轮验证。

## 本次恢复与操作入口

2026-10-28 18:02:25，在第二张4090上使用原 `bevspread` 环境恢复。恢复点为第一阶段第一轮第4000批次、1000次参数更新。正式进程关闭算子异常追踪模式。

```bash
cd /home/kemove/devdata1/zrl/InfraUni
screen -r infrauni_sparsedrive_camera_20261028
# Ctrl-A 后按 D：退出查看，保持训练运行
tail -f experiments/sparsedrive_camera_resolve_20261028/console.log
```

停止后重新启动并自动续训：

```bash
cd /home/kemove/devdata1/zrl/InfraUni
screen -dmS infrauni_sparsedrive_camera_20261028 bash scripts/train_sparsedrive_resolve.sh
```

同名训练已经运行时不要重复启动。

检查点：`stage1/latest.pth`、`stage1/recovery.pth`、`stage1/epoch_*.pth`；第二阶段对应 `stage2/`。每轮结束保存epoch文件，按完整验证mAP保存 `best_map.pth`。

恢复权重、优化器、步数及主进程/CUDA RNG。时序缓存从恢复点重新建立，数据加载进程的增强状态重新初始化，因此断点恢复不是逐位完全相同的重放。

独立重放失败批次：

```bash
cd /home/kemove/devdata1/zrl/InfraUni
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
PYTHONPATH=third_party/SparseDrive/.runtime:$PWD \
/home/kemove/anaconda3/envs/bevspread/bin/python tools/checks/diagnose_sparsedrive_failure.py \
experiments/sparsedrive_camera_resolve_20261028/diagnostics/failing_batch_4078.pth --mode fixed
```

也可将 `fixed` 换成 `legacy` 或 `projection-only` 运行对照；为避免与正式训练争用GPU，建议训练停止时执行。
