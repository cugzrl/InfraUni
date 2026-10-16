# TransFusion 候选 + DeTra 统一解码

最终检测不再使用 TransFusion 的类别和分数。TransFusion-LC 继续负责多模态特征和初始候选，DeTra 的最后一轮 Refinement 同时给出类别、存在置信度、三维框和未来轨迹。这些输出共用同一个 Query 下标。

## 和串联结构的差别

旧的 B、C、D、F 路径仍保留在 `JointTransFusionDeTra`。那条路径会改写检测头最后一层的 `forward`，并用 TransFusion 的类别和分数发布检测，DeTra 只替换框的一部分坐标。F0 的有限权重仍是 `experiments/ablation_f0_mini/epoch_5.pt`。正在进行的诊断只把第 6 个 epoch 写到 `experiments/ablation_f0_diag_mini`，用来确认非有限梯度会不会再次出现。旧结构的 15 epoch 复训已经停下，避免占住 GPU。

`QueryBank` 和 `MotionPrior` 还在 `infrauni/joint/memory.py`，统一解码的第一轮训练不调用它们。

## 数据流

`infrauni/joint/unified.py` 不替换任何模块的 `forward`。它调用官方 `extract_feat` 和 `pts_bbox_head`。图像 Query Decoder 仍然在 TransFusion 内部完成几何引导融合。补丁 `patches/transfusion_fused_query.diff` 只在返回字典里多放一个 `fused_query`，也就是融合预测头的输入，形状 `[B, 256, N]`。检测损失原来读取的 `center`、`heatmap`、`dim`、`rot`、`height`、`vel` 没有改含义。

这个融合 Query 和 TransFusion 解码出的粗框进入 MC-DeTra。粗框是残差的起点，也是辅助监督里的候选。评估读取的框、类别和分数来自 DeTra：

- 类别是 `class_logits` 的 argmax
- 分数是 `sigmoid(object_logit)` 乘上最大类别概率
- 框是 PoseUpdate 的 7 维框加上 `pred_velocity`
- 轨迹是同一组 Query 的多模式未来点

默认匹配 `class_gate` 仍在 TransFusion 候选上做一次，后面的 Refinement block 共用这次 Query 到真值的对应。`u_cost` 改用 DeTra 第 0 个 block 的类别概率、存在性和框回归代价，不再用 TransFusion 的硬类别门，距离超过 2 m 的配对仍然丢弃。两种分配都保证一个 Query 只对应一个真值。数字见 `docs/UNIFIED_DETRA_EXPERIMENT.md`。

## 梯度

DeTra 自己的 `detach_pose_refinement` 仍然切断 block 之间的位姿梯度，这是原方法的更新方式。预测损失另外把未来点写成“偏移 + 断开的当前中心”，所以它不能靠挪动当前中心来减小轨迹误差。检测损失仍然更新这个中心、尺寸、朝向、速度、类别和存在性。

融合 Query 没有被 `detach`。阶段 A 里骨干参数 `requires_grad=False`，梯度停在 DeTra 和 `query_proj`、`bev_proj`。阶段 B 再按 V4 的分组学习率解冻图像 Query Decoder，这一轮还没有开始。

对照变体：

| 变体 | 设置 |
| --- | --- |
| `u0` | 阶段 A，预测损失打开，当前 Query 仍可读未来 Query |
| `u_nofore` | 预测损失权重为 0，只训检测 |
| `u_causal` | t=0 不读 t>0，用来分开多任务监督和预测对当前 Query 的前向影响 |

第 1 个训练步会分别反传检测损失和预测损失，并把梯度范数写进日志。mini 上这一步的实测是：

| 模块 | 检测损失 | 预测损失 |
| --- | ---: | ---: |
| `class_head` | 5.309 | 0 |
| `object_head` | 1.166 | 0 |
| `box_head` | 0.482 | 0 |
| `future_head` | 0 | 0.515 |
| `time_attn` | 5.740 | 1.005 |
| `query_proj` | 6.355 | 0.410 |

预测损失没有进入 `box_head`。检测损失没有进入 `future_head`。共享的时间注意力和 Query 投影两边都有梯度。可训练参数 2228321，和冻结检测器的旧 DeTra 相同，没有把跨帧记忆算进去。零初始化的框头在这一步没有移动中心：候选中心误差 0.107 m，解码后仍是 0.107 m。当前框和轨迹起点对齐。这一步的候选召回是该帧上的数字，不是整个验证集。

## 对照

同一套 mini 划分、同一份 LC 初始化和 2 epoch 预算下的 mAP、NDS、EPA、minADE、minFDE、MR 写在 `docs/UNIFIED_DETRA_EXPERIMENT.md`。候选检测仍是 mAP 0.530、NDS 0.508。DeTra 发布的检测低于这个候选，中心几乎没有被修正。
