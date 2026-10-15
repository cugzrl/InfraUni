# 统一评估协议

同一批模型输出同时计算检测和预测。推理阶段只用检测器自己的框生成轨迹，不用未来真值，也不用真值框去生成 query。

## 检测

使用 nuScenes devkit，经 BEVFusion 的 `NuScenesDataset.evaluate` 调用。不要自行重写 mAP 或 NDS。

- 版本：`detection_cvpr_2019`
- mini 验证集：`mini_val`
- 框在当前 LiDAR 坐标系，底面中心，SECOND 朝向
- 评估器内部会做 `-yaw - pi/2`，再变到全局坐标，并按类别距离过滤
- 类别顺序与 TransFusion 一致：car、truck、construction_vehicle、bus、trailer、barrier、motorcycle、bicycle、pedestrian、traffic_cone
- 输出：mAP、NDS，以及 mATE、mASE、mAOE、mAVE、mAAE

换到 RESOLVE 或 V2X-Seq 时，检测必须改用该数据集自己的正式协议。那个分数不能再叫做 nuScenes NDS。

## 预测与联合指标

设置与 MASAR 对齐：6 个模式，12 个未来关键帧，2 Hz，共 6 秒。只统计最后一帧未来真值有效的目标。

匹配：

- 按类别分别做二分图匹配
- 代价是当前时刻 BEV 中心的 L2 距离
- 距离大于 2.0 m 的配对丢弃
- 未匹配的预测是假阳性

在匹配成功的目标上：

- minADE：六个模式里，全程平均 L2 的最小值，再对目标取平均
- minFDE：最后一帧 L2 的最小值
- MR：minFDE 大于 2.0 m 的比例

EPA 采用 ViP3D 的定义，α = 0.5：

- 匹配成功且 minFDE ≤ 2.0 m，记为命中
- 未匹配预测记为假阳性
- 每个类别 `EPA = (命中数 - 0.5 × 假阳性数) / 真值数`
- 表格里的 EPA 是有真值的类别的平均

预测指标使用分数不低于 0.1 的框。检测 mAP/NDS 仍把范围内的全部框交给官方评估器，由 PR 曲线自己使用分数。

`metrics.json` 里的 `forecast` 是上面这套协议。`forecast_all_scores` 去掉分数阈值，只用于检查低分模型有没有学到位移。低分框会全部变成假阳性，这个 EPA 不能放进论文表格。
