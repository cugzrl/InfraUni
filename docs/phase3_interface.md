# 第三阶段：路侧数据接口

这一阶段没有加入新的路侧模块。检测器和 DeTra 解码器只依赖统一样本，不读取 nuScenes 的路径。

每个样本需要这些字段，定义在 `infrauni/data/sample.py`：

| 字段 | 含义 |
| --- | --- |
| points | 当前 LiDAR 系点云，含强度和时间 |
| img | V 路已经归一化的图像 |
| img_meta | `lidar2img`、图像缩放、三维增广、样本 token |
| gt_bboxes_3d | 底面中心的 3D 框 |
| gt_labels_3d | 类别下标 |
| gt_fut | 未来中心，xy，当前 LiDAR 系 |
| gt_fut_mask | 该未来时刻是否存在 |

`ResolveDataset` 和 `V2XSeqDataset` 目前只保留这个接口。接入时要补标定、时间戳和身份关联，并把未来轨迹变到当前传感器坐标系。

检测指标必须换成对应数据集的正式协议。预测侧可以继续用同一套匹配距离、6 个模式、6 秒、EPA、minADE、minFDE 和 MR。
