# 当前结构

默认研究线有两条。

nuScenes 联合模型：TransFusion-LC 提取多模态特征和 200 个初始候选，DeTra 统一解码器输出类别、存在概率、三维框和 6 秒轨迹。代码在 `infrauni/joint/unified.py`。它不替换官方 `forward`。融合 Query 来自 `patches/transfusion_fused_query.diff`。

RESOLVE 检测基线：官方 OpenPCDet TransFusion-L，只吃高分辨率 LiDAR。入口是 `tools/train_resolve_l.py`，模型配置仍在 RESOLVE 仓库。

以后的 LiDAR 联合模型不把两套框架的 Tensor 或 checkpoint 混用。`infrauni/resolve/interface.py` 规定要传出的量：BEV 特征 `spatial_features_2d`、解码 Query、9 维初始框和热图分数。官方 `predict` 现在只返回框头，Query 特征要在适配器里保留。RESOLVE 轨迹先做 3 秒、2 Hz 的 6 个点，时间用 info 里的浮点时间戳，不用点云里的 float32 时间，也不套 nuScenes 的 6 秒 12 点。这一步先不训练，也不接相机和跨帧记忆。

## 目录

| 路径 | 作用 |
| --- | --- |
| `infrauni/joint/unified.py` | 当前 nuScenes 联合网络 |
| `infrauni/joint/loss.py` | 检测和预测损失，含 `class_gate` 与 `cost` 两种匹配 |
| `infrauni/joint/model.py` | 旧的串联网络，用来复现 B、C、D、F |
| `infrauni/joint/memory.py` | 可关闭的跨帧记忆，统一解码训练不调用 |
| `infrauni/resolve/` | RESOLVE 候选接口，不含检测器实现 |
| `third_party/TransFusion` | nuScenes 官方检测代码 |
| `third_party/MC-DeTra` | 轨迹解码代码 |
| `/home/kemove/devdata1/zrl/RESOLVE` | 路侧官方检测代码，不放进 InfraUni |

`tools/train_joint.py` 仍能选出旧变体，是为了读取已经训好的 checkpoint。新的实验用 `u0`、`u_nofore`、`u_causal`、`u_cost`。历史实验记录在 `docs/archive/`。

## 两种 RESOLVE 划分

随机逐帧划分只服务 TransFusion-L 检测基线。`infrauni/data/resolve_scene_split.json` 按完整序列划分，给以后的检测和预测联合实验用。两套 info 文件名不同，结果不能混报。
