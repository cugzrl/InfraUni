# SparseDrive Camera-only：RESOLVE 检测与轨迹预测适配

工作区：`/home/kemove/devdata1/zrl/InfraUni`。

## 来源与方法边界

- 官方源码：https://github.com/swc-17/SparseDrive 。本次克隆提交 `fbadea693cbef3f3daea7705e521c0dd3321605c`，完整 Git 元数据保留在 `third_party/SparseDrive`。因服务器直连 GitHub TLS 失败，本地完成 git clone 后上传完整仓库。
- 使用 SparseDrive-S 的 ResNet50、FPN、900 个检测锚点、6 层 Sparse4D 解码及实例时序记忆；保留对象运动模式锚点、时序队列、模式分类和轨迹回归。
- RESOLVE 没有地图和自车规划监督，本适配明确关闭地图预测、地图交叉注意力及自车规划分支。它是 SparseDrive 在路侧检测+预测任务上的适配版，不能标成已完整复现 nuScenes 原始全任务实验。
- 严格 camera-only 模型输入：四路图像、内外参投影矩阵、有效视角掩码、时刻和静态路侧位姿。不输入点云，也关闭原版使用点云监督的 dense depth 辅助分支；三维检测框、速度和未来轨迹只作为训练监督。
- 使用已缓存的 ImageNet ResNet50 V1 `resnet50-0676ba61.pth`；不加载 nuScenes 检测或预测预训练检查点。
- 官方可变形聚合 CUDA 算子已在现有环境编译。FlashAttention 换为标准 MultiheadAttention，保留注意力计算形式。FP16 在实际梯度检查中出现溢出，当前正式训练使用 FP32。

## 环境

复用 `/home/kemove/anaconda3/envs/bevspread`，没有新建 conda 环境。PyTorch 1.13.1/CUDA11.7、MMCV-full1.7.0、MMDetection2.28.2。

共享环境中的 charset_normalizer 有导入问题，因此兼容依赖仅安装到 `third_party/SparseDrive/.runtime`，由启动脚本优先加载；没有覆盖共享环境中的包。该目录还提供 NumPy1.23.5、yapf0.33.0、prettytable、einops。

## 完整数据划分及相机关联

- 直接沿用 `third_party/MC-DeTra/splits/resolve_forecast_split.json`，SHA256：`e42ba4f375c5a749ee37b2fc6b267aa0b1907f4344059cf3141a382d0f2fa410`。
- train/val/test：11946/3393/1703 帧，114/54/34 个互斥序列。不按相机是否完整缩减验证集，不沿用 nuScenes 的55米圆形过滤。
- 共享 `data/resolve_protocol` 的框、类别、速度，并与 `data/resolve_cache` 的原始数组逐帧核对；未来点和掩码也来自同一缓存。
- 已从原始10个 ZIP 中补齐45458张未解压相机图片，约33.24GB。图片只解压到原始数据目录，新的 SparseDrive 清单引用这些文件，不复制点云或图片。
- 原始 JSON 与标定文件的图像关联存在差异。本次优先使用标定文件名明确给出的“当前 LiDAR 帧—相机图像”配对；缺少精确标定时使用同一传感器0.201秒内的标定记录。关联方式、差异和数量记录在 `data/resolve_sparsedrive/manifest.json`，不按跨传感器原始时间戳直接做最近邻。
- 最后4个训练帧缺少 axis1 的同步图像：保留这些帧并显式屏蔽该视角的注意力贡献，使用其余3路真实图像。未用错时图像替代，也未把占位视角当作有效观测。验证和测试的全部帧均可用4路图像。
- 某些相机外参随帧变化，因此保留逐帧标定，不能统一替换为一个序列的首帧矩阵。
- 模型图像分辨率704×384，四路完整视野均缩放保留，并同步缩放投影矩阵；不把 LiDAR 训练中的旋转坐标应用到相机上。
- 两个任务的锚点仅由训练集生成：检测900个，轨迹每类别6种；没有从验证/测试标签学习锚点。
- 训练按序列连续取帧，序列边界和时间倒退会清空模型记忆。推理时不提供 GT、真实轨迹或真实跟踪ID。

## 统一评估

调用现有 `infrauni/eval/resolve_detection.py` 和 `resolve_forecast.py`，不调用 SparseDrive 自带 nuScenes EPA/范围过滤器。

- 检测：10类，0.5/1/2/4米阈值，官方AP曲线、TP误差；主分数使用论文 NDS9，另保存 NDS_legacy10。
- 预测：5秒、2Hz、10个未来点、6种模式；采用现有同类2米匹配、0.10分数门限、2米FDE命中与0.5误检惩罚的 EPA；minADE/minFDE 使用同一有效未来点及匹配协议。它是项目的统一预测扩展，RESOLVE没有官方预测基准。
- 检测框与轨迹共用同一个 query 下标选择和排序，避免两套独立后处理导致框与轨迹串位。
- 第一阶段只有检测指标；第二阶段同时输出检测和预测指标。没有预测头时不会制造零误差结果。

## 本次训练设置

该日程是 RESOLVE 适配设置，并非官方 nuScenes 的100轮+10轮日程。

1. 感知预训练20轮，训练相机检测和时序检测记忆。
2. 读取第一阶段自己的最终权重，自动进行10轮检测+运动预测联合训练。

每批2帧，累积4批，有效batch8；AdamW，主学习率1e-4，图像主干学习率为其0.1倍，权重衰减0.001；500步warmup、cosine衰减、梯度裁剪25。FP32。每250次参数更新保存轮次内恢复点，每轮保存检查点；每阶段第1轮、第5轮及之后每5轮和最终轮执行完整验证。两阶段各自保存 latest 和 best_map 链接；同一命令重新启动会自动从完整轮次检查点续训。

## 启动与查看

```bash
cd /home/kemove/devdata1/zrl/InfraUni
screen -dmS infrauni_sparsedrive_camera_20261028 bash scripts/train_sparsedrive_resolve.sh
screen -r infrauni_sparsedrive_camera_20261028
# Ctrl-A D：退出查看并保持训练运行
tail -f experiments/sparsedrive_camera_resolve_20261028/console.log
```

第二张4090通过 CUDA_VISIBLE_DEVICES=1 选择，对应 GPU UUID `GPU-bacc945a-d539-1c30-4e2c-691ee6493c87`。不要在同一 screen 已运行时重复启动。

检查点目录：

```text
/home/kemove/devdata1/zrl/InfraUni/experiments/sparsedrive_camera_resolve_20261028/stage1/
/home/kemove/devdata1/zrl/InfraUni/experiments/sparsedrive_camera_resolve_20261028/stage2/
```

包含 `epoch_001.pth` 等、`latest.pth`、`best_map.pth`、`metrics_epoch_005.json` 等。检查点含优化器和随机数状态。尚未完成第一轮时没有轮次检查点。

离线完整评估：

```bash
PYTHONPATH=third_party/SparseDrive/.runtime:$PWD CUDA_VISIBLE_DEVICES=1 \
 /home/kemove/anaconda3/envs/bevspread/bin/python tools/eval/eval_sparsedrive_resolve.py \
 --checkpoint experiments/sparsedrive_camera_resolve_20261028/stage2/latest.pth \
 --split val --output experiments/sparsedrive_camera_resolve_20261028/stage2/metrics_manual.json
```

## 代码与验证记录

- 中央适配：`infrauni/resolve/sparsedrive/`。
- 配置：`configs/resolve/sparsedrive_camera.py`。
- 数据准备：`tools/data/prepare_sparsedrive_resolve.py`；配置生成：`tools/data/configure_sparsedrive_resolve.py`。
- 训练：`tools/train/train_sparsedrive_resolve.py`；评估：`tools/eval/eval_sparsedrive_resolve.py`。
- 集成检查：`tools/checks/check_sparsedrive_resolve.py`，结果在 `data/resolve_sparsedrive/validation.json`。
- 已验证实际两阶段有限损失及有限梯度、预测损失回传图像主干、无GT推理、300个检测候选、6×10×2轨迹输出、未来位移目标往返和训练采样器不丢帧。
- 训练已启动不代表训练收敛，最终结果应从完整验证评估文件读取。

Environment, upstream revision, source hashes and integration checks: experiments/sparsedrive_camera_resolve_20261028/run_manifest.json. Runtime patch: patches/sparsedrive_resolve_runtime.diff.

## 2026-10-28 数值稳定修复与训练恢复

相机投影除法的反向传播 NaN 已用真实失败批次复现并修复。详见 docs/SPARSEDRIVE_TRAINING_FIX_20261028.md。新增正常可见投影等价检查、极端投影/尺寸回归检查、异常时序缓存恢复检查；当前使用安全投影与采样几何、异常缓存重置、float64梯度范数及轮次内检查点。断点后时序缓存重新建立，数据加载增强重新初始化。

## 第4轮loss审计与统计修正

详见 docs/SPARSEDRIVE_LOSS_AUDIT_20261028.md。完整验证集mAP从第1轮0.02123提升到第4轮0.11738，NDS从0.04377提升到0.18340。新增rolling100、明确统计起始批次的epoch_avg、完整轮次统计、匹配框诊断和裁剪前梯度标记，保持训练目标和协议。旧avg_since_resume名称已退役。
