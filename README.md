# InfraUni

第一阶段在nuScenes-mini上验证基础数据流和双向联合训练。


## 当前网络

`InfraUniBase`（约 1.49M 参数）的一次前向：

1. **LiDAR**：PointPillars 风格的 pillar BEV（不依赖 spconv），再经卷积得到热力图和粗 BEV token。
2. **Camera**：前视相机轻量 CNN。检测 query 用标定把 3D 中心投影到图像上采样特征，并与图像 token 做 cross-attention。这一步对应 TransFusion 的图像融合，但是局部采样加全局注意力的简化版，不是原论文的 deformable attention。
3. **Object Query**：热力图 top-k 初始化 query 和参考点，先做自注意力，再交叉到 LiDAR BEV，输出第一阶段 3D 框。
4. **轨迹 refinement**：每个检测 query 展开成 `3` 个模态 × `6` 个未来时刻（2 Hz，共 3 秒）。时间注意力和模态注意力之后，在预测航点处采样 BEV，再回归相对当前检测中心的位移。对应 DeTra 的 query 轨迹体，但没有 HD map、多帧 sweep 和 deformable attention。
5. **反向更新检测**：未来 query 的隐状态，加上预测航点本身的编码，通过门控写回检测 query，再出一版修正框。检测损失因此能更新轨迹分支，轨迹损失也能更新检测骨干。

联合损失：热力图 focal、三阶段分类 focal、框的 L1、未来航点 winner-takes-all L1、模态交叉熵。检测与轨迹共用同一次 Hungarian 匹配（匹配用相机融合后的框）。

## 目录

```text
configs/nuscenes_mini.yaml      范围、query 数、损失权重
infrauni/data/                  nuScenes-mini 解析与样本接口
infrauni/models/                pillar / 图像编码、联合网络、损失
infrauni/engine/                简易指标与 BEV 可视化
tools/train.py                  训练
tools/infer.py                  验证集推理
setup_env.sh                    复制conda环境
experiments/nuscenes_mini/      日志、权重、可视化
```

接入RESOLVE或V2X-Seq时，样本字典保持与 `infrauni/data/nuscenes_mini.py` 顶部相同的字段：`points`、`image`、`lidar2cam`、`K`、`boxes`、`labels`、`traj`、`traj_mask`


## 运行

```bash
$PY tools/train.py --config configs/nuscenes_mini.yaml
$PY tools/infer.py --ckpt experiments/nuscenes_mini/last.pt --split val --vis 6
```

## 第一阶段实验（nuScenes-mini，24 epoch）

从零训练，batch size 4，AdamW，学习率 `3e-4`，余弦退火

训练损失从epoch 1的33.5降到epoch 24的11.7：

| 项 | epoch 1 | epoch 24 |
| --- | ---: | ---: |
| 总损失 | 33.49 | 11.66 |
| 热力图 | 7.57 | 2.24 |
| 框回归 | 28.34 | 11.41 |
| 轨迹 | 20.26 | 5.02 |

第一步梯度拆分（见 `experiments/nuscenes_mini/grad_check.json`）：

- 只反传轨迹损失时，LiDAR pillar、图像编码器和轨迹头有梯度，运动回写模块为0
- 只反传检测损失时，运动门控、航点编码和修正检测头有梯度，轨迹头也有梯度

因此检测和预测除了共享特征，两边的损失都能更新对方的参数

`last.pt` 在验证集上的结果。下面的检测指标是自写的中心距离匹配，不是官方nuScenes mAP。轨迹的minADE/minFDE只在匹配成功且有未来真值的目标上计算，未来 6 步、3 秒，取 3 个模态中的最优

分类分数几乎都挤在0.22附近，阈值0.15会留下几乎全部120个query，所以精度很低：

| 设置 | Precision | Recall | F1 | minADE | minFDE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 分类分数 ≥ 0.15，同类，2 m | 0.034 | 0.097 | 0.051 | 2.94 | 3.13 |
| 分类分数 top-40，不区分类，2 m | 0.103 | 0.096 | 0.099 | 2.88 | — |
| 分类分数 top-40，不区分类，4 m | 0.261 | 0.244 | 0.253 | 2.91 | — |

不看分数、只看120个预测里离每个真值最近的框：验证集中位距离 **3.07 m**，26% 的真值在2m内，68%在4m内。匹配上的真阳性中心误差约 **1.32 m**。训练集和验证集的定位误差接近，说明当前瓶颈是模型和分数标定，不是mini过拟合。

BEV可视化在 `experiments/nuscenes_mini/infer_val/`。绿色是真值框和未来轨迹，红色是热力图分数最高的20个预测框，橙色是模态0的预测轨迹

## 已完成 / 已验证 / 未实现

已完成并实际跑通：

- 项目骨架、样本接口、nuScenes-mini 的框与3秒未来轨迹
- LiDAR BEV、前视图像、object query、3D框、多模态未来轨迹的一次前向
- 联合损失，以及检测损失与轨迹损失的双向反传
- 24epoch 训练、验证推理和BEV图

后续：

- TransFusion/DeTra的严格复现（deformable attention、多尺度BEV、HD map、多帧sweep、GRU轨迹头）
- 6个相机、历史轨迹、MASAR的appearance-guided past refinement
- 官方检测与预测评测（nuscenes-devkit 未安装）
- 分类分数标定。当前分数拉不开背景query，2m检测召回仍然低
- RESOLVE、V2X-Seq，以及路侧专用模块。跟踪不在本阶段范围内

## 下一步

1. 把分类头和热力图分数分开监督，并加query级的背景降权，先把2m召回提上来
2. 检测框改为相对热力图参考点的残差，减少绝对坐标回归的负担
3. 保持现在的样本字段，再接RESOLVE/V2X-Seq
4. 在这个可训练骨架上加路侧模块
