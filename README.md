# InfraUni

面向固定路侧场景的 LiDAR-Camera 3D 检测与轨迹预测

```bash
cd /home/kemove/devdata1/zrl/InfraUni
bash scripts/train_transfusion_l.sh
bash scripts/test_transfusion_l.sh experiments/transfusion_voxel_L_mini/epoch_1.pth
bash scripts/train_transfusion_lc.sh
bash scripts/test_transfusion_lc.sh experiments/transfusion_voxel_LC_mini/epoch_1.pth
```

## 目录

- `infrauni/data`：nuScenes 样本，以及 RESOLVE / V2X-Seq 的统一接口
- `infrauni/models/lidar_backbone.py`：体素、Sparse Encoder、SECOND、SECONDFPN
- `infrauni/models/camera_backbone.py`：冻结的 ImageNet ResNet-50 和可训练 FPN
- `infrauni/models/transfusion.py`：官方 TransFusion 检测头
- `infrauni/models/detra.py`：Object / Time / Mode 注意力和多轮修正
- `infrauni/eval`：官方 mAP、NDS，以及 EPA、minADE、minFDE、MR
- `docs/`：各阶段和评估协议

## 命令

当前检测基线是上面的官方 TransFusion 脚本。下面是早期自写流程，不用于正式检测复现。

```bash
cd /home/kemove/devdata1/zrl/InfraUni
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python

$PY tools/build_infos.py
$PY tools/smoke.py
$PY tools/train.py --config configs/mini.yaml --mode det --epochs 3
$PY tools/train.py --config configs/mini.yaml --mode pred_head --epochs 2 --init experiments/mini_det/last.pt
$PY tools/train.py --config configs/mini.yaml --mode joint --epochs 2 --init experiments/mini_det/last.pt
$PY tools/infer.py --config configs/mini.yaml --checkpoint experiments/mini_joint/last.pt
```

推理会读取权重里保存的 `mode`。检测走 nuScenes 官方评估，预测和 EPA 用同一批框。协议见 `docs/eval_protocol.md`
