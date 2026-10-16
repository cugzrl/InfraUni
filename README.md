# InfraUni

面向固定路侧场景的 LiDAR-Camera 3D 检测与轨迹预测。当前检测基线是官方 TransFusion，轨迹部分复用 MC-DeTra 的修正模块。记录见 `docs/TRANSFUSION_REPRODUCTION.md`、`docs/DETRA_JOINT.md`、`docs/DETRA_JOINT_V2.md`、`docs/DETRA_JOINT_V3.md` 和 `docs/DETRA_JOINT_V4.md`。跨帧记忆见 `docs/FORESIGHT_STREAMING.md`，统一解码见 `docs/DETRA_UNIFIED.md`，路侧数据见 `docs/RESOLVE_ADAPTATION.md`。

```bash
cd /home/kemove/devdata1/zrl/InfraUni
bash scripts/test_transfusion_lc.sh experiments/transfusion_voxel_LC_mini/epoch_1.pth
bash scripts/train_joint.sh --variant b --epochs 2
bash scripts/eval_joint.sh --checkpoint experiments/ablation_b_mini/epoch_2.pt
bash scripts/eval_joint.sh --variant cv
```
