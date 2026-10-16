# InfraUni

面向固定路侧场景的 LiDAR-Camera 3D 检测与轨迹预测。当前检测基线是官方 TransFusion，轨迹部分复用 MC-DeTra 的修正模块。记录见 `docs/TRANSFUSION_REPRODUCTION.md` 和 `docs/DETRA_JOINT.md`。

```bash
cd /home/kemove/devdata1/zrl/InfraUni
bash scripts/test_transfusion_lc.sh experiments/transfusion_voxel_LC_mini/epoch_1.pth
bash scripts/train_joint.sh --mode pred_head --steps 40
bash scripts/train_joint.sh --mode joint --steps 40 --blocks 2
bash scripts/eval_joint.sh --checkpoint experiments/joint_pred_head_mini/last.pt
bash scripts/eval_joint.sh --checkpoint experiments/joint_joint_mini/last.pt
```
