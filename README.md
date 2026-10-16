# InfraUni

固定路侧场景的 3D 检测和轨迹预测。当前有两条主线。

nuScenes 上，TransFusion-LC 提供多模态特征和初始候选，DeTra 统一解码器输出类别、存在概率、三维框和未来轨迹。RESOLVE 上，先用官方 OpenPCDet 复现 LiDAR-only TransFusion-L。结构见 `docs/ARCHITECTURE.md`，实验索引见 `docs/EXPERIMENT_INDEX.md`。

```bash
cd /home/kemove/devdata1/zrl/InfraUni
# nuScenes 统一解码，代价匹配
python tools/train_joint.py --variant u_cost --epochs 2 --work-dir experiments/unified_ucost_mini
python tools/eval_joint.py --checkpoint experiments/unified_ucost_mini/epoch_2.pt --metrics-name metrics_e02.pkl

# RESOLVE 单序列数据检查，然后官方 TransFusion-L
python tools/convert_resolve_lidar.py --bag rosbag2_2026_01_13-16_05_56
python tools/check_resolve_loader.py /home/kemove/devdata1/zrl/dataset/RESOLVE_lidar_smoke
CUDA_VISIBLE_DEVICES=1 python tools/train_resolve_l.py oneseq --epochs 1 --batch_size 1 --workers 0 --ckpt_save_interval 1
```

记录：

- `docs/UNIFIED_DETRA_EXPERIMENT.md`
- `docs/RESOLVE_TRANSFUSION_L.md`
- `docs/DETRA_UNIFIED.md`
- `docs/RESOLVE_ADAPTATION.md`

更早的串联实验和跨帧记忆在 `docs/archive/`。
