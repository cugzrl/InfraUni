# InfraUni

固定场景的3D检测和轨迹预测

```bash
cd /home/kemove/devdata1/zrl/InfraUni
python tools/train/train_joint.py --variant u_cost --epochs 2 --work-dir experiments/unified_ucost_mini
python tools/eval/eval_joint.py --checkpoint experiments/unified_ucost_mini/epoch_2.pt --metrics-name metrics_e02.pkl

# RESOLVE单序列数据检查，TransFusion-L
python tools/data/convert_resolve_lidar.py --bag rosbag2_2026_01_13-16_05_56
python tools/checks/check_resolve_loader.py /home/kemove/devdata1/zrl/dataset/RESOLVE_lidar_smoke
python tools/train/train_resolve_l.py oneseq --epochs 1 --batch_size 1 --workers 0 --ckpt_save_interval 1
```
