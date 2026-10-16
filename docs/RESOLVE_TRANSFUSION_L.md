# RESOLVE TransFusion-L

检测基线使用 `/home/kemove/devdata1/zrl/RESOLVE` 里的官方 OpenPCDet TransFusion-L，配置是 `tools/cfgs/sunlakes_models/transfusion.yaml`。InfraUni 只做 LiDAR 数据转换和启动参数，不改检测器。

环境是 `pcdet`：Python 3.8.20、PyTorch 2.4.1+cu118、spconv 2.3.6、torch-scatter 2.1.2。官方 README 写的是 PyTorch 2.8 和 CUDA 12.8。仓库里原来的 CUDA 扩展是 Python 3.10 的 `.so`，已用 CUDA 11.8、`TORCH_CUDA_ARCH_LIST=8.6` 为当前环境重新编译。`shapely` 装在这个环境里。`nuscenes` 从 bevfusion 环境的 site-packages 导入，因为转换脚本在 `use_cam=False` 时仍会导入它。

这个 checkout 的 `pcdet/models/detectors/__init__.py` 引用了不存在的雷达检测器，`backbones_3d` 也会导入用不到的 LION。这两处改成导入失败就跳过，TransFusion 本体没有改。全量旋转跳过 128 帧，和标注里缺少点云的帧数一致。

## LiDAR-only 转换

官方命令行的 `--use_cam` 和 `--with_cam_gt` 默认打开，而且 `store_true` 关不掉。`tools/convert_resolve_lidar.py` 直接调用 `rotate_point_cloud` 和 `create_dataset_from_json`，传入 `use_cam=False`、`use_lidar=True`。传感器是 `os128`，点云绕 z 轴旋转 -142.5 度。缺少点云的 128 帧直接跳过，不用相邻帧填补。

先转换完整序列 `rosbag2_2026_01_13-16_05_56`，27 帧全部有点云。核对 177 个保留框后，旋转中心、尺寸、朝向和官方类别的误差为 0。训练 info 22 条，验证 info 5 条。这一小划分只用于检查数据流，`m=1`，不是论文划分。

全量转换写到 `/home/kemove/devdata1/zrl/dataset/RESOLVE_layout`。官方随机帧划分是 `v1.0-trainval/sunlakes_infos_train.pkl` 和 `sunlakes_infos_val.pkl`，`seed=42`，`m=4`，对应官方 batch 4。按场景的划分写到同目录的 `sunlakes_infos_scene_{train,val,test}.pkl`，场景名单是 `infrauni/data/resolve_scene_split.json`。检测数字只用随机帧划分。联合预测以后只用场景划分，不用在随机划分上训过的权重做时序测试。

## 点、框和范围

点特征是 x、y、z、intensity、timestamp。DataLoader 里再加一维 batch 下标，所以 batch 点是 6 列，体素特征仍是 5 通道。一个训练 batch 读到 70706 个点、41610 个体素，体素形状 `(41610, 20, 5)`。

模型配置覆盖了数据集 yaml 里的范围。实际使用：

| 项 | 值 |
| --- | --- |
| 点云范围 | [-60, -60, -8, 88.8, 88.8, 0] |
| 体素 | [0.3, 0.3, 0.2] |
| 网格 | [496, 496, 40] |
| Backbone 输入 | 5 |
| 数据集 yaml 里的另一套范围 | [-60, -60, -8, 90, 80, 0]，体素 0.1，已被模型配置覆盖 |

框是 x、y、z、l、w、h、yaw、vx、vy。速度用旋转后的中心和文件名里的真实时间差，dt 小于等于 0 或大于等于 2 秒时记 0。info 里的时间戳保持 Python 浮点。点云第 5 维按官方写法存成 float32。绝对 Unix 时间约 1.76e9，float32 的间距是 128 秒，整段序列会塌成同一个值。`rosbag2_2026_01_13-16_05_56` 的 27 个时间戳在 float32 点特征里只有 1 个不同值，和 info 时间相差约 9.5 秒。正式基线保持这个编码。以后如果要改点特征，必须单独命名，不能悄悄替换基线。

## 类别

`infrauni/data/resolve_class_maps.json` 分开记了两套名字。

检测基线沿用官方映射：`van` 到 `barrier`，`golf cart` 到 `traffic_cone`，`construction vehicle` 到 `construction_vehicle`。预测以后保留 van 和 golf cart 自己的动态类别，不把高尔夫球车当成锥桶。

## 训练

启动脚本是 `tools/train_resolve_l.py`。它在 RESOLVE 的 `tools` 目录调用官方 `train.py`，模型、损失、优化器和增强仍是 `transfusion.yaml`。官方设置是 batch 4、20 epoch、Adam one-cycle、学习率 0.001、权重衰减 0.01、梯度裁剪 35。评估项在 yaml 里是 `kitti`。

当前唯一的训练差异是关掉 `gt_sampling`，因为真值库还没生成。单序列 `rosbag2_2026_01_13-16_05_56` 已用官方训练器跑完 1 个 epoch：22 个 iter，4.6 秒，损失从 965.5 降到 22.62。one-cycle 在这 22 步里从 1.00e-4 收到 1.26e-5，这是 1 个 epoch 被压缩后的学习率，不是 20 epoch 的曲线。checkpoint `checkpoint_epoch_1.pth` 可以完整载入 307 个参数。5 帧验证的 mAP 和 NDS 都是 0，平均输出 100 个框，符合未收敛的随机初始化，不能当作基线。

全量 LiDAR info 已写完。旋转成功 17042 帧，跳过 128 帧。没有点落入框内的帧被官方函数丢掉，最后保留 16380 帧。官方随机划分是训练 13104、验证 3276，`m=4`，没有余数。场景划分是训练 11556、验证 2472、测试 2352，没有未知场景。

20 epoch 已在 GPU 1 上开始。batch 4 占用约 9.2 GB。第 1 个 iter 损失 980.8，第 500 个 iter 损失 4.076，学习率约 1.01e-4，仍在 one-cycle 的上升段，峰值是官方的 0.001。稳定后每个 batch 约 0.25 秒，一个 epoch 约 15 分钟，20 个 epoch 大约 5 小时。每 5 个 epoch 存一次 checkpoint。`gt_sampling` 仍然关闭。权重目录是 `output/sunlakes_models/transfusion/infrauni_official/`。验证和误检可视化在第 20 个 epoch 结束后由官方评估写出。

权重会写在 RESOLVE 的 `output/` 下面，tag 是 `infrauni_oneseq` 或 `infrauni_official`。不要把 nuScenes 的 LC 权重载进这个头。
