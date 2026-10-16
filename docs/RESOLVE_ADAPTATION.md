# RESOLVE 适配准备

数据在 `/home/kemove/devdata1/zrl/dataset/RESOLVE`，官方代码在 `/home/kemove/devdata1/zrl/RESOLVE`。本轮只做完整性、标定、轨迹统计和目录对接。基础网络还在做稳定性对照，没有开始 RESOLVE 上的联合训练。

## 压缩包和解压

三个未解压包都在数据根目录，解压前用 `unzip -l` 看过内部是 `rosbag2_*` 目录，和已经存在的包没有同名冲突：

| 文件 | 大小 |
| --- | ---: |
| `seq141_seq160.zip` | 8.2 GB |
| `seq161_seq180.zip` | 7.1 GB |
| `seq181_seq202.zip` | 11 GB |

解压使用 `unzip -n`，不覆盖已有文件，也不删除压缩包。三个包都已解开，压缩包仍在原位。根目录现在有 202 个 `rosbag2_*` 目录，与标注里的 202 个序列一致。磁盘剩余约 1.7 TB。

标注是根目录的 `HSQJN_3.json`，17170 帧，202 个 `batchId`，每个 `batchId` 对应一个 bag。官方 README 把 json 和 bag 放在 `2026-01-04_18_02_01-export` 下面。当前下载把 bag 直接放在数据根目录。

## 和官方读取路径的差别

官方 `tools/create_sunlakes_data_v3.py` 从 `data_prefix/HSQJN_3.json` 取标注，再到 `data_prefix/resource/<bag>/128_128b_label/<stamp>.bin` 读点云。下载的点云在 `<bag>/128_128b_sync/`。抽查 `rosbag2_2025_10_17-09_29_02/128_128b_sync/1760718487-550083650.bin` 存在，同名的 `128_128b_label` 不存在。

`tools/link_resolve_layout.py` 在 `/home/kemove/devdata1/zrl/dataset/RESOLVE_layout/2026-01-04_18_02_01-export` 建立软链接：`128_128b_label`、`64_64b_label`、`16_16b_label` 以及四路 `axis*_label` 分别指向对应的 `*_sync`。解压完成后共 934 条链接、按 bag 复制的标定 808 个文件。原始 bag 和压缩包没有移动。

点云预处理是绕雷达原点的 `Rz(rot_pc)`，默认 `-142.5` 度。标定写成 `lidar2camera = extrinsic @ Rz.T`，使图像、框和旋转后的点云落在同一坐标系。适配时必须连这组旋转一起用。只旋转点云、不改外参，投影会错。

## 四路相机和标定

相机顺序由官方 `CAM_TYPES` 固定：

| 标注目录 | 名称 |
| --- | --- |
| `axis1_label` | `CAM_FRONT` |
| `axis3_label` | `CAM_FRONT_LEFT` |
| `axis4_label` | `CAM_FRONT_RIGHT` |
| `axis2_label` | `CAM_BACK` |

这不是 nuScenes 的六相机顺序。InfraUni 的 TransFusion-LC 不能按六路假设去索引 RESOLVE。

官方脚本期望每路相机一个总标定文件，例如 `axis1_os128b.txt`。数据里是逐帧文件 `calibration_txt/axis1__ouster128b/<image>__<lidar>_calibration.txt`。同一个 bag 内旋转矩阵逐帧相同。不同 bag 之间平移大约差 1 m，例如 `t_z` 从 41.43 到 42.45。一个全局标定文件不能代表全部 202 个序列。布局脚本按 bag 复制该 bag 的第一帧标定，写到 `per_bag/<bag>/axis*_os128b.txt`，本次 736 个文件。官方转换若一次处理全部序列，需要改成按 bag 读标定。这一步还没有改官方源码。

图像文件名也不总是和 json 一致。同一帧里 `axis1_sync` 和 `axis2_sync` 能对上，`axis3_sync` 最近文件相差约 0.13 s。官方读取使用 json 里的原路径。精确软链接补不上这个时间差，最近邻配对还没有做，避免把相邻帧图像配错。

## 类别、范围和权重

标注框是 9 个数：`x, y, z, roll, pitch, yaw, length, width, height`。官方转换使用 yaw 和后三维尺寸，忽略 roll、pitch，再绕 z 旋转中心和 yaw。速度由同一 `(batchId, id)` 的相邻帧中心差分得到，时间间隔异常时写 0。

类别计数：car 63083，truck 31059，golf cart 5329，trailer 5262，pedestrian 4812，van 4536，motorcycle 3723，bicycle 3228，bus 2361，construction vehicle 896。官方把 `van` 映射成 `barrier`，把 `golf cart` 映射成 `traffic_cone`。检测基线若要和官方数字对比，需要保留这个映射。预测和跨帧评估不应该把高尔夫球车当成锥桶，适配层要单独记录原始类名。

官方 TransFusion 配置是 OpenPCDet 的 LiDAR-only，不是六相机 LC。`transfusion.yaml` 的点云范围是 `[-60, -60, -8, 88.8, 88.8, 0]`，体素 `[0.3, 0.3, 0.2]`，网格 `[496, 496, 40]`。数据集 yaml 的范围是 `[-60, -60, -8, 90, 80, 0]`，体素 `[0.1, 0.1, 0.2]`。点特征是 `x, y, z, intensity, timestamp`。类别顺序把 barrier 放在 motorcycle 前面，和 nuScenes TransFusion 的顺序不同。

nuScenes LC 权重的相机数、点云范围和体素都对不上，不能直接加载。可以迁移的是与几何无关的头部结构超参：200 个 proposal、隐层 128。图像分支要按 4 路重建。LiDAR 体素编码器按 RESOLVE 的范围重新初始化。

## 未来轨迹和划分

原生采样间隔中位数是 0.100 s，也就是大约 10 Hz，不是 nuScenes 的 2 Hz。1963 条至少两帧的轨迹里，原生时间跨度中位数约 5.3 s。抽到名义 2 Hz 之后，跨度中位数约 5.0 s。同时满足 6 s 和 12 个间隔的轨迹有 787 条。同时满足 3 s 和 6 个间隔的有 1507 条。间隔大于 1 s 的断裂很少。目标 id 在同一个 `batchId` 内使用，不跨 bag。

RESOLVE 的预测时长先用 3 s。若沿用现有 2 Hz、6 模式的 DeTra 头，对应 6 个未来点，覆盖约 77% 的多帧轨迹。6 s、12 点只在 787 条长轨迹上作为补充协议，不作为统一监督长度。真值使用和检测相同的旋转后雷达坐标。序列中断、遮挡和提前消失用 mask 标出，缺终点的目标不记成假阳性，这和 nuScenes V3 协议的处理一致。

官方 `split_dataset_divisible` 按帧随机划分，种子 42，并丢掉余数使长度能被 batch 整除。这个划分只留给检测基线和官方表对比。跨帧记忆和轨迹预测使用按完整 `batchId` 的划分，种子同样是 42：train 142、val 30、test 30，写在 `infrauni/data/resolve_scene_split.json`。相邻帧不会同时出现在训练和验证里。

解压完成后，17170 帧里仍有 128 帧在 `128_128b_sync` 下没有同名文件，分布在 67 个 bag：58 个 bag 各缺 1 帧，5 个各缺 2 帧，另外 4 个 bag 缺 13 到 17 帧。抽查缺帧的最近文件大约差 0.1 秒，也就是相邻一帧，不是压缩包没解开。这 128 帧先从检测和轨迹监督里丢掉，不做最近邻硬配。

## 官方检测基线还差什么

`pcdet` 环境是 torch 2.4.1+cu118，能够导入 `/home/kemove/devdata1/zrl/RESOLVE/pcdet`。官方 README 写的是 torch 2.8 和 CUDA 12.8。少量样本可以先在现有环境里试，失败再决定是否另建环境。

还没有跑官方 TransFusion-L。原因有三件同时在占用机器：稳定性对照占着一张 GPU，最后一个压缩包还在写盘，以及单包标定和图像时间戳还没有接进一次完整的 `create_sunlakes_data_v3.py`。下一步用一个已经解压完的 bag、它自己的 `per_bag` 标定和软链接布局生成 info，再跑几个 iteration 的官方 LiDAR TransFusion。这一步只验证坐标和标注，不作为最终检测数字。多模态 LC 等这次检测读取正确、并且 nuScenes 上的联合训练不再产生非有限权重之后再接。
