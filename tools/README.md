# 工具目录导航

所有命令从仓库根目录 /home/kemove/devdata1/zrl/InfraUni 执行。业务 Python 脚本均已按用途放入子目录；工具根目录不再放业务 Python 文件。

| 目录 | 用途 | 脚本 |
|---|---|---|
| data/ | 数据转换、清单和缓存准备 | build_future_tracks.py、cache_resolve_cameras.py、convert_resolve_lidar.py、link_resolve_layout.py、summarize_resolve.py、configure_resolve.py、prepare_resolve.py |
| train/ | 训练入口 | train_joint.py、train_resolve_l.py、train_transfusion.py |
| eval/ | 保存结果与联合模型评估 | eval_joint.py、evaluate_resolve.py |
| checks/ | 数据、指标、梯度和对齐检查 | check_detra_core.py、check_future_tracks.py、check_joint_grad.py、check_resolve_loader.py、test_joint_alignment.py、diagnose_conflict.py、check_resolve_protocol.py |
| visualization/ | 相机投影和检测可视化 | visualize_resolve_cameras.py、visualize_resolve_detections.py、visualize_resolve_transfusion.py |
| checkpoints/ | 权重格式转换 | convert_fusion_init.py |
| launch/ | 第三方模块通用启动器 | run_mcdetra.py |
| resolve/ | 保留现有 screen 启动 Shell | start_transfusion.sh；调用已更新至 train/train_transfusion.py |

## 当前 RESOLVE 常用命令

```bash
/home/kemove/anaconda3/envs/pcdet/bin/python tools/data/configure_resolve.py
/home/kemove/anaconda3/envs/pcdet/bin/python tools/data/prepare_resolve.py
/home/kemove/anaconda3/envs/pcdet/bin/python tools/checks/check_resolve_protocol.py
/home/kemove/anaconda3/envs/pcdet/bin/python tools/eval/evaluate_resolve.py --help
/home/kemove/anaconda3/envs/forecast_mae/bin/python tools/launch/run_mcdetra.py detra_repro.train_cached --help
/home/kemove/anaconda3/envs/forecast_mae/bin/python tools/visualization/visualize_resolve_cameras.py --num-samples 1 --output-dir samples/resolve_layout_check
```

当前 TransFusion 的完整参数仍在 tools/resolve/start_transfusion.sh。不要在已有 screen 正在训练时重复启动。
共同评估说明见 ../README_RESOLVE.md。

## 旧实验入口

train/train_joint.py、eval/eval_joint.py 以及相关 joint 检查用于旧 nuScenes 实验，需要其对应模型版本和环境。
data/build_future_tracks.py 与 checks/check_future_tracks.py 也是 nuScenes 工具。
train/train_resolve_l.py、data/convert_resolve_lidar.py、data/summarize_resolve.py、checks/check_resolve_loader.py 保留旧 RESOLVE 转换和划分流程；当前统一序列基线使用 prepare_resolve.py 和 train_transfusion.py。
目录迁移只修正组织和路径，不代表历史模型版本已经兼容当前 third_party。补丁状态见 ../patches/README.md。

## TransFusion-L 检查点投影可视化

新增入口 tools/visualization/visualize_resolve_transfusion.py，使用第20轮检查点对验证集不同序列中目标较多的10帧重新推理。输出目录仅包含图片，每帧4路相机图和1张拼图，序号01至10。默认置信度显示阈值0.10，按类别着色，不在目标上标注文字；点云按距离着色。运行选择信息、预测数值及检查点哈希另存 experiments/visualization/。

命令（GPU按空闲情况选择，重复运行请换一个空目录）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pcdet/bin/python tools/visualization/visualize_resolve_transfusion.py --num-samples 10 --score-thresh 0.1 --output-dir visual/transfusion-l
```

## 固定样例对比

当前两模型主图使用同一清单 experiments/visualization/transfusion_l_20261028_105558/manifest.json，均为10帧、0.10显示阈值，无目标类别或分数文字。输出分别为 visual/transfusion-l 与 visual/mc-detra，仅包含图片。新入口参数 --sample-manifest 固定样例；--overwrite 会把旧图归档到 experiments/visualization/，再生成新图。resolve_viz_common.py 为二者共用的绘图工具，不是独立命令入口。

漏检核查及低阈值对照见 experiments/diagnostics/transfusion_mcdetra_20261028/report.md。0.10不是所有类别可靠的工作阈值；bus等类别的真阳性分数可能远低于此。正式评估继续使用原统一AP协议。

## 相同样例的标注真值

使用 /home/kemove/anaconda3/envs/forecast_mae/bin/python tools/visualization/visualize_resolve_gt.py，结果在 visual/gt，仅含50张图片，与两模型的样例、相机底图和文件名一致。真值直接读取共享验证标签并逐帧与原缓存核对，无分数过滤，也不在目标上绘制文字。
