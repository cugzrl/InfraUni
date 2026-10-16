# 训练稳定性与同轮次对照

本轮先停住网络扩容。F0 在第 5 个 epoch 的权重是有限的，第 6 个 epoch 起旧训练器把非有限值写进了图像 Query Decoder。后面的 `epoch_6.pt` 和 `epoch_15.pt` 含有 NaN，不能当作训练结果。

## 旧训练器实际做了什么

旧循环只检查损失标量。官方检测头一旦输出非有限值，匈牙利匹配会报 `matrix contains invalid numeric entries`。当时的处理是跳过这一步。`clip_grad_norm_` 不能去掉已经是 NaN 的梯度，跳过发生在一次错误更新之后时，参数已经坏了。随后 attention softmax 申请失败，日志里连续出现约 2909 次显存跳过，对应第 1937 步到第 4845 步。这些跳过没有把模型训回去。

第 5 个 epoch 的有效结果仍是 `experiments/ablation_f0_mini/epoch_5.pt`：mAP 0.5243，NDS 0.5116，EPA 0.3493，minADE 0.6505，minFDE 1.0758，MR 0.1229。检测略低于 1 epoch LC（mAP 0.5300，NDS 0.5080），预测是目前最好的一档。中心平均移动 0.067 m。

## 现在的更新规则

`optimizer.step()` 之前检查全部可训练梯度。出现 NaN 或 Inf 时恢复这一步之前的参数副本，写 `nonfinite.json`，进程以状态码 2 退出。记录里有模块名、epoch、step、损失、学习率和相关梯度范数。参数在 `step` 之后变成非有限值时同样回滚并退出。检测输出或官方匹配本身非有限时，不调用 `step`。

显存不足只清空缓存并重试当前步一次。再失败就停止，不再跳过后续几千步。

学习率分成两组。新增的 DeTra 和运动先验用 `--module-lr`，默认 `1e-4`。名字以 `detector.` 开头的图像融合 Decoder 和检测头用 `--detector-lr`，默认 `1e-6`。两组都线性衰减到 `1e-6`。检测梯度单独裁到 `--detector-clip`，默认 0.5；其余梯度裁到 1.0。诊断运行把检测学习率设回旧的 `1e-5`、裁剪设回 1.0，用来看旧设置会不会再次写出非有限值。

checkpoint 现在保存模型、优化器、`global_step`、epoch、PyTorch / CUDA / NumPy / Python 随机状态，以及 `finite: true`。旧的 `epoch_5.pt` 没有优化器状态，恢复时只加载权重，Adam 矩从零开始。这和当初连续训到第 6 个 epoch 的优化器不是同一条轨迹。

`memory_mode` 为 `none` 时不创建运动先验参数。F0 的可训练参数仍是 3032575，和 V3 一致。单步冒烟里当前框和轨迹 t=0 对齐，梯度范数与 V3 第 1 步同一量级：`box_head` 1.331，`decoder.1` 0.542，`prediction_heads` 0.354，`future_gru` 0.115，`future_head` 0.052。

## 诊断和第 5、10、15 epoch 对照

诊断从有限的 F0 `epoch_5.pt` 接着训第 6 个 epoch，总日程仍按 15 epoch 计算学习率。第 1620 步时两组学习率约为 `6.69e-5` 和 `6.99e-6`，`loss_det` 0.649，`loss_forecast` 2.520，官方检测损失 0.476。到第 1700 步损失仍有限，`loss_det` 0.411，`loss_forecast` 1.997，切换率 0。工作目录是 `experiments/ablation_f0_diag_mini`。这一轮还在跑。一旦出现非有限梯度，结果只保留 `nonfinite.json`，不覆盖原来的 `epoch_5.pt`。

稳定复训使用检测学习率 `1e-6`、检测梯度裁剪 0.5，从同一份 LC `epoch_1.pth` 重新开始，写到 `experiments/ablation_f0_stable_mini`。B 用同一数据顺序和种子重新训 15 epoch，写到 `experiments/ablation_b_long_mini`，避免覆盖原来只有 2 epoch 的 B。D1 已有同协议的 5、10、15 epoch 曲线，检测始终是 mAP 0.5300、NDS 0.5080。它在第 5 个 epoch 恢复时重置过 Adam，第 6 到第 15 epoch 的预测仍在变好。

| 版本 | epoch | mAP | NDS | EPA | minADE | minFDE | MR | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B | 2 | 0.5300 | 0.5080 | 0.2715 | 0.9724 | 1.6116 | 0.2689 | 已完成，V3 协议 |
| D1 | 5 | 0.5300 | 0.5080 | 0.3100 | 0.7306 | 1.1555 | 0.1580 | 框固定 |
| D1 | 10 | 0.5300 | 0.5080 | 0.3168 | 0.6980 | 1.1915 | 0.1580 | 框固定 |
| D1 | 15 | 0.5300 | 0.5080 | 0.3130 | 0.6711 | 1.1487 | 0.1548 | 按 minADE 选取 |
| F0 | 5 | 0.5243 | 0.5116 | 0.3493 | 0.6505 | 1.0758 | 0.1229 | 最后一份有限权重 |
| F0 | 6 及以后 | — | — | — | — | — | — | 旧 checkpoint 含 NaN，作废 |

B 和稳定 F0 的 5、10、15 epoch 由当前这轮训练写出后再评估。评估在训练进程退出后单独跑，避免父进程占着显存时 spconv 报 `cudaErrorMemoryAllocation`。数字出来之前，不以最后一轮或含 NaN 的文件代替上表。

## 检测和预测还缺什么

V3 已经说明：预测损失把修正后的中心当作轨迹原点时，检测 mAP 会掉。D1 断开这个原点后，检测保持 LC 的水平，预测在 5 epoch 之后超过 2 epoch 的独立 MLP。F0 的第 5 epoch 预测更好，检测没有超过 LC。单步梯度余弦接近 0，伤害来自多步积累的中心移动，不是匹配切换。

跨帧记忆默认关闭。它不参加上面的 B、D1、F0 对照。设计见 `docs/FORESIGHT_STREAMING.md`。
