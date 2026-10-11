# 训练记录 · Training log summary

[中文](#中文) · [English](#english)

## 中文

使用 AdamW，学习率 1e-4、权重衰减 1e-4、梯度裁剪 1.0。E/S/B 掩蔽 MSE 权重为 1.0，片段活性 MSE 权重为 0.2。
前向使用 bfloat16，参数与优化器使用 float32；每条流批量为 32。

每轮遍历 347,241 条 E/S/B 和 531,860 条片段，共 16,868 步。
训练经历两次断电，随后恢复模型、优化器、随机状态和游标。
两轮于 2026-10-04 13:15:17（UTC+08:00）完成，global_step=33736。
早期两次吞吐测试与正式训练记录分别保存。

| 检查点 | SHA-256 |
|---|---|
| epoch 1 | `eac54b5bf5884d02cd07b2f5da67dc5e0a181be9ad69cc32be3e951622f067e6` |
| epoch 2 | `61e685a8eae9d51b3fa878b7481c21178dc90b60f1d34440e59d2f43bf3633e7` |

DEV 规则选中 epoch 1，两轮 TEST 结果均保留。原始源码对应关系见[源码指南](ORIGINAL_SOURCES.md)，指标见[性能记录](PERFORMANCE_zh.md)。

---

## English

Training uses AdamW with learning rate 1e-4, weight decay 1e-4 and gradient clipping at 1.0. Masked E/S/B MSE has weight 1.0, and fragment-activity MSE has weight 0.2.
Forward passes use bfloat16; parameters and optimizer state use float32. Each stream has batch size 32.

Each epoch traverses 347,241 E/S/B records and 531,860 fragments in 16,868 steps.
Two power interruptions were followed by restoration of the model, optimizer, random states and sampler cursor.
Both epochs completed at 2026-10-04 13:15:17 (UTC+08:00), with global_step=33736.
The two early throughput tests are recorded separately from the training run.

| Checkpoint | SHA-256 |
|---|---|
| Epoch 1 | `eac54b5bf5884d02cd07b2f5da67dc5e0a181be9ad69cc32be3e951622f067e6` |
| Epoch 2 | `61e685a8eae9d51b3fa878b7481c21178dc90b60f1d34440e59d2f43bf3633e7` |

The DEV rule selects epoch 1, and TEST results for both epochs are retained. See the [source guide](ORIGINAL_SOURCES.md#english) for code mappings and the [performance report](PERFORMANCE_zh.md#english) for metrics.
