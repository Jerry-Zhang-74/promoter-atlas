# batch57 · 训练锁、输入校验与 checkpoint 修复 / Training-lock, input, and checkpoint repairs

[中文](#中文) · [English](#english)

## 中文

修复 batch53 核心的进程锁身份、接受清单哈希、初始化一致性、top-10% 平局规则、优化器步数、逐 epoch 配对顺序、checkpoint 校验和序列输入处理。旧版尚未实现的恢复入口在预约计算前给出明确错误。

修订实现集中在 shared/batch57_model_core_rc2，保持既有模型与配置接口兼容；README 逐项列出 D1–D12 改动和软件检查记录。完整的跨进程恢复机制随后在 batch58 实现。

- [D1–D12 修订](../shared/runs/code/batch57_model_core_rc2/README.md)
- [checkpoint 实现](../shared/runs/code/batch57_model_core_rc2/checkpoint.py)
- [后续恢复实现](../batch58/README.md)

---

## English

Repair batch53 process-lock identity, acceptance-manifest hashes, initialization matching, top-decile ties, optimizer-step accounting, per-epoch pair ordering, checkpoint checksums, and sequence input handling. The unfinished recovery entry point fails explicitly before reserving computation.

The revised implementation is in shared/batch57_model_core_rc2 and preserves the existing model/configuration interface. Its README lists D1–D12 repairs and software checks. Batch58 subsequently implements full cross-process recovery.

- [D1–D12 repairs](../shared/runs/code/batch57_model_core_rc2/README.md)
- [Checkpoint implementation](../shared/runs/code/batch57_model_core_rc2/checkpoint.py)
- [Subsequent recovery implementation](../batch58/README.md)
