# batch58 · 吞吐选择与跨进程训练恢复 / Throughput selection and cross-process recovery

[中文](#中文) · [English](#english)

## 中文

在修正后的 531,860 条 TRAIN 片段上评测有效 batch，并验证受控停止后跨进程恢复。新 checkpoint 保存优化器、随机状态、采样位置与覆盖信息，采用两代保留和原子指针更新。

实测选择有效 batch64；两次完整 epoch 共覆盖 1,063,720 条样本，记录 16,622 个优化器步，训练耗时约 205 秒。`TRAINING_DONE.json` 保存覆盖与恢复结论，current_training_status 记录配置、资源与 checkpoint 来源。

- [训练与恢复结果](reports/TRAINING_DONE.json)
- [资源与 checkpoint 记录](reports/current_training_status.json)
- [恢复机制说明](../shared/runs/code/batch58_throughput/README.md)

---

## English

Profile effective batch sizes on the corrected 531,860-fragment TRAIN set and verify recovery in a new process after a controlled stop. The checkpoint stores optimizer, RNG, sampling position, and coverage, with two-generation retention and atomic pointer updates.

Effective batch64 was selected. Two complete epochs cover 1,063,720 samples and 16,622 optimizer steps in about 205 seconds. `TRAINING_DONE.json` records coverage and recovery; current_training_status retains configuration, resource use, and checkpoint provenance.

- [Training/recovery result](reports/TRAINING_DONE.json)
- [Resource/checkpoint record](reports/current_training_status.json)
- [Recovery mechanism](../shared/runs/code/batch58_throughput/README.md)
