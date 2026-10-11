# batch30 · SuRE 原生长度输入与训练恢复修复 / SuRE native-length input and recovery repairs

[中文](#中文) · [English](#english)

## 中文

统一训练、验证、测试和 FASTA 的输入构建：按真实长度组成批次，避免补齐区域进入池化；按组大小设置权重，并修订单样本 BatchNorm、checkpoint 保存和恢复状态。

同时检查跨染色体块的相似序列，按整组处理后冻结 TRAIN chr1–16、DEV chr17–19、TEST chr20–22/X/Y。该批保存修复代码与划分准备记录，具体入口为 `b30_data.py`、`b30_train.py` 和 `split_freeze.json`。

- [原生长度输入实现](../shared/runs/code/batch30/b30_data.py)
- [训练与恢复实现](../shared/runs/code/batch30/b30_train.py)
- [划分冻结](reports/split_freeze.json)

---

## English

Unify input construction across training, validation, testing, and FASTA: batch sequences at their actual length so padding stays outside pooling. Repair group weighting, singleton BatchNorm handling, checkpoint saving, and resume state.

Audit similar sequences crossing chromosome blocks and freeze TRAIN chr1–16, DEV chr17–19, and TEST chr20–22/X/Y after whole-group handling. This batch retains repaired code and split-preparation records; start with `b30_data.py`, `b30_train.py`, and `split_freeze.json`.

- [Native-length input](../shared/runs/code/batch30/b30_data.py)
- [Training and recovery](../shared/runs/code/batch30/b30_train.py)
- [Frozen split](reports/split_freeze.json)
