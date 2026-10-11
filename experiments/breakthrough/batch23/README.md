# batch23 · 本地构建体序列与荧光端点对齐 / Local construct sequences and fluorescence endpoints

[中文](#中文) · [English](#english)

## 中文

从已有实验记录找回 14 个本地构建体的 DNA 序列和 D28 荧光端点，修正此前序列可用性的判断。以完整实际插入序列输入 batch22 保存的五折读出，检查每个构建体的预测与端点对应关系。

工作分为本地序列/端点对齐、冻结模型诊断和公开来源盘点，分别由阶段清单和公开来源目录记录。该步骤为后续 293FT 应用情境审计提供输入。

- [序列与端点对齐](../shared/runs/code/batch23/b23_reconcile.py)
- [本地诊断清单](reports/stage1_manifest.json)
- [公开数据来源](reports/public_source_inventory.json)

---

## English

Recover DNA sequences and D28 fluorescence endpoints for 14 local constructs from existing records, correcting the earlier assessment of sequence availability. Feed complete measured inserts into the five saved batch22 readouts and inspect construct-level prediction/endpoint alignment.

Work is divided into local sequence/endpoint reconciliation, frozen-model diagnostics, and public-source inventory, recorded in stage manifests and the source catalog. These records support the later 293FT assay-context audit.

- [Sequence/endpoint reconciliation](../shared/runs/code/batch23/b23_reconcile.py)
- [Local diagnostic manifest](reports/stage1_manifest.json)
- [Public-source inventory](reports/public_source_inventory.json)
