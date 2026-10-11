# batch20 · 原始序列输入的双任务推理入口 / Raw-sequence inference for two tasks

[中文](#中文) · [English](#english)

## 中文

为已选中的基因 E/S/B 模型和 MPRA 片段模型建立原始序列输入入口。分别实现序列预处理、特征生成和任务路由，并把输入哈希、模型来源与输出绑定。

该批重点是从 FASTA/序列到已有模型预测的可用流程。`artifact_manifest.json` 中的基因模型回放指标与 rc2 对齐；初步诊断记录在 `preliminary_metrics.json`，对应源码位于共享历史代码区。

- [原始序列命令入口](../shared/runs/code/batch20/b20_cli.py)
- [模型与回放清单](reports/artifact_manifest.json)

---

## English

Create raw-sequence entry points for the selected gene E/S/B and MPRA fragment models. Separate preprocessing, feature construction, and task routing, and attach input hashes and model provenance to outputs.

This batch makes the FASTA/sequence-to-prediction workflow usable with existing models. Gene-model replay metrics in `artifact_manifest.json` agree with rc2; `preliminary_metrics.json` retains diagnostics. Source is in the shared historical code area.

- [Raw-sequence CLI](../shared/runs/code/batch20/b20_cli.py)
- [Model and replay manifest](reports/artifact_manifest.json)
