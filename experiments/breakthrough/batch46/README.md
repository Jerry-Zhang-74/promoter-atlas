# batch46 · 同代候选的模型选择与实测收益 / Model selection among candidates from the same generation

[中文](#中文) · [English](#english)

## 中文

将同一优化谱系、同一代的候选构成固定三元组，比较按 K562 PARM 分数、GC 含量或 HepG2 分数选择的候选，其实测活性相对于均匀选择的变化。另计算实测最优候选与各策略之间的差距。

来源为 GSE301246 的合成启动子文库，保留 batch45 的 barcode 聚合方法。`metric_definitions.json` 定义 argmax、平局和三元组差值，`source_identity.json` 核对原始文件与 889 条 DNA 的身份。

- [候选选择定义](reports/metric_definitions.json)
- [原始来源身份](reports/source_identity.json)

---

## English

Form fixed candidate triplets within an optimization lineage and generation. Compare measured activity after selecting by K562 PARM score, GC content, or HepG2 score against uniform selection, and calculate the gap to the measured-best candidate.

The source is the GSE301246 synthetic-promoter library, using batch45 barcode aggregation. `metric_definitions.json` defines argmax, ties, and triplet contrasts; `source_identity.json` verifies the raw file and 889 DNA identities.

- [Candidate-selection definitions](reports/metric_definitions.json)
- [Raw-source identity](reports/source_identity.json)
