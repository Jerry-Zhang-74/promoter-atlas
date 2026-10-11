# batch38 · MAPT 设计身份与基因组位置复查 / MAPT design identity and genomic-position audit

[中文](#中文) · [English](#english)

## 中文

进一步核对 MAPT 设计序列的身份、可变位置、基因组一致性和 tiling 关系，将位置关系用于分组检查。分析同时回查上一批预测缓存，并检查另一套寡核苷酸工作表的数据准备条件。

代码按设计概览、已知变体身份、基因组比对、分组和最终审计拆分。阅读 `step9_finalize.py` 可追踪审计结论与缓存对账如何汇总，`sha_manifest.json` 保存该轮材料的校验索引。

- [审计结论汇总](code/batch38/step9_finalize.py)
- [基因组一致性](code/batch38/step2d_genome_consistency.py)
- [材料索引](reports/sha_manifest.json)

---

## English

Recheck MAPT design identity, variable positions, genomic consistency, and tiling relationships, then use positional relationships for group auditing. The analysis also reconciles the preceding prediction cache and examines readiness of another oligo worksheet.

Modules cover design overview, known-variant identity, genomic alignment, grouping, and final auditing. `step9_finalize.py` shows how the audit decision and cache reconciliation are assembled; `sha_manifest.json` indexes this stage's materials.

- [Audit assembly](code/batch38/step9_finalize.py)
- [Genomic consistency](code/batch38/step2d_genome_consistency.py)
- [Artifact index](reports/sha_manifest.json)
