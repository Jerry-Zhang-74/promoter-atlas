# Motif 扫描规则 v1 · Motif scanning rules v1

[中文](#中文) · [English](#english)

## 中文

PWM 来源为 JASPAR 2026 CORE vertebrates nonredundant，文件为 `data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt`。

TF 名称精确且唯一匹配时自动确定 matrix ID；歧义或缺失匹配标记 pending_manual_review。

扫描正反双链。relative_score 为观察分数除以理论最大分数，宽松阈值0.80、严格阈值0.90。本阶段冻结 schema，后续生成 promoter motif hits、aggregate 和 grammar 表。

---

## English

Use the JASPAR 2026 CORE vertebrate nonredundant collection at `data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt`.

An exact, unique TF-name match resolves the matrix ID automatically. Ambiguous or missing matches receive `pending_manual_review`.

Scan both strands. The relative score is the observed score divided by its theoretical maximum; loose and strict thresholds are 0.80 and 0.90. This record freezes the schema. Subsequent scanning produces promoter motif-hit, aggregate, and grammar tables.
