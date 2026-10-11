# 标签构建记录 · Label-build notes

[中文](#中文) · [English](#english)

## 中文

本记录采用 `candidate_E_long_table_B_gated_detectable_tissue_MAD`。

输入为 gene×tissue 的 `mean_tpm` 长表。同一 gene_id+tissue 有重复且同时含零值和非零值时，优先保留非零观测进行归并。S_raw 未定义时 Score_S 保持缺失，对应 Score_HKG 也保持缺失。

---

## English

The active label route in this record is `candidate_E_long_table_B_gated_detectable_tissue_MAD`.

The input is a gene-by-tissue long table using `mean_tpm`. When duplicate gene/tissue rows contain both zero and positive values, collapse them with preference for the nonzero observation. Undefined `S_raw` produces a missing `Score_S`; the corresponding `Score_HKG` also remains missing.
