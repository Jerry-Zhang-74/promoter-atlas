# Motif 特征流程 · Motif feature pipeline

[中文](#中文) · [English](#english)

## 中文

### Phase 6b 特征设计

以 up200 的 `record_id` 和已有序列 memmap 为基础，方案包含18个 motif、10对 motif 和256个特征。分层为 A_core（优先候选）、B_support（广度/稳定性支持）和 C_control（QC对照）。

单 motif 特征为 `count_loose`、`count_strict`、`presence_strict`、`density_loose`、`best_score_norm`、`best_pos_rel_tss`。位置分为远端[-200,-101]、中段[-100,-41]和近端[-40,-1]。

配对特征为 `present`、`min_spacing_bp`、`spacing_bin`、`same_orientation_flag`。

下一步确定 PWM accession、归一化和阈值，扫描 memmap/FASTA，在 features/motif 输出 hit、aggregate、grammar 表；下游训练入口为 train_promoter_model_v2.py。

---

## English

### Phase 6b design

The schema uses up200 `record_id` values and the existing sequence memmap. It contains 18 motifs, 10 motif pairs, and 256 features. Motif tiers are `A_core` (priority candidates), `B_support` (breadth/stability support), and `C_control` (QC controls).

Single-motif features: `count_loose`, `count_strict`, `presence_strict`, `density_loose`, `best_score_norm`, and `best_pos_rel_tss`. Position zones are distal `[-200,-101]`, middle `[-100,-41]`, and proximal `[-40,-1]`.

Pair features: `present`, `min_spacing_bp`, `spacing_bin`, and `same_orientation_flag`.

Next steps are to resolve PWM accession IDs, define normalization and thresholds, scan the memmap/FASTA, and write hit, aggregate, and grammar tables under `features/motif`. The downstream training entry point is `train_promoter_model_v2.py`.
