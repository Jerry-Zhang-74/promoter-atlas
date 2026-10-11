# 基因级基线与集成对照 · Gene-level baselines and ensemble comparisons

## 中文

这里保存两张可直接阅读的评估表，用于检查模型复杂度增加之后，E、S、B 三个轴分别获得了多少收益。

| 文件 | 怎么读 |
|---|---|
| [baselines_kmer_gc.tsv](baselines_kmer_gc.tsv) | 5-mer Ridge 与 GC-only 的逐轴基因级 Spearman、R² 和 Spearman 区间；用作序列组成基线 |
| [genelevel_ensemble_bootstrap.tsv](genelevel_ensemble_bootstrap.tsv) | 历史集成模型的基因级 Spearman、R²、样本量和 bootstrap 区间 |

这些表各自保留评估口径和样本量。与某个模型版本比较时，先核对测试集合、基因聚合和区间计算方式；详细版本评估放在[各版 evaluation 目录](../README.md)。

## English

These two evaluation tables help examine what added model complexity contributes to E, S and B separately.

| File | How to use it |
|---|---|
| [baselines_kmer_gc.tsv](baselines_kmer_gc.tsv) | Per-axis gene-level Spearman, R² and Spearman intervals for 5-mer Ridge and GC-only sequence-composition baselines |
| [genelevel_ensemble_bootstrap.tsv](genelevel_ensemble_bootstrap.tsv) | Gene-level Spearman, R², sample counts and bootstrap intervals for a historical ensemble |

Each table retains its evaluation definitions and counts. For comparisons with a model version, align the test set, gene aggregation and interval calculation. Detailed version evaluations are linked from the [model catalog](../README.md#english).
