# batch24 · HEK293 核表示与线性读出的比较 / HEK293 kernel representations versus linear readouts

[中文](#中文) · [English](#english)

## 中文

沿用 batch22 的 673 条 HEK293 序列和五折划分，比较 H1/H2 核表示与缓存的 R1 线性基线。每个核方案五次 Ridge 拟合，固定记录与此前结果的配对差异。

OOF Spearman 为 H1 0.5264、H2 0.5200，缓存 R1 为 0.4566；对应 MAE 为 0.5397、0.5417、0.5717。该阶段属于已有队列上的迭代比较，结果中保留完整选择条件和逐折表现。

- [核表示比较](reports/results.json)
- [拟合次数记录](reports/attempts_summary.json)

---

## English

Reuse batch22's 673 HEK293 sequences and five folds to compare H1/H2 kernel representations with cached linear baseline R1. Each kernel recipe uses five Ridge fits, with paired differences against the earlier predictions.

OOF Spearman is 0.5264 for H1, 0.5200 for H2, and 0.4566 for cached R1; MAE is 0.5397, 0.5417, and 0.5717. This is an iterative comparison on the existing cohort; results retain selection conditions and fold-level performance.

- [Kernel comparison](reports/results.json)
- [Fit accounting](reports/attempts_summary.json)
