# batch22 · HEK293 片段活性的五折读出 / Five-fold readouts of HEK293 fragment activity

[中文](#中文) · [English](#english)

## 中文

在 673 条 HEK293 片段上比较常数读出 R0、86 维表示 R1 和 214 维表示 R2，使用五折分组交叉验证、固定 alpha=100。目标单位为 `log2(hek exp/input)`。

共执行 10 次 Ridge 拟合。合并 OOF Spearman 为 R1 0.4566、R2 0.4617；MAE 为 0.5717、0.5946。完整记录包括各折误差、top-decile 和分组 bootstrap，便于同时考察排序与数值误差。

- [五折读出结果](reports/results.json)
- [拟合清单](reports/fits_summary.json)

---

## English

Compare constant readout R0, 86-feature R1, and 214-feature R2 on 673 HEK293 fragments, using five grouped folds and fixed alpha=100. The target is `log2(hek exp/input)`.

Ten Ridge fits were executed. Pooled OOF Spearman is 0.4566 for R1 and 0.4617 for R2; MAE is 0.5717 and 0.5946. Records include fold-level errors, top-decile metrics, and grouped bootstrap for both ranking and quantitative error.

- [Five-fold readout results](reports/results.json)
- [Fit summary](reports/fits_summary.json)
