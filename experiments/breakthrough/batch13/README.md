# batch13 · 稳定度特征的五折组间验证 / Five-fold grouped validation of stability features

[中文](#中文) · [English](#english)

## 中文

将 18,769 个有效 S 基因按组件分为五折，成对比较 45 维基线与完整位置特征 HGB；每折的标准化和模型均仅使用其余四折。

独立复算得到 pooled OOF Spearman：基线 0.4032、完整特征 0.4571，差值 0.0539。五折的 Spearman 差值均为正，逐折误差和合并结果分别记录。

- [五折结果](reports/metrics.json)
- [独立复算](runtime/batch13/review_independent/verify_result.json)

---

## English

Split the 18,769 eligible S genes into five component-preserving folds and compare 45-feature baseline HGB against the complete positional representation. Each fold fits transforms and models on the other four folds.

Independent recomputation gives pooled OOF Spearman 0.4032 for the baseline and 0.4571 for complete features, a gain of 0.0539. All five fold-level Spearman differences are positive; both per-fold errors and pooled results are retained.

- [Five-fold results](reports/metrics.json)
- [Independent recomputation](runtime/batch13/review_independent/verify_result.json)
