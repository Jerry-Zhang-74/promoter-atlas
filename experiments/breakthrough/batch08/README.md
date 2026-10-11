# batch08 · 稳定度标签的缩放与并列值审计 / Stability-label scaling and tie audit

[中文](#中文) · [English](#english)

## 中文

从 58,988 个基因的标签构建过程追踪 Score_S，核对其中 18,947 个有定义 S 的基因。检查缺失值参与排名后形成的有效区间、六位小数舍入产生的并列，以及仿射重标度对 Spearman、MAE、MSE、R² 的影响。

审计给出有效 Score_S 区间约 [0.678811, 1]；12 项尺度性质检查均有逐项结果。另记录 B_raw 的并列结构和 E×S 排序对 S 缩放的敏感性，供后续标签与目标选择使用。

- [标签审计](reports/audit_summary.json)
- [尺度性质检查](reports/scale_invariance_results.json)

---

## English

Trace Score_S construction across 58,988 genes, including 18,947 with defined S. Audit the effective range after missing-value ranking, ties introduced by six-decimal rounding, and the effects of affine rescaling on Spearman, MAE, MSE, and R².

The defined Score_S range is approximately [0.678811, 1]. Twelve scale-property checks are recorded individually. The audit also examines B_raw ties and sensitivity of E×S ranking to S scaling, informing subsequent target choices.

- [Label audit](reports/audit_summary.json)
- [Scale-property checks](reports/scale_invariance_results.json)
