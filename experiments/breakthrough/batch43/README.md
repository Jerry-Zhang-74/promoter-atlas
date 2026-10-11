# batch43 · 片段预测误差的组内与组间分解 / Within- and between-group error decomposition

[中文](#中文) · [English](#english)

## 中文

重放 batch42 已保存的 R0、R1、Rnull 预测，将总均方误差分解为组间与组内部分，同时比较按行加权和各组等权两种汇总。TRAIN 与 TEST 分开报告。

`phaseB_summary.json` 给出各模型的误差比例、分解恒等式残差和 318 条测试预测的回放误差。分析用于定位误差来自组均值偏差还是同组片段差异。

- [误差分解结果](reports/phaseB_summary.json)
- [输入校验清单](reports/sha_manifest.json)

---

## English

Replay saved batch42 R0, R1, and Rnull predictions and decompose mean-squared error into between- and within-group components, using both row-weighted and group-equal summaries. TRAIN and TEST are reported separately.

`phaseB_summary.json` gives error proportions, numerical residuals of the decomposition identity, and replay differences for all 318 test predictions. The analysis distinguishes group-mean bias from variation among fragments within a group.

- [Error decomposition](reports/phaseB_summary.json)
- [Input checksums](reports/sha_manifest.json)
