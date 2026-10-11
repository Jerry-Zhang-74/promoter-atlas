# batch54 · HEK293 模型的独立指标复算 / Independent metrics for HEK293 models

[中文](#中文) · [English](#english)

## 中文

独立读取 R0、Nabs 和 Njoint 的 TEST 预测，重算活性 Spearman、组等权 MAE、RMSE、固定 ID 的 top-10%、配对 MAE 与严格符号准确率。真实零差值和预测平局单独处理，bootstrap 保留重复抽样组的次数。

三个 JSON 分别保存各模型的结果，可与 batch52 汇总逐项对照；数据与权重检查、数值夹具和评估实现集中在 batch54_review。

- [R0 独立结果](reports/R0_TEST_parent_metrics.json)
- [Nabs 独立结果](reports/Nabs_TEST_parent_metrics.json)
- [Njoint 独立结果](reports/Njoint_TEST_parent_metrics.json)
- [指标实现说明](../shared/runs/code/batch54_review/README.md)

---

## English

Independently read TEST predictions from R0, Nabs, and Njoint and recompute activity Spearman, group-equal MAE, RMSE, fixed-ID top-decile metrics, pair MAE, and strict directional accuracy. True-zero differences and prediction ties are handled explicitly, and bootstrap retains repeated group draws.

Three JSON files store model-specific results for direct comparison with batch52. Data/weight checks, numerical fixtures, and evaluation code are collected in batch54_review.

- [Independent R0 metrics](reports/R0_TEST_parent_metrics.json)
- [Independent Nabs metrics](reports/Nabs_TEST_parent_metrics.json)
- [Independent Njoint metrics](reports/Njoint_TEST_parent_metrics.json)
- [Metric implementation](../shared/runs/code/batch54_review/README.md)
