# batch42 · HEK293FT 片段读出的留出组测试 / Held-out group testing of HEK293FT readouts

[中文](#中文) · [English](#english)

## 中文

在 5 条染色体、21 个组、318 条留出片段上比较 R1、R0、打乱对照和 TRAIN 常数基线，目标使用作者 MPRAnalyze getAlpha 原始单位。报告同时给出总体与组等权误差、组内排序、top-10% 和配对区间。

R1 的 pooled Spearman 为 0.0911、MAE 为 0.1662；R0 为 0.1584、0.1072，TRAIN 常数基线 MAE 为 0.0976。结果保留读出未超过简单基线的情形，供下一批误差分解分析。

- [留出组测试结果](reports/metrics.json)
- [误差分解](../batch43/README.md)

---

## English

Compare R1, R0, a shuffled control, and a TRAIN constant baseline on 318 held-out fragments spanning 21 groups and five chromosomes, using original MPRAnalyze getAlpha units. Report pooled and group-equal error, within-group ranking, top-decile metrics, and paired intervals.

R1 has pooled Spearman 0.0911 and MAE 0.1662; R0 has 0.1584 and 0.1072. The TRAIN constant baseline has MAE 0.0976. These outcomes motivate the following error-decomposition analysis.

- [Held-out group results](reports/metrics.json)
- [Error decomposition](../batch43/README.md)
