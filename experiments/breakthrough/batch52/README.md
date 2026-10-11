# batch52 · 修正 HEK293 目标的活性与配对模型 / Activity and pairwise models with corrected HEK293 targets

[中文](#中文) · [English](#english)

## 中文

构建完整的参考序列片段队列，并将目标固定为作者的 `Log2RPM_HEK293`。比较 R0 序列统计 Ridge、Nabs 绝对活性 CNN 与加入 short−long 配对损失的 Njoint；cohort_v2 记录修正后的标签、分组和划分。

最终 TEST 有 117,155 条片段。Nabs/Njoint 活性 Spearman 为 0.5749/0.5799；配对组平均 MAE 为 0.8398/0.8388。预设的配对提升条件未全部满足，最终记录为 LIMITED；独立复算见 batch54。

- [最终 TEST 比较](reports/final_test_metrics.json)
- [修正队列构建](code/batch52_data_v2/build_cohort_v2.py)
- [发布来源记录](code/batch52_data_v2/release_provenance.py)

---

## English

Build the full reference-sequence fragment cohort with the author's `Log2RPM_HEK293` target. Compare sequence-statistic Ridge R0, absolute-activity CNN Nabs, and Njoint with an additional short−long pairwise loss. cohort_v2 records corrected labels, groups, and splits.

The final TEST has 117,155 fragments. Nabs/Njoint activity Spearman is 0.5749/0.5799, with pair-group MAE 0.8398/0.8388. The preset pair-improvement conditions were not all met and the final record is LIMITED. Batch54 independently recomputes the metrics.

- [Final TEST comparison](reports/final_test_metrics.json)
- [Corrected-cohort construction](code/batch52_data_v2/build_cohort_v2.py)
- [Release provenance](code/batch52_data_v2/release_provenance.py)
