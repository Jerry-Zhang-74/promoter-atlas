# batch31 · PARM 外部模型迁移与 SuRE 重复性诊断 / PARM transfer and SuRE replicate diagnostics

[中文](#中文) · [English](#english)

## 中文

用冻结 PARM 模型预测 673 条 HEK293 序列及本地 293FT 构建体，与已有 R1/H1 基线比较。公开 HEK293 队列上的 PARM Spearman 为 0.6906；本地 ACTB、GAPDH、CMV 家族的方向准确率分别为 0、0.2、1，家族宏平均为 0.4。

另一项分析只读取 SuRE 的冻结 TRAIN 行，检查固定代理标签的 RNA 重复一致性：7,690 行的重复 Spearman 为 0.1176，双零比例约 76.8%。该诊断用于判断测量噪声与后续训练的关系。

- [PARM 迁移结果](reports/metrics.json)
- [SuRE 重复性](reports/sure_qc_metrics.json)
- [冻结权重推理](code/batch31/run_parm_inference.py)

---

## English

Apply frozen PARM to 673 HEK293 sequences and local 293FT constructs, comparing with existing R1/H1 baselines. PARM Spearman on the public HEK293 cohort is 0.6906; local directional accuracy is 0, 0.2, and 1 for ACTB, GAPDH, and CMV, respectively, with a family macro average of 0.4.

A separate analysis reads only frozen SuRE TRAIN rows to assess RNA-replicate agreement under the fixed proxy target. Across 7,690 rows, replicate Spearman is 0.1176 and approximately 76.8% have two zero counts. This diagnostic informs assessment of measurement noise before further training.

- [PARM transfer results](reports/metrics.json)
- [SuRE repeatability](reports/sure_qc_metrics.json)
- [Frozen-weight inference](code/batch31/run_parm_inference.py)
