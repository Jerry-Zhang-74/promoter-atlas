# batch37 · MAPT 变体效应的冻结模型诊断 / Frozen-model diagnostics of MAPT variant effects

[中文](#中文) · [English](#english)

## 中文

对 batch36 的 ref/alt 序列计算冻结模型预测差值，分别检查已知变体和饱和变体，并与 ΔGC、Δ长度及固定方向基线比较。报告按有效序列、退化输入、SNV 等子集拆分。

已知变体全部有限标签的 Spearman 为 0.0399；饱和变体为 −0.1599。`metrics.json` 保留这些低相关及负相关结果、预测零差值和缺失标签数量，逐项复算记录位于 runtime/review_scratch。

- [变体效应诊断](reports/metrics.json)
- [ref/alt 推理](code/batch37/infer.py)
- [指标复算](runtime/batch37/review_scratch/recompute_metrics_result.json)

---

## English

Compute frozen-model ref/alt prediction differences on batch36 sequences, examining known variants and saturation variants against ΔGC, Δlength, and fixed-direction baselines. Reports separate valid sequences, degenerate inputs, SNVs, and other subsets.

Spearman is 0.0399 for known variants with finite labels and −0.1599 for saturation variants. `metrics.json` retains these weak/negative results along with zero prediction differences and missing-label counts; independent recomputation is under runtime/review_scratch.

- [Variant-effect diagnostics](reports/metrics.json)
- [Ref/alt inference](code/batch37/infer.py)
- [Metric recomputation](runtime/batch37/review_scratch/recompute_metrics_result.json)
