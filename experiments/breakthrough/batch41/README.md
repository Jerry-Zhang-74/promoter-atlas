# batch41 · HEK293FT 寡核苷酸的外部模型比较 / External-model comparison on HEK293FT oligos

[中文](#中文) · [English](#english)

## 中文

在已匹配作者设计与实测读出的 HEK293FT 寡核苷酸队列上，计算 PARM、H1 和 GC 基线预测。评估代码按原始 MPRAnalyze 活性单位和冻结序列身份对齐，并输出逐项诊断与图表。

`diagnostic_manifest.json` 索引本轮模型和输入；`metrics.py`/`metrics2.py` 保存指标计算，独立复核的来源与结果分别位于 `rv01_source.json` 和 `rv03_metrics.json`。

- [模型和输入清单](reports/diagnostic_manifest.json)
- [指标实现](code/batch41/metrics2.py)
- [独立指标复核](runtime/batch41_review/rv03_metrics.json)

---

## English

Compute PARM, H1, and GC-baseline predictions for HEK293FT oligos whose author designs have been joined to measured readouts. Evaluation aligns frozen sequence identities with the original MPRAnalyze activity units and produces diagnostics and figures.

`diagnostic_manifest.json` indexes models and inputs. `metrics.py`/`metrics2.py` implement evaluation; independent source and metric checks are retained in `rv01_source.json` and `rv03_metrics.json`.

- [Model/input manifest](reports/diagnostic_manifest.json)
- [Metric implementation](code/batch41/metrics2.py)
- [Independent metric review](runtime/batch41_review/rv03_metrics.json)
