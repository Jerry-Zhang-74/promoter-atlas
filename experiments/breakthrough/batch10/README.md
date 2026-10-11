# batch10 · 位置 k-mer 对稳定度预测的贡献 / Positional k-mers for stability prediction

[中文](#中文) · [English](#english)

## 中文

在 45 维基线特征之外，为 400 bp 和 2000 bp 窗口增加 1,024 维位置三/四核苷酸特征，评估序列表示是否限制了 S_raw 预测。特征按基因汇总，跨 bin 的 k-mer 不计入，歧义碱基打断 k-mer。

Ridge 与 HGB 使用相同的 18,769 个有效基因和冻结划分。`preflight_summary.json` 核验序列编码与基因对齐，`tests_summary.json` 检查频率守恒、特征顺序和 TRAIN 拟合变换，`metrics.json` 保存比较结果。

- [位置特征配置](models/config.json)
- [预测结果](reports/metrics.json)
- [特征检查](reports/tests_summary.json)

---

## English

Add 1,024 positional tri-/tetranucleotide features from the 400 bp and 2000 bp windows to the 45-feature baseline, testing whether representation limits S_raw prediction. Features are aggregated by gene; k-mers crossing bin boundaries are excluded and ambiguous bases interrupt a k-mer.

Ridge and HGB use the same 18,769 eligible genes and frozen split. Preflight verifies encoding and gene alignment; tests cover frequency conservation, feature order, and TRAIN-fitted transforms. `metrics.json` stores the model comparison.

- [Positional-feature configuration](models/config.json)
- [Prediction results](reports/metrics.json)
- [Feature checks](reports/tests_summary.json)
