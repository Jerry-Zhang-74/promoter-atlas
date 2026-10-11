# batch11 · 三核苷酸与四核苷酸特征消融 / Tri- and tetranucleotide feature ablation

[中文](#中文) · [English](#english)

## 中文

拆开 batch10 的位置特征，分别比较三核苷酸、四核苷酸、完整特征和 45 维基线。两个消融表示各为 557 维，并分别拟合 TRAIN 标准化；Ridge 与 HGB 共享目标和划分。

完整特征 HGB 的 TEST S_raw Spearman 为 0.4744。逐臂训练、验证、测试结果及特征名检查保留在 reports，变换参数保存在单次拟合目录中。

- [消融结果](reports/metrics.json)
- [特征分组核对](reports/preflight_summary.json)

---

## English

Separate batch10's positional features into trinucleotide and tetranucleotide representations and compare them with the complete representation and 45-feature baseline. Each ablation has 557 features and its own TRAIN-fitted scaler; Ridge and HGB share the target and split.

The complete-feature HGB has TEST S_raw Spearman 0.4744. Reports retain per-arm TRAIN/DEV/TEST results and feature-name checks; the fit directory stores transformation parameters.

- [Ablation results](reports/metrics.json)
- [Feature-group verification](reports/preflight_summary.json)
