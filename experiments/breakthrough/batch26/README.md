# batch26 · STR 变体的序列重建与对照训练 / STR-variant reconstruction and controlled training

[中文](#中文) · [English](#english)

## 中文

将变体表与设计探针逐行对齐，重建 STR 序列并进行无标签的家族分组与划分。训练设计包含四次拟合：一个组成 Ridge、两个真实标签 CNN 种子，以及一个打乱标签的 CNN 对照。

`selection_manifest.json` 记录选择结果，`test_metrics.json` 保存最终测试。各次尝试在 runtime/attempts 中单独留档，可对应到真实标签和打乱标签对照。

- [模型选择记录](reports/selection_manifest.json)
- [测试结果](reports/test_metrics.json)
- [变体序列重建](../shared/runs/code/batch26/b26_data.py)

---

## English

Join variant rows to design probes, reconstruct STR sequences, and form label-free family groups and splits. The four-fit design comprises a composition Ridge, two CNN seeds with real labels, and a shuffled-label CNN control.

`selection_manifest.json` records selection and `test_metrics.json` stores final testing. Individual attempts under runtime/attempts map results to real-label and shuffled-label controls.

- [Model selection](reports/selection_manifest.json)
- [Test results](reports/test_metrics.json)
- [Variant reconstruction](../shared/runs/code/batch26/b26_data.py)
