# 交付源码快照 · Delivery source snapshots

## 中文

[`submission-source/`](submission-source/) 保存竞赛源码交付时收集的原始脚本副本。文件名的 `orig_` 前缀对应交付包里的来源快照，便于将交付实现与研究过程逐项对照。

| 文件组 | 对应工作 |
|---|---|
| `orig_download_*`、`orig_fetch_*`、`orig_build_*` | 取得基因组与直系同源关系，构建标签、序列窗口和训练划分 |
| `orig_b51_*`、`orig_b52_*` | 构建并核对片段 cohort，保存修正后的标签与划分 |
| `orig_61_*`、`orig_62_*` | 双流联合训练的架构、数据适配和检查点实现 |
| `orig_b63_*` | 发布模型对应的联合训练循环与数据读取实现 |

阅读某轮实验时，可从[实验目录](../experiments/README.md)查看完整的代码、配置和报告关联；日常预测与训练使用[当前 Python 包](../src/promoter_atlas/README.md)及[复现指南](../docs/REPRODUCIBILITY.md)。交付快照的来源记录见 [`PUBLIC_EXPORT_PROVENANCE.json`](../artifacts/manifests/PUBLIC_EXPORT_PROVENANCE.json)。

## English

[`submission-source/`](submission-source/) preserves original script copies collected for the competition source delivery. The `orig_` prefix identifies the source snapshots bundled with that delivery, allowing its implementation to be traced back to the research workflow.

| File group | Associated work |
|---|---|
| `orig_download_*`, `orig_fetch_*`, `orig_build_*` | Acquire genomes and ortholog relationships; construct labels, sequence windows and training splits |
| `orig_b51_*`, `orig_b52_*` | Build and verify the fragment cohort, including corrected labels and splits |
| `orig_61_*`, `orig_62_*` | Dual-stream joint-training architecture, data adapters and checkpoint handling |
| `orig_b63_*` | Joint-training loop and data readers associated with the released model |

For a complete experiment's code, configuration and reports, start with the [experiment catalog](../experiments/README.md#english). Use the [current Python package](../src/promoter_atlas/README.md#english) and [reproduction guide](../docs/REPRODUCIBILITY.md#english) for prediction and training. Delivery provenance is recorded in [`PUBLIC_EXPORT_PROVENANCE.json`](../artifacts/manifests/PUBLIC_EXPORT_PROVENANCE.json).
