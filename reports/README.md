# 复现记录与模型汇总 · Reproduction records and model summaries

## 中文

这里保存跨目录检查和跨模型汇总。单个模型的评估结果随版本保存在 [`models/versions/`](../models/README.md)，单轮实验的日志、分析与图表保存在对应的 [`experiments/`](../experiments/README.md) 目录。

### 发布与复现检查

| 记录 | 检查对象 |
|---|---|
| [`reproducibility/initial_inference.json`](reproducibility/initial_inference.json) | 首次公开时的权重身份、8 条 E/S/B 与 3 条活性预测、CPU 环境版本 |
| [`reproducibility/publication_validation.json`](reproducibility/publication_validation.json) | 公开源码指纹、训练归档恢复、训练输入计数、暖启动检查与迁移后的示例预测 |
| [`reproducibility/layout_20261011.json`](reproducibility/layout_20261011.json) | 目录整理后的文件完整性、旧路径解析、历史入口、数据计数和示例预测 |

每份 JSON 的日期、对象和检查项描述该次执行的结果。当前仓库的示例复现可运行 `python scripts/verify_release.py`；数据恢复后的检查可运行 `python scripts/check_training_ready.py --data-root .`，详见[复现指南](../docs/REPRODUCIBILITY.md)。

### 历史路线汇总

[`research/model_registry_phase6.tsv`](research/model_registry_phase6.tsv) 汇总早期序列、架构特征和融合路线，列出当时的模型状态、选型规则、指标来源及 E/S/B 指标。表内使用历史模型名称和路径，可通过[版本索引](../docs/VERSION_HISTORY.md)及[路径清单](../artifacts/manifests/PATH_MAP.json)定位当前文件。

## English

This directory holds checks spanning the repository and summaries spanning models. Version-specific evaluations live with [`models/versions/`](../models/README.md#english); individual experiment logs, analyses and figures live with the corresponding [`experiments/`](../experiments/README.md#english) directory.

### Publication and reproduction checks

| Record | Checked material |
|---|---|
| [`reproducibility/initial_inference.json`](reproducibility/initial_inference.json) | Initial public weight identity, eight E/S/B predictions, three activity predictions and CPU runtime versions |
| [`reproducibility/publication_validation.json`](reproducibility/publication_validation.json) | Public source hashes, training-archive restoration, input counts, warm-start checks and post-migration example predictions |
| [`reproducibility/layout_20261011.json`](reproducibility/layout_20261011.json) | File integrity, old-path resolution, historical entry points, data counts and example predictions after the layout change |

Each JSON document's date, objects and checks describe that execution. To check examples in the current checkout, run `python scripts/verify_release.py`. After restoring training inputs, run `python scripts/check_training_ready.py --data-root .`. See the [reproduction guide](../docs/REPRODUCIBILITY.md#english) for the workflow.

### Historical model-route summary

[`research/model_registry_phase6.tsv`](research/model_registry_phase6.tsv) summarizes early sequence, architecture-feature and fusion routes, including their recorded status, selection rules, metric sources and E/S/B metrics. It uses historical model names and paths. The [version index](../docs/VERSION_HISTORY.md#english) and [path manifest](../artifacts/manifests/PATH_MAP.json) locate their current files.
