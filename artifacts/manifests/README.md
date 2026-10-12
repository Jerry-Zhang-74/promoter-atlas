# 模型与数据清单

| 文件 | 内容 |
| --- | --- |
| [WEIGHTS_MANIFEST.json](WEIGHTS_MANIFEST.json) | v6.6.0 单文件权重的真实发布资产 URL、字节数、SHA-256 和资产编号；供 `scripts/download_weights.py` 使用 |
| [TRAINING_ASSETS.json](TRAINING_ASSETS.json) | `training-data-20261012` 中实际归档资产的下载 URL、字节数和 SHA-256 |
| [MODEL_VERSION_REGISTRY.json](MODEL_VERSION_REGISTRY.json) | 当前版本与 27 个历史版本的模型路径、原始名称、源码入口和文件指纹 |
| [PATH_MAP.json](PATH_MAP.json) | 训练源码使用的项目相对路径映射 |
| [UPSTREAM_DATA.json](UPSTREAM_DATA.json) | 原始公开数据的来源与下载记录 |

原始训练文件的路径、字节数与 SHA-256 见 [DATA_SOURCE_MANIFEST.json](../../data/DATA_SOURCE_MANIFEST.json)。归档资产校验值与解包后的原始文件校验值分别保存。训练与演示步骤见 [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)。
