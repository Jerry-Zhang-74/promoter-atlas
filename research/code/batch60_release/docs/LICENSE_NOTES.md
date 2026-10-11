# 许可与来源 · License and provenance

[中文](#中文) · [English](#english)

## 中文

该包包含项目编写的模型结构、训练配置和预测代码，来源为 batch52/batch53/batch55。模型以作者发布的 GSE301246 PARM 训练库 fold0 的 HEK293 测量从随机初始化训练，目标列为 `Log2RPM_HEK293`。

固定核心包含 `spec.py`、`model_core.py`、`normalizer.py`、`checkpoint.py` 和 `io_utils.py`，各文件 SHA-256 记录在清单中。第三方运行时 PyTorch 和 NumPy 按各自 BSD-3-Clause 许可安装。

发布资产围绕项目模型参数、配置、归一化器、代码与评估组织。上游测量表和参考基因组按各自来源获取。编码序列为参考基因组代理，HEK293 是测量来源，293FT 迁移待评估。性能数据来自冻结的独立评估 JSON。

完整训练 checkpoint 含 optimizer/scheduler/sampler/RNG 状态，SHA-256 为 `{SHA}`。此包提供推理流程；预测读取模型参数及固定 config/normalizer，并从 `model/*.sha256.json` 读取 `checkpoint_sha256`。其中记录的来源路径用于追溯。

---

## English

The bundle contains project-authored architecture, training configuration, and prediction code from batch52/batch53/batch55. The model was trained from random initialization on HEK293 measurements from author-published GSE301246 PARM training-library fold0, targeting `Log2RPM_HEK293`.

Pinned core modules are `spec.py`, `model_core.py`, `normalizer.py`, `checkpoint.py`, and `io_utils.py`, with per-file SHA-256 in the manifest. Install PyTorch and NumPy under their respective BSD-3-Clause licenses.

Release assets center on project parameters, configuration, normalizer, code, and evaluation. Obtain upstream measurement tables and reference genomes from their respective sources. Encoded sequences are reference-genome proxies; HEK293 is the measurement source and 293FT transfer awaits evaluation. Performance values come from frozen independent-evaluation JSON.

The complete training checkpoint includes optimizer/scheduler/sampler/RNG state and has SHA-256 `{SHA}`. This bundle provides an inference workflow: prediction reads parameters and pinned config/normalizer, plus `checkpoint_sha256` from `model/*.sha256.json`. Recorded provenance paths support traceability.
