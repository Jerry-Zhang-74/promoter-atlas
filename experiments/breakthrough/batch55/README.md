# batch55 · 片段模型打包器与预测接口 / Fragment-model staging and prediction interface

[中文](#中文) · [English](#english)

## 中文

为 HEK293 片段模型建立独立预测和发布打包流程。打包器核对目标、源码、checkpoint、normalizer 和模型训练记录，按白名单收集文件并生成校验清单；预测器读取 FASTA 并检查长度、重复 ID 和碱基字符。

本目录 `fixture_staging_demo` 保存人工模型夹具的打包测试材料。stager 交接状态及真实权重接入要求见 shared/batch55_submission 的说明，正式模型包在后续 batch59/60 形成。

- [打包与预测协议](../shared/runs/code/batch55_submission/README.md)
- [测试夹具清单](reports/fixture_staging_demo/staged/MODEL_MANIFEST.json)

---

## English

Provide standalone prediction and packaging for HEK293 fragment models. The stager verifies target, source, checkpoint, normalizer, and training records, collects whitelisted files, and writes checksums. The FASTA predictor validates lengths, duplicate IDs, and alphabet.

`fixture_staging_demo` retains staging tests with artificial model fixtures. The shared/batch55_submission README describes handoff status and integration of trained weights. Formal model bundles follow in batch59/60.

- [Staging and prediction contract](../shared/runs/code/batch55_submission/README.md)
- [Fixture manifest](reports/fixture_staging_demo/staged/MODEL_MANIFEST.json)
