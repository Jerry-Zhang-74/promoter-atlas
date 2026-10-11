# batch60 · 可独立运行的片段模型包布局 / Standalone fragment-model package layout

[中文](#中文) · [English](#english)

## 中文

调整片段模型包的解压布局：预测入口、固定代码、模型清单和使用指南位于明确位置，处理任意工作目录及中文/空格路径。配置和 normalizer 随包固定，results 保存相应评估。

`build_final.py` 组装包，`verify_final.py` 核对实际产物，`delivery_manifest.json` 列出交付文件。docs/README 是打包时填入版本、模型臂和权重名的文档模板。

- [最终交付清单](reports/delivery_manifest.json)
- [解压包核对](code/batch60_release/verify_final.py)
- [包文档模板](code/batch60_release/docs/README.md)

---

## English

Refine the extracted fragment-model layout with clear locations for the predictor, pinned code, model manifest, and guide, supporting arbitrary working directories and non-ASCII/space-containing paths. Configuration and normalizer are pinned with evaluation records under results.

`build_final.py` assembles bundles, `verify_final.py` checks them, and `delivery_manifest.json` lists deliverables. docs/README is the packaging template populated with the version, model arm, and weight filename.

- [Final delivery manifest](reports/delivery_manifest.json)
- [Extracted-bundle verification](code/batch60_release/verify_final.py)
- [Bundle-document template](code/batch60_release/docs/README.md)
