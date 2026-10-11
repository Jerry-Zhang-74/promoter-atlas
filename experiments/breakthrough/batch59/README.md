# batch59 · HEK293 片段模型发布验证 / HEK293 fragment-model release verification

[中文](#中文) · [English](#english)

## 中文

把已训练的片段模型整理为可交付包，收集固定核心、权重、配置、normalizer、用户指南和模型卡，并在打包前验证实际预测入口。代码分为已审查材料构建、ZIP 生成、GPU 冒烟和发布核对。

`delivery_manifest.json` 记录交付文件及其哈希。该批建立首次包结构，下一批进一步调整为解压后即可定位预测器的布局。

- [交付清单](reports/delivery_manifest.json)
- [发布验证](code/batch59_release/verify_release.py)
- [历史使用指南](code/batch59_release/docs/USER_GUIDE_zh.md)

---

## English

Assemble trained fragment models into deliverable bundles containing pinned code, weights, configuration, normalizer, user guidance, and model cards, and verify the predictor before packaging. Modules cover reviewed inputs, ZIP creation, GPU smoke inference, and release checks.

`delivery_manifest.json` records delivered files and hashes. This stage establishes the first package layout; batch60 refines the extracted layout so the predictor is easy to locate.

- [Delivery manifest](reports/delivery_manifest.json)
- [Release verification](code/batch59_release/verify_release.py)
- [Historical user guide](code/batch59_release/docs/USER_GUIDE_zh.md)
