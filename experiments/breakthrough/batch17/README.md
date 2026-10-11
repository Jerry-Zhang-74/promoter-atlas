# batch17 · 独立 E/S/B 模型的首次发布包 / First package of the independent E/S/B composite

[中文](#中文) · [English](#english)

## 中文

把 batch16 选中的独立 E/S/B 组合整理为 `v6.15.1-release.rc1` 历史候选包。包内文档定义输入数组、训练变换、各轴单位和示例输出，并保留共享模型消融结果。

`corrected_release_metrics.json` 对齐发布数字与实际选择规则；`reviewer_provenance_correction.json` 记录审查来源修订。使用此候选时从模型卡和 schema 读取接口，下一批 rc2 补充输入边界处理。

- [候选包接口](models/release_v6.15.1-release.rc1/README.md)
- [发布指标](reports/corrected_release_metrics.json)

---

## English

Package the independent E/S/B composite selected in batch16 as historical candidate `v6.15.1-release.rc1`. Its documents define input arrays, training transforms, axis units, and example outputs, while retaining shared-model ablations.

`corrected_release_metrics.json` aligns published metrics with selection, and `reviewer_provenance_correction.json` records review-provenance corrections. Read the model card and schema for this candidate's interface; batch18 rc2 adds input-boundary handling.

- [Candidate interface](models/release_v6.15.1-release.rc1/README.md)
- [Release metrics](reports/corrected_release_metrics.json)
