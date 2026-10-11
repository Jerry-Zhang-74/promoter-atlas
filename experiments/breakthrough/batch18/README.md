# batch18 · 发布包输入校验与解压回放 / Release input validation and extraction replay

[中文](#中文) · [English](#english)

## 中文

将独立 E/S/B 组合包修订为 `v6.15.1-release.rc2`，明确处理空 batch 等输入错误，并保存解压后的检查副本。模型沿用 batch16 的选中权重与变换，负向消融结果随包保存。

主包位于 `models/release_v6.15.1-release.rc2`；`models/extract_check` 是解压核对记录。两者中的 schema、示例预期输出和版本清单可用于检查包装前后是否一致。

- [rc2 接口](models/release_v6.15.1-release.rc2/README.md)
- [解压核对副本](models/extract_check/release_v6.15.1-release.rc2/README.md)

---

## English

Revise the independent E/S/B package to `v6.15.1-release.rc2`, handling input errors such as empty batches and retaining an extracted verification copy. The package uses the selected batch16 weights and transforms and includes negative ablations.

The main package is under `models/release_v6.15.1-release.rc2`; `models/extract_check` preserves extraction verification. Schema, expected example outputs, and version manifests allow comparison before and after packaging.

- [rc2 interface](models/release_v6.15.1-release.rc2/README.md)
- [Extracted verification copy](models/extract_check/release_v6.15.1-release.rc2/README.md)
