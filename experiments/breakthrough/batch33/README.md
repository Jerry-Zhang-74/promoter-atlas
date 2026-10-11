# batch33 · 293FT 测量时间点与实验情境对照 / 293FT timepoints and assay context

[中文](#中文) · [English](#english)

## 中文

把原始 CytoFLEX 导出表与冻结的构建体记录关联，整理细胞背景、采样时间、荧光门和可比较的端点。对 ACTB/GAPDH/CMV 的家族内方向进行时间点比较，并与保存的模型分数对应。

代码还构建文献证据矩阵，区分 293FT 在研究中的测量用途。`build_context.py` 负责端点和纳入条件，`build_compare.py` 负责方向比较；详细模块说明见代码 README。

- [分析模块说明](code/batch33/README.md)
- [测量情境构建](code/batch33/build_context.py)
- [来源校验清单](reports/sha_manifest.json)

---

## English

Join original CytoFLEX exports to frozen construct records, organizing cell context, sampling time, fluorescence gates, and comparable endpoints. Compare within-family ACTB/GAPDH/CMV directions across timepoints and align them with saved model scores.

The code also builds a literature evidence matrix describing how 293FT is used in assays. `build_context.py` handles endpoints and eligibility; `build_compare.py` handles directional comparisons. The code README explains each module.

- [Analysis modules](code/batch33/README.md)
- [Assay-context construction](code/batch33/build_context.py)
- [Source checksums](reports/sha_manifest.json)
