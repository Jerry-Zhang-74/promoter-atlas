# batch51 · HEK293 嵌套片段坐标队列 / HEK293 nested-fragment coordinate cohort

[中文](#中文) · [English](#english)

## 中文

从 GSE301246 PARM 文库的 MH4 文件构建 100–600 bp 嵌套片段队列。按坐标和包含关系形成组，并用固定哈希决定划分，逐项记录短/长片段端点。

本阶段代码首先准备坐标层；参考基因组序列的获取和后续完整队列在 batch52 延续。`build_cohort.py` 定义筛选和分组，`finalize_b51.py` 汇总来源核对与准备状态。

- [坐标队列构建](code/batch51/build_cohort.py)
- [来源与准备状态](code/batch51/finalize_b51.py)
- [完整队列与训练](../batch52/README.md)

---

## English

Construct a 100–600 bp nested-fragment coordinate cohort from the MH4 GSE301246 PARM library. Group records by coordinates and containment and use fixed hashes for split assignment, retaining short/long endpoint relationships.

This stage prepares the coordinate layer; reference-sequence retrieval and the full cohort continue in batch52. `build_cohort.py` defines filtering and grouping; `finalize_b51.py` assembles source checks and readiness records.

- [Coordinate-cohort construction](code/batch51/build_cohort.py)
- [Source and readiness](code/batch51/finalize_b51.py)
- [Full cohort and training](../batch52/README.md)
