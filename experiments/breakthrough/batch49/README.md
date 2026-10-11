# batch49 · 表达强度与稳定度的 Pareto 目录筛选 / Pareto catalog selection for expression and stability

[中文](#中文) · [English](#english)

## 中文

在观察标签目录上增加 `gene-rank`：使用 E_raw 和 S_raw 两个独立设计轴，B_raw 作为纳入阈值，支持表达排序、稳定度排序和精确 Pareto 前沿。排序与并列处理固定，显示数量限制作用于最终展示。

该功能与原有 `fragment`、`gene-observed` 入口并列。`b49_gene_rank.py` 实现前沿和筛选规则，`b49_cli.py` 暴露参数，清单复核记录随本批保存。

- [Pareto 与筛选算法](../shared/runs/code/batch49/b49_gene_rank.py)
- [查询命令入口](../shared/runs/code/batch49/b49_cli.py)
- [清单复核](reports/manifest_verification.json)

---

## English

Add `gene-rank` over the observed-label catalog, using E_raw and S_raw as separate design axes and B_raw as an eligibility threshold. Support expression sorting, stability sorting, and an exact Pareto frontier with deterministic ties; the display limit applies after selection.

The new command sits alongside `fragment` and `gene-observed`. `b49_gene_rank.py` implements the frontier and filters; `b49_cli.py` exposes parameters. Artifact-verification records are retained with the batch.

- [Pareto and filtering](../shared/runs/code/batch49/b49_gene_rank.py)
- [Catalog CLI](../shared/runs/code/batch49/b49_cli.py)
- [Manifest verification](reports/manifest_verification.json)
