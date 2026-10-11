# batch33 实验情境整理 · batch33 assay-context records

[中文](#中文) · [English](#english)

## 中文

版本 `v6.22.1-assay-context.rc1`，本批整理已有测量与文献证据。

- `build_context.py`：解析原始 CytoFLEX CSV 与冻结 cellline_v2 表，按原端点公式生成 `frozen_eligibility_matrix.tsv`、`context_registry.tsv` 和 `exclusions.tsv`；通过 ASCII 子串匹配中文列名，从 `source_inventory.json` 读取源路径。
- `build_compare.py`：按家族/时间点比较 D28，读取已有冻结模型分数生成方向表。
- `partB_search.py`：PubMed E-utilities 检索，上限 6 次查询、30 MB。
- `partB_evidence.py`：整理检索证据矩阵和结论。
- `make_figures.py`：使用 matplotlib Agg 生成最多 3 张图。

从仓库根目录使用 `python scripts/run_historical.py code/batch33/build_context.py` 准备旧路径工作副本并运行端点整理；其他模块同样传入原始脚本路径。本轮使用保存的端点和模型分数，D28 方向判断为 FAIL。

---

## English

Version `v6.22.1-assay-context.rc1` organizes existing measurements and literature evidence.

- `build_context.py`: parse original CytoFLEX CSVs and frozen cellline_v2 tables; apply the recorded endpoint formula and write `frozen_eligibility_matrix.tsv`, `context_registry.tsv`, and `exclusions.tsv`. Match Chinese column names through ASCII substrings and read source paths from `source_inventory.json`.
- `build_compare.py`: compare families/timepoints against D28 and build a direction grid from saved frozen-model scores.
- `partB_search.py`: PubMed E-utilities search, capped at six queries and 30 MB.
- `partB_evidence.py`: assemble the retrieved evidence matrix and conclusion.
- `make_figures.py`: create up to three figures with matplotlib Agg.

From the repository root, use `python scripts/run_historical.py code/batch33/build_context.py` to prepare the original-layout workspace and assemble endpoints. Pass the corresponding original script path for other modules. The batch uses saved endpoints and scores; the D28 verdict remains FAIL.
