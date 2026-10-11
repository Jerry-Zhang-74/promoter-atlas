# 历史研究档案 · Research archive

[中文](#中文) · [English](#english)

## 中文

此目录沿用原项目的科学文件结构。可从[版本沿革](../docs/VERSION_HISTORY.md)进入具体模型路线。

| 路径 | 内容 |
|---|---|
| `runs/scripts/` | 早期与 v6 训练、评估和数据构建 |
| `code/` | 后期冻结批次实现，包括联合训练 |
| `runs/breakthrough_20261002/`、`runs/overnight_20261002/` | 实验分支与运行记录 |
| `models/cards/`、`models/eval/`、`models/logs/`、`models/archives/` | 模型卡、指标、日志和版本档案 |
| `labels/spec/` | 标签定义 |
| `reports/` | 科学结果与配置 |
| `submission_snapshot/` | 早期提交的源码快照 |

在仓库根目录运行 `scripts/restore_training_assets.py`，即可将大文件恢复到此目录。
当前维护的程序入口位于 `src/promoter_atlas/`；历史笔记描述各自记录时间的项目状态。

2026-10-11 整理了历史文档的中英双语版本。原文件时间、来源 SHA、修订前后公开文档 SHA 及编辑日期记录在 `artifacts/manifests/SOURCE_HISTORY_MANIFEST.json`；各版本 ZIP 保存当时归档的文件。双语阅读版可从仓库浏览或通过主页的文档包下载。

---

## English

This directory follows the scientific file layout of the original project. Use the [version history](../docs/VERSION_HISTORY.md#english) to find a model route.

| Path | Contents |
|---|---|
| `runs/scripts/` | Early and v6 training, evaluation and data construction |
| `code/` | Later frozen batch implementations, including joint training |
| `runs/breakthrough_20261002/`, `runs/overnight_20261002/` | Experimental branches and run records |
| `models/cards/`, `models/eval/`, `models/logs/`, `models/archives/` | Model cards, metrics, logs and version archives |
| `labels/spec/` | Target definitions |
| `reports/` | Scientific results and configurations |
| `submission_snapshot/` | Source snapshots from early submissions |

Run `scripts/restore_training_assets.py` from the repository root to restore large files here.
The maintained program lives in `src/promoter_atlas/`. Historical notes describe the project at the time of each record.

The historical documentation received Chinese-first bilingual editing on 2026-10-11. `artifacts/manifests/SOURCE_HISTORY_MANIFEST.json` records original file times, source SHA, public-document SHA before and after editing, and the editorial date. Version ZIPs retain the files archived at publication. Browse the bilingual reading edition here or download the documentation bundle linked from the project homepage.
