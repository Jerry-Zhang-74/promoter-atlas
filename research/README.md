# 历史研究档案

此目录保留原项目科学文件的相对目录，便于旧脚本和数据路径对应。
从 [版本沿革](../docs/VERSION_HISTORY.md) 进入具体路线。

- `runs/scripts/`：早期和 v6 训练、评估、构建代码。
- `code/`：后期冻结批次实现，包括联合训练。
- `runs/breakthrough_20261002/`、`runs/overnight_20261002/`：实验分支代码与记录。
- `models/cards/`、`models/eval/`、`models/logs/`、`models/archives/`：已有模型证据。
- `labels/spec/`：标签定义。
- `reports/`：已有科学结果与配置。
- `submission_snapshot/`：早期公开提交中的原始脚本快照；使用原目录源码时优先上述目录。

运行仓库根目录的 `scripts/restore_training_assets.py` 将大文件恢复到此目录。
当前维护的程序入口位于 `src/promoter_atlas/`；这里的旧脚本保留历史逻辑与局限，
不要将旧文档中的当时“当前/最佳”措辞当作最新项目结论。
