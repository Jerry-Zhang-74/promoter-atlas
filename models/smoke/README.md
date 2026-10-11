# 训练流程短程检查 · Short training-flow checks

## 中文

这些目录保存模型开发时的短程运行记录，方便定位数据加载、前向计算、损失、评估和检查点写入发生问题的位置。

- `v2_smoke/`：早期卷积训练流程。
- `v2_long_smoke/`：2000 bp 长上下文流程。
- `v6_smoke/`、`v6_round2_smoke/`：多物种多任务初版及第二轮试跑。
- `v6_1_smoke/`、`v6_2_smoke/`、`v6_3_smoke/`：相应迭代的短程检查。

进入某个目录后，先读 `logs/` 中的运行过程，再用 `cards/` 和 `evaluation/` 核对配置与输出。完整训练版本及其性能记录见 [versions/](../versions/README.md)。

## English

These directories retain short development runs for locating problems in data loading, forward passes, losses, evaluation and checkpoint writing.

- `v2_smoke/`: the earlier convolutional training flow.
- `v2_long_smoke/`: the 2000 bp long-context flow.
- `v6_smoke/` and `v6_round2_smoke/`: initial multispecies multitask and second-round trial runs.
- `v6_1_smoke/`, `v6_2_smoke/` and `v6_3_smoke/`: short checks for the corresponding iterations.

Start with a directory’s `logs/`, then use `cards/` and `evaluation/` to inspect its configuration and outputs. See [versions/](../versions/README.md#english) for complete training runs and their performance records.
