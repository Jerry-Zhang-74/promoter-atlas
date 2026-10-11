# 训练与评估结果

同一人类 TEST 按基因汇总后计算 Spearman。E/B 各 5,603 个基因，S 为 1,855 个基因。

| 指标 | 原双窗口基线 | 第一轮 | 第二轮 |
|---|---:|---:|---:|
| E | 0.667523 | 0.648740 | 0.639034 |
| S | 0.362601 | 0.373182 | 0.366478 |
| B | 0.668783 | 0.650672 | 0.640421 |
| 均值 | 0.566302 | 0.557531 | 0.548644 |

DEV 三轴均值依次为 0.559574、0.549958、0.542326。预定规则要求每轴达到原基线减 0.02，再比较均值，最终选择第一轮。S 高于原基线，E/B 与均值低于原基线。

HEK293 片段 TEST 117,155 条，第一轮 Spearman 0.519427，第二轮 0.515579。这是同一权重共享局部塔的片段分支成绩。

first_epoch_test_metrics.json、second_epoch_test_metrics.json、first_epoch_activity_metrics.json、second_epoch_activity_metrics.json 保存完整指标和 SHA；checkpoint_selection.json 保存 DEV 选择。
