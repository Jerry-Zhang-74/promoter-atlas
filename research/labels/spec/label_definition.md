# 标签定义：早期设计规范 · Label definitions: early design specification

[中文](#中文) · [English](#english)

## 中文

本规范记录早期标签设计原则；公式与实际训练输入的对应关系见当前数据说明和构建脚本。

### 目标与输出

项目研究启动子表达规律，支持可实验检验的连续启动子文库设计。分别保留表达强度、跨组织稳定性和表达广度，便于解释和复核。

| 输出 | 含义与来源 | 用途 |
|---|---|---|
| `Score_E` | 连续表达强度，来源于 GTEx 基因级表达聚合 | 表达强度主轴 |
| `Score_S` | 连续跨组织稳定性，衡量可检出组织间的离散程度 | 稳定性主轴，覆盖高表达与中等表达启动子 |
| `Score_B` | 可检出组织比例或其连续得分 | 候选准入门控与分层 |
| `Score_HKG` | 由 E/S/B 产生的管家基因偏好排序 | 下游候选排序与实验优先级 |

主要设计空间为 **E × S**，B 用于门控和分层，HKG 用于下游筛选。S 保留独立的稳定性含义，E 保留强度含义，各轴分别报告。

### 建模与输入

建立仅用启动子序列预测 E/S/B 的多输出基线，先评估 up200，再在相同 QC 条件下成对比较 up2000。每条记录附带不确定性和 QC 标记。

首个基线取 `labels/build/promoter_analysis_set.tsv` 中 `core_model_pass = 1` 的记录；扩展上下文取 `extended_context_pass = 1`，QC 失败记录保留在审核集合。

项目验收同时关注生物学解释和实验设计表现，AUC、R²、Spearman 用于描述相应评估任务。

---

## English

This specification records the early label-design principles. Current data documentation and build scripts describe how formulas map to actual training inputs.

### Goals and outputs

The project studies promoter expression rules and supports experimentally testable continuous promoter libraries. Keep expression strength, cross-tissue stability, and breadth as separate, interpretable outputs.

| Output | Meaning and source | Role |
|---|---|---|
| `Score_E` | Continuous expression strength from GTEx gene-level aggregation | Primary strength axis |
| `Score_S` | Continuous cross-tissue stability, measured by dispersion among detectable tissues | Primary stability axis, covering high and moderate expression |
| `Score_B` | Detectable-tissue fraction or its continuous score | Entry gate and stratification |
| `Score_HKG` | Housekeeping preference derived from E/S/B | Downstream ranking and experimental priority |

The primary design space is **E × S**. B provides gating and stratification; HKG supports downstream selection. S retains its stability meaning, E retains its strength meaning, and each axis is reported separately.

### Modeling and inputs

Build a promoter-sequence multi-output E/S/B baseline, evaluate up200 first, and compare up2000 under the same QC policy. Attach uncertainty and QC tags to each record.

The first baseline uses rows with `core_model_pass = 1` in `labels/build/promoter_analysis_set.tsv`. Extended-context analysis uses `extended_context_pass = 1`; QC failures remain in the review set.

Project acceptance considers biological interpretation and experimental design performance. AUC, R², and Spearman describe the corresponding evaluation tasks.
