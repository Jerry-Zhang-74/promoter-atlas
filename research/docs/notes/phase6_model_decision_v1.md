# Phase 6 模型决策 v1 · Phase 6 model decision v1

[中文](#中文) · [English](#english)

## 中文

### 历史对照

| 模型 | E R² | S R² | B R² | 预测 E–S 相关 |
|---|---:|---:|---:|---:|
| v1（models/checkpoints/promoter_model_v1.pt） | 0.2367 | 0.0642 | 0.2268 | NA |
| v4_arch_only | 0.2297 | 0.0791 | 0.2114 | 0.3494 |
| v4_arch2_reduced | 0.2257 | 0.0577 | 0.2062 | 0.3751 |

Motif 与配对特征路线低于预设目标，结构特征提供了独立 S 信号，下一步探索序列/结构融合。

这一阶段按主要结构族分版本，族内保留路线名称，v4 对应结构特征路线。公开版本映射表记录旧名称与 Release 的对应关系。

---

## English

### Recorded comparison

| Model | E R² | S R² | B R² | Predicted E–S correlation |
|---|---:|---:|---:|---:|
| v1 (`models/checkpoints/promoter_model_v1.pt`) | 0.2367 | 0.0642 | 0.2268 | NA |
| v4_arch_only | 0.2297 | 0.0791 | 0.2114 | 0.3494 |
| v4_arch2_reduced | 0.2257 | 0.0577 | 0.2062 | 0.3751 |

Motif and pair-feature routes fell short of the objective. Architecture features provided an independent S signal, motivating stronger sequence/structure fusion.

The version convention used at this stage grouped major architecture families together and retained internal route names within a family; v4 referred to the architecture route. The public version mapping documents how these historical names map to published releases.
