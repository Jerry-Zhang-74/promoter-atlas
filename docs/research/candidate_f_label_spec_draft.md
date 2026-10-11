# Candidate F 标签草案 v0.1 · Candidate F label draft v0.1

[中文](#中文) · [English](#english)

## 中文

Candidate F 是在已冻结 Candidate E 旁新增的时间标签分支。本文件为资源清点和构建实施前的规范草案。

### 标签含义与公式

E 为表达强度，S_tissue 为跨组织稳定性，B 为表达广度，D_time 为时间持久性，R_time 为晚期相对早期的保留率。S_tissue 和 D_time 分别保留。`x` 为组织平均 TPM，`y` 为启动子在某生物情境和时间点的活性，`C_p` 为有效情境集合，epsilon 为正的小常数。

```text
E_raw(g) = mean_t log2(1 + x_(g,t))
B_raw(g) = mean_t I(x_(g,t) >= 1 TPM)
D(g) = {t: x_(g,t) >= 1 TPM}
S_tissue_raw(g) = -MAD({log2(1+x_(g,t)): t in D(g)})
    domain: B_raw >= 0.30 and |D(g)| >= 3
D_c_raw(p) = -MAD({log2(1+y_(p,c,tau)): tau in Tau_c})
D_time_raw(p) = median_(c in C_p) D_c_raw(p)
R_c(p) = (y_(p,c,tau_late)+epsilon)/(y_(p,c,tau_early)+epsilon)
R_time(p) = median_(c in C_p) R_c(p)
Score_Q = rank(Q_raw)/(n-1)
Score_HKG_3 = exp(0.40*log(Score_E+epsilon)
                 +0.35*log(Score_S_tissue+epsilon)+0.25*log(Score_B+epsilon))
Score_HKG_star = exp(0.25*log(Score_E+epsilon)
                    +0.25*log(Score_S_tissue+epsilon)+0.15*log(Score_B+epsilon)
                    +0.35*log(Score_D+epsilon))
```

rank 使用从 0 开始的秩，有效观测数为 n；每一轴在其定义域内缩放到 [0,1]，缺失值保持缺失。现行 Candidate E 的实现与后续草案提议分别记录在构建脚本和版本说明中。

### 时间数据准入与缺失值

情境需有明确生物学意义、有序时间点、可比测量单位、清晰的启动子/TSS 映射、足够的有效时间点、重复处理规则，以及有解释意义的晚期阶段。优先使用启动子原生活性测定。

无有效情境、映射待解决、全部情境未通过准入或时间结构过稀时，D_time_raw 与 Score_D 保持缺失。无有效早晚配对时 R_time 保持缺失。S_tissue 未定义时 Score_S 与 Candidate E 风格的 HKG 保持缺失。

持久性还需活动门控，例如 `max_tau y >= tau_min`，可进一步要求至少 m 个时间点达到阈值。这样可将持续活动与低信号平坦曲线区分。R≈1 表示晚期活动接近早期，R≪1 表示早期激活后衰减。

### 排序与实验设计

主设计轴保持 E × S_tissue，B 用于门控和分层。时间标签达到覆盖率和质量要求后，将 D_time 与 R_time 用于实验候选筛选。三轴排序建议权重为 0.40/0.35/0.25；四轴草案为 0.25/0.25/0.15/0.35，总和为 1。加权几何均值强调各维度共同满足需求，弱维度对结果有较大影响。这些权重是设计提案，等待实验评估。

### 数据流程

Candidate E：GTEx → `gene_expression_table.tsv` → `gene_expression_label_table.tsv` → `promoter_label_table.tsv` → `promoter_model_input_core.tsv` / `promoter_model_input_extended.tsv`。

Candidate F：FANTOM5 等时间数据 → 情境清单 → `promoter_temporal_mapping_table.tsv` → `context_level_durability_table.tsv` → `promoter_durability_table.tsv`，按稳定启动子 ID 左连接现有标签。

FANTOM5 的 CAGE 数据提供启动子/TSS 活性及部分时间序列，GTEx 提供组织表达汇总。时间数据按各自情境计算后再聚合；短期响应与长期维持分别解释。

拟输出 `labels/build/` 下上述时间标签表、`labels/qc/candidate_f_durability_missingness_summary.tsv`、`labels/qc/candidate_f_context_coverage_summary.tsv` 及 `docs/notes/candidate_f_temporal_data_inventory.md`。

下一步依次整理资源、冻结启动子/TSS 映射、情境准入及早晚配对规则，再实现构建脚本。

---

## English

Candidate F adds a temporal-label branch alongside frozen Candidate E. This is a specification draft preceding resource inventory and implementation.

### Label meanings and equations

E is expression strength; S_tissue is cross-tissue stability; B is breadth; D_time is temporal durability; R_time is late-to-early retention. Keep S_tissue and D_time separate. `x` is mean tissue TPM, `y` is promoter activity in a biological context at a timepoint, `C_p` is the set of valid contexts, and epsilon is a small positive constant.

```text
E_raw(g) = mean_t log2(1 + x_(g,t))
B_raw(g) = mean_t I(x_(g,t) >= 1 TPM)
D(g) = {t: x_(g,t) >= 1 TPM}
S_tissue_raw(g) = -MAD({log2(1+x_(g,t)): t in D(g)})
    domain: B_raw >= 0.30 and |D(g)| >= 3
D_c_raw(p) = -MAD({log2(1+y_(p,c,tau)): tau in Tau_c})
D_time_raw(p) = median_(c in C_p) D_c_raw(p)
R_c(p) = (y_(p,c,tau_late)+epsilon)/(y_(p,c,tau_early)+epsilon)
R_time(p) = median_(c in C_p) R_c(p)
Score_Q = rank(Q_raw)/(n-1)
Score_HKG_3 = exp(0.40*log(Score_E+epsilon)
                 +0.35*log(Score_S_tissue+epsilon)+0.25*log(Score_B+epsilon))
Score_HKG_star = exp(0.25*log(Score_E+epsilon)
                    +0.25*log(Score_S_tissue+epsilon)+0.15*log(Score_B+epsilon)
                    +0.35*log(Score_D+epsilon))
```

Rank starts at zero and n is the number of valid observations. Scale each axis to [0,1] within its defined domain and preserve missing values. Build scripts and version documentation distinguish the implemented Candidate E policy from subsequent proposals.

### Temporal-data eligibility and missing values

An eligible context has coherent biological meaning, ordered timepoints, comparable measurement units, explicit promoter/TSS mapping, sufficient observations, documented replicate handling, and an interpretable late phase. Prefer promoter-native activity assays.

D_time_raw and Score_D remain missing when valid contexts are absent, mapping is unresolved, every context fails eligibility, or the time structure is too sparse. R_time remains missing without a valid early/late pair. Undefined S_tissue produces missing Score_S and Candidate E-style HKG.

Durability also requires an activity gate, such as `max_tau y >= tau_min`, optionally with at least m timepoints above threshold. This separates sustained activity from flat low-signal traces. R≈1 means late activity is close to early activity; R≪1 indicates decay after initial activation.

### Ranking and experimental design

Keep E × S_tissue as the primary design space and use B for gating and stratification. Add D_time and R_time to experimental nomination after coverage and label quality are adequate. Proposed three-axis weights are 0.40/0.35/0.25; four-axis weights are 0.25/0.25/0.15/0.35, summing to 1. A weighted geometric mean emphasizes joint suitability and penalizes weak dimensions. These weights are design proposals awaiting experimental evaluation.

### Data flow

Candidate E: GTEx → `gene_expression_table.tsv` → `gene_expression_label_table.tsv` → `promoter_label_table.tsv` → `promoter_model_input_core.tsv` / `promoter_model_input_extended.tsv`.

Candidate F: temporal sources including FANTOM5 → context inventory → `promoter_temporal_mapping_table.tsv` → `context_level_durability_table.tsv` → `promoter_durability_table.tsv`, followed by a left join on stable promoter IDs.

FANTOM5 CAGE provides promoter/TSS activity and selected time courses; GTEx provides tissue-expression summaries. Calculate temporal statistics within each context before aggregation and interpret short-term response separately from long-term maintenance.

Planned outputs include the temporal tables above under `labels/build/`, `labels/qc/candidate_f_durability_missingness_summary.tsv`, `labels/qc/candidate_f_context_coverage_summary.tsv`, and `docs/notes/candidate_f_temporal_data_inventory.md`.

Next, inventory resources and freeze promoter/TSS mapping, context eligibility, and early/late pairing rules before implementing the build scripts.
