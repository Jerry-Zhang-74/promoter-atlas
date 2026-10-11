# FANTOM5 资源与映射草案 v0.1 · FANTOM5 inventory and mapping draft v0.1

[中文](#中文) · [English](#english)

## 中文

### 资源定位

本草案定义 Candidate F 的 FANTOM5 接入流程。FANTOM5 是由多种生物状态和部分时间实验组成的 CAGE 启动子/TSS 活性资源，作为时间持久性的情境性代理；Candidate E 保持独立。

优先整理 robust peaks、启动子 counts/TPM/max TPM/total TPM，以及原代细胞、组织、细胞系、分级样本和时间序列轨道。permissive peaks 用于探索。时间标签审核采用有明确时间顺序的子集，记录样本、情境、时间点、测定类型、物种、参考组装和重复信息。

### 清单与映射对象

先清点资源，再解析 peak 和样本，之后建立映射与情境活动表。字段约定如下，分别描述资源、情境、CAGE peak、现有启动子、两者映射及逐时间点活动：

```text
resource: resource_id, source_group, file_name, file_role, species, genome_build,
          assay_type, signal_unit, sample_category, contains_timecourse,
          contains_peak_coordinates, contains_expression_values, download_status,
          parse_status, notes
context: context_id, context_name, sample_category, biological_process_type, species,
         genome_build, timepoint_count, timepoint_labels, time_order_defined,
         replicate_info_available, signal_unit, promoter_level_ready,
         candidate_for_durability, exclusion_reason, notes
peak: fantom_peak_id, chrom, start, end, strand, peak_class,
      associated_gene_symbol, associated_gene_id, source_build
promoter: promoter_id
mapping: fantom_peak_id, promoter_id, overlap_bp,
         peak_center_to_promoter_anchor_distance, strand_match, mapping_class,
         mapping_confidence, mapping_rule_version
activity: context_id, timepoint_label, time_rank, fantom_peak_id, promoter_id,
          activity_value, signal_unit, replicate_aggregation_rule, activity_qc_flag
```

### 映射规则

依次优先链一致的直接区间重叠、peak 中心落在启动子锚定窗口，以及在前两者不可用且歧义低时的最近兼容启动子。记录源组装、目标组装、liftover/坐标协调方法及协调后的 peak ID 稳定性。

多启动子同等支持、链冲突、密集双向区域归属待定和组装未对齐的记录保留为待解决。置信度为 `high`（唯一且链/锚点一致重叠）、`medium`（唯一且距离适当的最近匹配）、`low`（弱支持或歧义）和 `reject`（不适合标签构建）。持久性构建使用 high 与经过审核的部分 medium。

### 情境准入和计算准备

情境需生物学一致、时间顺序明确、有可解释晚期、测量单位可比、启动子映射合格且重复处理完整。分化、发育、持续响应或恢复时间序列是候选。急性单次响应、时间含义不清的稀疏两点数据和 TSS 归属不稳定的记录进入单独审核。

计算 D_c_raw 前明确 context_id、time_rank、启动子活动、单位、缺失值和重复聚合；计算 R_c 前固定 early/late、间隔时间点的 QC 用途、低早期值处理和 epsilon。

### 分阶段输出与 QC

```text
data/external/fantom5/fantom5_resource_inventory.tsv
data/external/fantom5/fantom5_context_inventory.tsv
docs/notes/fantom5_candidate_f_mapping_notes.md
data/interim/fantom5/fantom5_peak_table.tsv
data/interim/fantom5/fantom5_sample_table.tsv
data/interim/fantom5/fantom5_timecourse_sample_table.tsv
labels/build/fantom_peak_to_promoter_mapping.tsv
labels/qc/fantom_peak_to_promoter_mapping_summary.tsv
labels/build/fantom_context_activity_long.tsv
labels/qc/fantom_context_activity_missingness.tsv
labels/build/context_level_durability_table.tsv
labels/build/promoter_durability_table.tsv
```

QC 汇总逐文件解析成功率、样本类别情境数、有序时间情境数、组装/peak 类别分布、映射比例、置信度、歧义数量及排除原因。

实验筛选关注持续活动、可解释的时间模式及其与序列编辑的联系。下一步细化情境审核表、TSS 协调、早晚配对和资源获取顺序。

---

## English

### Resource role

This draft defines FANTOM5 intake for Candidate F. FANTOM5 combines CAGE promoter/TSS activity across biological states and selected time courses, providing context-specific temporal proxies alongside the independent Candidate E branch.

Prioritize robust peaks; promoter counts, TPM, max TPM, and total TPM; and tracks from primary cells, tissues, cell lines, fractionation, and time courses. Use permissive peaks for exploration. Review explicitly ordered temporal subsets and record sample, context, timepoint, assay, species, assembly, and replicate metadata.

### Inventory and mapping objects

Inventory resources first, parse peaks and samples next, then construct mappings and context-activity tables. Fields below describe resources, contexts, CAGE peaks, existing promoters, their mappings, and activity at each timepoint:

```text
resource: resource_id, source_group, file_name, file_role, species, genome_build,
          assay_type, signal_unit, sample_category, contains_timecourse,
          contains_peak_coordinates, contains_expression_values, download_status,
          parse_status, notes
context: context_id, context_name, sample_category, biological_process_type, species,
         genome_build, timepoint_count, timepoint_labels, time_order_defined,
         replicate_info_available, signal_unit, promoter_level_ready,
         candidate_for_durability, exclusion_reason, notes
peak: fantom_peak_id, chrom, start, end, strand, peak_class,
      associated_gene_symbol, associated_gene_id, source_build
promoter: promoter_id
mapping: fantom_peak_id, promoter_id, overlap_bp,
         peak_center_to_promoter_anchor_distance, strand_match, mapping_class,
         mapping_confidence, mapping_rule_version
activity: context_id, timepoint_label, time_rank, fantom_peak_id, promoter_id,
          activity_value, signal_unit, replicate_aggregation_rule, activity_qc_flag
```

### Mapping rules

Prefer strand-consistent direct interval overlap, then a peak center within the promoter anchor window, then a nearest compatible promoter when the first two options are unavailable and ambiguity is low. Record source and target assemblies, liftover/coordinate harmonization, and peak-ID stability after harmonization.

Equal support for multiple promoters, strand conflicts, unresolved dense bidirectional regions, and unaligned assemblies remain unresolved. Confidence classes are `high` (unique overlap with strand/anchor consistency), `medium` (unique nearest compatible mapping at an acceptable distance), `low` (weak or ambiguous support), and `reject` (unsuitable for label building). Durability builds use high and reviewed medium mappings.

### Context eligibility and computation preparation

Contexts require biological coherence, ordered timepoints, an interpretable late phase, comparable units, qualified promoter mappings, and documented replicates. Differentiation, development, sustained response, and recovery are candidates. Acute single-pulse responses, sparse two-point data with unclear temporal meaning, and unstable TSS assignments receive separate review.

Before computing D_c_raw, define context_id, time_rank, promoter activity, units, missingness, and replicate aggregation. Before R_c, freeze early/late selection, the QC role of intermediate timepoints, low-early-value handling, and epsilon.

### Staged outputs and QC

```text
data/external/fantom5/fantom5_resource_inventory.tsv
data/external/fantom5/fantom5_context_inventory.tsv
docs/notes/fantom5_candidate_f_mapping_notes.md
data/interim/fantom5/fantom5_peak_table.tsv
data/interim/fantom5/fantom5_sample_table.tsv
data/interim/fantom5/fantom5_timecourse_sample_table.tsv
labels/build/fantom_peak_to_promoter_mapping.tsv
labels/qc/fantom_peak_to_promoter_mapping_summary.tsv
labels/build/fantom_context_activity_long.tsv
labels/qc/fantom_context_activity_missingness.tsv
labels/build/context_level_durability_table.tsv
labels/build/promoter_durability_table.tsv
```

QC summarizes parse success by file, contexts by sample category, ordered temporal contexts, assembly/peak-class counts, mapped fraction, confidence distribution, ambiguity counts, and exclusion reasons.

Experimental nomination focuses on sustained activity, interpretable temporal patterns, and links to sequence edits. Next, refine the context checklist, TSS reconciliation, early/late pairing, and acquisition order.
