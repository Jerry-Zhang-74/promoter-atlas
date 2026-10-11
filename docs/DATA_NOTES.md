# 数据说明 · Data notes

[中文](#中文) · [English](#english)

## 中文

模型使用 E/S/B 启动子记录与 MPRA 片段活性两套数据。以下路径均相对于恢复后的 `research/`。

### E/S/B 启动子

| 项目 | 内容 |
|---|---|
| 训练输入表 | `labels/build/promoter_atlas_v6_input.tsv` |
| 全表记录数 | 1,746,138；TRAIN 1,533,642 / DEV 105,364 / TEST 107,132 |
| 物种 | 人类及 18 个直系同源物种 |
| 人类参考 | GRCh38 / GENCODE v49 转录本启动子 |
| 窗口 | 局部 `[-200,+200)` 400 bp；上游 `[-2000,0)` 2000 bp |
| 人类序列 | `promoters/memmap/promoter_{sym400,up2000}_uint8.memmap` |
| 其他物种序列 | `data/multispecies/memmap/<key>_{sym400,up2000}_uint8.memmap` |
| TRAIN 均值、标准差 | E (0.7360959, 0.2327551)；S (0.8307955, 0.0871514)；B (0.7283229, 0.2314467) |

三轴标签派生自 GTEx 的组织平均表达量 `mean_tpm`，人类标签通过直系同源映射形成多物种记录。
定义见 `labels/spec/label_definition.md`，产物为 `labels/build/gene_expression_label_table.tsv`。

```text
E_raw = mean_tissues(log2(1 + mean_tpm))
B_raw = count(tissues with mean_tpm >= 1) / count(all tissues)
S_raw = -MAD_detectable_tissues(log2(1 + mean_tpm))
Score_X = rank_ascending_from_zero(X_raw) / (n - 1)
```

S 在 B_raw≥0.30 且至少 3 个组织可检出时定义。秩归一化使用完整的 58,988 基因目录；
同值按原表顺序稳定排序。S 缺失项在排序阶段置底，最终输出恢复为空。
缺失标签通过 `finite_mask=0` 排除于对应损失计算。保留的 split 值为 `train, val, test`。
同一基因下的转录本共享该基因的 E/S/B 标签。

人类使用冻结的 `promoter_split_core.tsv`，其他物种分别做物种内基因级划分。
同源家族在不同物种的记录可能分布于不同集合，具体评估条件见[评估说明](LIMITATIONS.md)。

### HEK293 片段活性

| 项目 | 内容 |
|---|---|
| 源文件 | `GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz` |
| 源文件大小 | 103,144,961 bytes |
| 源 SHA-256 | `1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96` |
| 目标 | `Log2RPM_HEK293`，v2_MH4 fold0 |
| 参考 | `Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz`，869,923,173 bytes |
| 参考 SHA-256 | `0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865` |
| 派生目录 | `data/breakthrough_20261002/batch52/corrected/cohort_v2/` |
| 记录数 | 761,655；TRAIN 531,860 / DEV 112,640 / TEST 117,155 |
| 分组数 | 5,332；TRAIN 3,724 / DEV 814 / TEST 794 |
| TRAIN 目标归一化 | mean=1.6329246759414673；std=1.3893377780914307 |

派生目录保存 `codes_int64.npy`、`endpoints.tsv`、`groups.tsv`、`manifest.json` 和 `build_report.json`。
片段长度为 100–600 bp；`A/C/G/T` 编码为 0–3，其他字符及 padding 编码为 4。
坐标采用 hg19/GRCh37 的 0-based 半开区间，负链取反向互补；正链 3′ 端为 `end`，负链为 `start`。

`union_id` 加盐 `b52-full-hek-v1|` 后哈希分组，组内记录进入同一集合。
目标值为 0 的 131,928 条记录完整保留。

### 双流采样与核验

seed=20261004。每个 epoch 确定性排列并遍历 347,241 条至少含一个有效标签的 ESB TRAIN 记录，
以及 531,860 条片段 TRAIN 记录。两流按计数均匀分布于 16,868 个训练步。DEV/TEST 用于评估。

```bash
python -m promoter_atlas.data_prepare --data-root research
python -m promoter_atlas.data_prepare --data-root research --deep --out data_check.json
```

检查程序核对结构和计数，`--deep` 增加源文件 SHA 校验。数据构建顺序见[源码指南](ORIGINAL_SOURCES.md)。

---

## English

The model uses E/S/B promoter records and MPRA fragment activity. Paths below are relative to the restored `research/` directory.

### E/S/B promoters

| Item | Details |
|---|---|
| Training-input table | `labels/build/promoter_atlas_v6_input.tsv` |
| All records | 1,746,138; TRAIN 1,533,642 / DEV 105,364 / TEST 107,132 |
| Species | Human and 18 orthologous species |
| Human reference | GRCh38 / GENCODE v49 transcript promoters |
| Windows | Local `[-200,+200)`, 400 bp; upstream `[-2000,0)`, 2000 bp |
| Human sequences | `promoters/memmap/promoter_{sym400,up2000}_uint8.memmap` |
| Other-species sequences | `data/multispecies/memmap/<key>_{sym400,up2000}_uint8.memmap` |
| TRAIN mean and standard deviation | E (0.7360959, 0.2327551); S (0.8307955, 0.0871514); B (0.7283229, 0.2314467) |

All three targets derive from GTEx tissue-average expression, `mean_tpm`. Human labels are transferred through orthology to form multispecies records.
Definitions are in `labels/spec/label_definition.md`; the resulting table is `labels/build/gene_expression_label_table.tsv`.

```text
E_raw = mean_tissues(log2(1 + mean_tpm))
B_raw = count(tissues with mean_tpm >= 1) / count(all tissues)
S_raw = -MAD_detectable_tissues(log2(1 + mean_tpm))
Score_X = rank_ascending_from_zero(X_raw) / (n - 1)
```

S is defined when B_raw≥0.30 and at least three tissues are detectable. Ranking uses the complete catalog of 58,988 genes, with ties ordered stably by their source-table positions.
Missing S values are placed at the bottom during ranking and restored to missing values in the output.
Missing targets receive `finite_mask=0` and are excluded from the corresponding loss. Accepted split labels are `train, val, test`.
Transcripts of the same gene share that gene's E/S/B targets.

Human records use the frozen `promoter_split_core.tsv`; the other species are split by gene within each species.
Orthologous families may have records in different sets across species. See the [evaluation notes](LIMITATIONS.md#english) for interpretation.

### HEK293 fragment activity

| Item | Details |
|---|---|
| Source file | `GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz` |
| Source size | 103,144,961 bytes |
| Source SHA-256 | `1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96` |
| Target | `Log2RPM_HEK293`, v2_MH4 fold0 |
| Reference | `Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz`, 869,923,173 bytes |
| Reference SHA-256 | `0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865` |
| Derived directory | `data/breakthrough_20261002/batch52/corrected/cohort_v2/` |
| Records | 761,655; TRAIN 531,860 / DEV 112,640 / TEST 117,155 |
| Groups | 5,332; TRAIN 3,724 / DEV 814 / TEST 794 |
| TRAIN target normalization | mean=1.6329246759414673; std=1.3893377780914307 |

The derived directory contains `codes_int64.npy`, `endpoints.tsv`, `groups.tsv`, `manifest.json` and `build_report.json`.
Fragments span 100–600 bp. `A/C/G/T` map to 0–3, and other characters and padding map to 4.
Coordinates use 0-based half-open hg19/GRCh37 intervals. Negative-strand sequences are reverse-complemented; the 3′ endpoint is `end` on the positive strand and `start` on the negative strand.

Groups are assigned by hashing `union_id` with salt `b52-full-hek-v1|`, keeping each group in one split.
All 131,928 zero-valued target records are retained.

### Dual-stream sampling and checks

The seed is 20261004. Each epoch deterministically permutes and traverses 347,241 ESB TRAIN records with at least one valid label and 531,860 fragment TRAIN records.
Both streams are spread across 16,868 training steps according to their counts. DEV and TEST serve evaluation.

```bash
python -m promoter_atlas.data_prepare --data-root research
python -m promoter_atlas.data_prepare --data-root research --deep --out data_check.json
```

The checker verifies structure and counts; `--deep` adds source-file SHA verification. See the [source guide](ORIGINAL_SOURCES.md#english) for construction order.
