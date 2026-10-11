# 原始训练与数据构建源码

本目录是项目原始源码的平铺归档，文件名加了 `orig_` 前缀以区分于交付包自身的模块。这里
的文件**不被 `main.py` 导入**，只用于说明交付权重的训练来源与数据的真实构建路径。

原始仓库路径以 `E:/promoter_atlas` 为根。`--data-root` 就指向该项目根目录，下面所有相对
路径都以它为基准。

## 1. 训练（交付权重由这批代码产出）

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_b63_train.py` | `code/batch63_joint_train/b63_train.py` | 真正产出交付权重的训练循环：双流计数调度、暖启动、断点续训、检查点保存 |
| `orig_b63_common.py` | `code/batch63_joint_train/b63_common.py` | 训练公共件（配置、z 统计、检查点格式） |
| `orig_b63_data.py` | `code/batch63_joint_train/b63_data.py` | 双流数据适配 |
| `orig_61_v61_arch.py` | `code/batch61_dual_esb_joint/v61_arch.py` | 架构与 `warmstart_from_v6_1`（先载 base，再设种子，再建活动头） |
| `orig_61_checkpointing.py` | `code/batch61_dual_esb_joint/checkpointing.py` | 保存/加载、RNG 捕获与恢复 |
| `orig_61_train_joint.py` | `code/batch61_dual_esb_joint/train_joint.py` | 早期联合训练入口 |
| `orig_61_data_adapter.py` | `code/batch61_dual_esb_joint/data_adapter.py` | 联合数据适配原型 |
| `orig_62_esb_stream.py` | `code/batch62_joint_data/esb_stream.py` | ESB 流（memmap 读取、掩蔽） |
| `orig_62_fragment_stream.py` | `code/batch62_joint_data/fragment_stream.py` | 片段流（长度过滤、组权重） |
| `orig_62_joint_adapter.py` | `code/batch62_joint_data/joint_adapter.py` | 两流合并 |

交付包里的 `train.py` 由这套循环整理为可移植入口；权重的原始日志与指标见
`TRAIN_LOG_SUMMARY.md` 与 `PERFORMANCE_zh.md`。

## 2. 数据构建（按依赖顺序）

### 2.1 E/S/B 标签（GTEx 基因表达）

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_build_candidate_e_labels.py` | `runs/scripts/build_candidate_e_labels.py` | 由 `gene_expression_table.tsv` 生成 `gene_expression_label_table.tsv`：`E_raw = mean(log2(1+mean_tpm))`、`B_raw = TPM>=1 组织占比`、`S_raw = -MAD(log2(1+mean_tpm))`（仅当 `B_raw>=0.30` 且可检出组织数 `>=3`），再对每个 `*_raw` 取全目录升序秩得到 `Score_*` |

依赖：`labels/build/gene_expression_table.tsv`（GTEx 长表，`gene_id, tissue, mean_tpm`）。
产出：`labels/build/gene_expression_label_table.tsv`（`E_raw/S_raw/B_raw` 与 `Score_E/S/B`）。

### 2.2 基因级切分

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_build_train_eval_split.py` | `runs/scripts/build_train_eval_split.py` | 生成冻结的 `labels/releases/promoter_split_core.tsv`（`gene_id_base, split`） |

### 2.3 序列与坐标

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_download_multispecies_genomes.py` | `runs/scripts/download_multispecies_genomes.py` | 从 Ensembl 拉取各物种基因组与 GTF |
| `orig_fetch_orthologs_biomart.py` | `runs/scripts/fetch_orthologs_biomart.py` | 取人→物种一对一 ortholog 表 |
| `orig_build_multispecies_promoters.py` | `runs/scripts/build_multispecies_promoters.py` | 由各物种 FASTA/GTF 构建 up2000 与 sym400 窗口，写 `data/multispecies/memmap/<species>_<window>_uint8.memmap` 与 `_index.tsv`，并向 `promoter_master_index.tsv` 追加 |
| `orig_build_human_sym400_promoters.py` | `runs/scripts/build_human_sym400_promoters.py` | 人类 sym400 窗口 |
| `orig_build_promoter_sequence_memmap.py` | `runs/scripts/build_promoter_sequence_memmap.py` | 人类 `promoters/memmap/promoter_{up200,up2000,sym400}_uint8.memmap` |
| `orig_build_promoter_sequence_memmap_up2000.py` | `runs/scripts/build_promoter_sequence_memmap_up2000.py` | 人类 up2000 窗口（补充分支） |

编码：`A/C/G/T -> 0..3`，其余（含 N）`-> 4`；负链取反向互补。

### 2.4 训练输入总表

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_build_v6_training_input.py` | `runs/scripts/build_v6_training_input.py` | 汇总 `labels/build/promoter_atlas_v6_input.tsv`：人类全转录本用冻结基因级切分 + 基因级 E/S/B；其他物种按物种内 90/5/5 随机切分，并用一对一 ortholog 迁移 `Score_E` |

依赖（人类）：`promoters/qc/gencode_v49_transcript_promoter_index.tsv`、
`labels/build/gene_expression_label_table.tsv`、`labels/releases/promoter_split_core.tsv`、
`features/omics/*`、`data/multispecies/promoter_master_index.tsv`、
`features/multispecies/ortholog_human_to_species.tsv`。
产出：`labels/build/promoter_atlas_v6_input.tsv`。

注意：`--cap-per-species` 默认 150000（物种内随机抽样，种子 20260912）；人类 507,365 行全量保留。

### 2.5 MPRA 片段 cohort

| 归档文件 | 原路径 | 作用 |
|---|---|---|
| `orig_b51_build_cohort.py` | `code/batch51/build_cohort.py` | 解析 GSE301246 BED，切取 GRCh37 参考序列 |
| `orig_b52_cohort_v2.py` | `code/batch52_data_v2/cohort_v2.py` | cohort v2 定义：长度过滤、编码、`union_id` 加盐切分 |
| `orig_b52_build_cohort_v2.py` | `code/batch52_data_v2/build_cohort_v2.py` | 构建 `data/breakthrough_20261002/batch52/corrected/cohort_v2/` |
| `orig_b52_run_all_v2.py` | `code/batch52_data_v2/run_all_v2.py` | 端到端运行 |
| `orig_b52_verify_v2.py` | `code/batch52_data_v2/verify_v2.py` | 产出后校验 |
| `orig_b52_finalize_v2.py` | `code/batch52_data_v2/finalize_v2.py` | 固化 manifest 与 datahash |

输入：`data/breakthrough_20261002/batch51/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz`、
`data/breakthrough_20261002/batch52/reference/GRCh37.primary_assembly.fa.gz`（SHA-256 见
`DATA_NOTES.md`）。产出：`cohort_v2/{codes_int64.npy, endpoints.tsv, groups.tsv, manifest.json}`。
切分盐串 `b52-full-hek-v1|`：对 `union_id` 加盐哈希在组级别决定 TRAIN/DEV/TEST。

## 3. 重建顺序与路径映射

1. GTEx 长表 -> `orig_build_candidate_e_labels.py` -> `gene_expression_label_table.tsv`；
2. `orig_build_train_eval_split.py` -> `promoter_split_core.tsv`；
3. `orig_download_multispecies_genomes.py` + `orig_fetch_orthologs_biomart.py` 取物种基因组与 ortholog；
4. `orig_build_multispecies_promoters.py`（逐物种）+ 人类 memmap 脚本 -> `promoters/memmap/` 与
   `data/multispecies/memmap/`；
5. `orig_build_v6_training_input.py` -> `labels/build/promoter_atlas_v6_input.tsv`；
6. MPRA 侧：`orig_b51_build_cohort.py` -> `orig_b52_*` -> `cohort_v2/`；
7. `train.py`（或原始 `orig_b63_train.py`）以 `--data-root <项目根>` 读取以上全部。

`data_prepare.py` 会逐项核对本目录所列产物的存在性、字节数与（深度模式下）源文件
SHA-256。完整数据获取方式见 `DATA_NOTES.md`。
