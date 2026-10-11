# 数据说明

本模型用到两套数据：ESB 启动子记录（主任务）与 MPRA 片段活性（辅助任务）。完整数据单独保存，来源、SHA-256、清洗与切分规则记录在此，`data_prepare.py`
可在本地副本上重新校验这些数字。真实的数据预处理实现见 `ORIGINAL_SOURCES.md 及 orig_*.py`。

## 1. ESB 启动子记录（项目内派生资产）

| 项目 | 值 |
|---|---|
| 表 | `labels/build/promoter_atlas_v6_input.tsv` |
| 行数 | 1,746,138（TRAIN 1,533,642 / DEV 105,364 / TEST 107,132） |
| 物种 | 19（人 + 18 个直系同源物种） |
| 坐标与序列 | GENCODE v49 转录本启动子。人用 GRCh38；其余 18 个物种使用各自装配。核心窗 `[-200,+200)` 400 bp、上游窗 `[-2000,0)` 2000 bp，分别以 uint8 记忆映射存放：人为 `promoters/memmap/promoter_{sym400,up2000}_uint8.memmap`，其余为 `data/multispecies/memmap/<key>_{sym400,up2000}_uint8.memmap` |
| 标签 | `Score_E` / `Score_S` / `Score_B`；TRAIN 集统计量：E (0.7360959, 0.2327551)、S (0.8307955, 0.0871514)、B (0.7283229, 0.2314467)（均值, 标准差） |

标签来源：三者都由 GTEx 基因表达量（`mean_tpm`）派生，人类标签经直系同源映射形成多物种训练记录。
按 `labels/spec/label_definition.md` 与构建脚本（产物 `labels/build/gene_expression_label_table.tsv`）：

* `E_raw = mean_tissues( log2(1 + mean_tpm) )` —— 表达强度；
* `B_raw = #{tissue: mean_tpm >= 1} / #tissue` —— 表达广度（可检出组织占比）；
* `S_raw = -MAD_detectable_tissues( log2(1 + mean_tpm) )` —— 跨组织稳定性，只在 `B_raw >= 0.30` 且
  可检出组织数 `>= 3` 时定义；
* `Score_X = rank_scale(X_raw)`，即从 0 开始的升序秩 / (n-1)，取自完整 58,988 基因目录；同值按源表顺序稳定排序，S 未定义项先置底排序，再将输出设为空；`S_raw` 未定义处 `Score_S` 为空。

清洗与切分：

* 缺失标签保留为缺失并被掩蔽（`finite_mask = 0`），不填 0、不做插补。
* `split` 值不在 `{train, val, test}` 内的行被排除，不并入 TRAIN。
* 同一基因下的转录本记录共享首轮 E/S/B 标签。
* ESB 切分为基因级：人用冻结的 `promoter_split_core.tsv`，18 个直系同源物种各自做物种内
  基因级切分。同一物种内没有同基因跨集；但直系同源关系跨物种，同一基因家族在不同物种的
  记录可能落在不同集，切分单位为物种内基因，跨物种直系同源关系保留。

## 2. 片段活性记录（MPRA，GEO GSE301246）

| 项目 | 值 |
|---|---|
| 源文件 | `GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz` |
| 源 SHA-256 | `1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96` |
| 源字节数 | 103,144,961 |
| 目标 | `Log2RPM_HEK293`（v2_MH4 fold0，HEK293） |
| 参考序列 | `Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz`，SHA-256 `0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865`，869,923,173 字节，来源 Ensembl FTP GRCh37 主装配 |
| 派生表 | `data/breakthrough_20261002/batch52/corrected/cohort_v2/`（`codes_int64.npy`、`endpoints.tsv`、`groups.tsv`、`manifest.json`、`build_report.json`） |
| 行数 | 761,655（TRAIN 531,860 / DEV 112,640 / TEST 117,155） |
| 组 | 5,332（TRAIN 3,724 / DEV 814 / TEST 794） |
| 目标归一化 | TRAIN 集 mean 1.6329246759414673，std 1.3893377780914307 |

清洗与切分：

* 片段长度限定 100–600 bp，超出行被丢弃；编码 `A/C/G/T -> 0..3`，其余（含 padding）为 4。
* 坐标为 hg19 / GRCh37 的 0-based 半开区间；负链取反向互补；片段 3' 端在正链为 `end`、
  负链为 `start`。
* 切分由 `union_id` 的加盐哈希（`b52-full-hek-v1|`）在**组级别**决定，5,332 组 -> TRAIN
  3,724 / DEV 814 / TEST 794；同一 `union_id` 组不跨集，因此 MPRA 片段在其组内无跨集泄漏。
* 含 131,928 个目标值为 0 的行照原样保留，不做过滤。

## 3. 两条数据流的采样

TRAIN 每个 epoch 按种子 20261004 的确定性排列遍历：ESB 侧遍历有监督（至少一个标签非缺失）
的记录（347,241 行），片段侧遍历 531,860 行，两者按计数均匀铺开，共 16,868 步。
DEV / TEST 不参与训练。

## 4. 复现

```bash
python data_prepare.py --data-root /path/to/promoter_atlas            # 结构 + 计数核对
python data_prepare.py --data-root /path/to/promoter_atlas --deep     # 追加源文件 SHA-256 核对
```

`--data-root` 是只读校验器；ESB 记忆映射与片段 cohort 的完整预处理实现（依赖、顺序与路径映射）
见 `ORIGINAL_SOURCES.md 及 orig_*.py`。运行 data_prepare.py --data-root <项目路径> --out data_check.json 可生成本地数据检查记录。
