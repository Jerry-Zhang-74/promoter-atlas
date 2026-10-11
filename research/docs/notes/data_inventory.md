# 早期数据清单 · Early data inventory

[中文](#中文) · [English](#english)

## 中文

### 早期清单

项目根目录为 promoter_atlas，legacy_data 已存在，包含 GRCh38 FASTA/FAI、Zone.Identifier、MD5SUMS 和 GENCODE v49 GTF。

| 类型 | 文件数 | 记录大小 |
|---|---:|---|
| 表达矩阵 | 0 | — |
| 启动子 BED | 0 | — |
| FASTA | 1 | 2.9 GB |
| 特征 | 0 | — |
| 数组 | 0 | — |
| 其他 | 4 | 3.1 GB |

FAI 为6.3KB，Zone.Identifier为25字节，MD5SUMS为3KB，GTF为3.1GB。

表达表需检查矩阵维度、单位和基因×组织/样本方向；启动子需检查链、长度、TSS；特征需检查行对应关系。目标定义由有效规范和构建脚本确定，复用旧 S/HKG 前先核对口径。

---

## English

### Inventory at the time of this record

`legacy_data` contained GRCh38 FASTA and FAI, a Zone.Identifier file, MD5SUMS, and a GENCODE v49 GTF.

| Category | Files | Recorded size |
|---|---:|---|
| Expression tables | 0 | — |
| Promoter BED | 0 | — |
| FASTA | 1 | 2.9 GB |
| Features | 0 | — |
| Arrays | 0 | — |
| Other files | 4 | 3.1 GB |

The FAI was 6.3 KB, Zone.Identifier 25 bytes, MD5SUMS 3 KB, and GTF 3.1 GB.

For expression tables, inspect matrix shape, units, and gene-by-tissue/sample orientation. For promoter sequences, verify strand, length, and TSS alignment. For features, verify row alignment. Build target definitions from the active specification and scripts, and resolve the historical S/HKG definitions before reuse.
