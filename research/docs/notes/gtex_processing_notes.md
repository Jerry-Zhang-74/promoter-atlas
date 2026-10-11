# GTEx 处理记录 · GTEx processing notes

[中文](#中文) · [English](#english)

## 中文

输入为 `data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz` 和 `data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt`。

输出为 `labels/build/gene_expression_table.tsv` 和 `labels/qc/gtex_expression_summary.tsv`。

逐基因流式读取压缩 GCT，按组织分组计算 TPM 均值与中位数，去除基因 ID 版本后缀。内存中保留样本-组织映射及单个基因行，结果逐步写入磁盘。

---

## English

Inputs:
- `data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz`
- `data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt`

Outputs:
- `labels/build/gene_expression_table.tsv`
- `labels/qc/gtex_expression_summary.tsv`

Stream the compressed GCT one gene at a time, group samples by tissue, and calculate mean and median TPM. Strip gene-ID version suffixes. Keep the sample-to-tissue map and one gene row in memory, and write results incrementally.
