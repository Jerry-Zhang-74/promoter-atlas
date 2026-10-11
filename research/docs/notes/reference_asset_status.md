# 早期参考资产状态 · Early reference-asset status

[中文](#中文) · [English](#english)

## 中文

### 早期参考资产

此阶段已取得参考基因组和 GTF，含78,691基因、507,365转录本。启动子 BED 使用考虑链方向的0-based半开区间，窗口为up200/up2000。

已生成 `promoters/bed/gencode_v49_transcript_promoters_up200.bed`、`gencode_v49_gene_promoters_merged_up200.bed`及相应up2000文件，QC汇总为 `promoters/qc/gencode_v49_promoter_summary.tsv`。

这份早期快照之后的准备步骤为 GTEx 处理、motif 特征和 memmap 构建。

---

## English

At this stage, the reference genome and GTF were available. Promoter BED files used strand-aware, zero-based half-open coordinates for up200 and up2000. The GTF contained 78,691 genes and 507,365 transcripts.

Generated files:
- `promoters/bed/gencode_v49_transcript_promoters_up200.bed`
- `promoters/bed/gencode_v49_gene_promoters_merged_up200.bed`
- `promoters/bed/gencode_v49_transcript_promoters_up2000.bed`
- `promoters/bed/gencode_v49_gene_promoters_merged_up2000.bed`
- `promoters/qc/gencode_v49_promoter_summary.tsv`

GTEx processing, motif features, and memmap generation were the next preparation steps in this early snapshot.
