# batch47 · 序列推理与观察标签分流 / Routing sequence inference and observed labels

[中文](#中文) · [English](#english)

## 中文

建立按输入类型分流的历史命令接口：`fragment` 调用冻结 PARM 片段引擎，`gene-observed` 按基因标识查询标签目录。两条路线分别保留模型来源或观察标签来源。

同时补齐 batch46 的 244 个三元组逐 RNA 重复最大/最小差值，四个 RNA 重复列均有 244 个有限值。标签字段解释与产物冻结顺序的后续修订见 batch48。

- [输入分流接口](../shared/runs/code/batch47/b47_cli.py)
- [逐重复差值补记](reports/b46_addendum_dmn_per_rna_rep_summary.json)
- [字段与冻结顺序修订](../batch48/README.md)

---

## English

Create historical input routing: `fragment` calls the frozen PARM fragment engine, while `gene-observed` looks up catalog labels by gene identifier. Each route preserves either model provenance or observed-label provenance.

Also complete per-RNA-replicate max/min contrasts for batch46's 244 triplets, with 244 finite values in each of four replicate columns. Batch48 subsequently revises label-field descriptions and artifact finalization order.

- [Input-routing CLI](../shared/runs/code/batch47/b47_cli.py)
- [Replicate-contrast addendum](reports/b46_addendum_dmn_per_rna_rep_summary.json)
- [Field/finalization revision](../batch48/README.md)
