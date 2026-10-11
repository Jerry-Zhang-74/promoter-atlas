# batch29 · SuRE 代理活性目标与序列基线 / SuRE proxy activity and sequence baselines

[中文](#中文) · [English](#english)

## 中文

为 SuRE 片段固定目标：先按文库总量归一化 RNA 与 `1 + plasmid`，再对各 RNA 重复计算 log2(1+ratio) 并取均值。按染色体及序列关系冻结划分，比较组成 Ridge 和序列 CNN，并保留源序列核对记录。

本目录保存划分表、来源解析和已执行尝试。后续 batch30 修正了 CNN 输入补齐区参与池化的问题，复查这批模型时可对照该修订。

- [来源解析](reports/source_resolution.json)
- [划分规则](reports/split.json)
- [代理目标计算](../shared/runs/code/batch29/b29_target.py)
- [输入修订](../batch30/README.md)

---

## English

Freeze a SuRE fragment target by normalizing RNA and `1 + plasmid` by library totals, computing log2(1+ratio) per RNA replicate, and averaging replicates. Freeze chromosome/sequence-based splits, compare composition Ridge with a sequence CNN, and retain source-sequence checks.

This directory stores split assignments, source resolution, and executed attempts. Batch30 subsequently repairs pooling over padded CNN inputs; consult that correction when examining this batch's models.

- [Source resolution](reports/source_resolution.json)
- [Split rules](reports/split.json)
- [Proxy-target computation](../shared/runs/code/batch29/b29_target.py)
- [Input correction](../batch30/README.md)
