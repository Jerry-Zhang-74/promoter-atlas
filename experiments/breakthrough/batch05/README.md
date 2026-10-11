# batch05 · 相似序列分组后的表达基线比较 / Expression baselines after sequence grouping

[中文](#中文) · [English](#english)

## 中文

在精确重复、反向互补和检测到的近似序列分组划分上，重新比较四槽重复/多启动子 CNN 与序列组成、启动子数量基线。特征包含 44 维双窗口组成统计和启动子数量；使用整组 bootstrap 保留组内依赖。

当前汇总中多启动子 CNN 的 TEST Spearman 为 0.7096，最佳组成/数量 Ridge 为 0.7231。较早的观察快照保存在 runtime 下；模型比较使用 reports 中的最终汇总及对应冻结清单。

- [分组比较结果](reports/consolidated_summary_v5.json)
- [冻结实验清单](runtime/batch05/frozen_manifest.json)

---

## English

Recompare repeated/diverse four-slot CNNs with composition and promoter-count baselines after grouping exact, reverse-complement, and detected near-identical sequences. Features comprise 44 composition statistics across two windows plus promoter count. Whole-component bootstrap preserves within-group dependence.

The consolidated TEST Spearman is 0.7096 for the diverse-promoter CNN and 0.7231 for the strongest composition/count Ridge. An earlier observation snapshot remains under runtime; use the report summary with its frozen manifest for the final comparison.

- [Grouped comparison](reports/consolidated_summary_v5.json)
- [Frozen experiment manifest](runtime/batch05/frozen_manifest.json)
