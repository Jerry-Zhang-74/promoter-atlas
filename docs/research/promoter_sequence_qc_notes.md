# 启动子序列 QC · Promoter sequence QC

[中文](#中文) · [English](#english)

## 中文

QC 标记阈值为 N 比例>0.10、base-4熵<1.20、同聚物长度≥12、二核苷酸重复比例≥0.30。

| 指标 | up200 | up2000 |
|---|---:|---:|
| 记录数 | 507365 | 507365 |
| 建议通过 | 492108 | 315445 |
| 通过率 | 0.969929 | 0.621732 |
| 高N | 13 | 147 |
| 低熵 | 107 | 4 |
| 同聚物 | 14212 | 191770 |
| 二核苷酸重复 | 1115 | 281 |
| 平均GC | 0.575897 | 0.486149 |
| 平均N比例 | 0.000015 | 0.000143 |
| 平均熵 | 1.898611 | 1.972940 |

两个窗口采用相同 QC 条件比较；候选先通过序列 QC，再进行生物学评估。

---

## English

QC flags use N fraction >0.10, base-4 entropy <1.20, homopolymer length ≥12, and dinucleotide-repeat fraction ≥0.30.

| Statistic | up200 | up2000 |
|---|---:|---:|
| Records | 507365 | 507365 |
| Recommended pass | 492108 | 315445 |
| Pass fraction | 0.969929 | 0.621732 |
| High N | 13 | 147 |
| Low entropy | 107 | 4 |
| Homopolymer | 14212 | 191770 |
| Dinucleotide repeat | 1115 | 281 |
| Mean GC | 0.575897 | 0.486149 |
| Mean N fraction | 0.000015 | 0.000143 |
| Mean entropy | 1.898611 | 1.972940 |

Compare the two windows under the same QC policy. Apply sequence QC before candidate nomination, followed by biological assessment.
