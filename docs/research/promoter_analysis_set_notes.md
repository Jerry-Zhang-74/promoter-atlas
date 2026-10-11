# 启动子分析集 · Promoter analysis sets

[中文](#中文) · [English](#english)

## 中文

分层为 core_clean（up200严格通过）、extended_clean（up2000严格通过）、qc_pass_but_review（建议通过且需复核）及 qc_fail。

额外标记包括 start0<2000、长度低于标称，以及有边界/高N/缩短问题的 CHM13 救援候选。

| 指标 | up200 | up2000 |
|---|---:|---:|
| 记录数 | 507365 | 507365 |
| 建议通过 | 492108 | 315445 |
| 建议通过率 | 0.969929 | 0.621732 |
| 严格通过 | 492104 | 315436 |
| 严格通过率 | 0.969921 | 0.621714 |
| core_clean | 492104 | 0 |
| extended_clean | 0 | 315436 |
| qc_fail | 15257 | 191920 |
| 边界标记 | 4 | 6 |
| 缩短 | 0 | 7 |
| 救援候选 | 0 | 156 |

基线用 core，扩展上下文用于成对窗口比较，中间层用于敏感性分析。救援候选进入设计前先单独评估。

---

## English

Analysis tiers are `core_clean` (strict-pass up200), `extended_clean` (strict-pass up2000), `qc_pass_but_review` (recommended pass requiring review), and `qc_fail`.

Additional flags cover `start0 < 2000`, shorter-than-nominal sequences, and CHM13 rescue candidates with boundary/high-N/shortening issues.

| Statistic | up200 | up2000 |
|---|---:|---:|
| Records | 507365 | 507365 |
| Recommended pass | 492108 | 315445 |
| Recommended fraction | 0.969929 | 0.621732 |
| Strict pass | 492104 | 315436 |
| Strict fraction | 0.969921 | 0.621714 |
| core_clean | 492104 | 0 |
| extended_clean | 0 | 315436 |
| qc_fail | 15257 | 191920 |
| Boundary flag | 4 | 6 |
| Shortened | 0 | 7 |
| Rescue candidates | 0 | 156 |

Use the core set for baseline training and the extended set for paired window comparisons. Review the intermediate tier in sensitivity analyses and assess rescue candidates before sequence design.
