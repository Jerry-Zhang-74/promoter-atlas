# batch09 · 原始稳定度与排名目标比较 / Raw stability versus rank targets

[中文](#中文) · [English](#english)

## 中文

在有 S 标签且具有启动子特征的 18,769 个基因上，比较稳定度目标的不同表达方式。输入沿用 45 维组成/数量特征，特征标准化、目标标准化和经验排名映射都由 TRAIN 拟合。

预检列出因缺少启动子特征而排除的 178 个基因；结果同时保留原始 S 单位的误差、秩相关和模型回放检查，使目标缩放的影响可以单独核对。

- [目标与模型配置](models/config.json)
- [稳定度比较结果](reports/metrics.json)
- [回放与数据检查](reports/tests_summary.json)

---

## English

Compare representations of the stability target on 18,769 genes with both defined S and promoter features. Inputs reuse the 45 composition/count features. Feature scaling, target scaling, and empirical rank mapping are fitted on TRAIN.

Preflight records 178 excluded genes lacking promoter features. Results retain errors in raw S units, rank correlations, and serialized-model replay checks so the effect of target scaling can be inspected directly.

- [Target and model configuration](models/config.json)
- [Stability comparison](reports/metrics.json)
- [Replay and data checks](reports/tests_summary.json)
