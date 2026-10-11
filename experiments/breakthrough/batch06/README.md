# batch06 · 组成基线之上的 CNN 残差学习 / CNN residual learning above composition baselines

[中文](#中文) · [English](#english)

## 中文

本实验检验序列 CNN 能否补充组成和启动子数量已经解释的表达信息。先在 TRAIN 内用五折整组划分产生 Ridge 的折外预测，再让 gene-bag CNN 学习 `Score_E − OOF Ridge`，最终将残差加回完整 TRAIN 拟合的 Ridge 预测。

方案固定两个随机种子、12 epochs，以及 HGB、Ridge 和零残差对照；`frozen_config_b6.json` 给出特征、选择规则和评估阈值，两个 residual 目录保存各次训练结果。

- [残差实验协议](reports/frozen_config_b6.json)
- [种子 20261011 结果](models/residual_s20261011/result.json)
- [种子 20261012 结果](models/residual_s20261012/result.json)

---

## English

Test whether a sequence CNN adds expression information beyond composition and promoter count. Five component-preserving folds within TRAIN produce out-of-fold Ridge predictions. A gene-bag CNN learns `Score_E − OOF Ridge`, and its residual is added to the Ridge prediction fitted on all TRAIN genes.

The protocol fixes two seeds, 12 epochs, and HGB, Ridge, and zero-residual comparators. `frozen_config_b6.json` specifies features, selection, and evaluation thresholds; the two residual directories retain each training result.

- [Residual-learning protocol](reports/frozen_config_b6.json)
- [Seed 20261011 result](models/residual_s20261011/result.json)
- [Seed 20261012 result](models/residual_s20261012/result.json)
