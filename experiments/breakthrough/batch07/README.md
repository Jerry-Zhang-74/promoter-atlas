# batch07 · 特征覆盖范围与选择性预测 / Feature coverage and selective prediction

[中文](#中文) · [English](#english)

## 中文

对 batch06 的冻结 HGB 预测器评估适用范围：在 TRAIN 特征上拟合标准化和 Ledoit–Wolf 协方差，用 Mahalanobis 距离衡量样本距训练特征中心的远近，以 TRAIN 距离第 90 百分位作为固定筛选阈值。

协议同时比较保留样本的误差、覆盖率和随机筛选对照；距离描述特征空间位置，输出解释围绕该几何量及实际误差展开。几何参数存于 `covergate_geometry.json`，接受条件见冻结配置。

- [覆盖率评估协议](reports/frozen_config_b7.json)
- [训练特征几何参数](models/covergate_geometry.json)

---

## English

Assess applicability of the frozen batch06 HGB predictor. Fit scaling and Ledoit–Wolf covariance on TRAIN features, measure Mahalanobis distance from the training-feature centre, and freeze the 90th TRAIN-distance percentile as the selection threshold.

The protocol compares retained-sample error, coverage, and random selection. The distance describes position in this feature geometry; interpretation uses that quantity together with observed error. Geometry is stored in `covergate_geometry.json`, with acceptance conditions in the frozen configuration.

- [Coverage evaluation protocol](reports/frozen_config_b7.json)
- [Training-feature geometry](models/covergate_geometry.json)
