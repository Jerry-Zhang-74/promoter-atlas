# batch16 · 独立模型与共享骨干的 E/S/B 比较 / Independent versus shared E/S/B models

[中文](#中文) · [English](#english)

## 中文

比较三个独立目标模型、共享骨干模型及增加辅助损失的共享模型，任务分别为 Score_E、原始稳定度 S_raw 和表达广度 B_raw。配置固定验证选择规则，并保留 HGB 特征基线。

共享臂 B/C 的平均验证增益分别为 −0.0061/−0.0049，最终部署选择独立组合 A。单 seed 20261016 的测试 E/S/B Spearman 为 0.7428/0.4342/0.7262；该值对应本批标签定义。provenance 子目录记录恢复执行时实际源码。

- [三臂配置](models/config.json)
- [选中模型](models/deployment_bundle.json)
- [指标独立复算](runtime/batch16/review/review_recompute_results.json)

---

## English

Compare independent target models, a shared backbone, and a shared backbone with auxiliary loss for Score_E, raw stability S_raw, and breadth B_raw. The configuration fixes the validation selection rule and retains HGB feature baselines.

Shared arms B/C have mean validation gains of −0.0061/−0.0049, leading to selection of independent composite A. For seed 20261016, TEST E/S/B Spearman is 0.7428/0.4342/0.7262 under this batch's target definitions. The provenance directory records the source used during recovery execution.

- [Three-arm configuration](models/config.json)
- [Selected model](models/deployment_bundle.json)
- [Independent metric recomputation](runtime/batch16/review/review_recompute_results.json)
