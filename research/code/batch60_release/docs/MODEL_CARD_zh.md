# 模型卡 MODEL_CARD（中文 · {VERSION} · 臂 {ARM}）

## 模型
- 版本：`{VERSION}`；结构：一热 ACGT 输入 + 残差 CNN（3 blocks, dilation 1/2/4）+ 掩码 mean/max/loglen/GC 汇合（130→64→1）。
- 目标量：`Log2RPM_HEK293`（作者公开列；**不是** `Log2Norm_HEK293`）。
- 训练：作者 fold0 全量 TRAIN，从零随机初始化；共完成 **6 个 epoch**，**选中的 checkpoint 为 0-based epoch 4（即第 5 个 epoch 之后）**。
- 权重 SHA256：`{SHA}`（本终版与 batch59 reviewed 逐字节一致）。
- 已知缺陷（如实保留，未改 checkpoint 掩盖）：`global_step=531860` 是历史样本数（sample count），**不是**真实优化器步数。
- checkpoint 为完整训练 checkpoint，其中含 optimizer/scheduler/sampler/RNG 状态条目；本包**不主张**已验证的“精确续训”，也不提供续训流程（预测仅依赖模型参数+config+normalizer，均已 SHA256 固定）。

## 真实留出 TEST 指标（公开 HEK293；retrospective 冻结分组；非前瞻、非本地 293FT）
| 模型 | activity rho ↑ | groupMAE ↓ | top10 精度 ↑ | pair groupMAE ↓ | pair macroSign ↑ |
|---|---|---|---|---|---|
| R0（组成 ridge 基线） | 0.5026 | 0.8628 | 0.3955 | 0.8732 | 0.5287 |
| **Nabs（绝对活性）** | 0.5749 | 0.7796 | 0.5389 | 0.8398 | 0.5619 |
| **Njoint（配对差值多任务）** | 0.5799 | 0.7806 | 0.5522 | 0.8388 | 0.5588 |

- 上表数值直接复制自独立评测器实际输出（`results/*_TEST_parent_metrics.json` 与本包 `results/`），无编造。
- **配对对比 FAIL（如实保留）**：Njoint 相对 Nabs 的 pair groupMAE 仅改善约 0.12%，**远低于**预注册的 10% 门槛；配对自助法 95% CI 含 0。
- 条件性分组自助法 CI 建立在声明分组之上，**不能当作独立家族/独立生物重复的置信区间**。

## 明确不声明（no over-claim）
- 不声明 293FT 迁移性能（本地 293FT 未验证；D28 实验 FAIL 保留）。
- 不声明前瞻性/盲测性能（公开数据此前可能已被接触，均为回顾性评估）。
- 不声明 >600 bp 支持（>600 明确 UNSUPPORTED，2000 bp 被拒）。
