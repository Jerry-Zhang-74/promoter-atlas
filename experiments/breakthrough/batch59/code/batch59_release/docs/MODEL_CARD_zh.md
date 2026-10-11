# 模型卡 {VERSION} · Model card {VERSION}

[中文](#中文) · [English](#english)

## 中文

### 模型与训练

版本 `{VERSION}`，臂 `{ARM}`，权重 SHA-256 `{SHA}`。结构为 ACGT 一热编码、3 个残差 CNN 块（dilation 1/2/4）、掩码 mean/max 与 loglen/GC 汇合、130→64→1 输出。目标为作者发布的 `Log2RPM_HEK293`。

使用作者 fold0 全量 TRAIN，从随机初始化训练 6 个 epoch，选中 0-based epoch 4（完成第 5 个 epoch 后）。历史 checkpoint 的 `global_step=531860` 记录样本数；优化器步数需按训练日志核对。checkpoint 同时保存 optimizer/scheduler/sampler/RNG 条目，该包提供的工作流程为推理，预测依赖权重、config 和 normalizer。

### HEK293 回顾性 TEST 评估

| 模型 | activity rho ↑ | groupMAE ↓ | top10 精度 ↑ | pair groupMAE ↓ | pair macroSign ↑ |
|---|---:|---:|---:|---:|---:|
| R0 | 0.5026 | 0.8628 | 0.3955 | 0.8732 | 0.5287 |
| Nabs | 0.5749 | 0.7796 | 0.5389 | 0.8398 | 0.5619 |
| Njoint | 0.5799 | 0.7806 | 0.5522 | 0.8388 | 0.5588 |

R0 是组成 ridge 基线，Nabs 预测绝对活性，Njoint 加入配对差值任务。指标来自独立评估 JSON。Njoint 相对 Nabs 的 pair groupMAE 改善约 0.12%，预设门槛为 10%，配对自助法 95% CI 包含 0，配对比较结论为 FAIL。区间以声明的分组为重采样单位，反映该分组下的不确定性。

### 应用范围

训练序列为 GRCh37/hg19 参考基因组代理，测定细胞背景为 HEK293。支持的训练长度范围为 100–600 bp，输入超过 600 bp 时返回错误。公开数据采用冻结分组的回顾性评估。293FT 迁移仍待评估，历史 D28 实验记录为 FAIL。

---

## English

### Model and training

Version `{VERSION}`, arm `{ARM}`, weight SHA-256 `{SHA}`. The architecture uses ACGT one-hot input, three residual CNN blocks (dilations 1/2/4), masked mean/max pooling with loglength/GC, and a 130→64→1 output. The target is the author-published `Log2RPM_HEK293`.

Training used all author-fold0 TRAIN rows, random initialization, and six epochs. The selected checkpoint is zero-based epoch 4, after the fifth epoch. Its historical `global_step=531860` records sample count; optimizer steps should be checked against the training log. The checkpoint also contains optimizer/scheduler/sampler/RNG entries. The provided workflow is inference, using weights, config, and normalizer.

### Retrospective HEK293 TEST evaluation

| Model | activity rho ↑ | groupMAE ↓ | top10 precision ↑ | pair groupMAE ↓ | pair macroSign ↑ |
|---|---:|---:|---:|---:|---:|
| R0 | 0.5026 | 0.8628 | 0.3955 | 0.8732 | 0.5287 |
| Nabs | 0.5749 | 0.7796 | 0.5389 | 0.8398 | 0.5619 |
| Njoint | 0.5799 | 0.7806 | 0.5522 | 0.8388 | 0.5588 |

R0 is the composition ridge baseline, Nabs predicts absolute activity, and Njoint adds pair-difference supervision. Metrics come from independent evaluation JSON. Njoint improved pair groupMAE by about 0.12% over Nabs, against a prespecified 10% threshold; the paired-bootstrap 95% CI includes zero and the pair comparison verdict is FAIL. Intervals use the declared groups as resampling units and describe uncertainty under that grouping.

### Application domain

Training sequences are GRCh37/hg19 reference-genome proxies and measurements come from HEK293. The empirical training domain is 100–600 bp; inputs above 600 bp return an error. Public-data evaluation is retrospective with frozen groups. Transfer to 293FT awaits evaluation; the historical D28 experiment records FAIL.
