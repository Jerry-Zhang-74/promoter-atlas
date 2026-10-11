# v6.15.1-release.rc2 · 独立 E/S/B 候选接口 / Independent E/S/B candidate interface

[中文](#中文) · [English](#english)

## 中文

这里保存独立 E/S/B 组合候选的接口文档、schema、模型卡与评估记录。该候选来自 batch16，采用三个独立 CNN；共享骨干及共享辅助任务的验证增益分别为 −0.0061 和 −0.0049，按固定规则选择独立组合。完整批次背景见 [batch18](../../README.md)。

### 输入与输出

- `seqs`：uint8 数组，批量形状 `(N,4,400)`、单基因形状 `(4,400)`；四个槽对应同基因启动子，A/C/G/T/N 编码为 0/1/2/3/4。
- `features`：1,069 维原始特征，批量形状 `(N,1069)`；顺序由训练特征定义固定，预测接口使用保存的 TRAIN 变换。
- E 输出为 `Score_E`，S 输出为可检出组织的 `S_raw = −MAD(log2(1+TPM))`，B 输出为可检出组织数/总组织数。接口携带 `is_model_prediction` 与 `is_catalog_observed` 来源字段。

原始包的预测类为 `PromoterAtlasGenePredictor`，提供 `predict_gene` 与 `predict_batch`。输入类型、形状及有限值错误使用 `PromoterAtlasInputError`；rc2 还显式处理 N=0 的空 batch。

### 阅读入口

- [MODEL_CARD.md](MODEL_CARD.md)：模型选择、目标和评估。
- [schema.json](schema.json)：输入输出结构。
- [example_expected_output.json](code/example_expected_output.json)：原始示例的预期输出。
- [negative_ablation_results.json](negative_ablation_results.json)：共享骨干实验的完整比较。
- [labels/README.md](labels/README.md)：原始观察标签及目标定义。

当前公开模型的安装、权重恢复和可执行示例见 [复现指南](../../../../../docs/REPRODUCIBILITY.md)。

---

## English

This directory retains the independent E/S/B candidate’s interface documentation, schema, model card, and evaluation records.The candidate comes from batch16 and uses three independent CNNs. Validation gains of −0.0061 and −0.0049 for the shared-backbone and shared-plus-auxiliary alternatives led to selection of the independent composite under the fixed rule. See [batch18](../../README.md) for context.

### Inputs and outputs

- `seqs`: uint8, `(N,4,400)` for a batch or `(4,400)` for one gene. The four slots hold promoters from the same gene; A/C/G/T/N use codes 0/1/2/3/4.
- `features`: 1,069 raw features, with batch shape `(N,1069)`. The training definition fixes their order; the predictor uses saved TRAIN transforms.
- E is `Score_E`; S is detectable-tissue `S_raw = −MAD(log2(1+TPM))`; B is detectable tissue count divided by total tissue count. The interface includes `is_model_prediction` and `is_catalog_observed` provenance fields.

The original package exposes `PromoterAtlasGenePredictor.predict_gene` and `predict_batch`. Type, shape, and nonfinite-value errors use `PromoterAtlasInputError`; rc2 also explicitly handles empty N=0 batches.

### Reading the records

- [MODEL_CARD.md](MODEL_CARD.md): selection, targets, and evaluation.
- [schema.json](schema.json): input/output structure.
- [example_expected_output.json](code/example_expected_output.json): expected original-example output.
- [negative_ablation_results.json](negative_ablation_results.json): shared-backbone comparisons.
- [labels/README.md](labels/README.md): observed labels and target definitions.

For installation, weight recovery, and runnable examples of the current public model, use the [reproduction guide](../../../../../docs/REPRODUCIBILITY.md).
