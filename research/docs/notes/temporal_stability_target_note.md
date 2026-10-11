# 时间持久性目标 · Temporal durability target

[中文](#中文) · [English](#english)

## 中文

跨组织稳定性与时间持久性分别描述组织差异和时间变化，作为独立输出保存。

`S_tissue = -MAD{log2(1+x_gt), t ∈ D_g}`

`D_time = -MAD{log2(1+y_pτ), τ ∈ T}`

`R_late/early = (y_late + ε)/(y_early + ε)`

现有 GTEx 标签支持 E、S_tissue、B。D_time 和保留率需要额外的纵向活动测量；完成数据整理后可增设对应预测头。

---

## English

Cross-tissue stability describes differences among tissues; temporal durability describes change over time. Store them as separate outputs.

`S_tissue = -MAD{log2(1+x_gt), t ∈ D_g}`

`D_time = -MAD{log2(1+y_pτ), τ ∈ T}`

`R_late/early = (y_late + ε)/(y_early + ε)`

The existing GTEx labels support E, S_tissue, and B. D_time and retention require additional longitudinal activity measurements. Corresponding prediction heads can be added after those data are prepared.
