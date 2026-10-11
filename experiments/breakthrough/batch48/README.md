# batch48 · 观察稳定度字段与产物冻结顺序修复 / Observed-stability fields and artifact finalization

[中文](#中文) · [English](#english)

## 中文

修订 batch47 的两个具体问题：按实际标签构建定义解释 `observedScore_S`；将所有纳入哈希的文件写完后再生成清单、执行两次只读复核，避免清单冻结后继续改写文件。

片段与基因观察值的路由沿用此前接口。`b48_gene_observed.py` 包含修订后的字段定义，`b48_finalize.py` 明确写入顺序，`manifest_verification.json` 保存两次文件复核结果。

- [观察标签字段](../shared/runs/code/batch48/b48_gene_observed.py)
- [冻结顺序实现](../shared/runs/code/batch48/b48_finalize.py)
- [两次校验结果](reports/manifest_verification.json)

---

## English

Repair two batch47 issues: describe `observedScore_S` according to the label-building definition, and write all hashed artifacts before creating the manifest and performing two read-only checks. This resolves changes made after the earlier manifest was frozen.

Fragment and observed-gene routing retain their existing interfaces. `b48_gene_observed.py` contains corrected field definitions; `b48_finalize.py` defines write order; `manifest_verification.json` records the two checks.

- [Observed-label fields](../shared/runs/code/batch48/b48_gene_observed.py)
- [Finalization order](../shared/runs/code/batch48/b48_finalize.py)
- [Two-pass verification](reports/manifest_verification.json)
