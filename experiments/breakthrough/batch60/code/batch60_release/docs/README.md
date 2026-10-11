# 片段活性预测包 {VERSION} · Fragment activity bundle {VERSION}

[中文](#中文) · [English](#english)

## 中文

本历史发布模板用于项目 `{ARM_DESC}` / `{ARM_ROLE}` 模型，臂 `{ARM}`，目标为 `Log2RPM_HEK293`。解压后根目录含 `predict.py` 和 `pinned_core/`，支持任意工作目录及中文、空格路径。

```text
predict.py
pinned_core/  PINNED_CORE_SHA256.json
MODEL_MANIFEST.json
model/       {CKPT} + .sha256.json
config/config.json
normalizer/normalizer.json
results/
USER_GUIDE.md  USER_GUIDE_zh.md  MODEL_CARD_zh.md
LICENSE_NOTES.md  requirements.txt  SHA256SUMS.txt
```

清单记录目标、checkpoint 路径和 SHA；config 与 normalizer 固定训练配置和尺度；results 保存 TEST 评估及复核。`SHA256SUMS.txt` 使用相对根目录路径。依赖 Python 3.12、torch 2.11、numpy 2.5、CUDA GPU。

运行 `python -X utf8 "<extracted-bundle>\predict.py" --bundle "<extracted-bundle>" --fasta "<input>.fa" --out "<out>.tsv"`，完整步骤见使用指南。

FASTA 校验空记录、全 N、重复 ID、非法碱基和 >600 bp。训练范围 100–600 bp，序列来自 GRCh37/hg19 代理，评估为公开 HEK293 冻结分组回顾性 TEST。293FT 迁移待评估。Njoint 配对指标改善约 0.12%，预设目标 10%，区间含 0，结论 FAIL。

---

## English

This historical release template packages the project `{ARM_DESC}` / `{ARM_ROLE}` model, arm `{ARM}`, targeting `Log2RPM_HEK293`. The extracted root contains `predict.py` and `pinned_core/`, supporting arbitrary working directories and paths with spaces or non-ASCII characters.

```text
predict.py
pinned_core/  PINNED_CORE_SHA256.json
MODEL_MANIFEST.json
model/       {CKPT} + .sha256.json
config/config.json
normalizer/normalizer.json
results/
USER_GUIDE.md  USER_GUIDE_zh.md  MODEL_CARD_zh.md
LICENSE_NOTES.md  requirements.txt  SHA256SUMS.txt
```

The manifest records target, checkpoint path, and SHA; config and normalizer freeze training settings and scale; results contain TEST evaluation and review. `SHA256SUMS.txt` uses root-relative paths. Dependencies are Python 3.12, torch 2.11, numpy 2.5, and a CUDA GPU.

Run `python -X utf8 "<extracted-bundle>\predict.py" --bundle "<extracted-bundle>" --fasta "<input>.fa" --out "<out>.tsv"`; see the user guide for full steps.

FASTA validation checks empty/all-N records, duplicate IDs, invalid bases, and lengths above 600 bp. Training covers 100–600 bp GRCh37/hg19 proxy sequences, evaluated retrospectively on public HEK293 frozen-group TEST data. Transfer to 293FT awaits evaluation. Njoint improved the pair metric by about 0.12% against a 10% target, with a CI including zero and a FAIL verdict.
