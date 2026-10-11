# batch55 打包交接 · batch55 packaging handoff

[中文](#中文) · [English](#english)

## 中文

### 打包和预测

`v6.38.0-first-model-submission-integration.rc1` 为 batch52/batch53 的 `Log2RPM_HEK293` 模型提供 stager 和独立预测器。交接时状态为 STAGER_READY_ONLY，等待 Nabs 权重。

predict.py 按自身路径加载 pinned_core，支持中文/空格路径；TRAINED 模型用 CUDA，SOFTWAREONLY 人工测试夹具可用 CPU。stage_submission.py 按白名单打包、扫描凭据、生成 SHA256SUMS 和 STAGED_COMPLETE/PARTIAL/REFUSED；make_fixture_model.py 构造测试夹具；tests/test_batch55.py 检查输入、哈希、路径和错误处理。

缺少权重、checkpoint SHA 不符、目标/源 SHA 不符、参数仍等于初始化、epoch/step/train_loss 空缺、占位结果、测试夹具被作为正式模型、凭据扫描命中都会返回 REFUSED。可选留出结果缺失时返回 STAGER_PARTIAL。

FASTA 接受 A/C/G/T/N，检查空文件/记录、全 N、重复 ID、非法字符及 >600 bp。正式输入清单需 kind=TRAINED、target_column=Log2RPM_HEK293、source SHA `1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96`、config SHA `12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157`、checkpoint 路径/SHA/epoch 及 normalizer 均值/标准差。

测试阶段使用 CPU 人工夹具；正式权重前向由独占 GPU 评估阶段执行。固定核心各文件 SHA 见 PINNED_CORE_SHA256.json。

```bash
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/tests/test_batch55.py
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/make_fixture_model.py --out <fixture_dir>
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/stage_submission.py --bundle <real_bundle_dir> --cohort-dir data/breakthrough_20261002/batch52/corrected/cohort_v2 --results <heldout_results.json> --version v6.38.0-first-model --out releases/batch52_first_model/v6.38.0-first-model
.venv/Scripts/python.exe -X utf8 <staged>/code/predict.py --bundle <staged> --fasta <in.fa> --out <preds.tsv>
```

---

## English

### Staging and prediction

`v6.38.0-first-model-submission-integration.rc1` provides a stager and standalone predictor for the batch52/batch53 `Log2RPM_HEK293` model. At handoff, status was STAGER_READY_ONLY, awaiting Nabs weights.

predict.py resolves pinned_core relative to itself and supports non-ASCII/space-containing paths. TRAINED models use CUDA; artificial SOFTWAREONLY fixtures may use CPU. stage_submission.py uses whitelist packaging, credential scanning, SHA256SUMS, and STAGED_COMPLETE/PARTIAL/REFUSED states. make_fixture_model.py creates fixtures; tests/test_batch55.py checks inputs, hashes, paths, and error handling.

Missing weights, checkpoint-SHA mismatch, target/source-SHA mismatch, unchanged initial parameters, empty epoch/step/train_loss, placeholder results, fixtures supplied as trained models, and credential matches return REFUSED. Missing optional held-out results return STAGER_PARTIAL.

FASTA accepts A/C/G/T/N and checks empty/all-N records, duplicate IDs, invalid characters, and lengths above 600 bp. The model manifest requires kind=TRAINED, target_column=Log2RPM_HEK293, source SHA `1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96`, config SHA `12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157`, checkpoint path/SHA/epoch, and normalizer mean/std.

Software checks used artificial CPU fixtures; real-weight forward checks belong to the exclusive GPU evaluation stage. Per-file pinned-core hashes are in PINNED_CORE_SHA256.json.

```bash
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/tests/test_batch55.py
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/make_fixture_model.py --out <fixture_dir>
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch55_submission/stage_submission.py --bundle <real_bundle_dir> --cohort-dir data/breakthrough_20261002/batch52/corrected/cohort_v2 --results <heldout_results.json> --version v6.38.0-first-model --out releases/batch52_first_model/v6.38.0-first-model
.venv/Scripts/python.exe -X utf8 <staged>/code/predict.py --bundle <staged> --fasta <in.fa> --out <preds.tsv>
```
