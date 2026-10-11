# batch53 模型实现与交接 · batch53 model implementation and handoff

[中文](#中文) · [English](#english)

## 中文

### 固定模型与训练协议

`v6.38.0-model-pipeline-code.rc1` 实现 batch52 的 R0、Nabs 与 Njoint，配置权威为 `spec.py`，目标为 `Log2RPM_HEK293`。

两神经臂 seed=20261004，初始化字节相同。ACGT 一热补零至 600，使用有效位掩码；Conv1d(4,64,k9,same)+GELU；3 个残差块，各有两层 Conv1d(64,64,k5,same)，dilation=1/2/4、GELU、dropout=0.1，每层卷积后掩码；masked mean/max 与标准化 loglength/GC 汇合为 130→64→1。

AdamW lr=1e-3、wd=1e-4、梯度裁剪 1.0、最多 6 个全量 epoch、逐 epoch 确定性排列、有效 batch32。Nabs 用组大小倒数加权且均值为 1 的 SmoothL1；Njoint 再加 0.5×同样加权的配对差值损失，配对截距为 0。DEV 按组平均配对 MAE 选 checkpoint，平局取最早 epoch。

R0 为 mono..tetra 340 维加 GC/loglength 共 342 维、alpha100 加权 ridge；TRAIN 拟合标准化和加权均值截距，配对差值为 short−long。

### 模块

data_adapter 读取并验证 TSV/FASTA 或 NPZ；model_core 定义 SeqCNN、encoder、param_sha256 和 predictor；normalizer 在 TRAIN 上拟合；train_neural 训练双臂；r0_baseline 实现 ridge；checkpoint 原子保存权重/优化器/RNG/采样器/配置/数据哈希/执行 ID；eval_replay 提供有锁的 ≤3 分钟 GPU 回放及 CPU 指标/bootstrap；pipeline_coordinator 按 R0→Nabs→Njoint 串行执行；epoch_cap 按吞吐选择共享 6/4/2 epoch 上限；gpu_lock 管理设备；exec_gate 管理台账；selfcheck/tests 检查软件行为。

数据要求：endpoints.tsv 包含 coord_id、union_id、split、length、Log2RPM_HEK293；groups.tsv 包含 union_id/split；pairs.tsv 包含同组短/长 coord_id；序列文件为 sequences.fa 或 sequences.npz；reference_provenance.json 声明 hg19 代理；manifest 可记录 source_sha256。核对 ID/长度、组互斥、同组同划分、配对 3′端点、短<long、目标与源 SHA。

### 执行与记录

启动时读取 batch52/PARENT_DATA_ACCEPTANCE.txt 与 PARALLEL_PIPELINE.txt。实际拟合需要 `--i-understand-this-is-a-real-fit`、匹配的 `--accepted-data-sha` 和执行槽位。台账上限 4，原计划为 R0/Nabs/Njoint 加一个失败重试位置。vendored gate SHA 为 `e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571`。

原 batch53 测试记录为 test_nofit 21/21、test_pipeline 4/4；其中包含一次小型 ridge solve，后续 batch57 改为纯软件检查并修订续训行为。旧实现的续训台账语义和修复见 batch57。

```bash
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch53_model_core/data_adapter.py <cohort_dir>
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch53_model_core/pipeline_coordinator.py --cohort-dir <cohort_dir> --out-root runs/breakthrough_20261002/batch52/models --ledger runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl --accepted-data-sha <source_sha256> --i-understand-this-is-a-real-fit
```

```text
data_adapter.load_cohort(dir, source_sha256=SOURCE_SHA256, validate_only=False) -> Cohort
train_neural.Trainer(...).run(max_seconds, cap_decider, cap_log_path)
pipeline_coordinator.run_pipeline(...)
eval_replay.replay_predictions(...); evaluate_split(...); paired_group_bootstrap(...); final_compare(...)
checkpoint.save_checkpoint(...); load_checkpoint(path, expect_config_sha, expect_datahash)
exec_gate.open_ledger(ledger_path, budget=4).reserve(...)
```

---

## English

### Frozen model and training protocol

`v6.38.0-model-pipeline-code.rc1` implements batch52 R0, Nabs, and Njoint. `spec.py` is the configuration source and the target is `Log2RPM_HEK293`.

Both neural arms use seed 20261004 and identical initial bytes. ACGT one-hot input is zero-padded to 600 with valid-position masks; a Conv1d(4,64,k9,same)+GELU stem feeds three residual blocks, each with two Conv1d(64,64,k5,same), dilations 1/2/4, GELU, dropout 0.1, and masks after each convolution. Masked mean/max plus standardized loglength/GC feed 130→64→1.

AdamW uses lr=1e-3, wd=1e-4, gradient clipping 1.0, up to six full epochs, deterministic epoch ordering, and effective batch32. Nabs uses inverse-group-size SmoothL1 weights normalized to mean 1. Njoint adds 0.5 times the weighted pair-difference loss with zero pair intercept. Select checkpoints by DEV group-mean pair MAE, taking the earliest epoch on ties.

R0 uses 340 mono-through-tetra features plus GC/loglength, 342 in total, with alpha100 weighted ridge. Fit scaling and the weighted-mean intercept on TRAIN; pair delta is short−long.

### Modules

data_adapter validates TSV/FASTA or NPZ; model_core defines SeqCNN, encoder, param_sha256, and predictor; normalizer fits TRAIN statistics; train_neural trains the arms; r0_baseline implements ridge; checkpoint atomically saves weights/optimizer/RNG/sampler/config/data hash/execution ID; eval_replay provides locked GPU replay up to three minutes and CPU metrics/bootstrap; pipeline_coordinator runs R0→Nabs→Njoint sequentially; epoch_cap selects a throughput-based shared 6/4/2-epoch cap; gpu_lock manages the device; exec_gate manages the ledger; selfcheck/tests check software behavior.

Data contract: endpoints.tsv contains coord_id, union_id, split, length, and Log2RPM_HEK293; groups.tsv contains union_id/split; pairs.tsv contains same-group short/long coord_id; sequences are in sequences.fa or sequences.npz; reference_provenance.json declares the hg19 proxy; manifest may record source_sha256. Validate IDs/lengths, disjoint groups, one split per group, shared pair 3′ endpoints, short<long, target, and source SHA.

### Execution and records

Startup reads batch52/PARENT_DATA_ACCEPTANCE.txt and PARALLEL_PIPELINE.txt. Fits require `--i-understand-this-is-a-real-fit`, matching `--accepted-data-sha`, and an execution reservation. The four-slot ledger originally planned R0/Nabs/Njoint plus one failed-run retry. Vendored gate SHA is `e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571`.

Original batch53 test records show test_nofit 21/21 and test_pipeline 4/4. That suite included a small ridge solve; batch57 subsequently adopted software-only checks and revised resume behavior. See batch57 for the old resume-accounting behavior and repairs.

```bash
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch53_model_core/data_adapter.py <cohort_dir>
.venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch53_model_core/pipeline_coordinator.py --cohort-dir <cohort_dir> --out-root runs/breakthrough_20261002/batch52/models --ledger runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl --accepted-data-sha <source_sha256> --i-understand-this-is-a-real-fit
```

```text
data_adapter.load_cohort(dir, source_sha256=SOURCE_SHA256, validate_only=False) -> Cohort
train_neural.Trainer(...).run(max_seconds, cap_decider, cap_log_path)
pipeline_coordinator.run_pipeline(...)
eval_replay.replay_predictions(...); evaluate_split(...); paired_group_bootstrap(...); final_compare(...)
checkpoint.save_checkpoint(...); load_checkpoint(path, expect_config_sha, expect_datahash)
exec_gate.open_ledger(ledger_path, budget=4).reserve(...)
```
