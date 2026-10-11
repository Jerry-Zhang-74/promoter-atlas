# batch53_model_core -- model implementation (v6.38.0-model-pipeline-code.rc1)

Substantive MODEL IMPLEMENTATION for batch52 (parent owns scientific protocol). Implements
BOTH fixed fresh neural arms (Nabs absolute, Njoint pair-augmented) and the R0 composition
ridge, plus the training/evaluation/pipeline entrypoints. NO new model choices or sweeps.

## Frozen spec (single source: spec.py)
- Target column **Log2RPM_HEK293** (PARENT_DATA_ACCEPTANCE); Log2Norm_HEK293 is forbidden.
- Architecture (identical bytes for both arms, seed 20261004): one-hot ACGT, zero pad to 600,
  valid-position mask; stem Conv1d(4,64,k9,same)+GELU; 3 residual blocks (two Conv1d(64,64,k5,
  same), dilations 1/2/4, GELU, dropout 0.1), masked after every conv; masked mean AND masked
  max pooled + standardized loglength/GC = 130 -> linear64 GELU dropout0.1 -> scalar.
- Optimizer AdamW lr1e-3 wd1e-4; grad clip 1.0; 6 full TRAIN epochs max; deterministic
  per-epoch shuffle; effective batch 32 (micro-batch divides 32, optional accumulation).
- Nabs loss = group-weighted SmoothL1 (weights inverse group size, mean 1).
  Njoint = same + 0.5 * group-weighted SmoothL1 pair difference (zero pair intercept).
- DEV checkpoint rule (both arms): best group-mean pair MAE, tie earliest epoch.
- R0: 342 features (mono..tetra 340 + GC + loglength), weighted ridge alpha 100, TRAIN-only
  standardization, weighted TRAIN mean intercept, ONE solve; pair delta = short - long.

## Entrypoints
- `data_adapter.py`    portable validated cohort loader (TSV/FASTA or NPZ); hard checks.
- `train_neural.py`    train Nabs/Njoint (CLI `main()`).
- `r0_baseline.py`     R0 features + weighted ridge.
- `checkpoint.py`      atomic checkpoints (optimizer/RNG/sampler/config/datahash/gate id).
- `eval_replay.py`     independent replay (GPU <=3 min, lock) + CPU metrics/bootstrap/compare.
- `pipeline_coordinator.py`  R0 -> Nabs -> Njoint on ONE GPU, gated, target-asserted.
- `epoch_cap.py`       one-time throughput-only 6/4/2 shared cap decision.
- `gpu_lock.py`        GPU-exclusive lock (PID+creation), bounded handoff.
- `exec_gate.py`       wrapper over the reviewed gate (vendored byte-identical, sha pinned).
- `tests/` artificial CPU no-fit software tests; `selfcheck.py` runs them.

## Guarantees
- Importing any module has NO side effects. The helper ran NO training/solve/gradient/GPU.
- Real fits require `--i-understand-this-is-a-real-fit` + accepted accepted-data SHA + a
  reserved exec_gate slot in the batch52 ledger (MAX 4). Tests use artificial data only.
