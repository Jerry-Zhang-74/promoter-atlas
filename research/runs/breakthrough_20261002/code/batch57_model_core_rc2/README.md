# batch57_model_core_rc2 -- v6.38.1-model-core-critical-repair.rc2

Repaired MODEL CORE. Drop-in replacement for `code/batch53_model_core` (v6.38.0 rc1) after the
canonical batch53 review returned compliance FAIL (D1..D12). Original batch53 bytes are preserved,
not modified in place. Same module names/APIs and the SAME `config_sha256` (12488fae...), so live
checkpoints written by the running fit remain strictly loadable (ABI verified read-only).

## Repairs
- **D1 gpu_lock**  distinct FILETIME buffers -> stable OS creation time; live holder never reclaimed
  (GetExitCodeProcess liveness); release only own identity. Two-process test proves it.
- **D2 resume**    fail-closed (`ResumeUnsupported`); every execution consumes a slot; checkpoints
  stay readable. Real restore not implemented (protocol-permitted fallback).
- **D3** best fixed-direction baseline = `max()` (was `min()`).
- **D4** `SOURCE_SHA256` = real 64-hex `1bbe5db7...d96`; `SOURCE_SHA1` separate (no mislabel).
- **D5** `assert_parent_specs` REQUIRES + asserts the acceptance-file sha256 (fail-closed).
- **D6** `actual_status()` reads the REAL ledger (no fictitious reserve); RUN_POLICY read dynamically.
- **D7** Njoint asserts shared init bytes vs recorded Nabs `init_sha` before its first gradient.
- **D8** top10 precision uses frozen coord/ID stable tie-breakers.
- **D9** checkpoints record TRUE optimizer steps (sample count kept separately).
- **D10** deterministic per-epoch pair shuffle + repeat counts (Njoint).
- **D11** selected checkpoint + `.sha256.json` sidecar.
- **D12** adapter: ragged-row reject; NPZ byte-id decode + declared-length slice; ambiguous-base
  consistent exclusion (default reject, `allow_ambiguous=True` uses valid-base length).

## Entrypoints (unchanged CLI except D5)
`data_adapter.py`, `train_neural.py`, `r0_baseline.py`, `eval_replay.py`, `pipeline_coordinator.py`,
`checkpoint.py`, `epoch_cap.py`, `gpu_lock.py`, `exec_gate.py`.

## Verification (artificial CPU no-fit only)
`python selfcheck_rc2.py` -> `reports/breakthrough_20261002/batch57/code_selfcheck.json`:
tests/test_nofit_rc2.py 21/21 PASS, tests/test_pipeline_rc2.py 6/6 PASS (incl. read-only ABI load of a
live checkpoint + artificial no-grad forward). ZERO training/solve/backward/optimizer.step/GPU/network.

See `OWNER_FIX_HANDOFF.txt` for what needs a new process vs is eval-only, and `CODE_READY.json`.
