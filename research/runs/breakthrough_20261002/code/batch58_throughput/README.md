# batch58 -- v6.39.0-throughput-checkpoint.rc1 (throughput resource variant + REAL resume)

RESOURCE CONFIGURATION variant. NOT evidence of better accuracy; NOT exact replication when the
effective batch changes. Parent explicitly authorized this variant and the b58-only cap-3 gate.

## What it is
* A fresh **Nabs** (activity-only) training of the reviewed frozen batch57 model, trained on the
  SAME corrected cohort (531860 TRAIN rows / all frozen groups / target `Log2RPM_HEK293`),
  SAME architecture / fp32 / lr / loss / seed / per-epoch deterministic order -- changing ONLY
  the **effective optimizer batch** (fewer optimizer steps per epoch, recorded explicitly).
* A **REAL checkpoint resume** (fresh `batch58.checkpoint.v2` schema) replacing the batch57
  fail-closed stub. Weights + AdamW state + python/numpy/torch-CPU/CUDA RNG + normalizer +
  pooled-feature(loglen/GC) state + epoch + exact `next_pos` + order seed + pair pointer +
  history + best + global optimizer steps + config/data hashes + device are persisted.

## Files
| file | role |
|---|---|
| `b58_gate.py` | NEW separate ledger, cap **3**, every reservation CONSUMING; re-verifies the vendored gate bytes `e1ff80a6...` |
| `_exec_gate_vendored.py` | byte-identical reviewed executor gate (sha `e1ff80a6...`) |
| `b58_core.py` | resume-capable trainer (schema v2, 2-generation retention, stopfile/SIG, OOM pre-step restore) |
| `b58_probe.py` | forward+backward (NO optimizer step) throughput/memory profiler, batches 32..2048 |
| `b58_run.py` | bounded GPU waiter + driver (profile -> train -> resume) + status artifacts |
| `selfcheck_b58.py` | artificial CPU selfcheck (gate/resume/retention/coverage/selector units) |
| `smoke_b58.py` | artificial CPU end-to-end stop+resume integration test |
| `dryrun_b58.py` | CPU dry-run over a small REAL cohort slice (validates the exact call path, no gate) |
| `freeze_code_ready.py` | re-freezes CODE_READY.json with current bytes + per-file sha256 |
| `finalize_b58.py` | honest memory-target assessment + TRAINING_DONE + worker status |
| `b58_manifest.py` | immutable manifest (excludes mutable progress/review/mail) |

## Executions (b58-only allowance, cap-3) -- all three CONSUMED
1. `b58-profile` -- gradient profile only, no optimizer step.
2. `b58-Nabs-throughput-variant` -- fresh training, controlled graceful stop inside epoch 0.
3. `b58-resume-continuation` -- NEW process resumes the saved generation and finishes 2 epochs.

The original batch52 ledger (MAX 4) is untouched; the frozen batch32 comparison is unchanged.

## Resume semantics (falsifiable)
* `next_pos` = position AFTER the last processed batch. An old-style mid checkpoint (b57 schema,
  whose `pos` points AT the processed batch) is REJECTED, never silently replayed.
* On resume the loader recomputes the deterministic order, checks the saved next/last batch id
  lists AND a running coverage-prefix hash (skip/repeat detection) before any gradient.
* Keep the last TWO valid immutable generations; sha256 + sidecar are written BEFORE `latest.json`
  is atomically updated.
* Stop only after a fully saved optimizer boundary (SIGINT/SIGTERM/stopfile).
* CUDA OOM -> restore whole pre-step state + RNG, retry with smaller micro-batch; retries recorded.
  No CPU neural fallback, no fake GPU load.

## GPU-use precondition
GPU is touched ONLY after the original healthy run's actual PID+OS-creation is gone, BOTH
`Nabs_ready.json` and `Njoint_ready.json` exist, `batch52/GPU_EVALUATION_DONE.json` (first-model
evaluation handoff) exists, and no competing python compute process remains. The shared repaired
b57 GPU lock is taken in `runs/breakthrough_20261002/batch52`.

## PARENT pre-GPU patch (2026-10-04 10:12) -- KEPT, never reverted
`PARENT_PRE_GPU_CORRECTION.json` corrected the batch selector to **samples/second**
(`batch/median_s`), made unmeasured memory fail-closed, and added the GPU handoff prerequisite +
shared lock. Verified by `selfcheck_b58.check_selector_throughput_units`
(batch32@0.010 s vs batch128@0.020 s -> 128 wins despite worse latency). `CODE_READY.json` was
re-frozen afterwards with the current bytes.

## Honest resource outcome (see reports/.../batch58/throughput_analysis.json)
The requested ~6.7-7.1 GiB device target was **NOT reached**: the corrected selector picks the
fastest measured memory-safe batch (64 -> ~2.8 GiB peak). Only batch 2048 approached the target
(7.61 GiB used) but left 0.015 GiB free (< 0.7 GiB floor) and was ~22x slower per step. Usage was
not padded. Resource result only -- no accuracy/perf-superiority claim.
