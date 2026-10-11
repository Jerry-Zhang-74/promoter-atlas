# batch54_review -- independent first-model acceptance evaluator

batch54 (v6.38.0-first-model-independent-acceptance.rc1) is the INDEPENDENT evaluator,
running in parallel with control62 (batch52 data/training) and control61 (batch53 model code).
Scope: writes ONLY under `runs/breakthrough_20261002/code/batch54_review` and
`reports/breakthrough_20261002/batch54`. No architecture/model-code edits. Zero fits, solves,
backward passes, optimizer steps, downloads. CPU share = 1 core, RAM <= 2 GB.

## What it does
- `indep_metrics.py` -- OWN implementations of: pooled Spearman, group-equal MAE, same-unit
  RMSE, top-k precision with fixed stable-ID tie-break, pair delta group MAE, macro strict
  sign accuracy (true-zero separate; predicted ties incorrect), and a paired group bootstrap
  that RETAINS SAMPLING MULTIPLICITY (>=30 groups). No producer evaluator is imported for
  metric values.
- `indep_cohort.py` -- OWN read-only cohort TSV parser (targets/group/split/pairs).
- `test_fixtures.py` -- artificial no-fit fixture tests (duplicate bootstrap groups, constant
  prediction, NA, true-zero/ties, fixed-ID top-k, brute-force agreement). Writes
  `fixture_evidence.json`.
- `model_checks.py` -- CPU read-only artifact checks: ready marker, selected checkpoint SHA
  sidecar, frozen config SHA, cohort datahash, exact target column, normalizer, per-epoch
  full-TRAIN coverage, DEV checkpoint-selection reproduction, shared init bytes, exec_gate
  ledger accounting, runnable load check (synthetic forward only).
- `run_acceptance.py` -- CLI: `fixtures | r0 | metrics | checks | accept`.
- `assemble_acceptance.py` -- writes `reports/.../batch54/acceptance.json` + `report.md`.
- `wait_for_modelA.py` -- bounded local (zero-LLM) poller for `Nabs_ready.json`, <=30 s poll
  until 12:20 Asia/Shanghai; writes `modelA_wait_event.json`.

## Explicit non-goals / limits
- Does NOT import the training evaluator as sole metric proof.
- Does NOT do a CPU neural fallback; independent GPU replay of saved weights is deferred to a
  granted exclusive short slot (<=60 s), acquiring the GPU lock with PID+creation identity.
- TEST targets are physically present in `endpoints.tsv`; TEST separation is procedural.
- Reference-encoded sequence is a proxy for donor HG02601 (HEK293 identity preserved);
  no actual donor/clone or local 293FT claim.

## Evidence files (this dir)
`fixture_evidence.json`, `provenance_evidence.json`, `modelA_wait.log`,
`modelA_wait_event.json` (on completion), `code_selfcheck.json` (optional), `README.md`.
