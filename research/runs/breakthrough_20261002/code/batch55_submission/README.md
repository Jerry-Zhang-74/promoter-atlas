# batch55 submission integration (v6.38.0-first-model-submission-integration.rc1)

Scope: package and predict with the NEW first model (batch52/batch53, target
`Log2RPM_HEK293`). This directory contains the *staging* code and a *standalone predictor*;
the actual trained model bundle is produced by `stage_submission.py` only when real weights
are ready. Code readiness is NOT model completion.

## Contents
| file | role |
| --- | --- |
| `predict.py` | standalone, cwd-independent predictor (strict FASTA validation) |
| `stage_submission.py` | stager: packages the model with HARD refusals + SHA manifest |
| `make_fixture_model.py` | builds an artificial SOFTWAREONLY no-fit fixture (tests only) |
| `pinned_core/` | frozen copy of the model core + `PINNED_CORE_SHA256.json` |
| `tests/test_batch55.py` | self-check: strict input, secret exclusion, artifact SHA, path independence, refusals |
| `USER_GUIDE.md` | end-user instructions with concrete PowerShell commands |

## Invariants
- Path-independent: the predictor resolves its pinned core from its own location.
- Device policy: real (kind=TRAINED) inference requires CUDA; no CPU fallback. Only a
  SOFTWAREONLY fixture may run on CPU and is loudly labelled as such.
- Strict FASTA: empty file/record, duplicate id, invalid base, all-ambiguous ("null") record,
  and length > 600 are rejected.
- The stager refuses missing weights, target/label mismatch, untrained checkpoint, source-SHA
  mismatch and placeholder results, and emits an actionable partial status instead of a
  completed claim.
- Never copies keys/credentials, genomes/raw data, old failed wrong-target R0, or proprietary
  external weights.

## Quick self-check (CPU, no fit, no GPU, no network)
```powershell
$env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
E:\promoter_atlas\.venv\Scripts\python.exe -X utf8 runs\breakthrough_20261002\code\batch55_submission\tests\test_batch55.py
```

