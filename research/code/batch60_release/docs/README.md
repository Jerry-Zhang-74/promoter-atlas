# Promoter-activity predictor bundle — {ARM_DESC} — {VERSION}

Self-contained **flat-layout** release of the {ARM_ROLE} project model (batch52/batch53, arm
`{ARM}`). Target quantity: the author-documented `Log2RPM_HEK293` (a public HEK293 MPRA
activity value) — **not** `Log2Norm_HEK293`.

This bundle is extracted as-is: `predict.py` sits at the archive **root** and finds its pinned
core in the sibling `pinned_core/`. There is no current-working-directory dependency and no
absolute workspace path — paths may contain spaces or non-ASCII characters.

## Contents (flat layout, root-relative)
| path | role |
| --- | --- |
| `predict.py` | standalone, cwd-independent predictor (strict FASTA validation) |
| `pinned_core/` | frozen model core modules + `PINNED_CORE_SHA256.json` (SHA-pinned) |
| `MODEL_MANIFEST.json` | bundle manifest (target, checkpoint path + SHA, evidence scope) |
| `model/` | selected trained checkpoint (`{CKPT}`) + its `.sha256.json` sidecar |
| `config/config.json` | training configuration (SHA-pinned) |
| `normalizer/normalizer.json` | frozen target normalizer |
| `results/` | real frozen held-out TEST metrics + independent reviews (copied verbatim) |
| `USER_GUIDE.md`, `USER_GUIDE_zh.md`, `MODEL_CARD_zh.md` | user guides and model card |
| `LICENSE_NOTES.md` | license / provenance / scope notes |
| `requirements.txt` | runtime pins (Python 3.12, torch 2.11, numpy 2.5) |
| `SHA256SUMS.txt` | SHA256 of every other file in this bundle, root-relative (flat paths) |

## Run (any cwd; paths may contain spaces or non-ASCII characters)
```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
python -X utf8 "<extracted-bundle>\predict.py" --bundle "<extracted-bundle>" --fasta "<input>.fa" --out "<out>.tsv"
```
`python` is any Python 3.12 interpreter with the pinned requirements installed; replace it with
the full path to your interpreter if needed. Real (`kind=TRAINED`) inference **requires a CUDA
GPU** — there is no CPU fallback.

## Invariants
- Path-independent: the predictor resolves `pinned_core/` from its own location.
- Device policy: real inference requires CUDA; no CPU fallback.
- Strict FASTA: empty file/record, duplicate id, invalid base, all-ambiguous ("null") record,
  and any record longer than 600 bp are rejected explicitly (exit code 2; no silent
  truncation or windowing).
- No keys/credentials, no genome/raw measurement tables, no external pretrained weights.

## Scientific scope (no over-claim)
Public HEK293 retrospective evaluation on a frozen-group TEST split; the encoded sequences are
a **GRCh37/hg19 reference proxy**, not the donor clone. **Local 293FT is UNVALIDATED.**
Supported length is 100–600 bp (empirical training domain); >600 bp is UNSUPPORTED. The
Njoint-vs-Nabs pair comparison FAIL is preserved (≈0.12% improvement, below the predeclared
10%; bootstrap CI includes 0).
