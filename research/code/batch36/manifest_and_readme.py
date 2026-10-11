"""Batch36: missing-value audit, README, SHA manifest (excl. self/mutable) + twice verification."""
from __future__ import annotations
import csv, hashlib, json, os, glob, datetime

ROOT = "E:/promoter_atlas"
OUTD = os.path.join(ROOT, "data/breakthrough_20261002/batch36/mapt")
REPD = os.path.join(ROOT, "reports/breakthrough_20261002/batch36")
ARMS = ["SAT_HEK293FT", "SAT_NEURON", "KV_SNV_HEK293FT", "KV_INDEL_HEK293FT",
        "KV_SNV_NEURON", "KV_INDEL_NEURON"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# missing-value audit
miss = {}
for arm in ARMS:
    rows = list(csv.DictReader(open(os.path.join(OUTD, "cohort_" + arm + ".tsv"), encoding="utf-8"), delimiter="\t"))
    miss[arm] = {k: sum(1 for r in rows if r[k] in ("", "NA", "nan")) for k in ["logFC", "P_Value", "adjP_Value", "base_ref", "base_alt"]}
json.dump({"generated_at": now(), "note": "base_ref NA = deletion alleles; a few NA estimates retained as reported (never filtered)",
           "missing": miss}, open(os.path.join(REPD, "missing_value_audit.json"), "w", encoding="utf-8"), indent=2)

readme = """# Batch36 - v6.25.0-293ft-label-recovery.rc1

Target chassis = **293FT**. Public sources keep their names (this study measured **HEK293FT** as the
activity host for the HEK arm; neurons = KOLF2.1J h-NGN2). ZERO fits / scoring / GPU in this batch.

## What was recovered
The study's **processed per-variant labels** were recovered from the **official bioRxiv supplement**
`media-1.xlsx` (73.4 MB, sha256 `bc6e05853091d2d1245c73afb5551eabe8fb5ad4bfba2bb61fa8b5136e62090e`),
tables:
* **s15 - MAPT prom HEKs** (HEK293FT saturation mutagenesis) - PRIORITY
* s13 - SNVs HEKs, s11 - InDels HEKs (known-variant MPRA, HEK293FT)
* s14 / s12 / s10 - the matching **neuron** sheets, kept SEPARATE
* s17 - PromoterAI is a **computed score, NOT measured truth** (not used as labels)

Labels are **bcalm logFC = variant vs reference-sequence control** (a regulatory-variant EFFECT, not
absolute activity; with t, P.Value, adj.P.Val, B). **All rows are retained** (negative / zero /
non-significant); significance is a flag, never a filter.

## Join (frozen, auditable)
Joined 1:1 to the frozen 36,000-record author design FASTA (GSM9640449). **All six arms matched 100%**
(0 unmatched). Duplicate design headers (900) carry **identical sequences** -> no many-to-many expansion.
No exact-sequence duplicates and no exact-RC pairs within any frozen arm.

## Sequence model (resolved from the author's own design)
The 249 bp synthesized oligo = **15 bp constant flank + 219 bp genomic variable insert + 15 bp constant
flank**. Determined empirically from the 30 constant columns across the author's 2000 reference oligos;
the ref-vs-alt mismatch sits at design offset **124 = 15 + 109** (centre of the 219 bp insert). Deletion
alleles are 244 bp (a centred 5 bp deletion). Both the **raw 249 bp design** and the **verified 219 bp
variable insert** are stored; no reference fill / truncation.

## Scope limits (unchanged, important)
* The saturation arm is **ONE 2 kb locus** (chr17:45,893,536-45,895,535) -> **no train/val/test split**;
  within-locus variant diagnostic ONLY, never a new-family autonomous promoter.
* The construct is a **fixed lentiMPRA CRE/enhancer reporter** (pLS-SceI); HEK293FT arm = plasmid
  transfection + activity readout; neuron arm = lentiviral transduction. **These are NOT autonomous promoters.**
* The known-variant HEK293FT arm spans ~48 chr17 CRE clusters and CAN support a parent-region group split.
* Study is a **PREPRINT**; labels are bcalm estimates (no per-replicate SE column recovered).
* No model is validated for new 293FT families (unchanged).

## Files
`data/.../batch36/mapt/media-1.xlsx`; `supp/` extracted sheets; `cohort_<ARM>.tsv` frozen cohorts;
`cohort_<ARM>_inserts.fasta` (219 bp inserts). Reports in `reports/.../batch36/`.

## Run the recovery
```bash
PYTHONUTF8=1 ./.venv/Scripts/python.exe -X utf8 code/batch36/extract_supp_tables.py
PYTHONUTF8=1 ./.venv/Scripts/python.exe -X utf8 code/batch36/freeze_cohort.py
PYTHONUTF8=1 ./.venv/Scripts/python.exe -X utf8 code/batch36/finalize_batch36.py
```
"""
open(os.path.join(REPD, "README.md"), "w", encoding="utf-8").write(readme)

# SHA manifest over immutable artifacts (exclude self, verification, status/selfcheck/review)
EXCL = {"sha_manifest.json", "sha_verify_result.json", "status.json", "worker_selfcheck.json",
        "independent_review.json", "missing_value_audit.json"}
targets = []
for pat in ["data/breakthrough_20261002/batch36/mapt/*.tsv", "data/breakthrough_20261002/batch36/mapt/*.fasta",
            "data/breakthrough_20261002/batch36/mapt/media-1.xlsx", "data/breakthrough_20261002/batch36/mapt/supp/*.tsv",
            "code/batch36/*.py", "reports/breakthrough_20261002/batch36/*.json",
            "reports/breakthrough_20261002/batch36/*.tsv", "reports/breakthrough_20261002/batch36/*.md",
            "reports/breakthrough_20261002/batch36/figures/*.png"]:
    targets += glob.glob(os.path.join(ROOT, pat))
files = {}
for t in sorted(set(targets)):
    b = os.path.basename(t)
    if b in EXCL:
        continue
    files[os.path.relpath(t, ROOT).replace("\\", "/")] = {"sha256": sha256(t), "bytes": os.path.getsize(t)}
json.dump({"batch": "batch36", "generated_at": now(), "n_files": len(files), "files": files},
          open(os.path.join(REPD, "sha_manifest.json"), "w", encoding="utf-8"), indent=2)
print("manifest files", len(files))

# read-only verification TWICE
def verify():
    m = json.load(open(os.path.join(REPD, "sha_manifest.json"), encoding="utf-8"))
    bad = []
    for rel, rec in m["files"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or sha256(p) != rec["sha256"]:
            bad.append(rel)
    return {"n": len(m["files"]), "mismatches": bad, "ok": not bad}

v1 = verify(); v2 = verify()
res = {"generated_at": now(), "verification_1": v1, "verification_2": v2,
       "consistent": v1 == v2, "all_match": v1["ok"] and v2["ok"]}
json.dump(res, open(os.path.join(REPD, "sha_verify_result.json"), "w", encoding="utf-8"), indent=2)
print("verify ok", res["all_match"], "consistent", res["consistent"], "n", v1["n"])

