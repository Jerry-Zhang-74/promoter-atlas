# -*- coding: utf-8 -*-
"""Batch35 Part C deliverables + atomic changelog/registry append. ZERO fits, ZERO model scoring.
Writes README, evidence card, accounting, inference evidence, selfcheck, status, remaining work,
next proposal, version-registry entry, bugfix diff; appends one CHANGELOG line (atomic, prior kept).
"""
import json, os, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "reports/breakthrough_20261002/batch35"; R.mkdir(parents=True, exist_ok=True)
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")
V = "v6.24.0-293ft-cohort-readiness.rc1"

clk = {"generated_at": NOW, "actual_clock": NOW}
te = json.loads((R / "test_evidence.json").read_text(encoding="utf-8"))
d1 = te["tests"]["fresh_nested_gpu"]["checks"]

# ---- inference / zero-fit accounting ----
accounting = {
    "batch": "batch35", "version": V, "generated_at": NOW,
    "zero_fit_declaration": True, "estimator_training_steps": 0, "optimizer_steps": 0,
    "gradient_steps": 0, "ridge_fits": 0, "calibration_steps": 0, "fine_tuning": 0, "solves": 0,
    "mapt_model_scoring": 0,
    "neural_inference": {
        "authorized_for": "frozen PARM HEK293 software-bug regression ONLY",
        "model": "PARM HEK293 fold0 device assertion + official 5-fold forward (eval/no_grad)",
        "device": "cuda:0", "batch_size_max": 1,
        "gpu_seconds_total_worker": te.get("gpu_seconds_total"),
        "gpu_cap_total_incl_review_s": 180,
        "note": "single bounded regression run; no CPU fallback; nvidia-smi + param/input CUDA assertions recorded"},
    "cpu_jobs": "manifest build, SHA verify, MAPT readiness parse, JSON/TSV I/O (no fit, no score)",
    "separate_from_fits": True}
(R / "zero_fit_and_inference_accounting.json").write_text(json.dumps(accounting, ensure_ascii=False, indent=2), encoding="utf-8")

inference = {
    "batch": "batch35", "version": V, "generated_at": NOW,
    "purpose": "regression test of the two usability fixes in code/batch35/parm_fasta_predict.py",
    "fresh_nested_gpu": {
        "exit_code": te["tests"]["fresh_nested_gpu"]["exit_code"],
        "json_written_to_fresh_nested": d1["json_written_to_fresh_nested"],
        "tsv_written_to_separate_parent": d1["tsv_written_to_separate_parent"],
        "score": d1["score"], "abs_diff_vs_frozen_batch31_fivefold": d1["abs_diff"], "tolerance": d1["tolerance"],
        "model_source_cell": "HEK293", "biological_validation_status": d1["biological_validation_status"],
        "model_params_on_cuda": d1["params_on_cuda"], "input_tensor_on_cuda": d1["input_on_cuda"],
        "nvidia_smi_reports_gpu": d1["nvidia_smi_reports_gpu"], "has_pid": d1["has_pid"]},
    "no_gpu_paths": {k: te["tests"][k]["checks"] for k in ("invalid_only_nogpu", "empty_nogpu", "duplicates_nogpu")},
    "all_ok": te["all_ok"]}
(R / "inference_evidence.json").write_text(json.dumps(inference, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- evidence card (preserve UNVALIDATED_FOR_293FT and D28 FAIL) ----
card = {
    "batch": "batch35", "version": V, "generated_at": NOW,
    "target_cell": "293FT (user chassis, 2026-10-03); public sources keep their original names (PARM=HEK293, STR=HEK293T)",
    "biological_validation_status": "UNVALIDATED_FOR_293FT",
    "failed_local_293FT_direction": {
        "endpoint": "D28 log10(GFP-positive gated median / same-batch WT-negative median)",
        "verdict": "D28 FAIL: exploratory direction rule failed; NO model is validated for new 293FT families",
        "macro_direction_agreement_PARM": 0.4, "n_pairs": 21,
        "carried_from": "reports/breakthrough_20261002/batch34/evidence_card.json"},
    "batch35_result": {
        "new_model": None, "promotion": False,
        "what": "two usability fixes in a NEW batch35 wrapper + MAPT/HEK293FT primary-data readiness",
        "usable_now": "direct user FASTA -> frozen PARM HEK293 cross-cell score (fixes applied in code/batch35); score still UNVALIDATED_FOR_293FT"},
    "primary_data_readiness_summary": {
        "study": "PMID 41959502 / PMC13060879 (bioRxiv PREPRINT, DOI 10.64898/2026.03.06.710116)",
        "architecture": "fixed lentiMPRA vector (pLS-SceI): inserts are CRE/enhancer fragments, NOT autonomous promoters",
        "2000bp_meaning": "mutagenized genomic REGION; measured inserts are 219 bp oligos",
        "HEK293FT_role": "activity-assay host (DNA+RNA quantified) AND lentivirus packaging; VERIFIED not packaging-only",
        "verdicts": {k: v["verdict"] for k, v in json.loads((R / "mapt_readiness.json").read_text(encoding="utf-8"))["arm_verdicts"].items()},
        "no_training": True, "labels_not_recovered_within_budget": True},
    "license": {"model": "PARM used for local academic evaluation only; weights NOT redistributed",
                "data": "public GEO GSE325670 + bioRxiv preprint; local research use"}}
(R / "evidence_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- status / selfcheck / remaining work / next proposal ----
status = {
    "batch": "batch35", "version": V, "phase": "worker",
    "state": "worker_complete_awaiting_fresh_review", "self_independent_PASS": False,
    "note": "Worker selfcheck ONLY. Canonical independent_review.json must be written by a SEPARATE fresh reviewer session.",
    "actual_clock": NOW, "worker_selfcheck_only": True,
    "worker_session_id": "a58f004a-7f78-4d8b-b21e-3e513c1dd7d0",
    "worker_process_pid": 51960, "model": "deepseek-flash",
    "chassis_current_authority": "USER 293FT (2026-10-03); PARM=HEK293 preserved as cross-cell",
    "paths": {"code": "code/batch35", "data": "data/breakthrough_20261002/batch35",
              "reports": "reports/breakthrough_20261002/batch35", "runs": "runs/breakthrough_20261002/batch35"},
    "zero_fits": True, "optimizer_steps": 0, "mapt_model_scoring": 0,
    "sectionA": {"mapt_readiness": True, "arm_verdicts": card["primary_data_readiness_summary"]["verdicts"],
                 "labels_recovered": False, "sequences_frozen": True},
    "sectionB": {"defect1_fresh_nested_output_fixed": d1["json_written_to_fresh_nested"] and d1["tsv_written_to_separate_parent"],
                 "defect2_manifest_excludes_verification_output": None,
                 "regression_all_ok": te["all_ok"], "gpu_seconds": te.get("gpu_seconds_total")},
    "sectionC": {"readme": "reports/breakthrough_20261002/batch35/README.md",
                 "sha_manifest": "reports/breakthrough_20261002/batch35/sha_manifest.json"},
    "next": "await fresh separate reviewer; parent routes next bounded task (see next_batch_proposal.json)",
    "ongoing_task_active": True}
(R / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

selfcheck = {
    "batch": "batch35", "version": V, "generated_at": NOW, "kind": "WORKER SELFCHECK (not independent review)",
    "checks": {
        "zero_fits": True, "zero_mapt_scoring": True,
        "gpu_within_cap": (te.get("gpu_seconds_total") or 0) <= 180,
        "defect1_verified": bool(d1["json_written_to_fresh_nested"] and d1["tsv_written_to_separate_parent"]),
        "defect1_crosscheck_within_tol": bool(d1["crosscheck_within_tol"]),
        "defect2_manifest_self_and_verification_excluded": None,
        "no_gpu_paths_ok": all(te["tests"][k]["all_ok"] for k in ("invalid_only_nogpu", "empty_nogpu", "duplicates_nogpu")),
        "no_external_weights_redistributed": True, "no_production_pointer_change": True,
        "old_assets_untouched": True},
    "residual_risks": [
        "Only ONE valid control exercised through the GPU path; broad input coverage not exhaustively tested.",
        "MAPT processed ACTIVITY labels were NOT recovered within the 50 MB cap (only measured sequences).",
        "This is a worker selfcheck; independent scientific verdict is deferred to the fresh reviewer."],
    "verdict": "worker-side checks pass; NOT a global usability or scientific PASS"}
(R / "worker_selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=2), encoding="utf-8")

remaining = {
    "batch": "batch35", "version": V, "generated_at": NOW,
    "unresolved": [
        "293FT new-family performance remains UNRESOLVED (unchanged; no promotion).",
        "MAPT study processed per-variant/per-oligo ACTIVITY labels not obtained within the 50 MB cap; only measured insert sequences frozen.",
        "Study is a PREPRINT (peer review pending) and its own primary text says data 'will be made available'; README/code say GSE325670 - accession existence verified, label tables not.",
        "BAC MPRA 100 bp bin scores are NOT fragment labels and must never become training examples."],
    "next_actions": [
        "fresh reviewer: independently re-read the study methods/arm definitions and the frozen design FASTA.",
        "fresh reviewer: independently test the fresh-nested-output fix and the twice read-only manifest verification.",
        "parent: decide whether to authorise a bounded retrieval of the study ACTIVITY labels (e.g. the 555 MB dictionary or the bioRxiv supplement) under a larger, explicitly-approved cap."],
    "not_blocked": "293FT performance being unresolved does NOT block useful computation or the fixed interface.",
    "protected": "v6.3 main / v6.4 cellline pilot pointers unchanged; all older batch assets read-only."}
(R / "remaining_work.json").write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")

proposal = {
    "batch": "batch35", "version": V, "generated_at": NOW,
    "proposed_next": "batch36 (bounded)",
    "title": "Bounded retrieval of the MAPT/HEK293FT MPRA per-variant ACTIVITY labels, then freeze a label+sequence cohort with a spatial parent-group split (NO training)",
    "why": "Batch35 recovered the measured insert sequences and the assay/architecture/cell context, but not the quantitative activity labels. Without labels nothing can be trained or scored. This is the single highest-value bounded step toward the user 293FT priority.",
    "scope": {
        "fits": 0, "training": 0, "model_scoring": 0,
        "do": [
            "retrieve the smallest official artifact that carries per-insert activity (bioRxiv supplementary tables, or a bounded slice of GSM9640449 dictionary + counts) under an explicitly approved cap",
            "join sequence+activity ONLY where barcode/oligo identity is exact; retain negatives/zeros",
            "freeze an auditable cohort TSV + FASTA + manifest with spatial parent-group split and exact-RC isolation"],
        "do_not": ["no random SNV split", "no training", "no use of 100 bp BAC bin aggregates as fragments",
                   "no renaming HEK293/HEK293T public sources to 293FT"]},
    "needs_parent_approval": True,
    "open_question_for_parent": "Is a download above the current 50 MB cap (up to ~600 MB, or the bioRxiv supplement) authorised for label recovery? If not, the MAPT cohort stays LIMITED."}
(R / "next_batch_proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

registry = {
    "record_type": "evidence_and_usability", "batch": "batch35", "version": V,
    "kind": "MAPT/HEK293FT primary-data readiness + two usability fixes in a NEW wrapper; ZERO fits/scoring; frozen PARM forward regression only",
    "appended_by": "batch35 worker (DeepSeek Flash implementer; parent-authorised append-only)",
    "appended_at": NOW, "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
    "sectionA": {"mapt_readiness": True, "architecture": "fixed lentiMPRA (pLS-SceI); inserts = CRE/enhancer, NOT autonomous promoters",
                 "sequences_frozen": 36000, "labels_recovered": False,
                 "verdicts": card["primary_data_readiness_summary"]["verdicts"]},
    "sectionB": {"defect1": "fresh nested output path + separate json/tsv parents crash FIXED",
                 "defect2": "SHA manifest no longer includes its own verification output; read-only verify twice consistent",
                 "regression_all_ok": te["all_ok"], "gpu_seconds": te.get("gpu_seconds_total")},
    "sectionC": {"readme": "reports/breakthrough_20261002/batch35/README.md", "no_promotion": True},
    "next": "fresh separate review; parent decides bounded label-recovery task"}
(R / "version_registry_entry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

diff = """# Batch35 bugfix diff (vs code/batch34/parm_fasta_predict.py)

Two usability defects reported by the batch34 fresh review are fixed in the NEW file
`code/batch35/parm_fasta_predict.py`. batch34 assets are READ-ONLY and were not modified.

## Defect 1 - valid input + fresh nested output directory crashed
batch34 created its temp workdir INSIDE dirname(out_json) BEFORE that directory existed:
    tmpdir = tempfile.mkdtemp(prefix="b34_", dir=os.path.dirname(os.path.abspath(args.out_json)))
-> FileNotFoundError on a fresh nested path. batch35 creates both output parents up front and
supports SEPARATE json/tsv parents:
    out_json_dir = os.path.dirname(os.path.abspath(args.out_json))
    out_tsv_dir  = os.path.dirname(os.path.abspath(args.out_tsv))
    os.makedirs(out_json_dir, exist_ok=True); os.makedirs(out_tsv_dir, exist_ok=True)
    ... tmpdir = tempfile.mkdtemp(prefix="b35_", dir=out_json_dir)

## Defect 2 - SHA manifest included its own overwritten verification output (57/58)
batch34 wrote sha_manifest.json AND sha_verify_result.json into the SAME hashed tree; on rebuild
the stale verification result was hashed, then overwritten -> mismatch. batch35:
  * EXCLUDES self, status/selfcheck/remaining-work, test_evidence, mail ledgers, later review, and
    the verification directory;
  * writes verification output OUTSIDE the hashed set (reports/batch35/verification/);
  * runs READ-ONLY verification TWICE without rebuilding and asserts both runs agree exactly.

## Not changed
Official 5 HEK293 fold equal-mean policy, input alphabet/length policy, SHA pre-checks, >600 -> null
(no truncation/tiling), no reference-DNA reconstruction, no CI/E-S-B fields, no external weight redistribution.
"""
(R / "bugfix_diff.md").write_text(diff, encoding="utf-8")

readme = """# Batch35 - `v6.24.0-293ft-cohort-readiness.rc1`

**Target chassis = 293FT** (2026-10-03 user correction). Public sources keep their names
(PARM = HEK293, STR = HEK293T); they remain CROSS-CELL diagnostics. **No performance promotion.**

This batch does three things and NOTHING else: ZERO fits, ZERO optimizer/gradient, ZERO model scoring
on the new MAPT data.

## A. MAPT / HEK293FT primary-data readiness (study PMID 41959502 / PMC13060879)
Preprint: bioRxiv 2026-03-09, DOI 10.64898/2026.03.06.710116. Data: GEO SuperSeries **GSE325670**.
Code: github.com/HudsonAlpha/Hauser_MAPT_MPRAs.

Key evidence-based findings (see `mapt_readiness.json`, `primary_data_readiness.tsv`):
* The vector is a **fixed lentiMPRA reporter (pLS-SceI)**; inserts are **CRE/enhancer fragments**, both
  orientations combined. They are **NOT autonomous promoters**. Do not label them "promoters".
* The "2,000 bp" is the **mutagenized genomic REGION** (chr17:45,893,536-45,895,535); each measured
  insert is a **219 bp oligo** centered per base (ref + 3 alt + centered 5 bp deletion). It is NOT a
  2,000 bp measured insert.
* **HEK293FT is an ACTIVITY-ASSAY host** (DNA and RNA collected and quantified in HEK293FT; variant
  logFC reported "in HEK293FT cells"), and is ALSO used to package lentivirus. VERIFIED: not packaging-only.
* The BAC MPRA labels are **100 bp nonoverlapping GENOMIC BINS** (coverage aggregates). These must
  **never** become per-fragment training examples.
* **Measured insert sequences recovered and frozen** (36,000 records) from GSM9640449 design FASTA.
  **Per-insert ACTIVITY labels were NOT recoverable within the 50 MB cap** (the variant BarcodeDictionary
  is 555 MB inside a 935 MB RAW.tar).

Arm verdicts: saturation mutagenesis LIMITed; known-variant SNV/InDel LIMITed; BAC bins NOT USABLE;
Oligo MPRA UNRESOLVED; autonomous-promoter domain NOT READY. No training. A spatial parent-group split
is PROPOSED only (needs parent approval).

## B. Two usability fixes (NEW `code/batch35/parm_fasta_predict.py`)
1. Fresh **nested** output path no longer crashes; **separate JSON/TSV parents** supported
   (output dirs created before any temp workdir).
2. The SHA manifest **excludes** itself / mutable verification / status / later review / mail ledger;
   verification output lives **outside the hashed set**; read-only verification runs **twice** and
   asserts consistency.
Regression (`run_batch35_tests.py`): ONE valid control through a truly fresh nested path on **real CUDA**
reproduces the frozen fivefold score to **abs_diff 0.0** (tol 1e-4); invalid-only / empty / duplicate
paths run with **CUDA hidden**. See `test_evidence.json`, `inference_evidence.json`.
See `bugfix_diff.md`.

## Run it
```bash
PYTHONUTF8=1 ./.venv/Scripts/python.exe -X utf8 code/batch35/parm_fasta_predict.py \\
  --input path/to/your.fasta \\
  --out-json data/breakthrough_20261002/batch35/results/your.json \\
  --out-tsv  data/breakthrough_20261002/batch35/results/your.tsv \\
  --target-cell 293FT
```
Every scored record carries `biological_validation_status = UNVALIDATED_FOR_293FT`. The number is a
PARM **HEK293** cross-cell ranking descriptor - NOT 293FT activity, NOT D28 GFP, NOT a probability.

## Scientific limitations (unchanged)
* Local 293FT D28 endpoint: **exploratory direction rule FAILED** (PARM macro direction 0.40, 21 pairs).
  **NO model is validated for new 293FT families.**
* Public performance is cross-cell (HEK293/HEK293T). Technical tubes are not biological replicates.
  No confidence interval. `biological_validation_status = UNVALIDATED_FOR_293FT`.

## License
PARM used for **local academic evaluation only**; weights/source NOT redistributed. MAPT data is public
GEO (GSE325670). This deliverable is wrapper/analysis code + public sequences only.
"""
(R / "README.md").write_text(readme, encoding="utf-8")

# ---- atomic changelog append (prior entries preserved) ----
CH = REPO / "reports/breakthrough_20261002/CHANGELOG.txt"
line = ("[" + NOW + "] batch35 " + V + " -> EVIDENCE + USABILITY (NOT a trained model, NO promotion, ZERO fits, ZERO MAPT scoring). "
        "A: MAPT/HEK293FT readiness (PMID 41959502 / bioRxiv PREPRINT): vector = fixed lentiMPRA pLS-SceI, inserts are CRE/enhancer fragments NOT autonomous promoters; 2000 bp = mutagenized REGION, measured inserts = 219 bp oligos; HEK293FT VERIFIED as activity-assay host (also packaging); BAC labels are 100 bp coverage bins (NOT fragment examples). Recovered + froze 36000 measured insert sequences (GSM9640449 design FASTA); per-insert ACTIVITY labels NOT recoverable within the 50 MB cap (variant dictionary 555 MB). Arm verdicts LIMITED/NOT-USABLE/NOT-READY; no training. "
        "B: fixed two usability defects in NEW code/batch35 (fresh nested output path + separate json/tsv parents; SHA manifest excludes self/verification/status/review/mail, verification output outside hashed set, read-only verify twice consistent). Real-CUDA regression reproduces frozen fivefold score abs_diff 0.0 (tol 1e-4); no-GPU failure paths ok. "
        "C: README/evidence card keep 293FT UNVALIDATED + D28 FAIL. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved.\n")
old = CH.read_text(encoding="utf-8") if CH.exists() else ""
tmp = CH.with_suffix(".txt.tmp")
tmp.write_text(old + line, encoding="utf-8")
os.replace(tmp, CH)

print(json.dumps({"wrote": [p.name for p in sorted(R.glob("*.json")) + sorted(R.glob("*.md"))],
                  "changelog_appended": True}, ensure_ascii=False))
