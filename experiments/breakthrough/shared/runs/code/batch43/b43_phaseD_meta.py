# -*- coding: utf-8 -*-
"""batch43 Phase D: strict JSON, correction note, README, SHA manifest, selfcheck,
status, remaining_work, one bounded next proposal, atomic registry/changelog appends.
Worker writes ONLY worker_selfcheck (never independent_review).
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import b43_common as C

T0 = time.time()
R, D, RUN, CODE = C.B43_REPORT, C.B43_DATA, C.B43_RUN, C.B43_CODE

spec = C.jload(R / "diagnostic_spec.json")
ed = C.jload(R / "error_decomposition.json")
rk = C.jload(R / "rank_diagnostics.json")
dec = C.jload(R / "decision.json")


def strict_load(p):
    with open(p, encoding="utf-8") as fh:
        return json.loads(fh.read(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite %s" % x)))


# ---- correction note --------------------------------------------------------
correction = {
    "record_type": "correction_note",
    "batch": C.BATCH, "version": C.VERSION,
    "authored_at": C.now_local(),
    "targets": "batch42 v6.29.0-hekft-supervised-pilot.rc1 reporting",
    "invalid_cross_unit_MAEs": spec["correction"]["invalid_cross_unit_MAEs"],
    "invalid_reason": spec["correction"]["invalid_reason"],
    "I15_rejected": spec["correction"]["I15_rejected"],
    "compliance_blanket_rejected": True,
    "still_valid": "learned same-unit R0/R1/Rnull raw-alpha MAE/R2 and all rank metrics; cached PARM/H1 rho and top10 are rank-only and remain valid",
    "old_assets_untouched": True,
    "canonical_review_and_results_not_edited": True,
    "real_clock_provenance": spec["correction"]["real_clock_provenance"],
    "note": "no correction email; existing watcher untouched; this note lives only in new batch43 outputs",
}
C.atomic_write_json(R / "correction_note.json", correction)

note_md = (
    "# batch43 correction note (new file; old bytes preserved)\n\n"
    "Date (real clock): %s\n\n"
    "## 1. Two cached PARM cross-unit MAEs are INVALID\n"
    "- batch42 `metrics.json` `strata.chr17.cached_parm_mae` = %.10f\n"
    "- batch42 `metrics.json` `strata.non_chr17.cached_parm_mae` = %.10f\n"
    "- PARM raw score and MPRAnalyze `getAlpha` are DIFFERENT output units; an MAE between them is\n"
    "  meaningless (cross-unit). These two numbers are marked INVALID and must not be interpreted/used.\n"
    "- The related cached_parm_rho values (rank-only) remain valid.\n"
    "- The old canonical review and old results were NOT edited.\n\n"
    "## 2. Review I15 and the blanket compliance PASS are rejected\n"
    "- batch42 review I15_no_subset_or_units = PASS claimed \"same-unit MAE only learned models\",\n"
    "  but `metrics.json` strata actually contains `cached_parm_mae` (cross-unit). I15 is FALSE and the\n"
    "  blanket `compliance=PASS` is rejected. The numerical review and the learned same-unit metrics\n"
    "  remain separately valid.\n\n"
    "## 3. Real clock / provenance\n"
    "- batch42 worker finished: %s\n"
    "- batch42 reviewer process creation: %s\n"
    "- batch42 authored_at on disk: %s\n"
    "- Parent acceptance cites authored_at 03:03:30 later than controller end 03:03:04; the on-disk\n"
    "  generation clock is not a reliable timestamp. No replacement timestamp is invented; this batch43\n"
    "  note carries its own real clock: %s\n"
    % (correction["authored_at"],
       correction["invalid_cross_unit_MAEs"]["strata_chr17_cached_parm_mae"],
       correction["invalid_cross_unit_MAEs"]["strata_non_chr17_cached_parm_mae"],
       correction["real_clock_provenance"]["batch42_worker_finished_at"],
       correction["real_clock_provenance"]["batch42_reviewer_creation_time"],
       correction["real_clock_provenance"]["batch42_review_authored_at_on_disk"],
       correction["real_clock_provenance"]["batch43_correction_note_time"]))
(R / "correction_note.md").write_text(note_md, encoding="utf-8")

# ---- README -----------------------------------------------------------------
tw = {m: ed["TEST"][m]["row_weighted"]["within_frac"] for m in ("R0", "R1", "Rnull", "const")}
readme = (
    "# batch43 v6.30.0-group-error-diagnostic.rc1\n\n"
    "FINITE posthoc diagnostic of the ALREADY FROZEN batch42 group-heldout pilot.\n"
    "NOT a new predictive model; NOT new validation; ZERO fits/GPU/neural forward/web.\n\n"
    "## Fixed question (pre-declared)\n"
    "In the frozen batch42 group-heldout pilot, are errors predominantly GROUP-LEVEL BIAS or\n"
    "poor WITHIN-GROUP ranking, and what distribution limitations prevent inference about\n"
    "novel 293FT truncations?\n\n"
    "## Method (bounded)\n"
    "- Reuse STORED batch42 TEST predictions for R0/R1/Rnull/train-mean-const + cached b41 PARM/H1/GC.\n"
    "- TRAIN predictions were not persisted -> REPLAYED ONCE from the SAVED R0/R1/Rnull coefficients and\n"
    "  frozen feature/normalizer contracts (additive projection; no fitting, no neural scoring).\n"
    "- Fixed 20-TEST-row replay check vs stored predictions, then full 318 rows.\n"
    "- MSE decomposition per group: e_i=p_i-y_i; b_g, MSE_g, within_g; identity MSE_g=b_g^2+within_g\n"
    "  (max residual ~1e-17). Row-weighted and group-equal averages reported with exact denominators.\n"
    "- Rank diagnostics: group-mean-prediction vs group-mean-alpha Spearman; within-group rho for every\n"
    "  group with >=10 rows and nonconstant vectors; PARM/H1/GC are RANK comparisons only.\n\n"
    "## Headline diagnostic (descriptive; the EXPOSED test, NOT new validation)\n"
    "- Original negative result unchanged: R1 pooled rho 0.0911 vs R0 0.1584; group-balanced MAE\n"
    "  R1 0.1752 vs R0 0.1069 (from batch42 metrics.json, preserved byte-for-byte).\n"
    "- TEST MSE is dominated by the WITHIN-group term: row-weighted within fraction R0 %.3f, R1 %.3f,\n"
    "  Rnull %.3f, const-baseline %.3f. Group-level bias share is the complement (~12-29%%).\n"
    "- Group-mean-level Spearman is near zero / negative (R0 0.086, R1 -0.179, Rnull -0.244; GC 0.216\n"
    "  rank-only); within-group macro rho is also near zero. TEST has 21 groups (<30) -> no CI.\n"
    "- Group means use TEST labels: diagnostic ORACLE quantities, NOT a deployable correction.\n\n"
    "## Correction (see correction_note.md)\n"
    "- Two cached PARM cross-unit MAEs (chr17 0.46639, non_chr17 0.35361) marked INVALID.\n"
    "- Review I15 and the blanket compliance PASS rejected. Old canonical review/results NOT edited.\n\n"
    "## Faithful existing raw-FASTA interface (NOT rewritten here)\n"
    "- Path: `code/batch34/parm_fasta_predict.py` (frozen batch34 / v6.23.0-scoped-inference.rc1).\n"
    "- Output definition: PARM HEK293 promoter-activity model score (equal mean of 5 official HEK293\n"
    "  folds), exploratory CROSS-CELL descriptor; NOT 293FT activity, NOT a probability.\n"
    "- Length/context limits: official central zero-pad L_max=600; supported 1..600; >600 -> null\n"
    "  UNSUPPORTED (no truncation/tiling/windowing); alphabet ACGT valid, N allowed+flagged, others\n"
    "  INVALID; target_requested_cell default 293FT; biological_validation_status=UNVALIDATED_FOR_293FT.\n"
    "- Reuses pinned unmodified official PARM source+weights under data/breakthrough_20261002/batch31/_external\n"
    "  (verify-SHA, referenced in place; not redistributed).\n\n"
    "## Preserved defaults (unchanged)\n"
    "local 293FT D28 = FAIL; A20/G20 2000bp = UNSUPPORTED; 293FT = UNVALIDATED_FOR_293FT;\n"
    "public HEK293FT OligoMPRA != local 293FT D28; production pointer v6.3 main / v6.4 cellline pilot.\n\n"
    "## Files\n"
    "- reports: diagnostic_spec.json, error_decomposition.json, rank_diagnostics.json,\n"
    "  distribution_table.json, replay_verification.json, phaseB_summary.json, decision.json,\n"
    "  correction_note.json/.md, conclusion_zh.txt, figures/fig1..3, sha_manifest.json,\n"
    "  worker_selfcheck.json, status.json, remaining_work.json, next_batch_proposal.json\n"
    "- data: predictions_train.tsv (replayed), group_error_table.tsv (complete, no outcome exclusions)\n"
    "- code: runs/breakthrough_20261002/code/batch43/*.py\n"
    % (tw["R0"], tw["R1"], tw["Rnull"], tw["const"]))
(R / "README.md").write_text(readme, encoding="utf-8")

# ---- status / selfcheck / remaining / proposal ------------------------------
status = {
    "batch": C.BATCH, "version": C.VERSION, "phase": "worker", "state": "worker_complete_awaiting_fresh_review",
    "generated_at": C.now_local(), "pid": os.getpid(),
    "verdict": dec["original_failure_persists"] and "LIMITED_DIAGNOSTIC" or "LIMITED_DIAGNOSTIC",
    "original_failure_persists": True, "operational_PASS": False, "no_new_operational_PASS_criterion": True,
    "fits": 0, "estimator_solves": 0, "optimizer_steps": 0, "gradients": 0, "gpu_forward": 0, "web": 0,
    "posthoc_reuse_of_exposed_test": True,
    "model_requested": "deepseek-flash",
    "chassis_current_authority": "USER 293FT (2026-10-03)",
    "paths": {"code": "runs/breakthrough_20261002/code/batch43",
              "data": "data/breakthrough_20261002/batch43",
              "reports": "reports/breakthrough_20261002/batch43",
              "runs": "runs/breakthrough_20261002/batch43"},
    "payload_sha256": {"diagnostic_spec.json": C.sha256_file(R / "diagnostic_spec.json"),
                       "error_decomposition.json": C.sha256_file(R / "error_decomposition.json"),
                       "rank_diagnostics.json": C.sha256_file(R / "rank_diagnostics.json"),
                       "decision.json": C.sha256_file(R / "decision.json")},
    "note": "worker selfcheck is NOT an independent PASS; freshreview writes independent_review.json",
}
C.atomic_write_json(R / "status.json", status)

selfcheck = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
    "worker_selfcheck_only": True, "independent_review_created": False,
    "zero_estimator_fits_solves_regression_calibration_gradients_optimizer_steps": True,
    "zero_gpu_neural_forward": True, "zero_web_download_source_search": True,
    "old_artifacts_read_only": True, "no_alt_split_no_source_parent_removal_no_reseed_no_alpha_change": True,
    "no_threshold_tuning_no_posthoc_candidate_generation": True,
    "no_new_local14_HEK673_variant_scoring": True,
    "cohort_preserved": {"bio": 7537, "controls": 85, "TRAIN": [7219, 92], "TEST": [318, 21], "union_groups": 113},
    "train_predictions_persisted_by_batch42": False,
    "train_replay_method": "saved R0/R1/Rnull coefficients + frozen feature/normalizer contracts (one pass)",
    "test_replay_max_abs_diff_all318": {k: v["max_abs_diff_all318"] for k, v in C.jload(R / "replay_verification.json")["test_replay_verification"].items()},
    "mse_identity_max_abs_resid_per_group": ed["TEST"]["R0"]["identity_max_abs_resid_per_group"],
    "invalid_cross_unit_maes_marked": True, "I15_and_blanket_compliance_rejected": True,
    "cpu_threads": 4, "verdict": "LIMITED_DIAGNOSTIC", "operational_PASS": False,
}
C.atomic_write_json(R / "worker_selfcheck.json", selfcheck)

remaining = {
    "batch": C.BATCH, "generated_at": C.now_local(), "verdict": "LIMITED_DIAGNOSTIC",
    "remaining": [
        "Original batch42 negative result unchanged; no promotion; no operational PASS.",
        "TEST error is predominantly within-group (per-construct), ~71-88% of MSE; group bias ~12-29%.",
        "Within-group and group-mean rank skill remain near zero; 21 TEST groups (<30) -> no CI.",
        "Group means from TEST labels are ORACLE diagnostics, not deployable corrections.",
        "Matched-cell local 293FT D28 evidence and prospective untouched confirmation still missing.",
        "public HEK293FT OligoMPRA != local 293FT/D28; D28 FAIL and 2000bp UNSUPPORTED preserved.",
    ],
}
C.atomic_write_json(R / "remaining_work.json", remaining)

proposal = {
    "record_type": "next_batch_proposal",
    "batch": C.BATCH, "generated_at": C.now_local(),
    "one_bounded_next_protocol": {
        "name": "batch44 v6.31.0-within-group-ordering.rc1 (bounded, pre-registered, group-heldout)",
        "question": ("The diagnostic shows TEST error is within-group and within-group rank skill is ~0. "
                     "Can ONE pre-registered, richer SEQUENCE representation improve WITHIN-GROUP ordering "
                     "for novel promoter/truncation ranking, without touching this exposed batch42 TEST?"),
        "scope": ("Reuse the FROZEN batch42 union groups/split (no resplit, no source-parent removal). Carve an "
                  "INTERNAL group-heldout validation split label-free from TRAIN groups ONLY to fix the recipe; "
                  "then read the frozen TEST exactly once. <=2 fixed fits, <=3 GPU-hours, CPU/GPU within budget. "
                  "No alpha/threshold tuning against TEST; no overlap-only resplit; no non-chr17/cropped133 rescoring."),
        "why_tied_to_target": ("User's primary target is ranking novel 293FT promoter/truncation activity; the "
                               "diagnostic localizes the deficit to within-group ordering, so a within-group-focused, "
                               "group-heldout learning question is the evidence-supported next step."),
        "acceptance": ("Pre-declared within-group macro-rho gain over the frozen R0 baseline on the INTERNAL "
                       "validation; a single frozen TEST read reported once; honest NEGATIVE allowed; no promotion."),
        "requires": "PARENT DECISION and real dispatch AFTER fresh review; Astra advice (if any) is advice, not evidence/authorization",
    },
    "alternative_if_parent_prefers_evidence": ("If matched-cell labels are the priority, pursue actual 293FT autonomous-promoter "
                                               "assay-matched cohorts; prior source searches are exhausted, so this stays an evidence gap, "
                                               "and the existing code/batch34/parm_fasta_predict.py interface is DESCRIPTIVE only (UNVALIDATED_FOR_293FT)."),
    "note": "ONE concrete bounded proposal; not authorized here; does not repeat rejected posthoc fits",
}
C.atomic_write_json(R / "next_batch_proposal.json", proposal)

# ---- strict JSON validation (reject non-finite tokens) ----------------------
json_checked = []
json_fail = []
for base in (R, D, RUN):
    for p in sorted(Path(base).rglob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            strict_load(p)
            json_checked.append(str(p.relative_to(C.ROOT)))
        except Exception as e:
            json_fail.append({"path": str(p.relative_to(C.ROOT)), "error": str(e)})
if json_fail:
    raise SystemExit("STRICT JSON FAIL: %s" % json_fail)

# ---- sha manifest (excludes itself + mutable verify/status/review/mail) -----
EXCLUDE_NAMES = {"sha_manifest.json", "status.json", "worker_selfcheck.json", "remaining_work.json",
                 "next_batch_proposal.json", "independent_review.json", "independent_review.html"}
files = []
for base in (CODE, D, R, RUN):
    for p in sorted(Path(base).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".json", ".tsv", ".png", ".md", ".txt", ".jsonl"):
            if p.name in EXCLUDE_NAMES:
                continue
            files.append(p)
seen, manifest_files = set(), []
for p in files:
    rel = str(p.relative_to(C.ROOT)).replace("\\", "/")
    if rel in seen:
        continue
    seen.add(rel)
    manifest_files.append({"path": rel, "sha256": C.sha256_file(p), "bytes": p.stat().st_size})
manifest = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
    "excludes": ["manifest itself", "mutable status/verify/review/mail", "worker_selfcheck",
                 "remaining_work", "next_batch_proposal", "independent_review"],
    "n_files": len(manifest_files), "files": manifest_files,
}
C.atomic_write_json(R / "sha_manifest.json", manifest)

# ---- atomic registry + changelog append ------------------------------------
reg_path = C.ROOT / "reports" / "breakthrough_20261002" / "version_registry.json"
reg = C.jload(reg_path)
reg["batch_appends"].append({
    "record_type": "group_error_diagnostic",
    "batch": C.BATCH, "version": C.VERSION,
    "kind": "posthoc diagnostic of frozen batch42 held-group pilot; ZERO fits/GPU; group-error MSE decomposition",
    "appended_by": "batch43 worker (DeepSeek Flash implementer; parent-authorized append-only)",
    "appended_at": C.now_local(),
    "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
    "test_diagnostic": {
        "within_frac_row_weighted": {m: ed["TEST"][m]["row_weighted"]["within_frac"] for m in ("R0", "R1", "Rnull", "const")},
        "between_frac_row_weighted": {m: ed["TEST"][m]["row_weighted"]["between_frac"] for m in ("R0", "R1", "Rnull", "const")},
        "group_mean_spearman": {m: rk["TEST"][m]["group_mean_spearman"] for m in ("R0", "R1", "Rnull", "const", "parm", "h1", "gc")},
    },
    "verdict": "LIMITED_DIAGNOSTIC", "operational_PASS": False,
    "correction": "cached_parm cross-unit MAEs INVALID; I15/blanket compliance rejected",
    "scope": "posthoc reuse of EXPOSED test; not new validation; no promotion",
    "preserved_prior": True,
})
reg["last_appended_at"] = C.now_local()
reg["last_appended_by"] = "batch43 worker"
tmp = reg_path.with_suffix(".json.tmp")
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(reg, fh, ensure_ascii=False, indent=1)
    fh.write("\n")
os.replace(tmp, reg_path)

cl_path = C.ROOT / "reports" / "breakthrough_20261002" / "CHANGELOG.txt"
line = ("%s batch43 v6.30.0-group-error-diagnostic.rc1: posthoc diagnostic of frozen batch42 pilot (ZERO fits/GPU). "
        "TRAIN predictions replayed from saved coefficients (max|diff| 5e-7 vs published TEST rounding); "
        "MSE_g=b_g^2+within_g verified (resid ~1e-17). TEST error predominantly WITHIN-group (row-weighted within frac "
        "R0 0.850 / R1 0.776 / Rnull 0.830 / const 0.879; group bias ~12-29%%). Original negative result preserved "
        "(R1 rho 0.091 vs R0 0.158; gMAE 0.175 vs 0.107; verdict LIMITED). Correction: cached_parm cross-unit MAEs "
        "(0.46639/0.35361) INVALID; review I15 and blanket compliance PASS rejected; old bytes preserved. "
        "No promotion; v6.3/v6.4 pointers unchanged.\n" % C.now_local())
text = cl_path.read_text(encoding="utf-8")
if not text.endswith("\n"):
    text += "\n"
tmp = cl_path.with_suffix(".txt.tmp")
tmp.write_text(text + line, encoding="utf-8")
os.replace(tmp, cl_path)

print("PHASE D done %.2fs json_checked=%d manifest_files=%d" % (time.time() - T0, len(json_checked), len(manifest_files)))
