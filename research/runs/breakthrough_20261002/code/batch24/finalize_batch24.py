#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 finalize: report.md, manifests, selfcheck, status, version card.
ZERO fits (reads saved artifacts only)."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C

PID = os.getpid()


def load(name):
    return json.load(open(C.REPORT_OUT / name, encoding="utf-8"))


def fmt(x, n=6):
    return ("%.*f" % (n, x)) if isinstance(x, float) else str(x)


def build_report(res, loc, gse, add):
    L = []
    A = L.append
    A("# batch24 - %s" % C.VERSION)
    A("")
    A("Retrospective reuse of the batch22 GSE188639 HEK293 core-promoter folds "
      "(group-heldout CV). **RETROSPECTIVE ADAPTIVE DEVELOPMENT, NOT independent "
      "prospective validation.** Units: log2(hek exp / input).")
    A("")
    A("## Question")
    A("Batch22 composition R1 (rho 0.4566) beat the null but adding frozen K562 features gave no "
      "substantive gain; the batch23 local-14 full-insert diagnostic stayed negative/OOD. Instead of "
      "more end-to-end neural training on 673 rows, this batch tests **two deterministic, fixed "
      "sequence kernels** (longer words + TSS-relative position) in a small-sample regularized "
      "estimator. ZERO neural fits; 10 kernel-ridge fits (5 fixed folds x 2 recipes); cached R1 has "
      "ZERO refits.")
    A("")
    A("## Kernels (fixed before any fit)")
    A("- forward-orientation k-mer counts over the ACTUAL full insert, fixed complete ACGT "
      "vocabulary; NO reverse-complement merging (orientation matters); L2 per vector (empty -> 0).")
    A("- `S_global = 0.5*cos(counts_k4) + 0.5*cos(counts_k5)` (dims 256 and 1024); **H1 = S_global**.")
    A("- `S_position` = mean of the k4 cosine kernels inside the three fixed intervals "
      "[0,44), [44,89), [89,133) (133 bp contract); **H2 = 0.5*S_global + 0.5*S_position**.")
    A("- Ridge: per outer train fold, center y by the TRAIN mean only; solve "
      "`(K_train + alpha*I) w = y_centered`, fixed **alpha = 1.0**; predict "
      "`K_holdout_train @ w + train_mean`. No full-cohort centering, no target-fit scaling, no "
      "alpha selection. float64 symmetric solve with residual checks.")
    A("- The 673x673 label-free Gram is computed ONCE: every feature mapping is fixed and "
      "per-sequence, with no population-estimated parameters.")
    A("")
    return L

def build_report2(L, res, loc, gse):
    A = L.append
    po = res["pooled_oof"]
    A("## Results (pooled OOF over fixed batch22 folds, n=673, 669 groups)")
    A("")
    A("| recipe | Spearman | 95% group-bootstrap CI | MAE | R2 | top-decile precision (k=68) |")
    A("|---|---|---|---|---|---|")
    for n, lab in (("R1_cached", "R1 cached baseline"), ("H1", "H1 global k4+k5 (primary)"),
                   ("H2", "H2 global+position (secondary)")):
        m = po[n]; b = m["spearman_ci_group_bootstrap"]
        A("| %s | %.6f | [%.4f, %.4f] | %.6f | %.6f | %.6f |"
          % (lab, m["spearman"], b["lo"], b["hi"], m["mae"], m["r2"],
             m["top_decile"]["precision"]))
    A("")
    A("Per-fold Spearman (folds 0-4):")
    A("")
    A("| recipe | f0 | f1 | f2 | f3 | f4 |")
    A("|---|---|---|---|---|---|")
    for n, lab in (("R1_cached", "R1"), ("H1", "H1"), ("H2", "H2")):
        A("| %s | %s |" % (lab, " | ".join(
            "%.4f" % res["per_fold"][str(f)][n]["spearman"] for f in range(C.N_FOLDS))))
    A("")
    A("### Paired group bootstrap (>=2000 group draws)")
    A("")
    A("| comparison | delta rho point | 95% CI | 97.5% CI (two-comparison) | MAE reduction 95% CI |")
    A("|---|---|---|---|---|")
    for c in ("H1", "H2"):
        b = res["paired_group_bootstrap"]["%s_minus_R1" % c]
        d = res["decisions"][c]
        A("| %s - R1 | %.6f | [%.5f, %.5f] | [%.5f, %.5f] | [%.5f, %.5f] |"
          % (c, d["delta_spearman_vs_R1"],
             b["delta_spearman"]["ci95"]["lo"], b["delta_spearman"]["ci95"]["hi"],
             b["delta_spearman"]["ci97_5_two_comparison"]["lo"],
             b["delta_spearman"]["ci97_5_two_comparison"]["hi"],
             b["mae_reduction"]["ci95"]["lo"], b["mae_reduction"]["ci95"]["hi"]))
    A("")
    A("The 97.5% intervals are a conservative two-comparison description, not a formal proof of "
      "significance.")
    A("")
    A("### Predeclared practical-gain decision (frozen before fit, NOT changed after)")
    A("")
    A("Criteria: rho >= +0.03 AND MAE reduction >= 10% AND top-decile precision not lower AND rho "
      "improves in >= 4/5 folds.")
    A("")
    A("| candidate | d(rho) | MAE reduction | top-decile not lower | positive folds | thresholds met |")
    A("|---|---|---|---|---|---|")
    for c in ("H1", "H2"):
        d = res["decisions"][c]
        A("| %s | %+.4f | %.2f%% | %s | %d/5 | %s |"
          % (c, d["delta_spearman_vs_R1"], 100 * d["mae_reduction_frac_vs_R1"],
             "yes" if d["top_decile_not_lower"] else "no", d["positive_folds"],
             "YES" if d["thresholds_met"] else "NO"))
    A("")
    A("**Honest outcome: the predeclared practical-gain rule is NOT met for either kernel.** "
      "3 of 4 criteria hold for both H1 and H2 (rho gain >= +0.03; top-decile precision not lower "
      "(0.3529 vs 0.2941); rho improves in 5/5 folds; the 97.5% two-comparison delta-rho CI "
      "excludes zero), but the **MAE reduction criterion fails** (5.59% for H1, 5.24% for H2, "
      "below the frozen 10% bar). Per the frozen rule this is a **mixed / no-substantive-improvement "
      "result**, not a validated win. The ranking improvement is consistent but the absolute-error "
      "gain is below the predeclared practical threshold, so no promotion and no practical-advantage "
      "claim is made.")
    A("")
    return L

def build_report3(L, res, loc, gse, add):
    A = L.append
    A("## Local-14 diagnostic (NO fit, NO tuning)")
    A("- H2 is **unsupported** on every one of the 14 constructs (none is 133 bp) - its positional "
      "contract is validated for exactly 133 bp; no truncation was applied.")
    A("- H1 produces an **exploratory_model_score** only (equal average of the 5 saved H1 fold "
      "models). `H1_activity_prediction_unsupported = true`; no validated activity claim.")
    p = loc["pooled_descriptive"]; f = loc["within_family_descriptive"]; s = loc["pairwise_sign_agreement"]
    A("- Descriptive pooled Spearman (exploratory score vs measured log10 GFP): %.4f (n=%d)."
      % (p["spearman_exploratory_vs_measured"], p["n"]))
    A("- Within family: ACTB n=6 rho=%.4f; GAPDH n=6 rho=%.4f; CMV n=2 (not computed, <3)."
      % (f["ACTB"]["spearman_exploratory_vs_measured"],
         f["GAPDH"]["spearman_exploratory_vs_measured"]))
    A("- Pairwise sign agreement: %d/%d = %.4f (score ties %d, measured ties %d)."
      % (s["n_agree"], s["n_pairs"], s["agreement"], s["n_tie_score"], s["n_tie_meas"]))
    A("- No cross-scale MAE/R2 and no local calibration was computed. All 5 fold outputs retained; "
      "fold spread is not a CI.")
    A("- Mapping availability vs certainty: the 14 insert sequences are available; the "
      "assay-to-plasmid barcode mapping is UNVERIFIED for all 14.")
    A("- CMVcore actual feature length 204 bp vs the historical 200 bp label; CMVfull (584 bp) is a "
      "distinct enhancer-containing construct; A6 and the 81 bp untransfected sample are excluded.")
    A("")
    A("## GSE306816 bounded qualification (<=25 min, <=75 MB, NO fit)")
    A("- Downloaded %.3f MB (bound 75 MB); the variant-effect table is COMPLETE, the rest are "
      "bounded partial extracts." % gse["downloaded_mb_total"])
    vt = gse["findings"]["variant_effect_table"]
    A("- Complete processed table: %d rows, %d parent loci, rows/locus %d-%d (median %d); "
      "%.1f%% of slopes positive; %.1f%% with p<0.05."
      % (vt["n_rows"], vt["n_parent_loci"], vt["rows_per_locus"]["min"],
         vt["rows_per_locus"]["max"], vt["rows_per_locus"]["median"],
         100 * vt["frac_beta1_positive"], 100 * vt["frac_p_lt_0.05"]))
    A("- Cell = **HEK293T** (sample metadata), 3 biological replicates; GFP reporter MPRA; the "
      "processed file name abbreviates the cell as 'HEK293'.")
    A("- The variable measured element is a promoter-proximal **STR** in a FIXED promoter/reporter "
      "backbone. The processed tables provide BC -> STR variant names only; **no construct-level DNA "
      "sequence and no fixed-backbone sequence** are supplied.")
    A("- Exact join check (partial extracts): STR_allele == Variant_name for %d shared variant names."
      % gse["findings"]["exact_join_check"]["n_exact_intersection"])
    A("- **Decision**: not usable for the current 133 bp promoter-autonomy domain (variant-effect "
      "labels are well defined, but sequence-level input is missing). It is NOT rejected because it "
      "is synthetic/engineered. This is not a claim that all public HEK data are unavailable - the "
      "specific missing context is listed in `gse306816_followup.json`.")
    A("")
    A("## Limitations")
    A("- Retrospective adaptive development: the batch22 folds and test outcomes have been reused "
      "before; this is NOT independent prospective validation and the larger OOF score is not a "
      "fresh generalization estimate.")
    A("- Published HEK293 is NOT HEK293T and NOT the local GFP assay.")
    A("- Group/source-exposure limits carry forward; exact-substring source-unseen is not proof of "
      "absence of all homology (no alignment-based screening).")
    A("- No wet-lab, clinical, or cell-transfer reliability claim is made.")
    A("")
    A("## batch23 closure errata")
    for c in add["corrections"]:
        A("- **%s** (%s)" % (c["id"], c["target"].split("(")[0].strip()))
        A("  - issue: %s" % c["issue"])
        A("  - correction: %s" % c["correct"])
    A("")
    A("## Artifacts")
    A("- fits: `models/breakthrough_20261002/batch24/attempt_fold{0..4}_{H1,H2}/`")
    A("- OOF: `data/breakthrough_20261002/batch24/oof_predictions_batch24.tsv`")
    A("- Grams: `data/breakthrough_20261002/batch24/gram_*.npy`")
    A("- figures: `reports/breakthrough_20261002/batch24/figures/`")
    A("")
    return L

def main():
    res = load("results.json"); loc = load("local14_exploratory.json")
    gse = load("gse306816_followup.json"); add = load("batch23_closure_addendum.json")
    att = load("attempts_summary.json")
    L = build_report(res, loc, gse, add)
    L = build_report2(L, res, loc, gse)
    L = build_report3(L, res, loc, gse, add)
    (C.REPORT_OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- next batch proposal -----------------------------------------------
    prop = {
        "generated_at_local": C.now_local(), "batch": C.batch if hasattr(C, "batch") else C.BATCH,
        "status": "proposed_only; parent must approve the exact frozen protocol",
        "basis": ("H1/H2 give a consistent ranking gain (5/5 folds, 97.5% CI excludes zero) but the "
                  "predeclared 10% MAE-reduction bar is not met; no promotion."),
        "primary_objective": ("decide whether the residual error is an estimator/representation limit "
                              "or a genuine domain limit, using ONLY prospective-family or new-data "
                              "evidence rather than further reuse of the same 673 rows"),
        "options": [
            {"id": "A", "name": "length-aware / variable-length kernel with a measured supervision "
                              "set that actually varies in length",
             "why": "H2's positional contract is 133 bp only; the local constructs are 100-2000 bp "
                    "and cannot be scored. Needs a length-varying measured cohort, which batch23 did "
                    "not locate within bound.",
             "cost": "no new fit authorized here"},
            {"id": "B", "name": "STR variant-effect task on GSE306816 (SEPARATE scope)",
             "why": "variant-effect slopes are defined for 299+ parent loci in HEK293T with 3 bio "
                    "reps; but no construct sequence is provided.",
             "blocker": "requires the fixed backbone/insertion context (paper methods) before any "
                        "sequence model can be applied; NOT authorized as new training here."},
            {"id": "C", "name": "prospective-family evaluation of the frozen H1 kernel",
             "why": "the only way to convert the retrospective ranking gain into real evidence is a "
                    "new sequence family / new measured cohort, untouched by batch22-24 selection.",
             "cost": "zero fits; evaluation-only once data exist"},
        ],
        "recommended": "C (if a new measured family can be sourced) else A (length-varying supervision), "
                       "with B only after the GSE306816 backbone context is obtained",
        "forbidden_here": ["architecture chosen from its name", "K562 label leakage into held-out HEK "
                           "groups", "lowering the predeclared thresholds after seeing results"],
    }
    C.atomic_write_json(C.REPORT_OUT / "next_batch_proposal.json", prop)

    # ---- immutable artifact manifest (built by write_manifest() LAST) -------
    def write_manifest():
        excl_names = {"sha_manifest.json", "status.json", "selfcheck.json",
                      "independent_review.json"}
        files = []
        for root, dirs, fns in os.walk(C.ROOT):
            if any(seg in root for seg in (".git", ".venv", "site-packages", "__pycache__",
                                           "node_modules")):
                continue
            rp = os.path.relpath(root, C.ROOT).replace("\\", "/")
            if not (rp.startswith("reports/breakthrough_20261002/batch24")
                    or rp.startswith("data/breakthrough_20261002/batch24")
                    or rp.startswith("models/breakthrough_20261002/batch24")
                    or rp.startswith("runs/breakthrough_20261002/code/batch24")
                    or rp.startswith("runs/breakthrough_20261002/batch24")):
                continue
            for fn in fns:
                if fn in excl_names or fn.endswith(".tmp"):
                    continue
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, C.ROOT).replace("\\", "/")
                files.append({"path": rel, "bytes": os.path.getsize(p),
                              "sha256": C.sha256_file(p)})
        cardpath = C.ROOT / "models" / "cards" / "promoter_atlas_v6_18_0_hek_kernel_rc1_card.txt"
        if cardpath.exists():
            files.append({"path": "models/cards/promoter_atlas_v6_18_0_hek_kernel_rc1_card.txt",
                          "bytes": cardpath.stat().st_size, "sha256": C.sha256_file(cardpath)})
        files.sort(key=lambda x: x["path"])
        man = {"generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
               "scope": ("batch24-produced artifacts only; all prior artifacts read-only. "
                         "Excludes this manifest, mutable statuses and the reviewer-owned "
                         "independent_review.json."),
               "n_artifacts": len(files), "artifacts": files}
        C.atomic_write_json(C.REPORT_OUT / "sha_manifest.json", man)
        print("manifest artifacts:", len(files))

    # ---- data manifest -----------------------------------------------------
    dman = {
        "generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
        "task": ("fixed-kernel ridge regression of the published HEK293 bulk core-promoter "
                 "activity hek_norm from 133 bp inserts"),
        "target": {"column": "hek_norm", "definition": "log2(hek exp / input)",
                   "units": "log2 RNA/DNA ratio", "negative_values_kept": True,
                   "extra_log_applied": False},
        "cell": "HEK293 (NOT HEK293T); assay: transient bulk MPRA, 2 replicates",
        "inputs_read_only": {
            "sequence_table.tsv": {"path": "data/breakthrough_20261002/batch22/sequence_table.tsv",
                                   "sha256": C.INPUT_SHA["sequence_table.tsv"], "rows": 673,
                                   "sequence_lengths": [133]},
            "fold_assignment.tsv": {"path": "data/breakthrough_20261002/batch22/fold_assignment.tsv",
                                    "sha256": C.INPUT_SHA["fold_assignment.tsv"], "n_groups": 669,
                                    "n_folds": 5, "seed": 20261025},
            "oof_predictions.tsv": {"path": "data/breakthrough_20261002/batch22/oof_predictions.tsv",
                                    "sha256": C.INPUT_SHA["oof_predictions.tsv"],
                                    "use": "cached R1 baseline predictions copied by exact prom_id; "
                                           "ZERO R1 refits"},
            "local_constructs.tsv": {"path": "data/breakthrough_20261002/batch23/local_constructs.tsv",
                                     "sha256": C.INPUT_SHA["local_constructs.tsv"], "rows": 14},
        },
        "split": "unchanged batch22 group folds; no new split chosen by score",
        "k562_role": "paired-cell diagnostic reference only; never a model input or target",
        "evidence_kind": ("RETROSPECTIVE ADAPTIVE DEVELOPMENT on reused batch22 folds/test outcomes, "
                          "NOT independent prospective validation"),
    }
    C.atomic_write_json(C.DATA_OUT / "data_manifest.json", dman)

    # ---- selfcheck (WORKER ONLY) ------------------------------------------
    resid = [json.load(open(C.MODEL_OUT / ("attempt_fold%d_%s" % (f, r)) / "stats.json",
                              encoding="utf-8")) for r in C.RECIPES for f in range(C.N_FOLDS)]
    selfcheck = {
        "generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
        "selfcheck_by_worker": True, "independent_review_written": False,
        "pid": PID,
        "checks": {
            "input_sha_verified_against_batch22_manifest": True,
            "ids_and_folds_aligned": "673 ids, 669 groups, 5 folds",
            "zero_neural_fits": True,
            "kernel_ridge_fits": att["n_attempts"],
            "fit_budget_max": att["max_ridge_fits_budget"],
            "all_fits_registered_preregister_plus_complete": True,
            "max_relative_solve_residual": max(x["relative_residual"] for x in resid),
            "all_predictions_finite": all(x["finite"] for x in resid),
            "train_only_target_centering": True,
            "alpha_fixed_at_1": True,
            "labels_not_used_for_kernel_or_scaling": True,
            "cached_R1_refits": 0,
            "cached_R1_metrics_reproduce_batch22": True,
            "local14_no_fit_no_tuning": True,
            "H2_marked_unsupported_on_non_133bp": True,
            "no_cross_endpoint_mae_r2_emitted": True,
            "gse306816_within_bound": True,
        },
        "unit_tests": {"file": "runs/breakthrough_20261002/code/batch24/test_batch24_kernels.py",
                       "n_tests": 11, "result": "ALL PASS",
                       "covers": ["vocabulary order/dims", "symmetry", "PSD tolerance",
                                  "orientation (non-RC-merged)", "empty-vector zeros",
                                  "H2 identity", "fixed intervals"]},
        "not_verified_by_worker": ["independent metric recomputation",
                                   "independent GSE306816 source reading"],
    }
    C.atomic_write_json(C.REPORT_OUT / "selfcheck.json", selfcheck)

    # ---- canonical status --------------------------------------------------
    status = {
        "batch": "batch24", "version": C.VERSION,
        "phase": "worker_complete_awaiting_independent_review",
        "state": "experiment_complete",
        "experiment_complete": True,
        "ongoing_task_active": True,
        "real_process_identity": {
            "fit_pids": sorted(set(x["pid"] for x in resid)),
            "finalize_pid": PID,
            "eval_pid_note": "evaluation/figure scripts each ran as their own short-lived process; "
                              "their outputs are hashed in sha_manifest.json",
            "role": "batch24 worker (kernel fits + evaluation + report)"},
        "started_at_local": "2026-10-03T16:24:41+08:00",
        "updated_at_local": C.now_local(),
        "protocol": "runs/breakthrough_20261002/batch24/PROTOCOL.txt",
        "science_status": "reports/breakthrough_20261002/batch24/results.json",
        "headline": {
            "H1_rho": res["pooled_oof"]["H1"]["spearman"],
            "H2_rho": res["pooled_oof"]["H2"]["spearman"],
            "R1_rho": res["pooled_oof"]["R1_cached"]["spearman"],
            "practical_gain_rule_met": False,
            "why": "MAE reduction below the predeclared 10% bar for both kernels",
        },
        "fits": {"kernel_ridge": att["n_attempts"], "neural": 0,
                 "cached_R1_refits": 0, "budget": 10},
        "remaining_work": [
            "fresh independent Flash review (batch24/REVIEW_PROTOCOL.txt)",
            "parent decides the next bounded task from the kernel outcomes",
        ],
        "selfcheck_path": "reports/breakthrough_20261002/batch24/selfcheck.json",
        "independent_review_path": None,
    }
    C.atomic_write_json(C.REPORT_OUT / "status.json", status)

    # ---- version card ------------------------------------------------------
    card = """promoter_atlas candidate card
candidate_version: {ver}
track/task: HEK293 core-promoter fragment activity -> fixed-kernel ridge (CPU)
parent_model_version: none (NOT a neural checkpoint; deterministic kernels)
data_version: data/breakthrough_20261002/batch22/sequence_table.tsv sha256 {d1}
split_version: data/breakthrough_20261002/batch22/fold_assignment.tsv sha256 {d2} (5 group folds, seed 20261025)
code_version: runs/breakthrough_20261002/code/batch24/*.py
evaluation_protocol_version: reports/breakthrough_20261002/batch24/results.json
run_id: batch24_kernels_g1
seed: none (deterministic solver; no random seeds)
attempt_id: 1..10 (5 folds x 2 recipes)
config_sha: {cfg}
creation_time: {ts}
metrics (pooled OOF, log2(hek exp/input), n=673, 669 groups):
  R1_cached baseline: rho 0.456622 MAE 0.571670 R2 0.191538 top-decile 0.294118
  H1 global k4+k5   : rho 0.526449 MAE 0.539729 R2 0.276528 top-decile 0.352941
  H2 global+position: rho 0.519965 MAE 0.541731 R2 0.285527 top-decile 0.352941
decision: NO PROMOTION. Predeclared practical-gain rule NOT met (MAE reduction 5.6%/5.2% < 10%);
          ranking gain is consistent (5/5 folds, 97.5% two-comparison delta-rho CI > 0) but is
          retrospective adaptive development, not prospective validation.
review_status: awaiting fresh independent review
status: experimental / unpromoted
scope_limits: published HEK293 (NOT HEK293T); reused batch22 folds; no cell-transfer or wet-lab claim
""".format(ver=C.VERSION, d1=C.INPUT_SHA["sequence_table.tsv"],
           d2=C.INPUT_SHA["fold_assignment.tsv"], cfg=att["config_sha"],
           ts=C.now_local())
    carddir = C.ROOT / "models" / "cards"
    carddir.mkdir(parents=True, exist_ok=True)
    (carddir / "promoter_atlas_v6_18_0_hek_kernel_rc1_card.txt").write_text(card, encoding="utf-8")
    C.atomic_write_json(C.REPORT_OUT / "version_registry_entry.json", {
        "note": ("batch24-local append-only registry entry. The shared "
                 "reports/breakthrough_20261002/version_registry.json and CHANGELOG.txt are prior "
                 "artifacts and were left read-only per the batch24 write-scope rule."),
        "candidate_version": C.VERSION, "track": "hek_kernel_fragment_activity",
        "task": "fixed-kernel ridge (H1/H2) vs cached batch22 R1", "status": "experimental_unpromoted",
        "parent_model_version": None, "code_version": "runs/breakthrough_20261002/code/batch24",
        "data_version": C.INPUT_SHA["sequence_table.tsv"],
        "split_version": C.INPUT_SHA["fold_assignment.tsv"],
        "config_sha": att["config_sha"], "run_id": "batch24_kernels_g1", "seed": None,
        "attempt_ids": list(range(1, att["n_attempts"] + 1)),
        "created_at_local": C.now_local(), "review_status": "awaiting_independent_review",
        "decision": "no_promotion_predeclared_rule_not_met",
        "card": "models/cards/promoter_atlas_v6_18_0_hek_kernel_rc1_card.txt",
    })
    write_manifest()
    print("finalized:", C.REPORT_OUT)


if __name__ == "__main__":
    main()
