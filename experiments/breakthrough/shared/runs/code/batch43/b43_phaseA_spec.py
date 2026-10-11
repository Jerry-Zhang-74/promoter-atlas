# -*- coding: utf-8 -*-
"""batch43 Phase A: freeze diagnostic_spec.json (ZERO fits).

Freezes exact input/cache/source/model/group/split SHA + the fixed definitions,
cohort and strata.  Reads the ACTUAL batch42 metrics + original coefficient
replay evidence (not just a PASS string).  Records the two cached PARM
cross-unit MAEs as INVALID and rejects the I15/compliance blanket, WITHOUT
editing the old canonical review or old results.
"""
from __future__ import annotations
import json
import os
import time

import b43_common as C

T0 = time.time()
C.set_thread_limits(4)

metrics = C.jload(C.B42_METRICS)
review = C.jload(C.B42_REVIEW)

orig = {
    "R1_pooled_spearman": metrics["R1"]["pooled_spearman"],
    "R0_pooled_spearman": metrics["R0"]["pooled_spearman"],
    "Rnull_pooled_spearman": metrics["Rnull"]["pooled_spearman"],
    "R1_group_balanced_mae": metrics["R1"]["group_balanced_mae"],
    "R0_group_balanced_mae": metrics["R0"]["group_balanced_mae"],
    "Rnull_group_balanced_mae": metrics["Rnull"]["group_balanced_mae"],
    "zero_fit_const_baseline": metrics["zero_fit_train_weighted_mean_baseline"],
    "decision_verdict": metrics["decision"]["verdict"],
    "decision_operational_PASS": metrics["decision"]["operational_PASS"],
    "train_diagnostic": metrics.get("train_diagnostic"),
}
invalid = {
    "strata_chr17_cached_parm_mae": metrics["strata"]["chr17"]["cached_parm_mae"],
    "strata_non_chr17_cached_parm_mae": metrics["strata"]["non_chr17"]["cached_parm_mae"],
}
valid_rank_only = {
    "strata_chr17_cached_parm_rho": metrics["strata"]["chr17"]["cached_parm_rho"],
    "strata_non_chr17_cached_parm_rho": metrics["strata"]["non_chr17"]["cached_parm_rho"],
}

i11 = next(c for c in review["integrity_checks"] if c["id"] == "I11_replay")
i15 = next(c for c in review["integrity_checks"] if c["id"] == "I15_no_subset_or_units")
replay_evidence = {
    "review_I11_replay": {"result": i11["result"], "detail": i11["detail"]},
    "review_replay_max_abs_diff_vs_published": review["independent_numbers"].get("replay_max_abs_diff_vs_published"),
    "review_recompute_max_abs_diff_vs_worker": review["independent_numbers"].get("recompute_max_abs_diff_vs_worker"),
    "review_I15_claim": i15["detail"],
    "review_verdict": review["verdict"],
    "review_authored_at_on_disk": review.get("authored_at"),
    "review_reviewed_at_on_disk": review.get("reviewed_at"),
}

spec = {
    "batch": C.BATCH,
    "version": C.VERSION,
    "record_type": "diagnostic_spec",
    "phase": "A_freeze",
    "generated_at": C.now_local(),
    "pid": os.getpid(),
    "posthoc_diagnostic_reuse": True,
    "not_new_validation": True,
    "fixed_question": ("In the already frozen batch42 group-heldout pilot, are errors predominantly "
                       "group-level bias or poor within-group ranking, and what distribution limitations "
                       "prevent inference about novel 293FT truncations?"),
    "authorized": ("posthoc diagnostic reuse of an EXPOSED test; NOT authorized to correct predictions "
                   "using test group means or to select a better model/subset"),
    "inputs_sha256": {
        "frozen_input_table.tsv": C.sha256_file(C.FROZEN_TABLE),
        "design_source_fa_gz": C.sha256_file(C.DESIGN_FA),
        "groups.tsv": C.sha256_file(C.GROUPS_TSV),
        "frozen_split.json": C.sha256_file(C.FROZEN_SPLIT),
        "predictions_test.tsv": C.sha256_file(C.PRED_TEST),
        "cache_h1_preds.json": C.sha256_file(C.H1_CACHE),
        "cache_parm_preds.json": C.sha256_file(C.PARM_CACHE),
        "batch42_metrics.json": C.sha256_file(C.B42_METRICS),
        "batch42_independent_review.json": C.sha256_file(C.B42_REVIEW),
        "models_R0.json": C.sha256_file(C.B42_DATA / "models" / "R0.json"),
        "models_R1.json": C.sha256_file(C.B42_DATA / "models" / "R1.json"),
        "models_Rnull.json": C.sha256_file(C.B42_DATA / "models" / "Rnull.json"),
        "code_b42_common_feature_contract": C.sha256_file(C.B42_CODE / "b42_common.py"),
        "code_b42_phaseC_eval.py": C.sha256_file(C.B42_CODE / "b42_phaseC_eval.py"),
    },
    "sha_expectations_ok": {
        "frozen_table": C.sha256_file(C.FROZEN_TABLE) == C.FROZEN_TABLE_SHA,
        "design": C.sha256_file(C.DESIGN_FA) == C.DESIGN_SHA,
        "feature_code": C.sha256_file(C.B42_CODE / "b42_common.py") == C.FEATURE_CODE_SHA,
        "batch42_metrics": C.sha256_file(C.B42_METRICS) == C.B42_METRICS_SHA,
    },
    "cohort": {
        "bio_rows": 7537, "controls": 85,
        "TRAIN": {"rows": 7219, "groups": 92},
        "TEST": {"rows": 318, "groups": 21},
        "union_groups": 113, "chroms": 14, "length_bp": 270,
        "preserved": "all original union groups/IDs; source-parent merging UNCHANGED (no removal)",
    },
    "strata": ["overall", "chr17", "non_chr17"],
    "definitions": {
        "error": "e_i = p_i - y_i (learned same-unit models only; p=prediction, y=raw alpha)",
        "group_mean_error": "b_g = mean over i in g of e_i",
        "group_MSE": "MSE_g = mean over i in g of e_i^2",
        "group_within": "within_g = mean over i in g of (e_i - b_g)^2",
        "identity": "MSE_g = b_g^2 + within_g (per group; verified to float tolerance)",
        "row_weighted": "(1/N) sum over i; between=(1/N) sum over g of n_g*b_g^2; within=(1/N) sum over g of sum over i in g of (e_i-b_g)^2",
        "group_equal": "(1/G) sum over g; between=(1/G) sum over g of b_g^2; within=(1/G) sum over g of within_g",
        "denominators": {"row_weighted": "N rows (TEST=318, TRAIN=7219)",
                         "group_equal": "G groups (TEST=21, TRAIN=92)"},
        "this_is": "MSE decomposition, NOT an additive MAE decomposition",
        "group_means_from_TEST_labels": "diagnostic ORACLE quantities, NOT deployable correction or improved predictions",
    },
    "rank_scope": {
        "group_mean_spearman": "group-mean prediction vs group-mean alpha (all groups, no CI/significance)",
        "within_group_rho": "every group with at least 10 rows and nonconstant vectors; count/coverage/undefined for others",
        "cross_unit_models_PARM_H1_GC": "rank comparisons ONLY (no MAE/MSE/R2 against alpha)",
        "constant_baseline": "rho undefined leads to null plus reason (NOT zero)",
    },
    "distribution_scope": "TRAIN/TEST x {overall, chr17, nonchr17}; fixed; no new strata or resampling selected by findings",
    "original_metrics": orig,
    "script_sha256": C.sha256_file(__file__),
    "elapsed_s": None,
}

spec["correction"] = {
    "invalid_cross_unit_MAEs": invalid,
    "invalid_reason": ("PARM raw score and alpha are different output units; comparing them by MAE is "
                       "INVALID (cross-unit). These two numbers must not be interpreted or used."),
    "valid_same_unit_rank_only": valid_rank_only,
    "I15_rejected": ("review I15_no_subset_or_units PASS is FALSE: batch42 metrics.json strata contains "
                     "cached_parm_mae (cross-unit). The blanket compliance PASS is rejected; numerical "
                     "review and same-unit learned metrics remain separately valid."),
    "old_assets_untouched": True,
    "real_clock_provenance": {
        "batch42_worker_finished_at": review["reviewer"]["worker_session"]["finished_at"],
        "batch42_reviewer_creation_time": review["reviewer"]["harness_process"]["creation_time"],
        "batch42_review_authored_at_on_disk": review.get("authored_at"),
        "note": ("parent acceptance cites review authored_at 03:03:30 later than controller end 03:03:04; "
                 "the on-disk authored_at differs. The on-disk generation clock cannot be treated as "
                 "reliable; this batch43 note carries its OWN real clock."),
        "batch43_correction_note_time": C.now_local(),
    },
}
spec["replay_evidence_read"] = replay_evidence

assert all(spec["sha_expectations_ok"].values()), "SHA expectation failed"
assert orig["decision_verdict"] == "LIMITED" and orig["decision_operational_PASS"] is False

spec["elapsed_s"] = round(time.time() - T0, 2)
C.atomic_write_json(C.B43_REPORT / "diagnostic_spec.json", spec)
C.atomic_write_json(C.B43_RUN / "phaseA_freeze.json",
                    {"batch": C.BATCH, "phase": "A_freeze", "generated_at": C.now_local(),
                     "elapsed_s": spec["elapsed_s"], "sha_expectations_ok": spec["sha_expectations_ok"],
                     "script_sha256": spec["script_sha256"]})
print("PHASE A ok elapsed=%.2fs" % spec["elapsed_s"])
print(json.dumps({"invalid": invalid, "verdict": orig["decision_verdict"]}, ensure_ascii=False))
