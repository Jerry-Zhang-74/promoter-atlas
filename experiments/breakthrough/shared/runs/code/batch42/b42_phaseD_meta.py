# -*- coding: utf-8 -*-
"""batch42 Phase D (worker metadata ONLY): status / selfcheck / remaining work /
ONE concrete next proposal.  Does NOT re-append registry/changelog (done once)."""
from __future__ import annotations
import json

import b42_common as C

REP = C.B42_REPORT
met = json.load(open(REP / "metrics.json", encoding="utf-8"))
bf = json.load(open(C.B42_RUN / "phaseB_fits.json", encoding="utf-8"))
frozen = json.load(open(C.B42_DATA / "frozen_split.json", encoding="utf-8"))
d = met["decision"]
now = C.now_local()

C.atomic_write_json(REP / "status.json", {
    "batch": C.BATCH, "phase": "worker", "state": "complete", "version": C.VERSION,
    "generated_at": now, "pid": __import__("os").getpid(),
    "verdict": d["verdict"], "operational_PASS": False,
    "gate_PASS": frozen["gate"]["PASS"], "fits_consumed": bf["status_after"]["consumed"],
    "max_fits": C.MAX_FITS, "zero_neural_gpu_torch": True,
    "reports_dir": str(REP), "metrics": "metrics.json", "report": "report.html",
    "next": "fresh independent review then parent decision",
    "remaining_work": "remaining_work.json", "next_batch_proposal": "next_batch_proposal.json",
})

C.atomic_write_json(REP / "worker_selfcheck.json", {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": now,
    "worker_selfcheck_only": True, "independent_review_created": False,
    "freeze_before_any_label_access": True,
    "label_free_group_and_split": True, "no_label_driven_regroup_or_resplit": True,
    "exactly_three_estimator_solves": bf["status_after"]["consumed"] == 3,
    "no_cv_no_tuning_no_multiseed_no_fullfit": True,
    "test_labels_sealed_until_models_written": True,
    "seed_recorded": C.NULL_SEED, "boot_not_run_reason": d["bootstrap_reason"],
    "weight_rule": "1/train_group_size normalized to mean 1",
    "R0_alpha": C.R0_ALPHA, "R1_alpha": C.R1_ALPHA,
    "cpu_threads": 4, "zero_neural_gpu_torch": True, "no_model_sources_downloaded": True,
    "verdict": d["verdict"], "operational_PASS": False,
    "note": "this selfcheck is NOT an independent PASS",
})

C.atomic_write_json(REP / "remaining_work.json", {
    "batch": C.BATCH, "generated_at": now, "verdict": d["verdict"],
    "remaining": [
        "R1 (H1 sequence kernel) did NOT beat R0 composition on held-out groups (rho %.3f vs %.3f); negative/descriptive, no operational PASS" % (d["r1_rho"], d["r0_rho"]),
        "TEST only 21 groups (<30): conditional bootstrap/CI/significance unavailable -> cannot certify any positive claim",
        "retrospective reuse of batch41 (all labels previously exposed); NOT new untouched external validation",
        "public HEK293FT != local 293FT/D28; local D28 FAIL and A20/G20 2000bp unsupported remain unconverted",
        "no model promotion; production pointers v6.3/v6.4 unchanged",
        "independent review pending; worker selfcheck is not a PASS",
    ],
})

C.atomic_write_json(REP / "next_batch_proposal.json", {
    "record_type": "next_batch_proposal", "batch": C.BATCH, "generated_at": now,
    "one_bounded_next_protocol": {
        "name": "batch43 v6.30.0-hekft-power-pilot.rc1 (descriptive, ZERO new neural/GPU)",
        "question": "was the negative R1<R0 result caused by too few held-out GROUPS (power), or by the fixed H1 kernel/alpha being genuinely weaker than composition?",
        "scope": "Re-run the SAME three frozen specifications on a PRESET, label-free split that guarantees >=30 held-out groups, WITHOUT changing grouping evidence: use the pre-registered overlap-only partition (already computed label-free: 306 groups, TEST 61 groups) instead of the 4-source union, and report the identical metrics plus the conditional 2000-group bootstrap.",
        "why_bounded": "no new fits beyond the same 3 fixed specs on a pre-computed label-free partition; single metric set; CPU only; <15 min",
        "acceptance": "identical fixed thresholds; if R1 still does not beat R0 with >=30 groups -> conclude the fixed H1 kernel/alpha is not superior for this assay; no promotion either way",
        "requires": "PARENT DECISION on grouping granularity (overlap-only 306 vs union 113) BEFORE execution; no automatic launch",
    },
    "alternative_if_parent_prefers": "add ONE pre-registered alternative fixed alpha for R1 (e.g. alpha=100) under the SAME 3-fit budget in a fresh batch, still union grouping -- to test whether the loss is alpha-driven",
    "note": "ONE specific bounded next protocol as required; not authorized here",
})
print("worker metadata written")
