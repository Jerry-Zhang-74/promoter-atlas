#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 v6.14.0-scv5.rc1 worker self-check (NOT independent review).

Run by the WORKER ONLY. Emits:
  runs/breakthrough_20261002/batch13/worker_self_check.json
  reports/breakthrough_20261002/batch13/worker_self_check.json
  reports/breakthrough_20261002/batch13/worker_self_check.txt
Read-only on frozen inputs; writes only batch13-owned files. Labels the result
awaiting independent review.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import (  # noqa: E402
    DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT, CANDIDATE_VERSION, BATCH, N_FOLDS,
    N_EXPECTED_ELIGIBLE, N_EXPECTED_COMPONENTS, EXPECTED_SHA, sha256_file,
    atomic_write_json, atomic_write_text, now_local)


def main():
    checks = {}

    # source SHAs
    src_files = {
        "features45.npz": "data/breakthrough_20261002/batch06/features45.npz",
        "features1024.npz": "data/breakthrough_20261002/batch10/attempt_001_extract/features1024.npz",
        "split_arrays.npz": "data/breakthrough_20261002/batch05/split_arrays.npz",
        "trackA_gene_promoters.npz": "data/breakthrough_20261002/batch03/trackA_gene_promoters.npz",
    }
    sha_ok = True
    for key, rel in src_files.items():
        p = Path(__file__).resolve().parents[4] / rel
        ok = sha256_file(p) == EXPECTED_SHA[key]
        sha_ok &= ok
        checks["sha_" + key] = ok
    checks["source_sha_all_ok"] = bool(sha_ok)

    # fold manifest present
    manifest = json.loads((DATA_OUT / "fold_manifest.json").read_text(encoding="utf-8"))
    checks["fold_manifest_frozen"] = bool(manifest.get("frozen_before_any_fit_or_heldout_metric"))
    checks["fold_gene_counts_sum"] = int(sum(manifest["fold_gene_counts"]))
    checks["fold_gene_counts_sum_ok"] = (sum(manifest["fold_gene_counts"]) == N_EXPECTED_ELIGIBLE)

    # ten fit checkpoints present (5 folds x 2 arms)
    attempt_dir = MODEL_OUT / "attempt_001_fit"
    present = 0
    for f in range(N_FOLDS):
        fold_dir = attempt_dir / ("fold%d" % f)
        for arm in ("base", "full"):
            if (fold_dir / ("%s_model.joblib" % arm)).exists() and \
               (fold_dir / ("%s_scaler.joblib" % arm)).exists():
                present += 1
    checks["fit_checkpoints_present"] = present
    checks["fit_checkpoints_present_ok"] = (present == 10)

    # metrics
    metrics = json.loads((REPORT_OUT / "metrics.json").read_text(encoding="utf-8"))
    checks["n_fits_executed"] = metrics["n_fits_executed_this_run"]
    checks["n_fits_executed_ok"] = (metrics["n_fits_executed_this_run"] == 10)
    p = metrics["primary_contrast"]
    checks["pooled_delta"] = p["pooled_delta"]
    checks["bootstrap_lo"] = p["bootstrap"]["lo"]
    checks["bootstrap_hi"] = p["bootstrap"]["hi"]
    checks["n_positive_folds"] = p["n_positive_folds"]
    checks["criterion_met"] = p["criterion_met"]

    # tests and replay
    tests = json.loads((REPORT_OUT / "tests_summary.json").read_text(encoding="utf-8"))
    checks["tests_all_pass"] = tests["all_pass"]
    replay = json.loads((DATA_OUT / "replay.json").read_text(encoding="utf-8"))
    checks["replay_all_match"] = replay["all_folds_all_arms_match"]

    all_ok = bool(sha_ok and checks["fold_gene_counts_sum_ok"]
                  and checks["fit_checkpoints_present_ok"]
                  and checks["n_fits_executed_ok"]
                  and checks["tests_all_pass"] and checks["replay_all_match"])
    checks["all_ok"] = all_ok

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "worker": "self_check (NOT independent review)",
        "review_status": "awaiting_review",
        "written_at": now_local(),
        "all_ok": bool(all_ok),
        "checks": checks,
        "note": ("worker self-check confirms integrity and key metrics; a fresh "
                 "independent reviewer must produce the top-level PASS/FAIL verdict "
                 "plus scientific_acceptance and acceptance_threshold_met separately"),
    }
    atomic_write_json(RUNS_OUT / "worker_self_check.json", summary)
    atomic_write_json(REPORT_OUT / "worker_self_check.json", summary)

    lines = [
        "worker_self_check %s: all_ok=%s (awaiting independent review)"
        % (CANDIDATE_VERSION, all_ok),
        "  pooled_delta       %.9f" % checks["pooled_delta"],
        "  bootstrap_lo       %.9f" % checks["bootstrap_lo"],
        "  bootstrap_hi       %.9f" % checks["bootstrap_hi"],
        "  n_positive_folds   %d" % checks["n_positive_folds"],
        "  criterion_met      %s" % checks["criterion_met"],
        "  n_fits_executed    %d" % checks["n_fits_executed"],
        "  fit_checkpoints    %d/10" % checks["fit_checkpoints_present"],
        "  tests_all_pass     %s" % checks["tests_all_pass"],
        "  replay_all_match   %s" % checks["replay_all_match"],
    ]
    atomic_write_text(REPORT_OUT / "worker_self_check.txt", "\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
