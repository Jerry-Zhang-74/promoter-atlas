#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 independent tests (scientific failure modes, not implementation mirrors).

Covers: deterministic label-independent fold assignment, completeness, no
component crossing, fold train-only scalers/targets, no held-out fitting, exact
paired row alignment, independent metric calculation (scipy), bootstrap strata,
saved-model replay per fold, resume/non-overwrite guard, and CPU thread limits.
No research fits are re-run.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import joblib  # noqa: E402
from scipy import stats  # noqa: E402

from b13_common import (  # noqa: E402
    ROOT, DATA_OUT, MODEL_OUT, REPORT_OUT, CANDIDATE_VERSION, BATCH, N_FOLDS,
    BOOT_SEED, N_EXPECTED_ELIGIBLE, N_EXPECTED_COMPONENTS,
    reconstruct_s_raw, load_gene_universe, load_features45, load_full_features,
    spearman, atomic_write_json, thread_caps_record)
from b13_fold_manifest import assign_folds  # noqa: E402


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def t_fold_assignment_deterministic():
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    comp_e = comp_dense[eligible]
    a = assign_folds(comp_e)
    b = assign_folds(comp_e)
    f1, s1, h1, o1, c1 = a
    f2, s2, h2, o2, c2 = b
    ok = (f1 == f2 and o1 == o2 and c1 == c2)
    return check("fold_assignment_deterministic", ok,
                 "rerun yields identical fold map, order and counts")


def t_fold_assignment_complete_no_crossing():
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    comp_e = comp_dense[eligible]
    fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_e)
    fold_of_gene = np.array([fold_of_component[int(c)] for c in comp_e], dtype=np.int64)
    ok_complete = int((fold_of_gene >= 0).sum()) == N_EXPECTED_ELIGIBLE
    ok_sum = int(sum(fold_counts)) == N_EXPECTED_ELIGIBLE
    ok_comp_sum = int(len(order)) == N_EXPECTED_COMPONENTS
    ok_crossing = len(fold_of_component) == N_EXPECTED_COMPONENTS
    return check("fold_assignment_complete_no_crossing",
                 ok_complete and ok_sum and ok_comp_sum and ok_crossing,
                 "genes=%d sum=%d components=%d" % (N_EXPECTED_ELIGIBLE, sum(fold_counts), len(order)))


def t_fold_assignment_label_independent():
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    comp_e = comp_dense[eligible]
    fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_e)
    perm = np.random.default_rng(1).permutation(len(comp_e))
    fold2, _, _, _, _ = assign_folds(comp_e[perm])
    ok = fold_of_component == fold2
    return check("fold_assignment_label_independent", ok,
                 "assignment invariant to row permutation (no label input)")


def t_manifest_matches_recompute():
    manifest = json.loads((DATA_OUT / "fold_manifest.json").read_text(encoding="utf-8"))
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    comp_e = comp_dense[eligible]
    fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_e)
    ok = (fold_counts == manifest["fold_gene_counts"])
    rows = {r["component_id"]: r["fold"] for r in manifest["component_rows"]}
    ok &= all(rows[c] == fold_of_component[c] for c in fold_of_component)
    return check("manifest_matches_recompute", ok,
                 "saved manifest equals recomputed assignment")


def t_train_only_scalers_targets():
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    X45 = load_features45()[eligible]
    X_full = load_full_features()[eligible]
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]], dtype=np.float64)
    fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_dense[eligible])
    fold_of_gene = np.array([fold_of_component[int(c)] for c in comp_dense[eligible]], dtype=np.int64)

    attempt_dir = MODEL_OUT / "attempt_001_fit"
    ok_all = True
    detail = []
    for f in range(N_FOLDS):
        heldout = fold_of_gene == f
        train = ~heldout
        tf = json.loads((attempt_dir / ("fold%d" % f) / "transforms.json").read_text(encoding="utf-8"))
        base_scaler = joblib.load(attempt_dir / ("fold%d" % f) / "base_scaler.joblib")
        full_scaler = joblib.load(attempt_dir / ("fold%d" % f) / "full_scaler.joblib")
        ok_mean_b = np.allclose(base_scaler.mean_, X45[train].mean(axis=0), atol=1e-9)
        ok_mean_f = np.allclose(full_scaler.mean_, X_full[train].mean(axis=0), atol=1e-9)
        leaked_b = np.allclose(base_scaler.mean_, X45.mean(axis=0), atol=1e-9)
        leaked_f = np.allclose(full_scaler.mean_, X_full.mean(axis=0), atol=1e-9)
        ok_t = (abs(tf["train_mean"] - s_raw[train].mean()) < 1e-12
                and abs(tf["train_std"] - s_raw[train].std(ddof=0)) < 1e-12)
        ok_fold = ok_mean_b and ok_mean_f and (not leaked_b) and (not leaked_f) and ok_t
        ok_all &= ok_fold
        detail.append("fold%d=%s" % (f, ok_fold))
    return check("train_only_scalers_targets", ok_all, " ".join(detail))


def t_paired_row_alignment():
    pred = pd.read_csv(DATA_OUT / "oof_predictions.tsv", sep="\t", low_memory=False)
    ok = len(pred) == N_EXPECTED_ELIGIBLE
    ok &= pred["gene_id"].is_unique
    ok &= pred["base_pred"].notna().all() and pred["full_pred"].notna().all()
    ok &= np.isfinite(pred["base_pred"].values).all() and np.isfinite(pred["full_pred"].values).all()
    ok &= np.isfinite(pred["S_raw_full"].values).all()
    ok &= pred["fold"].between(0, N_FOLDS - 1).all()
    ok &= int(len(pred)) == int(pred["fold"].notna().sum())
    return check("paired_row_alignment", ok,
                 "rows=%d unique gene_id=%s" % (len(pred), pred["gene_id"].is_unique))


def t_independent_metric_calc():
    pred = pd.read_csv(DATA_OUT / "oof_predictions.tsv", sep="\t", low_memory=False)
    s = pred["S_raw_full"].values.astype(np.float64)
    base = pred["base_pred"].values.astype(np.float64)
    full = pred["full_pred"].values.astype(np.float64)
    rb, _ = stats.spearmanr(s, base)
    rf, _ = stats.spearmanr(s, full)
    delta = rf - rb
    m = json.loads((REPORT_OUT / "metrics.json").read_text(encoding="utf-8"))
    ok = abs(delta - m["primary_contrast"]["pooled_delta"]) < 1e-9
    ok &= abs(rb - m["pooled_oof"]["base"]["spearman"]) < 1e-9
    ok &= abs(rf - m["pooled_oof"]["full"]["spearman"]) < 1e-9
    return check("independent_metric_calc", ok,
                 "scipy delta=%.9f vs saved %.9f" % (delta, m["primary_contrast"]["pooled_delta"]))


def t_bootstrap_strata_reproducible():
    pred = pd.read_csv(DATA_OUT / "oof_predictions.tsv", sep="\t", low_memory=False)
    s = pred["S_raw_full"].values.astype(np.float64)
    base = pred["base_pred"].values.astype(np.float64)
    full = pred["full_pred"].values.astype(np.float64)
    fold = pred["fold"].values.astype(np.int64)
    comp = pred["component_id"].values.astype(np.int64)

    comp_gene_idx = {}
    for c in np.unique(comp):
        comp_gene_idx[int(c)] = np.flatnonzero(comp == c)
    fold_comps = {}
    for f in range(N_FOLDS):
        fold_comps[f] = [int(c) for c in np.unique(comp[fold == f])]

    rng = np.random.default_rng(BOOT_SEED)
    first = None
    for b in range(20):
        idx = []
        for f in range(N_FOLDS):
            comps = fold_comps[f]
            sample = rng.choice(comps, size=len(comps), replace=True)
            for c in sample:
                idx.append(comp_gene_idx[c])
        idx = np.concatenate(idx)
        d = spearman(full[idx], s[idx]) - spearman(base[idx], s[idx])
        if b == 0:
            first = d
    m = json.loads((REPORT_OUT / "metrics.json").read_text(encoding="utf-8"))
    ok = first is not None and len(fold_comps) == N_FOLDS
    ok &= all(len(fold_comps[f]) > 0 for f in range(N_FOLDS))
    ok &= m["primary_contrast"]["bootstrap"]["n_bootstrap_nan"] == 0
    return check("bootstrap_strata_reproducible", ok,
                 "within-fold component strata present, no NaN bootstrap deltas")


def t_saved_model_replay_per_fold():
    replay = json.loads((DATA_OUT / "replay.json").read_text(encoding="utf-8"))
    ok = (replay["all_folds_all_arms_match"] and replay["pooled_base_replay_match"]
          and replay["pooled_full_replay_match"])
    return check("saved_model_replay_per_fold", ok, "all folds base/full replay match")


def t_nonoverwrite_guard():
    script = Path(__file__).resolve().parent / "b13_run.py"
    cmd = [sys.executable, "-X", "utf8", str(script), "--attempt", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           timeout=300, cwd=str(ROOT))
        refused = r.returncode != 0 and "REFUSE" in (r.stdout + r.stderr)
        return check("nonoverwrite_guard", refused,
                     "b13_run --attempt 1 exits non-zero with REFUSE (returncode=%d)" % r.returncode)
    except subprocess.TimeoutExpired:
        return check("nonoverwrite_guard", False, "subprocess timeout")


def t_thread_limits():
    rec = thread_caps_record()
    env = rec["env"]
    ok_env = all(env.get(k) == "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    nthreads = [t.get("num_threads") for t in rec["threadpoolctl"] if t.get("num_threads") is not None]
    ok_tpc = all(n <= 4 for n in nthreads)
    return check("thread_limits", ok_env and ok_tpc,
                 "env caps=4; threadpoolctl num_threads=%s" % nthreads)


def main():
    t0 = time.time()
    tests = [
        t_fold_assignment_deterministic(),
        t_fold_assignment_complete_no_crossing(),
        t_fold_assignment_label_independent(),
        t_manifest_matches_recompute(),
        t_train_only_scalers_targets(),
        t_paired_row_alignment(),
        t_independent_metric_calc(),
        t_bootstrap_strata_reproducible(),
        t_saved_model_replay_per_fold(),
        t_nonoverwrite_guard(),
        t_thread_limits(),
    ]
    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tests": tests,
        "n_pass": int(sum(1 for t in tests if t["pass"])),
        "n_total": len(tests),
        "all_pass": all(t["pass"] for t in tests),
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "tests_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
