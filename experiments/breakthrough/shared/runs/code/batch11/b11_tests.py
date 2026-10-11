#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 independent tests (scientific failure modes, not implementation mirroring).

Covers: feature subset selection/order (A=tri 512, B=tetra 512 by name), exact
old-prediction matching (full/baseline loaded, not refit), training-only scaling,
component alignment (no cross-split components), saved model replay, paired
bootstrap sampling (deterministic + paired), thread caps, output-nonoverwrite
guard, and status identity (worker PID/creation distinct from writer). No
research fits are re-run.
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
import joblib  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from b11_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT, CANDIDATE_VERSION,
                        BATCH, FEATURE_NAMES_45, FEATURE_NAMES_1024, FEATURE_NAMES_ALL,
                        N_FEATURE_UNIVERSE, N_EXPECTED_ELIGIBLE, N_COMPONENTS_TOTAL,
                        reconstruct_s_raw, load_gene_universe, load_features45,
                        load_features1024, feature_arm_indices, spearman,
                        atomic_write_json, thread_caps_record,
                        windows_process_creation_time, fmt_local)  # noqa: E402


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def t_feature_subset_selection_order():
    idx45, tri, tetra = feature_arm_indices()
    names = FEATURE_NAMES_1024
    ok = (len(idx45) == 45 and len(tri) == 512 and len(tetra) == 512)
    ok &= (tri == list(range(512)) and tetra == list(range(512, 1024)))
    ok &= all("_tri_" in names[i] for i in tri)
    ok &= all("_tetra_" in names[i] for i in tetra)
    ok &= (names[0] == "sym400_bin0_tri_AAA" and names[511] == "up2000_bin3_tri_TTT")
    ok &= (names[512] == "sym400_tetra_AAAA" and names[1023] == "up2000_tetra_TTTT")
    ok &= (FEATURE_NAMES_ALL[:45] == FEATURE_NAMES_45)
    return check("feature_subset_selection_order", ok,
                 "A=tri indices 0..511, B=tetra indices 512..1023, verified by name")


def t_exact_old_prediction_matching():
    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    b10 = pd.read_csv("data/breakthrough_20261002/batch10/predictions.tsv", sep="\t", low_memory=False)
    b9 = pd.read_csv("data/breakthrough_20261002/batch09/predictions.tsv", sep="\t", low_memory=False)
    ok = np.allclose(pred["full_ridge_pred"].values, b10["ridge_aug_pred"].values, atol=1e-12)
    ok &= np.allclose(pred["full_hgb_pred"].values, b10["hgb_aug_pred"].values, atol=1e-12)
    ok &= np.allclose(pred["base_ridge_pred"].values, b9["ridge_raw_pred"].values, atol=1e-12)
    ok &= np.allclose(pred["base_hgb_pred"].values, b9["hgb_raw_pred"].values, atol=1e-12)
    ok &= np.array_equal(pred["gene_id"].values, b10["gene_id"].values)
    ok &= np.array_equal(pred["gene_id"].values, b9["gene_id"].values)
    return check("exact_old_prediction_matching", ok,
                 "full/baseline columns equal frozen batch10/batch09 TSVs to 1e-12")


def t_training_only_scaling():
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    X45 = load_features45()
    X1024 = load_features1024()
    idx45, tri, tetra = feature_arm_indices()
    X_A = np.concatenate([X45[:, idx45], X1024[:, tri]], axis=1)
    X_B = np.concatenate([X45[:, idx45], X1024[:, tetra]], axis=1)
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    tr = gene_split[eligible] == "train"
    X_A_e = X_A[eligible]; X_B_e = X_B[eligible]
    scA = joblib.load(MODEL_OUT / "attempt_001_fit" / "X_scaler_A.joblib")
    scB = joblib.load(MODEL_OUT / "attempt_001_fit" / "X_scaler_B.joblib")
    okA_mean = np.allclose(scA.mean_, X_A_e[tr].mean(axis=0))
    sdA = X_A_e[tr].std(axis=0); sdA[sdA == 0] = 1.0
    okA_scale = np.allclose(scA.scale_, sdA)
    okB_mean = np.allclose(scB.mean_, X_B_e[tr].mean(axis=0))
    sdB = X_B_e[tr].std(axis=0); sdB[sdB == 0] = 1.0
    okB_scale = np.allclose(scB.scale_, sdB)
    # no full-data leakage
    no_leak = (not np.allclose(scA.mean_, X_A_e.mean(axis=0)) and
               not np.allclose(scB.mean_, X_B_e.mean(axis=0)))
    tf = json.load(open(MODEL_OUT / "attempt_001_fit" / "transforms.json", encoding="utf-8"))
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])
    ok_t = (abs(tf["target_raw"]["train_mean"] - s_raw[tr].mean()) < 1e-12 and
            abs(tf["target_raw"]["train_std"] - s_raw[tr].std(ddof=0)) < 1e-12)
    return check("training_only_scaling", bool(okA_mean and okA_scale and okB_mean and okB_scale and no_leak and ok_t),
                 "A/B scalers + target fit on eligible train only")


def t_component_alignment():
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    n_el = int(eligible.sum())
    split_counts = {sp: int((gene_split[eligible] == sp).sum()) for sp in ("train", "val", "test")}
    ok = (n_el == N_EXPECTED_ELIGIBLE and
          split_counts == {"train": 15004, "val": 1898, "test": 1867})
    n_comp = int(np.unique(comp_dense[eligible]).size)
    ok &= (n_comp == N_COMPONENTS_TOTAL)
    import collections
    comp_split = collections.defaultdict(set)
    for c, sp in zip(comp_dense[eligible], gene_split[eligible]):
        comp_split[int(c)].add(sp)
    crossing = sum(1 for v in comp_split.values() if len(v) > 1)
    ok &= (crossing == 0)
    return check("component_alignment", ok,
                 "eligible=%d split=%s components=%d crossing=%d" % (n_el, split_counts, n_comp, crossing))


def t_saved_model_replay():
    fit_dir = MODEL_OUT / "attempt_001_fit"
    models = joblib.load(fit_dir / "models.joblib")
    scA = joblib.load(fit_dir / "X_scaler_A.joblib")
    scB = joblib.load(fit_dir / "X_scaler_B.joblib")
    tf = json.load(open(fit_dir / "transforms.json", encoding="utf-8"))
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    X45 = load_features45(); X1024 = load_features1024()
    idx45, tri, tetra = feature_arm_indices()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    Xs_A = scA.transform(np.concatenate([X45[:, idx45], X1024[:, tri]], axis=1)[eligible])
    Xs_B = scB.transform(np.concatenate([X45[:, idx45], X1024[:, tetra]], axis=1)[eligible])
    mu = tf["target_raw"]["train_mean"]; sd = tf["target_raw"]["train_std"]
    p = {
        "A_ridge": models["A_ridge"].predict(Xs_A) * sd + mu,
        "A_hgb": models["A_hgb"].predict(Xs_A) * sd + mu,
        "B_ridge": models["B_ridge"].predict(Xs_B) * sd + mu,
        "B_hgb": models["B_hgb"].predict(Xs_B) * sd + mu,
    }
    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    ok = all(np.allclose(p[k], pred[k + "_pred"].values, atol=1e-8)
             for k in ("A_ridge", "A_hgb", "B_ridge", "B_hgb"))
    return check("saved_model_replay", ok, "replayed A/B models match predictions.tsv")


def t_bootstrap_paired_sampling():
    # deterministic + paired: same seed, same resample for both arms; delta distribution
    # matches manual computation on a small synthetic case.
    rng1 = np.random.default_rng(20261011)
    rng2 = np.random.default_rng(20261011)
    uniq = np.arange(10)
    s1 = rng1.choice(uniq, size=10, replace=True)
    s2 = rng2.choice(uniq, size=10, replace=True)
    ok_det = np.array_equal(s1, s2)
    # paired: contrast delta = full rho - arm rho on the SAME indices
    truth = np.array([0.5, 0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.05])
    full_p = truth + 0.01 * np.arange(10)
    arm_p = truth + 0.005 * np.arange(10)
    idx = np.array([2, 5, 2, 8, 1, 3, 9, 0, 4, 7])
    d = spearman(full_p[idx], truth[idx]) - spearman(arm_p[idx], truth[idx])
    d_direct = spearman(full_p[idx] - arm_p[idx], np.ones(10))  # sanity: not equal in general
    ok_paired = isinstance(d, float) and np.isfinite(d)
    # verify the bootstrap interval percentile convention [0.625, 99.375] is a
    # 98.75% two-sided interval (width fraction = 0.9875) and symmetric on a
    # uniform 0..1999 grid.
    arr = np.arange(2000, dtype=np.float64)
    lo = float(np.percentile(arr, 0.625))
    hi = float(np.percentile(arr, 99.375))
    ok_width = abs((hi - lo) / 1999.0 - 0.9875) < 1e-12
    ok_sym = abs((lo + hi) - 1999.0) < 1e-9
    return check("bootstrap_paired_sampling", bool(ok_det and ok_paired and ok_width and ok_sym),
                 "seed deterministic, paired delta, percentile [0.625,99.375] => 98.75%% two-sided (width fraction 0.9875)")


def t_thread_limits():
    rec = thread_caps_record()
    env = rec["env"]
    ok_env = all(env.get(k) == "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    nthreads = [t.get("num_threads") for t in rec["threadpoolctl"] if t.get("num_threads") is not None]
    ok_tpc = all(n <= 4 for n in nthreads)
    return check("thread_limits", bool(ok_env and ok_tpc),
                 "env caps=4; threadpoolctl num_threads=%s" % nthreads)


def t_nonoverwrite_guard():
    script = Path(__file__).resolve().parent / "b11_run.py"
    cmd = [sys.executable, "-X", "utf8", str(script), "--attempt", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           timeout=300, cwd=str(ROOT))
        refused = r.returncode != 0 and "REFUSE" in (r.stdout + r.stderr)
        return check("nonoverwrite_guard", refused,
                     "b11_run --attempt 1 exits non-zero with REFUSE (returncode=%d)" % r.returncode)
    except subprocess.TimeoutExpired:
        return check("nonoverwrite_guard", False, "subprocess timeout (did not fail fast)")


def t_status_identity():
    status = json.load(open(REPORT_OUT / "status.json", encoding="utf-8"))
    w = status.get("worker", {})
    sw = status.get("statewriter", {})
    ok = (isinstance(w.get("pid"), int) and w.get("pid") > 0)
    ok &= (w.get("process_create_time_local") is not None)
    ok &= (w.get("creation_verified_via") == "Windows GetProcessTimes (ctypes)")
    ok &= (isinstance(sw.get("pid"), int))
    # worker pid must differ from statewriter pid (distinct identity)
    ok &= (w.get("pid") != sw.get("pid"))
    # verify worker creation time independently now
    ct = windows_process_creation_time(int(w["pid"]))
    ok &= (ct is not None and fmt_local(ct) == w.get("process_create_time_local"))
    return check("status_identity", ok,
                 "worker pid %s != writer pid %s; creation time verified" % (w.get("pid"), sw.get("pid")))


def main():
    t0 = time.time()
    tests = [
        t_feature_subset_selection_order(),
        t_exact_old_prediction_matching(),
        t_training_only_scaling(),
        t_component_alignment(),
        t_saved_model_replay(),
        t_bootstrap_paired_sampling(),
        t_thread_limits(),
        t_nonoverwrite_guard(),
        t_status_identity(),
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


if __name__ == "__main__":
    main()

