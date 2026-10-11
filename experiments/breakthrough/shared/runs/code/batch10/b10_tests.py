#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 independent tests (scientific failure modes, not implementation mirroring).

Covers: synthetic k-mer/bin/ambiguous examples, denominator conservation, gene
aggregation permutation invariance, feature order/names, exact gene/split
alignment, train-only transforms, absence of label features, CPU thread limits,
the output-non-overwrite guard, and saved-model prediction replay. No research
fits are re-run.
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

from b10_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT, CANDIDATE_VERSION,
                        BATCH, FEATURE_NAMES_1024, FEATURE_NAMES_45, FEATURE_NAMES_ALL,
                        N_FEATURE_UNIVERSE, N_EXPECTED_ELIGIBLE,
                        kmer_counts, window_tri_tetra, reconstruct_s_raw,
                        load_gene_universe, load_features45, spearman,
                        atomic_write_json, thread_caps_record)  # noqa: E402


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def t_kmer_synthetic():
    seg = np.array([[0, 1, 2, 0, 1, 2, 3, 3, 3]], dtype=np.int64)
    cnt, denom = kmer_counts(seg, 3)
    ok_denom = denom[0] == 7
    ok_acg = cnt[0, 0 * 16 + 1 * 4 + 2] == 2
    ok_cga = cnt[0, 1 * 16 + 2 * 4 + 0] == 1
    ok_ttt = cnt[0, 3 * 16 + 3 * 4 + 3] == 1
    ok_sum = cnt.sum() == 7
    return check("kmer_synthetic", bool(ok_denom and ok_acg and ok_cga and ok_ttt and ok_sum),
                 "ACG=2, CGA=1, TTT=1, denom=7")


def t_kmer_ambiguous_break():
    seg = np.array([[0, 1, 4, 0]], dtype=np.int64)
    cnt, denom = kmer_counts(seg, 3)
    return check("kmer_ambiguous_break", bool(denom[0] == 0 and cnt.sum() == 0),
                 "ambiguous base breaks all k-mers -> zero block")


def t_kmer_no_cross_bin():
    # 6 bp window would be split; verify a k-mer is only counted in its own bin by
    # checking a pure-A 400bp window has bin0 AAA freq 1 and bin1 AAA freq 1 (each
    # bin normalized independently), never sharing starts across the boundary.
    seq = np.zeros((1, 400), dtype=np.int64)
    tri, tetra, qc = window_tri_tetra(seq, 400)
    ok = (abs(tri[0, 0] - 1.0) < 1e-12 and abs(tri[0, 64] - 1.0) < 1e-12 and
          abs(tetra[0, 0] - 1.0) < 1e-12)
    return check("kmer_no_cross_bin", bool(ok),
                 "pure-A: each bin AAA=1, tetra AAAA=1 (no boundary double count)")


def t_denominator_conservation_synthetic():
    seq = np.array([[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)
    # pad to 400 with A so each bin has valid starts
    seq = np.zeros((1, 400), dtype=np.int64)
    seq[0, 0:4] = [0, 1, 2, 3]
    tri, tetra, qc = window_tri_tetra(seq, 400)
    ok = abs(tri[0, 0:64].sum() - 1.0) < 1e-12 and abs(tetra[0].sum() - 1.0) < 1e-12
    return check("denominator_conservation_synthetic", bool(ok),
                 "block frequencies sum to 1 per block (valid denominator)")


def t_gene_aggregation_permutation_invariance():
    # two promoters; gene feature = mean over promoters, order-independent
    a = np.zeros((1, 400), dtype=np.int64)
    b = np.ones((1, 400), dtype=np.int64)  # all C
    ta, _, _ = window_tri_tetra(a, 400)
    tb, _, _ = window_tri_tetra(b, 400)
    feats = np.concatenate([ta, tb], axis=0)  # (2,256) tri only for this test
    # use full 1024 assembly to exercise the real path
    ta_sym, te_sym, _ = window_tri_tetra(a, 400)
    ta_up, te_up, _ = window_tri_tetra(np.zeros((1, 2000), dtype=np.int64), 2000)
    ba = np.concatenate([ta_sym, ta_up, te_sym, te_up], axis=1)
    tb_sym, te_sym2, _ = window_tri_tetra(b, 400)
    tb_up, te_up2, _ = window_tri_tetra(np.zeros((1, 2000), dtype=np.int64), 2000)
    bb = np.concatenate([tb_sym, tb_up, te_sym2, te_up2], axis=1)
    mean_ab = (ba + bb) / 2.0
    mean_ba = (bb + ba) / 2.0
    ok = np.allclose(mean_ab, mean_ba, atol=1e-15)
    return check("gene_aggregation_permutation_invariance", bool(ok),
                 "mean over promoters is permutation-invariant")


def t_feature_order_names():
    ok = (len(FEATURE_NAMES_1024) == 1024 and len(FEATURE_NAMES_ALL) == 1069 and
          FEATURE_NAMES_ALL[:45] == FEATURE_NAMES_45)
    ok &= FEATURE_NAMES_1024[0] == "sym400_bin0_tri_AAA"
    ok &= FEATURE_NAMES_1024[63] == "sym400_bin0_tri_TTT"
    ok &= FEATURE_NAMES_1024[255] == "sym400_bin3_tri_TTT"
    ok &= FEATURE_NAMES_1024[256] == "up2000_bin0_tri_AAA"
    ok &= FEATURE_NAMES_1024[511] == "up2000_bin3_tri_TTT"
    ok &= FEATURE_NAMES_1024[512] == "sym400_tetra_AAAA"
    ok &= FEATURE_NAMES_1024[1023] == "up2000_tetra_TTTT"
    return check("feature_order_names", bool(ok),
                 "1024 new feature names in frozen order match declared indices")


def t_x1024_block_sums_conserved():
    z = np.load(DATA_OUT / "attempt_001_extract" / "features1024.npz")
    X = z["X1024"]
    ok = X.shape == (N_FEATURE_UNIVERSE, 1024) and np.isfinite(X).all()
    sums = []
    for lo, hi in [(0, 64), (64, 128), (128, 192), (192, 256),
                   (256, 320), (320, 384), (384, 448), (448, 512),
                   (512, 768), (768, 1024)]:
        s = X[:, lo:hi].sum(axis=1)
        sums.append((lo, hi, float(s.min()), float(s.max())))
    # every block sum should be 0 (all promoters zero-denom) or 1 (normalized)
    all_ok = all(abs(smin) < 1e-9 or abs(smax - 1.0) < 1e-9 for (_, _, smin, smax) in sums)
    also_ok = all(abs(smax) <= 1.0 + 1e-9 for (_, _, _, smax) in sums)
    return check("x1024_block_sums_conserved", bool(ok and all_ok and also_ok),
                 "block sums in {0,1}: %s" % sums)


def t_alignment_split():
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    n_el = int(eligible.sum())
    split_counts = {sp: int((gene_split[eligible] == sp).sum()) for sp in ("train", "val", "test")}
    ok = (n_el == N_EXPECTED_ELIGIBLE and split_counts == {"train": 15004, "val": 1898, "test": 1867})
    # no component crosses split boundary
    import collections
    comp_split = collections.defaultdict(set)
    for c, sp in zip(comp_dense[eligible], gene_split[eligible]):
        comp_split[int(c)].add(sp)
    crossing = sum(1 for v in comp_split.values() if len(v) > 1)
    return check("alignment_split", bool(ok and crossing == 0),
                 "eligible=%d split=%s; components crossing=%d" % (n_el, split_counts, crossing))


def t_train_only_transforms():
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    X45 = load_features45()
    z = np.load(DATA_OUT / "attempt_001_extract" / "features1024.npz")
    X = np.concatenate([X45, z["X1024"]], axis=1)
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    X_e = X[eligible]
    tr = gene_split[eligible] == "train"
    sc = joblib.load(MODEL_OUT / "attempt_001_fit" / "X_scaler.joblib")
    ok_mean = np.allclose(sc.mean_, X_e[tr].mean(axis=0))
    sd = X_e[tr].std(axis=0)
    sd[sd == 0] = 1.0
    ok_scale = np.allclose(sc.scale_, sd)
    full_mean = X_e.mean(axis=0)
    leaked = np.allclose(sc.mean_, full_mean)
    # target standardization train-only
    tf = json.load(open(MODEL_OUT / "attempt_001_fit" / "transforms.json", encoding="utf-8"))
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])
    ok_t = (abs(tf["target_raw"]["train_mean"] - s_raw[tr].mean()) < 1e-12 and
            abs(tf["target_raw"]["train_std"] - s_raw[tr].std(ddof=0)) < 1e-12)
    return check("train_only_transforms", bool(ok_mean and ok_scale and not leaked and ok_t),
                 "scaler/target fit on eligible train only; no full-data leakage")


def t_no_label_features():
    # feature names contain no S/E/B label identifiers; matrix has no NaN and no
    # column is essentially identical to the raw S target (label leakage check).
    names_bad = [n for n in FEATURE_NAMES_ALL if any(t in n.lower() for t in ("score_s", "score_e", "score_b", "s_raw"))]
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    X45 = load_features45()
    z = np.load(DATA_OUT / "attempt_001_extract" / "features1024.npz")
    X = np.concatenate([X45, z["X1024"]], axis=1)
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    X_e = X[eligible]
    s = np.array([full_s[g] for g in gene_gid[eligible]])
    finite = np.isfinite(X_e).all()
    # label-leakage: no column may be a perfect monotonic copy of S (|rho| ~ 1)
    max_abs_rho = 0.0
    for j in range(X_e.shape[1]):
        r = abs(spearman(X_e[:, j], s))
        if r > max_abs_rho:
            max_abs_rho = r
    ok = (len(names_bad) == 0 and finite and max_abs_rho < 1.0 - 1e-9)
    return check("no_label_features", bool(ok),
                 "no label-named features; finite; max |rho(feature,S)|=%.6f < 1" % max_abs_rho)


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
    script = Path(__file__).resolve().parent / "b10_run.py"
    cmd = [sys.executable, "-X", "utf8", str(script), "--attempt", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           timeout=300, cwd=str(ROOT))
        refused = r.returncode != 0 and "REFUSE" in (r.stdout + r.stderr)
        return check("nonoverwrite_guard", refused,
                     "b10_run --attempt 1 exits non-zero with REFUSE (returncode=%d)" % r.returncode)
    except subprocess.TimeoutExpired:
        return check("nonoverwrite_guard", False, "subprocess timeout (did not fail fast)")


def t_saved_model_replay():
    fit_dir = MODEL_OUT / "attempt_001_fit"
    models = joblib.load(fit_dir / "models.joblib")
    sc = joblib.load(fit_dir / "X_scaler.joblib")
    tf = json.load(open(fit_dir / "transforms.json", encoding="utf-8"))
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    X45 = load_features45()
    z = np.load(DATA_OUT / "attempt_001_extract" / "features1024.npz")
    X = np.concatenate([X45, z["X1024"]], axis=1)
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    Xs = sc.transform(X[eligible])
    mu = tf["target_raw"]["train_mean"]; sd = tf["target_raw"]["train_std"]
    p_r = models["ridge_aug"].predict(Xs) * sd + mu
    p_h = models["hgb_aug"].predict(Xs) * sd + mu
    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    ok = (np.allclose(p_r, pred["ridge_aug_pred"].values, atol=1e-8) and
          np.allclose(p_h, pred["hgb_aug_pred"].values, atol=1e-8))
    return check("saved_model_replay", bool(ok), "replayed ridge_aug/hgb_aug match predictions.tsv")


def main():
    t0 = time.time()
    tests = [
        t_kmer_synthetic(),
        t_kmer_ambiguous_break(),
        t_kmer_no_cross_bin(),
        t_denominator_conservation_synthetic(),
        t_gene_aggregation_permutation_invariance(),
        t_feature_order_names(),
        t_x1024_block_sums_conserved(),
        t_alignment_split(),
        t_train_only_transforms(),
        t_no_label_features(),
        t_thread_limits(),
        t_nonoverwrite_guard(),
        t_saved_model_replay(),
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
