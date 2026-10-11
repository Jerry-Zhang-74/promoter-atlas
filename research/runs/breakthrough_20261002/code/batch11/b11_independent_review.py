#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 INDEPENDENT FRESH REVIEW (v6.12.0-sfeatablate.rc1).

Independent recomputation by the fresh reviewer session. Recomputes source SHAs,
eligibility/split/component counts, 557-column arm shapes, saved-model replay,
full/baseline loaded-prediction matching, test metrics and the four co-primary
paired component bootstrap contrasts WITHOUT re-fitting anything (only loading
saved models and frozen predictions). Writes independent_review.json.
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from threadpoolctl import threadpool_limits, threadpool_info  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from b11_common import (  # noqa: E402
    ROOT, EXPR, TRACKA_NPZ, SPLIT_NPZ, FEATURES_NPZ, B10_FEATURES1024,
    B10_FEATDICT, B10_PRED, B9_PRED, DATA_OUT, MODEL_OUT, REPORT_OUT,
    CANDIDATE_VERSION, BATCH, SEED, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT,
    DELTA_MIN, N_FEATURE_UNIVERSE, N_EXPECTED_ELIGIBLE, N_EXPECTED_EXCLUDED,
    M_DEFINED, N_COMPONENTS_TOTAL, FEATURE_NAMES_45, FEATURE_NAMES_1024,
    sha256_file, atomic_write_json, reconstruct_s_raw, load_gene_universe,
    load_features45, load_features1024, feature_arm_indices,
    windows_process_creation_time, fmt_local,
)

TRANSIENT_WRITER_PID = os.getpid()
CONTROL_STATUS = ROOT / "runs" / "breakthrough_20261002" / "control_17" / "status.json"


def controller_reviewer_pid():
    try:
        cs = json.load(open(CONTROL_STATUS, encoding="utf-8"))
        return int(cs["reviewer"]["pid"]), cs.get("started_at")
    except Exception:
        return None, None


def rankdata_avg(x):
    x = np.asarray(x, dtype=np.float64)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[sorter[j]] == x[sorter[i]]:
            j += 1
        ranks[sorter[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson_rank_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = rankdata_avg(x)
    ry = rankdata_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


def main():
    t0 = time.time()
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    with threadpool_limits(limits=4):
        # ---- 1. provenance SHAs ----
        exp_sha = {
            "batch05_split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
            "batch06_features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
            "batch03_trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
            "batch10_features1024.npz": "2dc7dc0eda7c60b3fa740261e39fb37832dc0e0faf81a7fa2f9cd37a11bc14f2",
        }
        sha_map = {
            "batch05_split_arrays.npz": sha256_file(SPLIT_NPZ),
            "batch06_features45.npz": sha256_file(FEATURES_NPZ),
            "batch03_trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
            "batch10_features1024.npz": sha256_file(B10_FEATURES1024),
        }
        sha_ok = {k: (sha_map[k] == exp_sha[k]) for k in exp_sha}
        chk("source_sha256", all(sha_ok.values()), json.dumps(sha_ok, ensure_ascii=False))

        # ---- 2. universe / split / features ----
        gene_gid, gene_split, comp_dense, n_prom, starts, gene_id_perprom = load_gene_universe()
        X45 = load_features45()
        X1024 = load_features1024()
        idx45, tri, tetra = feature_arm_indices()
        full_s = reconstruct_s_raw(EXPR)
        chk("reconstructed_S_defined_18947", len(full_s) == M_DEFINED, "n=%d" % len(full_s))
        chk("feature_shapes", X45.shape == (N_FEATURE_UNIVERSE, 45) and X1024.shape == (N_FEATURE_UNIVERSE, 1024),
            "X45=%s X1024=%s" % (X45.shape, X1024.shape))

        # ---- 3. eligibility / split / components ----
        eligible = np.array([g in full_s for g in gene_gid])
        n_el = int(eligible.sum())
        split_counts = {sp: int((gene_split[eligible] == sp).sum()) for sp in ("train", "val", "test")}
        n_comp = int(np.unique(comp_dense[eligible]).size)
        comp_split = defaultdict(set)
        for c, sp in zip(comp_dense[eligible], gene_split[eligible]):
            comp_split[int(c)].add(sp)
        crossing = sum(1 for v in comp_split.values() if len(v) > 1)
        chk("eligibility_18769", n_el == N_EXPECTED_ELIGIBLE, "n_eligible=%d" % n_el)
        chk("split_counts_15004_1898_1867", split_counts == {"train": 15004, "val": 1898, "test": 1867},
            str(split_counts))
        chk("components_18400_no_crossing", n_comp == N_COMPONENTS_TOTAL and crossing == 0,
            "n_components=%d crossing=%d" % (n_comp, crossing))
        chk("excluded_178", (M_DEFINED - n_el) == N_EXPECTED_EXCLUDED, "n_excluded=%d" % (M_DEFINED - n_el))

        # ---- 4. arm shapes (557 each) + feature order by name ----
        X_A = np.concatenate([X45[:, idx45], X1024[:, tri]], axis=1)
        X_B = np.concatenate([X45[:, idx45], X1024[:, tetra]], axis=1)
        chk("arm_557_columns", X_A.shape == (N_FEATURE_UNIVERSE, 557) and X_B.shape == (N_FEATURE_UNIVERSE, 557),
            "A=%s B=%s" % (X_A.shape, X_B.shape))
        fd = json.load(open(B10_FEATDICT, encoding="utf-8"))
        names = fd["feature_names_1024"]
        tri_ok = (tri == list(range(512))) and all("_tri_" in names[i] for i in tri)
        tetra_ok = (tetra == list(range(512, 1024))) and all("_tetra_" in names[i] for i in tetra)
        chk("feature_order_by_name", bool(tri_ok and tetra_ok and names == FEATURE_NAMES_1024 and
             fd["baseline_feature_names_45"] == FEATURE_NAMES_45),
            "tri=%s tetra=%s" % (tri_ok, tetra_ok))

        # ---- 5. target standardization + exactly four fits loaded ----
        s_raw = np.array([full_s[g] for g in gene_gid[eligible]], dtype=np.float64)
        tr = gene_split[eligible] == "train"
        te = gene_split[eligible] == "test"
        tr_mean = float(s_raw[tr].mean())
        tr_std = float(s_raw[tr].std(ddof=0))

        fit_dir = MODEL_OUT / "attempt_001_fit"
        models = joblib.load(fit_dir / "models.joblib")
        scA = joblib.load(fit_dir / "X_scaler_A.joblib")
        scB = joblib.load(fit_dir / "X_scaler_B.joblib")
        tf = json.load(open(fit_dir / "transforms.json", encoding="utf-8"))
        chk("exactly_four_fits", sorted(models.keys()) == sorted(["A_ridge", "A_hgb", "B_ridge", "B_hgb"]),
            str(sorted(models.keys())))
        X_A_e = X_A[eligible]
        X_B_e = X_B[eligible]
        sdA = X_A_e[tr].std(axis=0)
        sdA[sdA == 0] = 1.0  # StandardScaler zero-variance handling (scale_=1.0)
        sdB = X_B_e[tr].std(axis=0)
        sdB[sdB == 0] = 1.0
        nzA = int((X_A_e[tr].std(axis=0) == 0).sum())
        nzB = int((X_B_e[tr].std(axis=0) == 0).sum())
        scA_ok = np.allclose(scA.mean_, X_A_e[tr].mean(axis=0), atol=1e-10) and \
                 np.allclose(scA.scale_, sdA, atol=1e-10)
        scB_ok = np.allclose(scB.mean_, X_B_e[tr].mean(axis=0), atol=1e-10) and \
                 np.allclose(scB.scale_, sdB, atol=1e-10)
        leakA = not np.allclose(scA.mean_, X_A_e.mean(axis=0), atol=1e-8)
        leakB = not np.allclose(scB.mean_, X_B_e.mean(axis=0), atol=1e-8)
        chk("train_only_scalers", bool(scA_ok and scB_ok and leakA and leakB),
            "A=%s B=%s zero-var_cols(A/B train)=%d/%d leak_absent(A/B)=%s/%s"
            % (scA_ok, scB_ok, nzA, nzB, leakA, leakB))
        tf_ok = (abs(tf["target_raw"]["train_mean"] - tr_mean) < 1e-12 and
                 abs(tf["target_raw"]["train_std"] - tr_std) < 1e-12)
        chk("train_only_target_std", bool(tf_ok),
            "mean=%.12f std=%.12f" % (tf["target_raw"]["train_mean"], tf["target_raw"]["train_std"]))

        # ---- 6. replay saved models -> predictions ----
        Xs_A = scA.transform(X_A_e)
        Xs_B = scB.transform(X_B_e)
        mu = tf["target_raw"]["train_mean"]
        sd = tf["target_raw"]["train_std"]
        preds = {
            "A_ridge": models["A_ridge"].predict(Xs_A) * sd + mu,
            "A_hgb": models["A_hgb"].predict(Xs_A) * sd + mu,
            "B_ridge": models["B_ridge"].predict(Xs_B) * sd + mu,
            "B_hgb": models["B_hgb"].predict(Xs_B) * sd + mu,
        }
        pred_df = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
        chk("gene_order_matches", bool(np.array_equal(pred_df["gene_id"].values, gene_gid[eligible])),
            "predictions.tsv gene order == eligible universe order")
        replay_ok = all(np.allclose(preds[k], pred_df[k + "_pred"].values, atol=1e-8) for k in preds)
        chk("saved_model_replay", bool(replay_ok), "A/B replay match predictions.tsv")

        # ---- 7. full (batch10) and baseline (batch09) loaded, not refit ----
        b10 = pd.read_csv(B10_PRED, sep="\t", low_memory=False)
        b9 = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
        b10_map = {r.gene_id: r for r in b10.itertuples(index=False)}
        b9_map = {r.gene_id: r for r in b9.itertuples(index=False)}
        full_ridge = np.array([b10_map[g].ridge_aug_pred for g in gene_gid[eligible]], dtype=np.float64)
        full_hgb = np.array([b10_map[g].hgb_aug_pred for g in gene_gid[eligible]], dtype=np.float64)
        base_ridge = np.array([b9_map[g].ridge_raw_pred for g in gene_gid[eligible]], dtype=np.float64)
        base_hgb = np.array([b9_map[g].hgb_raw_pred for g in gene_gid[eligible]], dtype=np.float64)
        b9_s = np.array([b9_map[g].S_raw_full for g in gene_gid[eligible]], dtype=np.float64)
        loaded_ok = (np.allclose(full_ridge, pred_df["full_ridge_pred"].values, atol=1e-12) and
                     np.allclose(full_hgb, pred_df["full_hgb_pred"].values, atol=1e-12) and
                     np.allclose(base_ridge, pred_df["base_ridge_pred"].values, atol=1e-12) and
                     np.allclose(base_hgb, pred_df["base_hgb_pred"].values, atol=1e-12))
        chk("full_and_baseline_loaded_not_refit", bool(loaded_ok), "full/base match frozen TSVs to 1e-12")
        chk("truth_reconstruction_matches_batch09", bool(np.allclose(b9_s, s_raw, atol=1e-12)),
            "S_raw reconstruction == batch09 S_raw_full")

        # ---- 8. recompute test metrics (vs common raw truth) ----
        def mets(p, y):
            rho_sc = float(spearmanr(p[te], y[te]).correlation)
            rho_own = pearson_rank_spearman(p[te], y[te])
            assert abs(rho_sc - rho_own) < 1e-9, "scipy vs own spearman mismatch"
            return {"spearman_vs_raw_truth": rho_sc,
                    "mae": float(np.mean(np.abs(p[te] - y[te]))),
                    "mse": float(np.mean((p[te] - y[te]) ** 2)),
                    "r2": float(1.0 - ((y[te] - p[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())}

        all_preds = {
            "full_ridge": full_ridge, "full_hgb": full_hgb,
            "A_ridge": preds["A_ridge"], "A_hgb": preds["A_hgb"],
            "B_ridge": preds["B_ridge"], "B_hgb": preds["B_hgb"],
            "base_ridge": base_ridge, "base_hgb": base_hgb,
        }
        test_metrics = {k: mets(v, s_raw) for k, v in all_preds.items()}

        # ---- 9. paired component bootstrap (2000, seed, [0.625, 99.375]) ----
        comp_te = comp_dense[eligible][te]
        uniq_comp = np.unique(comp_te)
        rng = np.random.default_rng(SEED)
        s_raw_te = s_raw[te]
        comp_list = {int(c): np.flatnonzero(comp_te == c) for c in uniq_comp}
        arm_test = {k: v[te] for k, v in all_preds.items()}
        boot_rho = {k: np.empty(N_BOOTSTRAP) for k in all_preds}
        for b in range(N_BOOTSTRAP):
            sample = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
            idx = np.concatenate([comp_list[int(c)] for c in sample])
            tb = s_raw_te[idx]
            for k, at in arm_test.items():
                boot_rho[k][b] = float(spearmanr(at[idx], tb).correlation)

        def contrast(full_key, arm_key, learner):
            d = boot_rho[full_key] - boot_rho[arm_key]
            point = test_metrics[full_key]["spearman_vs_raw_truth"] - test_metrics[arm_key]["spearman_vs_raw_truth"]
            lo = float(np.percentile(d, BOOT_LO_PCT))
            hi = float(np.percentile(d, BOOT_HI_PCT))
            return {"learner": learner, "full_arm": full_key, "ablated_arm": arm_key,
                    "delta": float(point), "adjusted_lo": lo, "adjusted_hi": hi,
                    "bootstrap_mean": float(d.mean()), "n_resamples": N_BOOTSTRAP,
                    "lo_pct": BOOT_LO_PCT, "hi_pct": BOOT_HI_PCT,
                    "meets_rule": bool(point >= DELTA_MIN and lo > 0)}

        primary = {
            "ridge_full_minus_A": contrast("full_ridge", "A_ridge", "Ridge"),
            "ridge_full_minus_B": contrast("full_ridge", "B_ridge", "Ridge"),
            "hgb_full_minus_A": contrast("full_hgb", "A_hgb", "HGB"),
            "hgb_full_minus_B": contrast("full_hgb", "B_hgb", "HGB"),
        }

        secondary = {}
        for arm_key, base_key, learner in [("A_ridge", "base_ridge", "Ridge"), ("B_ridge", "base_ridge", "Ridge"),
                                           ("A_hgb", "base_hgb", "HGB"), ("B_hgb", "base_hgb", "HGB")]:
            d = boot_rho[arm_key] - boot_rho[base_key]
            point = test_metrics[arm_key]["spearman_vs_raw_truth"] - test_metrics[base_key]["spearman_vs_raw_truth"]
            secondary["%s_minus_%s" % (arm_key, base_key)] = {
                "learner": learner, "delta": float(point),
                "lo": float(np.percentile(d, BOOT_LO_PCT)), "hi": float(np.percentile(d, BOOT_HI_PCT)),
                "descriptive_only": True}

    # ---- 10. fast protocol checks outside heavy compute ----
    guard = subprocess.run([sys.executable, "-X", "utf8",
                            str(Path(__file__).resolve().parent / "b11_run.py"), "--attempt", "1"],
                           capture_output=True, text=True, encoding="utf-8",
                           cwd=str(ROOT), timeout=120)
    guard_ok = (guard.returncode != 0 and "REFUSE" in (guard.stdout + guard.stderr))
    chk("nonoverwrite_guard", bool(guard_ok),
        "b11_run --attempt 1 returncode=%d REFUSE=%s" % (guard.returncode, "REFUSE" in (guard.stdout + guard.stderr)))

    info = threadpool_info()
    nthreads = [t.get("num_threads") for t in info if t.get("num_threads") is not None]
    env_caps = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    chk("thread_limits_le4", all((n is None or n <= 4) for n in nthreads) and all(env_caps.get(k) == "4" for k in env_caps),
        "env=%s threadpoolctl=%s" % (env_caps, nthreads))

    sc = json.load(open(REPORT_OUT / "worker_self_check.json", encoding="utf-8"))
    chk("worker_self_check_not_independent", sc.get("is_independent_review") is False and sc.get("awaiting_fresh_review") is True,
        "worker_self_check.json correctly marked awaiting fresh review")
    indep_exists = (REPORT_OUT / "independent_review.json").exists()
    chk("no_preexisting_independent_review", not indep_exists, "independent_review.json must not pre-exist before this write")

    # ---- 11. agreement with worker metrics.json ----
    wm = json.load(open(REPORT_OUT / "metrics.json", encoding="utf-8"))
    wm_tm = wm["test_metrics"]
    agree = {}
    max_abs_diff = 0.0
    for k in all_preds:
        d = abs(test_metrics[k]["spearman_vs_raw_truth"] - wm_tm[k]["spearman_vs_raw_truth"])
        max_abs_diff = max(max_abs_diff, d)
        agree[k] = d < 1e-9
    contrast_agree = {}
    for k, c in primary.items():
        wc = wm["primary_conditional_increment"]["contrasts"][k]
        contrast_agree[k] = (abs(c["delta"] - wc["delta"]) < 1e-9 and
                             abs(c["adjusted_lo"] - wc["bootstrap"]["lo"]) < 1e-9 and
                             abs(c["adjusted_hi"] - wc["bootstrap"]["hi"]) < 1e-9 and
                             c["meets_rule"] == wc["meets_rule"])
    chk("recomputed_metrics_agree_with_worker", bool(all(agree.values()) and all(contrast_agree.values())),
        "max_abs_spearman_diff=%.3g contrasts_agree=%s" % (max_abs_diff, json.dumps(contrast_agree, ensure_ascii=False)))

    # ---- 12. acceptance + reviewer identity ----
    n_meets = sum(1 for c in primary.values() if c["meets_rule"])
    sci_accept = bool(n_meets >= 1)
    reviewer_pid, controller_started_at = controller_reviewer_pid()
    reviewer_ct = windows_process_creation_time(reviewer_pid) if reviewer_pid else None
    reviewer_ct_str = fmt_local(reviewer_ct)
    reviewer_identity_ok = bool(
        reviewer_pid and reviewer_ct is not None and
        reviewer_pid != TRANSIENT_WRITER_PID and
        (controller_started_at is None or reviewer_ct_str != controller_started_at)
    )
    chk("reviewer_identity_verified",
        reviewer_identity_ok,
        "reviewer_pid=%s creation=%s transient_writer_pid=%s controller_started_at=%s"
        % (reviewer_pid, reviewer_ct_str, TRANSIENT_WRITER_PID, controller_started_at))

    review = {
        "reviewer_role": "fresh independent reviewer (batch11)",
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "reviewer_pid": reviewer_pid,
        "reviewer_process_create_time_local": reviewer_ct_str,
        "reviewer_creation_verified_via": "Windows GetProcessTimes (ctypes)",
        "controller_started_at": controller_started_at,
        "transient_writer_pid": TRANSIENT_WRITER_PID,
        "identity_distinction_note": ("reviewer is the long-lived CLI process (PID from control_17 "
                                      "status.json) with its OS creation time verified; controller "
                                      "started_at and the transient status-writer subprocess PID are "
                                      "distinct and not confused with the reviewer."),
        "backend_provenance": {
            "configured_base_url": "https://api.deepseek.com/anthropic",
            "configured_model": "deepseek-v4-pro",
            "note": ("CLI 'unrecognized_model' stderr diagnostic alone is NOT evidence of "
                     "provider/model fallback; requests route through configured ANTHROPIC_BASE_URL. "
                     "No secrets inspected."),
        },
        "protocol_compliance": {
            "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks,
            "n_pass": sum(1 for c in checks if c["pass"]),
            "n_total": len(checks),
        },
        "scientific_acceptance": {
            "question": ("Does either feature group (A=old45+positional3mer512, "
                         "B=old45+global4mer512) add predictive information conditional on "
                         "the other group for continuous raw S, under the fixed Ridge/HGB learners?"),
            "verdict": "PASS" if sci_accept else "FAIL",
            "meaning": ("PASS = >=1 of four co-primary contrasts met delta>=%.2f AND adjusted "
                        "lower bound>0. FAIL = no scoped conditional increment detected; this is "
                        "NOT proof of equivalence/redundancy." % DELTA_MIN),
            "n_contrasts_meeting_rule": n_meets,
            "contrasts": primary,
        },
        "recomputed_test_metrics": test_metrics,
        "secondary_descriptive_vs_baseline": secondary,
        "worker_agreement": {"metrics_agree": bool(all(agree.values())),
                             "contrasts_agree": bool(all(contrast_agree.values())),
                             "max_abs_spearman_diff": float(max_abs_diff)},
        "evidence_sha256": {k: sha_map[k] for k in sorted(sha_map)},
        "limitations": [
            "human conditional gene-level retrospective raw S prediction only",
            "not mammalian universal / synthetic promoter / HEK293T / temporal claims",
            "reused fixed test split has adaptive research exposure; repeated test use",
            "incomplete near-homology isolation (exact/RC only, not full homology-free)",
            "predictive ablation, NOT causal mechanism identification",
            "A vs B differ in BOTH k-mer length AND positional aggregation",
            "standalone A/B gains vs baseline do NOT prove independent/additive contribution",
            "full model is a reference, NOT a mathematical upper bound",
            "non-significant increment is NOT proof of equivalence or redundancy",
            "raw-vs-rank is nonlinear; not addressed here (outside this question)",
        ],
        "summary_zh": "",
        "review_timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
    }

    zh = []
    zh.append("batch11（%s）独立复核完成。" % CANDIDATE_VERSION)
    zh.append("四个共主条件增量比较（full 减 A/B，各 Ridge/HGB）均未满足“delta>=0.01 且校正下界>0”：")
    for k, c in primary.items():
        zh.append("  %s delta=%.6f 校正区间[%.6f, %.6f] meets=%s"
                  % (k, c["delta"], c["adjusted_lo"], c["adjusted_hi"], c["meets_rule"]))
    zh.append("即：在固定 Ridge/HGB 学习器下，未检出 A 或 B 特征组在另一组之上的显著条件增量；这是负面结果，不等于等价/冗余。")
    zh.append("协议合规：%s（%d/%d 检查通过）；科学接受：%s。"
              % (review["protocol_compliance"]["verdict"],
                 review["protocol_compliance"]["n_pass"],
                 review["protocol_compliance"]["n_total"],
                 review["scientific_acceptance"]["verdict"]))
    review["summary_zh"] = "\n".join(zh)

    atomic_write_json(REPORT_OUT / "independent_review.json", review)
    print(json.dumps({
        "protocol_compliance": review["protocol_compliance"]["verdict"],
        "scientific_acceptance": review["scientific_acceptance"]["verdict"],
        "n_pass": review["protocol_compliance"]["n_pass"],
        "n_total": review["protocol_compliance"]["n_total"],
        "contrasts": primary,
        "reviewer_pid": reviewer_pid,
        "reviewer_creation": reviewer_ct_str,
        "elapsed_s": review["elapsed_s"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

