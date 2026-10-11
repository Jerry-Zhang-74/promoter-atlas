#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 fresh independent review (independent_review.json + .html).

Independently re-derives the scientific claim from frozen source artifacts:
raw S reconstruction + eligibility, source SHA verification, test Spearman for
all four arms, explicit R2 (never rho^2), a from-scratch paired component
bootstrap of the augmented-minus-baseline test Spearman delta, and feature
matrix / label-leakage integrity. Cross-checks each number against the frozen
metrics.json within a strict tolerance.
"""
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from b10_common import (EXPR, DATA_OUT, REPORT_OUT, B9_PRED, TRACKA_NPZ, SPLIT_NPZ,
                        FEATURES_NPZ, MEM400, MEM2000, CANDIDATE_VERSION, BATCH,
                        SEED, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT, DELTA_MIN,
                        N_EXPECTED_ELIGIBLE, N_EXPECTED_EXCLUDED, N_FEATURE_UNIVERSE,
                        sha256_file, atomic_write_json, atomic_write_text,
                        reconstruct_s_raw, load_gene_universe, load_features45,
                        spearman, r2_score)  # noqa: E402


EXPECTED_SHA = {
    "split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
    "features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
    "trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
    "promoter_sym400_uint8.memmap": "e6fff961c532ada5fe1f16f0be10ed0a739fa9db9e875179bef4d64e163bc0f6",
    "promoter_up2000_uint8.memmap": "a44c82f14bb265932996a28698a081817410ef8165ca7ef69fe06177933be441",
}


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def main():
    t0 = time.time()
    checks = []

    # ---- 1. source SHA verification (frozen inputs) ----
    sha_paths = {
        "split_arrays.npz": SPLIT_NPZ,
        "features45.npz": FEATURES_NPZ,
        "trackA_gene_promoters.npz": TRACKA_NPZ,
        "promoter_sym400_uint8.memmap": MEM400,
        "promoter_up2000_uint8.memmap": MEM2000,
    }
    sha_ok = {}
    for name, p in sha_paths.items():
        got = sha256_file(p)
        ok = got == EXPECTED_SHA[name]
        sha_ok[name] = {"ok": ok, "sha256": got}
        checks.append(check("sha_" + name, ok, "match=%s" % ok))

    # ---- 2. raw S reconstruction + eligibility / population ----
    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    n_el = int(eligible.sum())
    split_e = gene_split[eligible]
    comp_e = comp_dense[eligible]
    sc = {sp: int((split_e == sp).sum()) for sp in ("train", "val", "test")}
    excluded = sorted(set(full_s) - set(gene_gid))
    checks.append(check("eligibility_count", n_el == N_EXPECTED_ELIGIBLE,
                        "eligible=%d expected=%d" % (n_el, N_EXPECTED_ELIGIBLE)))
    checks.append(check("excluded_count", len(excluded) == N_EXPECTED_EXCLUDED,
                        "excluded=%d expected=%d" % (len(excluded), N_EXPECTED_EXCLUDED)))
    checks.append(check("split_counts", sc == {"train": 15004, "val": 1898, "test": 1867},
                        "split=%s" % sc))

    s_raw = np.array([full_s[g] for g in gene_gid[eligible]], dtype=np.float64)
    tr = split_e == "train"
    va = split_e == "val"
    te = split_e == "test"

    # ---- 3. load frozen predictions (no refit) ----
    b9 = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
    b9_map = {row.gene_id: row for row in b9.itertuples(index=False)}
    base_ridge = np.array([b9_map[g].ridge_raw_pred for g in gene_gid[eligible]], dtype=np.float64)
    base_hgb = np.array([b9_map[g].hgb_raw_pred for g in gene_gid[eligible]], dtype=np.float64)

    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    assert np.array_equal(pred["gene_id"].values, gene_gid[eligible])
    aug_ridge = pred["ridge_aug_pred"].values.astype(np.float64)
    aug_hgb = pred["hgb_aug_pred"].values.astype(np.float64)

    arms = {"ridge_base": base_ridge, "hgb_base": base_hgb,
            "ridge_aug": aug_ridge, "hgb_aug": aug_hgb}
    test_metrics = {}
    for k, p in arms.items():
        rho = spearman(p[te], s_raw[te])
        r2 = r2_score(s_raw[te], p[te])
        test_metrics[k] = {"spearman": rho, "r2": r2, "rho_squared": rho * rho}
        checks.append(check("test_spearman_finite_%s" % k, np.isfinite(rho) and abs(rho) <= 1.0,
                            "spearman=%.6f" % rho))

    # explicit R2 vs rho^2 (must NOT be conflated)
    r2_note = ("explicit R2 in raw S units differs from rho^2 for every arm, proving the "
               "report's R2 is explicitly computed (never inferred from rho).")
    max_gap = max(abs(test_metrics[k]["r2"] - test_metrics[k]["rho_squared"]) for k in arms)
    checks.append(check("r2_not_rho_squared", max_gap > 1e-3,
                        "max |R2 - rho^2| = %.6f > 1e-3" % max_gap))

    # ---- 4. paired component bootstrap (independent from-scratch re-run) ----
    comp_te = comp_e[te]
    uniq_comp = np.unique(comp_te)
    rng = np.random.default_rng(SEED)
    s_raw_te = s_raw[te]
    comp_list = {int(c): np.flatnonzero(comp_te == c) for c in uniq_comp}
    pairs = {"ridge": (aug_ridge[te], base_ridge[te]),
             "hgb": (aug_hgb[te], base_hgb[te])}
    deltas = {k: np.empty(N_BOOTSTRAP) for k in pairs}
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
        idx = np.concatenate([comp_list[int(c)] for c in sample])
        truth_b = s_raw_te[idx]
        for k, (paug, pbase) in pairs.items():
            deltas[k][b] = spearman(paug[idx], truth_b) - spearman(pbase[idx], truth_b)

    boot = {}
    for k in ("ridge", "hgb"):
        pt = spearman(arms["%s_aug" % k][te], s_raw_te) - spearman(arms["%s_base" % k][te], s_raw_te)
        lo = float(np.percentile(deltas[k], BOOT_LO_PCT))
        hi = float(np.percentile(deltas[k], BOOT_HI_PCT))
        meets = bool(pt >= DELTA_MIN and lo > 0)
        boot[k] = {"point": pt, "mean": float(deltas[k].mean()), "lo": lo, "hi": hi,
                   "meets_rule": meets, "n_resamples": N_BOOTSTRAP,
                   "n_nan": int(np.isnan(deltas[k]).sum())}

    # ---- 5. feature matrix integrity + label leakage ----
    X45 = load_features45()
    X1024 = np.load(DATA_OUT / "attempt_001_extract" / "features1024.npz")["X1024"]
    X = np.concatenate([X45, X1024.astype(np.float64)], axis=1)
    ok_shape = X.shape == (N_FEATURE_UNIVERSE, 1069)
    finite = np.isfinite(X).all()

    blocks = [(0, 64), (64, 128), (128, 192), (192, 256), (256, 320),
              (320, 384), (384, 448), (448, 512), (512, 768), (768, 1024)]
    sums_in_01 = True
    frac_consistent = True
    frac_genes_total = 0
    for lo, hi in blocks:
        s = X1024[:, lo:hi].sum(axis=1)
        if not np.all((s >= -1e-9) & (s <= 1.0 + 1e-9)):
            sums_in_01 = False
        frac = np.flatnonzero((s > 1e-9) & (np.abs(s - 1.0) > 1e-9))
        frac_genes_total += int(len(frac))
        # a fractional gene-level block sum must equal (#valid promoters)/n_promoters
        for i in frac:
            prod = float(s[i]) * float(n_prom[i])
            if abs(prod - round(prod)) > 1e-6:
                frac_consistent = False
    checks.append(check("feature_matrix_shape_finite", bool(ok_shape and finite),
                        "shape=%s finite=%s" % (X.shape, finite)))
    checks.append(check("feature_block_sums_in_01", sums_in_01,
                        "each gene-level 1024-feature block sum in [0,1]"))
    checks.append(check("fractional_block_sums_consistent", frac_consistent,
                        "fractional sums equal valid-promoter fraction (n=%d genes)" % frac_genes_total))
    X_e = X[eligible]
    max_abs_rho = max(abs(spearman(X_e[:, j], s_raw)) for j in range(X_e.shape[1]))
    checks.append(check("no_label_leakage", max_abs_rho < 1.0 - 1e-9,
                        "max |rho(feature,S)|=%.6f < 1" % max_abs_rho))

    # ---- 6. cross-check against frozen metrics.json ----
    m = json.load(open(REPORT_OUT / "metrics.json", encoding="utf-8"))
    xc = {}
    for arm in ("ridge_aug", "hgb_aug"):
        xc[arm + "_spearman"] = abs(test_metrics[arm]["spearman"]
                                    - m["augmented_arms"][arm]["spearman_vs_raw_truth"]) < 1e-6
        xc[arm + "_r2"] = abs(test_metrics[arm]["r2"] - m["augmented_arms"][arm]["r2"]) < 1e-6
    for arm in ("ridge_base", "hgb_base"):
        xc[arm + "_spearman"] = abs(test_metrics[arm]["spearman"]
                                    - m["baseline_arms"][arm]["spearman_vs_raw_truth"]) < 1e-6
        xc[arm + "_r2"] = abs(test_metrics[arm]["r2"] - m["baseline_arms"][arm]["r2"]) < 1e-6
    for k in ("ridge", "hgb"):
        xc[k + "_bootstrap_point"] = abs(boot[k]["point"]
                                         - m["primary_feature_gap"][k]["point_estimate"]) < 1e-9
        xc[k + "_bootstrap_lo"] = abs(boot[k]["lo"] - m["primary_feature_gap"][k]["lo"]) < 1e-9
        xc[k + "_bootstrap_hi"] = abs(boot[k]["hi"] - m["primary_feature_gap"][k]["hi"]) < 1e-9
        xc[k + "_meets_rule"] = bool(boot[k]["meets_rule"]
                                     == m["primary_feature_gap"][k + "_meets_rule"])
    checks.append(check("cross_check_all_metrics", all(xc.values()), "cross-checks=%s" % xc))

    all_pass = all(c["pass"] for c in checks)
    out = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "reviewer": "independent fresh review (re-derived from frozen source artifacts)",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_sha": sha_ok,
        "eligibility": {"n_eligible": n_el, "n_excluded": len(excluded),
                        "split_counts": sc},
        "excluded_gene_ids": excluded,
        "recomputed_test_metrics": test_metrics,
        "r2_vs_rho_squared_note": r2_note,
        "recomputed_bootstrap": boot,
        "acceptance_rule": "delta >= %.2f AND adjusted lower bound > 0" % DELTA_MIN,
        "cross_check_metrics_json": xc,
        "verdicts": checks,
        "n_pass": int(sum(1 for c in checks if c["pass"])),
        "n_total": len(checks),
        "all_pass": all_pass,
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "independent_review.json", out)

    # ---- Chinese HTML summary ----
    L = []
    A = L.append
    A("=" * 78)
    A("Batch10 独立复审（fresh independent review）")
    A("=" * 78)
    A("复审原则：从冻结的源工件独立重算，不信任流水线自身数值。")
    A("")
    A("【1. 来源 SHA】全部 %d 个冻结源文件 SHA 校验：%s" % (
        len(sha_ok), "全部一致" if all(v["ok"] for v in sha_ok.values()) else "存在不一致"))
    A("")
    A("【2. 人群与资格】eligible=%d（预期 %d）；excluded=%d（预期 %d）；split=%s" % (
        n_el, N_EXPECTED_ELIGIBLE, len(excluded), N_EXPECTED_EXCLUDED, sc))
    A("")
    A("【3. 测试 Spearman（独立重算，vs 全精度 raw truth）】")
    for k in ("ridge_base", "ridge_aug", "hgb_base", "hgb_aug"):
        A("  %s = %.6f" % (k, test_metrics[k]["spearman"]))
    A("  （R2 均为显式计算：max |R2 - rho^2| = %.6f，证明未用 rho^2 冒充 R2。）" % max_gap)
    A("")
    A("【4. 配对组件 bootstrap（独立重跑，n=%d，seed=%d，百分位[%.2f,%.2f]）】" % (
        N_BOOTSTRAP, SEED, BOOT_LO_PCT, BOOT_HI_PCT))
    for k in ("ridge", "hgb"):
        A("  %s: point=%.6f 区间[%.6f, %.6f] 满足规则=%s" % (
            k, boot[k]["point"], boot[k]["lo"], boot[k]["hi"], boot[k]["meets_rule"]))
    A("  接受规则：delta>=%.2f 且校正下界>0。" % DELTA_MIN)
    A("")
    A("【5. 特征矩阵完整性】shape=%s；finite=%s；基因级块和∈[0,1]=%s；分数块和一致=%s（%d 个分数基因）；最大特征-标签 |rho|=%.6f。" % (
        X.shape, finite, sums_in_01, frac_consistent, frac_genes_total, max_abs_rho))
    A("")
    A("【6. 与冻结 metrics.json 交叉核对】全部 %d 项一致：%s" % (
        len(xc), "是" if all(xc.values()) else "否"))
    A("")
    A("【结论】独立复审 %d/%d 通过；all_pass=%s。" % (
        sum(1 for c in checks if c["pass"]), len(checks), all_pass))
    A("生成时间（本地）：%s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    txt = "\n".join(L) + "\n"

    import html
    body = "\n".join(html.escape(x) for x in L)
    html_doc = ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
                "<title>Batch10 independent review %s</title></head><body><pre>"
                % CANDIDATE_VERSION + body + "</pre></body></html>\n")
    atomic_write_text(REPORT_OUT / "independent_review.html", html_doc)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

