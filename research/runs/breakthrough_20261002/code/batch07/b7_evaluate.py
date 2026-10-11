#!/usr/bin/env python3
"""Batch07 evaluation: fixed coverage gate vs frozen HGB (primary) + references.

Loads the FROZEN geometry artifact (verifies its SHA), applies the pre-registered
90th-percentile train gate to heldout genes, and reports full/retained/rejected
metrics, component bootstrap, random-mask baseline, diagnostic quantiles and
descriptive strata. No predictor refit, no threshold grid, no GPU.
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    REPORT, MODEL_DIR, MODEL_DIR06, atomic_write_json, sha256_file,
    metrics_dict, high_error_rate, describe_range, mahalanobis_sq,
    component_bootstrap_gated_mae_diff, fast_spearman, mae, r2, mse,
    CANDIDATE, PRIMARY_QUANTILE, DIAGNOSTIC_QUANTILES,
    MIN_TEST_COVERAGE, MIN_REL_MAE_REDUCTION, BOOT_N, BOOT_SEED,
    RANDOM_MASK_N, RANDOM_MASK_SEED, HIGH_ERROR_THRESHOLD,
)
from b7_data import (  # noqa: E402
    load_labels_split, load_features, load_saved_predictions, build_split_rows,
)


def gated_metrics(y, p, dist, threshold):
    """Full/retained/rejected metrics for one predictor under one fixed gate."""
    retained = dist <= threshold
    rejected = ~retained

    def m(yy, pp):
        d = metrics_dict(yy, pp)
        d["high_error_rate_gt_020"] = high_error_rate(yy, pp, HIGH_ERROR_THRESHOLD)
        d["target_range"] = describe_range(yy)
        d["pred_range"] = describe_range(pp)
        return d

    out = {
        "n_full": int(len(y)),
        "n_retained": int(retained.sum()),
        "n_rejected": int(rejected.sum()),
        "coverage": float(retained.mean()),
        "full": m(y, p),
        "retained": m(y[retained], p[retained]) if retained.sum() else None,
        "rejected": m(y[rejected], p[rejected]) if rejected.sum() else None,
    }
    return out, retained


def random_mask_maes(y, p, K, n=RANDOM_MASK_N, seed=RANDOM_MASK_SEED):
    """Deterministic fixed-seed random retention masks of EXACTLY K rows."""
    rng = np.random.default_rng(seed)
    N = len(y)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        perm = rng.permutation(N)
        mask = np.zeros(N, dtype=bool)
        mask[perm[:K]] = True
        out[i] = mae(p[mask], y[mask])
    return out


def fixed_n_strata(y, p, n):
    return {
        "n=1": metrics_dict(y[n == 1], p[n == 1]) if (n == 1).sum() >= 2 else None,
        "n=2-3": metrics_dict(y[(n >= 2) & (n < 4)], p[(n >= 2) & (n < 4)]) if ((n >= 2) & (n < 4)).sum() >= 2 else None,
        "n>=4": metrics_dict(y[n >= 4], p[n >= 4]) if (n >= 4).sum() >= 2 else None,
    }


def gc_tertile_strata(y, p, gc, edges):
    lo = gc <= edges[0]
    mid = (gc > edges[0]) & (gc <= edges[1])
    hi = gc > edges[1]
    return {
        "gc_tertile_1": metrics_dict(y[lo], p[lo]) if lo.sum() >= 2 else None,
        "gc_tertile_2": metrics_dict(y[mid], p[mid]) if mid.sum() >= 2 else None,
        "gc_tertile_3": metrics_dict(y[hi], p[hi]) if hi.sum() >= 2 else None,
    }


def main():
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    X45, comp44, log1p_count, gc_gene = load_features()

    geom = np.load(MODEL_DIR / "covergate_geometry.npz")
    geom_sha = sha256_file(MODEL_DIR / "covergate_geometry.npz")
    geom_meta = json.loads((MODEL_DIR / "covergate_geometry.json").read_text(encoding="utf-8"))
    assert geom_sha == geom_meta["geometry_sha256"], "geometry SHA mismatch"
    mean = geom["mean"]; scale = geom["scale"]
    location = geom["location"]; precision = geom["precision"]
    threshold = float(geom["threshold"])
    diag_q = geom["diagnostic_quantiles"]
    diag_thr = geom["diagnostic_thresholds"]

    import joblib
    hgb = joblib.load(MODEL_DIR06 / "hgb_baseline.joblib")
    hgb_pred_all = hgb.predict(X45)
    fr = np.load(MODEL_DIR06 / "finalridge.npz")
    coef = fr["coef"]; intercept = fr["intercept"]; mu = fr["mu"]; sd = fr["sd"]
    fr_pred_all = ((X45 - mu) / sd) @ coef + intercept

    b6, b5r_map, b5b2_map = load_saved_predictions()

    Xs_all = (X45.astype(np.float64) - mean) / scale
    dist_all = mahalanobis_sq(Xs_all, location, precision)

    train_idx = np.flatnonzero(gene_split == "train")
    gc_edges = [float(x) for x in np.quantile(gc_gene[train_idx], [1.0 / 3.0, 2.0 / 3.0])]
    count_edges = [float(x) for x in np.quantile(n_prom[train_idx], [1.0 / 3.0, 2.0 / 3.0])]

    results = {
        "candidate_version": CANDIDATE,
        "geometry_sha256": geom_sha,
        "primary_gate": {"quantile": PRIMARY_QUANTILE, "threshold": threshold,
                         "quantile_method": "linear"},
        "diagnostic_quantiles": {f"{q:.2f}": float(t) for q, t in zip(diag_q, diag_thr)},
        "strata_thresholds_train": {"gc_tertile_edges": gc_edges, "count_tertile_edges": count_edges},
        "by_split": {},
    }

    raw_rows = []

    for split_name in ["val", "test"]:
        gid, y, comp, nsub, idx, _ = build_split_rows(split_name)
        dist = dist_all[idx]
        hgbp = hgb_pred_all[idx]
        frp = fr_pred_all[idx]
        b2 = np.array([b5b2_map.get(g, np.nan) for g in gid], dtype=float)
        cfm = np.array([b6[g]["combined_fixedmean"] for g in gid], dtype=float)

        split_res = {"n_genes": int(len(y)), "n_components": int(len(np.unique(comp)))}

        prim, prim_mask = gated_metrics(y, hgbp, dist, threshold)
        split_res["hgb_primary_gate"] = prim
        split_res["hgb_primary_retained_mask"] = prim_mask.astype(np.uint8).tolist()

        diag = {}
        for q, t in zip(diag_q, diag_thr):
            gm, _ = gated_metrics(y, hgbp, dist, float(t))
            diag[f"{q:.2f}"] = gm
        split_res["hgb_diagnostic_quantiles"] = diag

        retained = prim_mask
        same_row = {
            "hgb": metrics_dict(y[retained], hgbp[retained]) if retained.sum() else None,
            "finalridge": metrics_dict(y[retained], frp[retained]) if retained.sum() else None,
            "b2_original": metrics_dict(y[retained], b2[retained]) if retained.sum() else None,
            "combined_fixedmean": metrics_dict(y[retained], cfm[retained]) if retained.sum() else None,
        }
        preds_by_name = {"hgb": hgbp, "finalridge": frp, "b2_original": b2, "combined_fixedmean": cfm}
        for k in preds_by_name:
            if same_row[k] is not None and retained.sum():
                same_row[k]["high_error_rate_gt_020"] = high_error_rate(
                    y[retained], preds_by_name[k][retained], HIGH_ERROR_THRESHOLD)
        split_res["same_row_retained"] = same_row

        split_res["hgb_promoter_count_strata"] = fixed_n_strata(y, hgbp, nsub)
        split_res["hgb_gc_tertile_strata"] = gc_tertile_strata(y, hgbp, gc_gene[idx], gc_edges)

        split_res["hgb_component_bootstrap_mae"] = component_bootstrap_gated_mae_diff(
            y, hgbp, comp, dist, threshold, n=BOOT_N, seed=BOOT_SEED)

        if split_name == "test":
            K = int(prim["n_retained"])
            rma = random_mask_maes(y, hgbp, K, n=RANDOM_MASK_N, seed=RANDOM_MASK_SEED)
            gate_mae = mae(hgbp[retained], y[retained]) if K else float("nan")
            gate_pct = float(np.mean(rma <= gate_mae)) * 100.0 if K else float("nan")
            split_res["random_mask_baseline"] = {
                "n_masks": RANDOM_MASK_N, "seed": RANDOM_MASK_SEED,
                "target_retained_count": K,
                "random_mae_mean": float(rma.mean()), "random_mae_std": float(rma.std()),
                "random_mae_p2_5": float(np.percentile(rma, 2.5)),
                "random_mae_p50": float(np.percentile(rma, 50)),
                "random_mae_p97_5": float(np.percentile(rma, 97.5)),
                "gate_mae": float(gate_mae) if K else float("nan"),
                "gate_percentile_vs_random": gate_pct,
                "note": "diagnostic only, NOT acceptance/tuning",
            }
            split_res["random_mask_maes"] = [float(x) for x in rma]

        results["by_split"][split_name] = split_res

        for i in range(len(y)):
            raw_rows.append({
                "gene_id": gid[i], "split": split_name,
                "component_id": int(comp[i]), "n_promoters": int(nsub[i]),
                "gc_sym400": float(gc_gene[int(idx[i])]),
                "Score_E": float(y[i]),
                "hgb": float(hgbp[i]), "hgb_error": float(hgbp[i] - y[i]),
                "mahalanobis_dist": float(dist[i]),
                "hgb_retained_90": bool(prim_mask[i]),
                "finalridge": float(frp[i]), "finalridge_error": float(frp[i] - y[i]),
                "b2_original": float(b2[i]) if np.isfinite(b2[i]) else None,
                "b2_error": float(b2[i] - y[i]) if np.isfinite(b2[i]) else None,
                "combined_fixedmean": float(cfm[i]),
                "combined_fixedmean_error": float(cfm[i] - y[i]),
            })

    t = results["by_split"]["test"]
    prim = t["hgb_primary_gate"]
    coverage = prim["coverage"]
    rel_mae_red = (prim["full"]["mae"] - prim["retained"]["mae"]) / prim["full"]["mae"]
    boot = t["hgb_component_bootstrap_mae"]
    acceptance = {
        "coverage": coverage, "coverage_ge_80pct": bool(coverage >= MIN_TEST_COVERAGE),
        "relative_mae_reduction": rel_mae_red,
        "relative_mae_reduction_ge_5pct": bool(rel_mae_red >= MIN_REL_MAE_REDUCTION),
        "component_bootstrap_lb": boot["bootstrap_diff_ci_low"],
        "component_bootstrap_lb_gt_0": bool(boot["bootstrap_diff_ci_low"] > 0),
        "all_mae": prim["full"]["mae"], "retained_mae": prim["retained"]["mae"],
        "n_degenerate_zero_retained_boot": boot["n_degenerate_zero_retained"],
        "acceptance_threshold_met": bool(
            coverage >= MIN_TEST_COVERAGE and rel_mae_red >= MIN_REL_MAE_REDUCTION
            and boot["bootstrap_diff_ci_low"] > 0),
        "note": "acceptance_threshold_met is a scientific gate result, distinct from review integrity PASS",
    }
    results["acceptance"] = acceptance

    import pandas as pd
    raw_df = pd.DataFrame(raw_rows)
    pred_path = REPORT / "predictions" / "pred_v6.8.0-covergate.rc1.tsv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(pred_path, sep="\t", index=False)

    results["artifacts"] = {
        "raw_membership_tsv": str(pred_path),
        "geometry_npz": str(MODEL_DIR / "covergate_geometry.npz"),
        "geometry_meta": str(MODEL_DIR / "covergate_geometry.json"),
        "written_at": datetime.datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(REPORT / "eval_covergate_b7.json", results)

    print("=== TEST primary HGB gate ===")
    print("coverage=%.6f  full_mae=%.6f  retained_mae=%.6f  rel_mae_red=%.6f" %
          (coverage, prim["full"]["mae"], prim["retained"]["mae"], rel_mae_red))
    print("bootstrap LB=%.6f  degenerate=%d" % (boot["bootstrap_diff_ci_low"], boot["n_degenerate_zero_retained"]))
    print(json.dumps(acceptance, indent=2, ensure_ascii=False, default=float))
    print("DONE")


if __name__ == "__main__":
    main()
