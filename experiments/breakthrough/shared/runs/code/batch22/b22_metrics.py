#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4: predeclared metrics, per-fold results, group bootstrap, source strata.

Primary decision (FROZEN before results; not lowered afterwards):
  R2 over R1 pooled OOF Spearman gain >= 0.03
  AND MAE reduction >= 10 percent
  AND top-decile precision not lower
  AND positive Spearman difference in >= 4 of 5 folds.
A paired group-bootstrap CI for the difference that crosses zero => the improvement
remains UNCERTAIN even if the point thresholds pass.
"""
from __future__ import annotations

import csv

import numpy as np

from b22_common import (DATA_OUT, REPORT_OUT, atomic_write_json, now_local,
                        sha256_text, spearman, mae, mse, r2, top_decile_precision,
                        group_bootstrap)

GAIN_RHO = 0.03
MAE_RED = 0.10
MIN_POS_FOLDS = 4
N_BOOT = 2000


def metrics_block(pred, obs, tie_keys):
    return {"n": int(len(pred)), "spearman": spearman(pred, obs), "mae": mae(pred, obs),
            "mse": mse(pred, obs), "r2": r2(pred, obs),
            "top_decile": top_decile_precision(pred, obs, tie_keys)}


def paired_group_bootstrap(pred_a, pred_b, obs, groups, n_boot=N_BOOT, seed=20261025):
    """Bootstrap independent GROUPS; return CI for (rho_a - rho_b) and (mae_b - mae_a)."""
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    drho, dmae = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), size=len(ug))
        idx = np.concatenate([idx_by[ug[j]] for j in pick])
        ra, rb = spearman(pred_a[idx], obs[idx]), spearman(pred_b[idx], obs[idx])
        if np.isfinite(ra) and np.isfinite(rb):
            drho.append(ra - rb)
        dmae.append(mae(pred_b[idx], obs[idx]) - mae(pred_a[idx], obs[idx]))
    drho = np.asarray(drho); dmae = np.asarray(dmae)
    def ci(v):
        return {"point": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5)), "n_boot": int(len(v)),
                "n_groups": int(len(ug))}
    return {"delta_spearman": ci(drho), "mae_reduction": ci(dmae)}


def run():
    z = np.load(DATA_OUT / "oof_predictions.npz", allow_pickle=True)
    y = z["y"]; fold = z["fold_of"]; gid = z["group_id"]; seen = z["source_seen"]
    preds = {"R0": z["R0"], "R1": z["R1"], "R2": z["R2"]}
    pids = [r["prom_id"] for r in csv.DictReader(
        open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"), delimiter="\t")]
    # stable SHA tie-breaking key (protocol): sha256(prom_id) prefix, deterministic
    sha = [sha256_text(p)[:16] for p in pids]
    k = int(fold.max()) + 1
    out = {"generated_at_local": now_local(), "units": "log2(hek exp/input)",
           "n": int(len(y)), "n_folds": k, "tie_break": "sha256(prom_id)[:16] ascending",
           "decision_rule": {"rho_gain_min": GAIN_RHO, "mae_reduction_min": MAE_RED,
                             "top_decile_not_lower": True,
                             "min_positive_folds": "%d of %d" % (MIN_POS_FOLDS, k),
                             "frozen": True},
           "pooled_oof": {}, "per_fold": {}, "strata": {}}
    for name, p in preds.items():
        blk = metrics_block(p, y, sha)
        blk["spearman_ci_group_bootstrap"] = group_bootstrap(
            p, y, gid, metric=spearman, n_boot=N_BOOT, seed=20261025)
        out["pooled_oof"][name] = blk
    # per-fold (group-heldout) results
    for f in range(k):
        m = fold == f
        out["per_fold"][str(f)] = {"n": int(m.sum()),
                                   "n_groups": int(len(np.unique(gid[m])))}
        for name, p in preds.items():
            out["per_fold"][str(f)][name] = metrics_block(p[m], y[m],
                                                          [sha[i] for i in np.where(m)[0]])
    # source-seen vs source-unseen strata (pooled OOF)
    for lab, mask in (("source_unseen", seen == 0), ("source_seen", seen == 1)):
        out["strata"][lab] = {"n": int(mask.sum()),
                              "n_groups": int(len(np.unique(gid[mask]))) if mask.sum() else 0}
        for name, p in preds.items():
            if mask.sum() >= 3:
                out["strata"][lab][name] = metrics_block(p[mask], y[mask],
                                                         [sha[i] for i in np.where(mask)[0]])
            else:
                out["strata"][lab][name] = {"n": int(mask.sum()), "note": "too few rows"}
    # paired group bootstrap R2 vs R1 and R1 vs R0
    out["paired_group_bootstrap"] = {
        "R2_minus_R1": paired_group_bootstrap(preds["R2"], preds["R1"], y, gid),
        "R1_minus_R0": paired_group_bootstrap(preds["R1"], preds["R0"], y, gid),
    }
    # decision
    r1 = out["pooled_oof"]["R1"]; r2 = out["pooled_oof"]["R2"]
    d_rho = r2["spearman"] - r1["spearman"]
    mae_red = (r1["mae"] - r2["mae"]) / r1["mae"] if r1["mae"] > 0 else float("nan")
    top_ok = r2["top_decile"]["precision"] >= r1["top_decile"]["precision"]
    pos_folds = sum(1 for f in range(k)
                    if out["per_fold"][str(f)]["R2"]["spearman"]
                    > out["per_fold"][str(f)]["R1"]["spearman"])
    boot = out["paired_group_bootstrap"]["R2_minus_R1"]["delta_spearman"]
    ci_excludes_zero = (boot.get("lo") is not None and boot["lo"] is not None
                        and boot["lo"] > 0)
    met = (d_rho >= GAIN_RHO) and (mae_red >= MAE_RED) and top_ok and (pos_folds >= MIN_POS_FOLDS)
    out["decision"] = {
        "delta_spearman_R2_minus_R1": d_rho, "mae_reduction_frac_R2_vs_R1": mae_red,
        "top_decile_not_lower": bool(top_ok), "positive_folds": int(pos_folds),
        "thresholds_met": bool(met),
        "bootstrap_ci_lo_gt_zero": bool(ci_excludes_zero),
        "verdict": ("improvement supported" if (met and ci_excludes_zero)
                    else "improvement uncertain (CI includes zero)" if met
                    else "no substantive improvement (honest negative)"),
    }
    atomic_write_json(REPORT_OUT / "results.json", out)
    print("[b22_metrics] R0 rho=%.4f R1 rho=%.4f R2 rho=%.4f | R1 mae=%.4f R2 mae=%.4f | d_rho=%.4f mae_red=%.3f posfolds=%d/%d => %s"
          % (out["pooled_oof"]["R0"]["spearman"], r1["spearman"], r2["spearman"],
             r1["mae"], r2["mae"], d_rho, mae_red, pos_folds, k, out["decision"]["verdict"]))
    return out


if __name__ == "__main__":
    run()
