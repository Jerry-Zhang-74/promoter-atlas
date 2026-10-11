#!/usr/bin/env python3
"""Batch06 evaluation: combined (finalridge + residual CNN) vs baselines."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_common import (  # noqa: E402
    REPORT, MODEL_DIR, DATA, atomic_write_json, sha256_file,
    fast_spearman, r2, mae, mse, metrics_dict,
    fast_component_bootstrap_diff, fast_component_bootstrap_ci,
    SEEDS, CANDIDATE,
)
from b6_model import GeneBagModel  # noqa: E402
from b6_train_common import predict_4slot, ARM  # noqa: E402
from b6_data import load_b6, make_memmap400  # noqa: E402

EXPOSED_META = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_prep_meta.json"
BATCH05_RIDGE_PRED = ROOT / "reports" / "breakthrough_20261002" / "batch05" / "predictions" / \
    "pred_ann_diagnostics_v5.tsv"
BATCH05_B2_PRED = ROOT / "reports" / "breakthrough_20261002" / "batch05" / "predictions" / \
    "pred_v6.6.0-scratchgene.rc7_s20261011.tsv"


def strata(y, pred, n, gc, gc_edges, mask):
    if int(mask.sum()) < 2:
        return {"n": int(mask.sum()), "spearman": None, "r2": None, "mae": None, "mse": None}
    return metrics_dict(y[mask], pred[mask])


def fixed_n_strata(y, pred, n):
    return {
        "n=1": strata(y, pred, n, None, None, n == 1),
        "n=2-3": strata(y, pred, n, None, None, (n >= 2) & (n < 4)),
        "n>=4": strata(y, pred, n, None, None, n >= 4),
    }


def main():
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()
    mm = make_memmap400()

    # features
    f = np.load(DATA / "features45.npz")
    comp44 = f["comp44"]; log1p_count = f["log1p_count"]
    X45 = np.concatenate([comp44, log1p_count[:, None]], axis=1)

    # finalridge
    fr = np.load(MODEL_DIR / "finalridge.npz")
    coef = fr["coef"]; intercept = fr["intercept"]; mu = fr["mu"]; sd = fr["sd"]
    def finalridge_pred(idx):
        return ((X45[idx] - mu) / sd) @ coef + intercept

    # hgb
    import joblib
    hgb = joblib.load(MODEL_DIR / "hgb_baseline.joblib")

    # exposed gene sets
    import json
    prep = json.loads(EXPOSED_META.read_text(encoding="utf-8"))
    exposed_val = set(prep["exposed_val_genes"]); exposed_test = set(prep["exposed_test_genes"])

    # residual CNN checkpoints
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resid_models = {}
    for seed in SEEDS:
        ck = torch.load(MODEL_DIR / ("residual_s%d" % seed) / "final.pt",
                        map_location=device, weights_only=False)
        model = GeneBagModel().to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        resid_models[seed] = model

    # build per-split canonical rows and predictions
    rows = {}
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        y = gene_score_e[idx]
        gid = gene_gid[idx]
        comp = comp_dense[idx]
        nsub = n_prom[idx]
        frp = finalridge_pred(idx)
        hgbp = hgb.predict(X45[idx])
        resid = {}
        for seed in SEEDS:
            resid[seed] = predict_4slot(resid_models[seed], mm, r400, starts, idx, device, ARM, epoch=0)
        combined = {seed: frp + resid[seed] for seed in SEEDS}
        fixed_mean = frp + 0.5 * (resid[SEEDS[0]] + resid[SEEDS[1]])
        rows[split_name] = dict(idx=idx, y=y, gid=gid, comp=comp, nsub=nsub,
                                frp=frp, hgbp=hgbp, resid=resid, combined=combined,
                                fixed_mean=fixed_mean)

    # existing batch05 ridge + B2 predictions (descriptive, matched by gene_id)
    b5_ridge = pd.read_csv(BATCH05_RIDGE_PRED, sep="\t")
    b5_ridge_map = dict(zip(b5_ridge["gene_id"].astype(str), b5_ridge["pred_E_annridge"].astype(float)))
    b5_b2 = pd.read_csv(BATCH05_B2_PRED, sep="\t")
    b5_b2_map = dict(zip(b5_b2["gene_id"].astype(str), b5_b2["pred_E_4slot"].astype(float)))

    results = {"candidate_version": CANDIDATE, "by_split": {}}
    for split_name in ["val", "test"]:
        r = rows[split_name]
        y = r["y"]; gid = r["gid"]; comp = r["comp"]; nsub = r["nsub"]
        frp = r["frp"]; hgbp = r["hgbp"]; fixed_mean = r["fixed_mean"]

        b5r = np.array([b5_ridge_map.get(g, np.nan) for g in gid], dtype=float)
        b2 = np.array([b5_b2_map.get(g, np.nan) for g in gid], dtype=float)

        # train-derived strata thresholds (GC tertiles on train)
        gc_gene = comp44[:, 20]
        train_idx = np.flatnonzero(gene_split == "train")
        gc_edges = [float(x) for x in np.quantile(gc_gene[train_idx], [1.0 / 3.0, 2.0 / 3.0])]
        count_edges = [float(x) for x in np.quantile(n_prom[train_idx], [1.0 / 3.0, 2.0 / 3.0])]

        model_preds = {
            "finalridge": frp,
            "hgb": hgbp,
            "zero_residual_finalridge": frp,
            "combined_s20261011": r["combined"][SEEDS[0]],
            "combined_s20261012": r["combined"][SEEDS[1]],
            "combined_fixedmean": fixed_mean,
        }
        # descriptive-only models (may have NaN if gene missing, but all genes present)
        model_preds_desc = {
            "batch05_ridge_existing": b5r,
            "b2_original": b2,
        }

        split_res = {"n_genes": int(len(y)), "n_components": int(len(np.unique(comp))),
                     "strata_thresholds": {"gc_tertile_edges_train": gc_edges,
                                           "count_tertile_edges_train": count_edges},
                     "models": {}}
        for mname, p in model_preds.items():
            d = {"full": metrics_dict(y, p),
                 "fixed_n_strata": fixed_n_strata(y, p, nsub)}
            d["unexposed"] = strata(y, p, nsub, None, None,
                                    ~np.isin(gid, list(exposed_val if split_name == "val" else exposed_test)))
            d["component_bootstrap_spearman"] = fast_component_bootstrap_ci(
                y, p, comp, fast_spearman, n=2000, seed=20266011)
            d["component_bootstrap_mae"] = fast_component_bootstrap_ci(
                y, p, comp, mae, n=2000, seed=20266012)
            split_res["models"][mname] = d
        for mname, p in model_preds_desc.items():
            ok = np.isfinite(p)
            d = {"full": metrics_dict(y[ok], p[ok]),
                 "fixed_n_strata": fixed_n_strata(y[ok], p[ok], nsub[ok]),
                 "n_matched": int(ok.sum())}
            d["unexposed"] = strata(y[ok], p[ok], nsub[ok], None, None,
                                    ~np.isin(gid[ok], list(exposed_val if split_name == "val" else exposed_test)))
            split_res["models"][mname] = d

        # paired deltas vs baselines
        paired = {}
        for base_name in ["finalridge", "hgb"]:
            base_pred = model_preds[base_name]
            paired[base_name] = {}
            for cand_name in ["combined_s20261011", "combined_s20261012", "combined_fixedmean"]:
                cpred = model_preds[cand_name]
                paired[base_name][cand_name] = {
                    "delta_spearman_point": fast_spearman(cpred, y) - fast_spearman(base_pred, y),
                    "delta_mae_point": mae(cpred, y) - mae(base_pred, y),
                    "relative_mae_reduction": (mae(base_pred, y) - mae(cpred, y)) / mae(base_pred, y),
                    "component_bootstrap_delta_spearman": fast_component_bootstrap_diff(
                        y, cpred, base_pred, comp, fast_spearman, n=2000, seed=20266011),
                    "component_bootstrap_delta_mae": fast_component_bootstrap_diff(
                        y, cpred, base_pred, comp, mae, n=2000, seed=20266012),
                }
        split_res["paired_vs_baselines"] = paired
        results["by_split"][split_name] = split_res

    # acceptance check on TEST
    test_res = results["by_split"]["test"]
    acc = {"candidates": {}, "thresholds": {
        "test_spearman_gain": 0.02, "component_bootstrap_95_lb_gt_0": True,
        "relative_mae_reduction": 0.03, "individual_seeds_nonnegative_gain": True}}
    for cand_name in ["combined_s20261011", "combined_s20261012", "combined_fixedmean"]:
        acc["candidates"][cand_name] = {}
        for base_name in ["finalridge", "hgb"]:
            pd_ = test_res["paired_vs_baselines"][base_name][cand_name]
            bs = pd_["component_bootstrap_delta_spearman"]
            met = {
                "delta_spearman_point": pd_["delta_spearman_point"],
                "bootstrap_ci_low": bs["ci_low"],
                "bootstrap_ci_high": bs["ci_high"],
                "spearman_gain_ge_0.02": pd_["delta_spearman_point"] >= 0.02,
                "bootstrap_lb_gt_0": bs["ci_low"] > 0,
                "relative_mae_reduction": pd_["relative_mae_reduction"],
                "mae_reduction_ge_3pct": pd_["relative_mae_reduction"] >= 0.03,
            }
            acc["candidates"][cand_name][base_name] = met
    # fixed mean must beat EACH baseline; individual seeds nonnegative gain
    fixed_ok = all(
        acc["candidates"]["combined_fixedmean"][b]["spearman_gain_ge_0.02"] and
        acc["candidates"]["combined_fixedmean"][b]["bootstrap_lb_gt_0"] and
        acc["candidates"]["combined_fixedmean"][b]["mae_reduction_ge_3pct"]
        for b in ["finalridge", "hgb"])
    seeds_nonneg = all(
        acc["candidates"]["combined_s20261011"][b]["delta_spearman_point"] >= 0 and
        acc["candidates"]["combined_s20261012"][b]["delta_spearman_point"] >= 0
        for b in ["finalridge", "hgb"])
    acc["acceptance_threshold_met"] = bool(fixed_ok and seeds_nonneg)
    acc["fixed_mean_beats_each_baseline"] = bool(fixed_ok)
    acc["individual_seeds_nonnegative"] = bool(seeds_nonneg)
    results["acceptance"] = acc

    # save predictions TSV (val + test, all models)
    out_rows = []
    for split_name in ["val", "test"]:
        r = rows[split_name]
        y = r["y"]; gid = r["gid"]; comp = r["comp"]; nsub = r["nsub"]
        for i in range(len(y)):
            row = {"gene_id": gid[i], "split": split_name, "component_id": int(comp[i]),
                   "n_promoters": int(nsub[i]), "Score_E": float(y[i]),
                   "finalridge": float(r["frp"][i]), "hgb": float(r["hgbp"][i]),
                   "combined_s20261011": float(r["combined"][SEEDS[0]][i]),
                   "combined_s20261012": float(r["combined"][SEEDS[1]][i]),
                   "combined_fixedmean": float(r["fixed_mean"][i]),
                   "residual_s20261011": float(r["resid"][SEEDS[0]][i]),
                   "residual_s20261012": float(r["resid"][SEEDS[1]][i])}
            out_rows.append(row)
    pred_df = pd.DataFrame(out_rows)
    pred_path = REPORT / "predictions" / "pred_v6.7.0-residualgene.rc1.tsv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)

    # save OOF fold memberships / provenance pointers
    results["artifacts"] = {
        "predictions": str(pred_path),
        "oof_residuals": str(DATA / "oof_residuals.npz"),
        "features45": str(DATA / "features45.npz"),
        "finalridge": str(MODEL_DIR / "finalridge.npz"),
        "hgb": str(MODEL_DIR / "hgb_baseline.joblib"),
    }
    atomic_write_json(REPORT / "eval_residual_b6.json", results)

    import json
    print("=== TEST full metrics ===")
    for mname in ["finalridge", "hgb", "combined_s20261011", "combined_s20261012", "combined_fixedmean"]:
        mm = test_res["models"][mname]["full"]
        print("%-22s sp=%.6f r2=%.6f mae=%.6f" % (mname, mm["spearman"], mm["r2"], mm["mae"]))
    print("=== acceptance ===")
    print(json.dumps(acc, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
