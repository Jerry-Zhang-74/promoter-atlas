#!/usr/bin/env python3
"""Batch05 annotation diagnostics (CPU, deterministic ridge baselines on frozen split).

Three predeclared feature variants, ridge-fit with the SAME alpha grid and a
SINGLE GROUP-PRESERVING train-inner fold (components never split), then
final-fit on FULL train genes (train-only scaling) and evaluated once on frozen
val/test:
  1. log1p(promoter_count)          (1 feature)
  2. 44 composition features        (mono+dinuc+GC+length, sym400+up2000)
  3. 44 composition + log1p(count)  (45 features)

Reports additive count information and strata metrics. Descriptive baseline set,
NOT a new NN arm, NOT causal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b5_common import (  # noqa: E402
    atomic_write_json, metrics_dict, component_bootstrap_ci, spearman, r2,
)
from b5_data import load_data_b5, make_memmap, inner_split_components  # noqa: E402

BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))
from baseline_seq_ridge import promoter_features  # noqa: E402

DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def gene_mean(pf, starts, idx):
    out = np.zeros((len(idx), pf.shape[1]), dtype=np.float64)
    for k, g in enumerate(idx):
        out[k] = pf[int(starts[g]):int(starts[g + 1])].mean(axis=0)
    return out


def ridge_inner_select(Xtr, ytr, Xval, yval, alpha_grid):
    from sklearn.linear_model import Ridge
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtrs = (Xtr - mu) / sd
    Xvals = (Xval - mu) / sd
    best_a, best_sp = None, -np.inf
    grid = {}
    for a in alpha_grid:
        m = Ridge(alpha=a, fit_intercept=True).fit(Xtrs, ytr)
        p = m.predict(Xvals)
        sp = spearman(p, yval)
        grid[str(a)] = {"alpha": a, "inner_val_spearman": sp}
        if sp > best_sp:
            best_sp, best_a = sp, a
    return best_a, mu, sd, grid


def strata_safe(y, pred, mask):
    if int(mask.sum()) < 2:
        return {"n": int(mask.sum()), "spearman": None, "r2": None, "mae": None, "mse": None}
    return metrics_dict(y[mask], pred[mask])


def fixed_n_strata(y, pred, n):
    return {
        "n=1": strata_safe(y, pred, n == 1),
        "n=2-3": strata_safe(y, pred, (n >= 2) & (n < 4)),
        "n>=4": strata_safe(y, pred, n >= 4),
    }


def gc_strata(y, pred, gc, edges):
    return {
        "gc_low": strata_safe(y, pred, gc < edges[0]),
        "gc_mid": strata_safe(y, pred, (gc >= edges[0]) & (gc < edges[1])),
        "gc_high": strata_safe(y, pred, gc >= edges[1]),
    }


def count_strata(y, pred, n, edges):
    return {
        "count_low": strata_safe(y, pred, n < edges[0]),
        "count_mid": strata_safe(y, pred, (n >= edges[0]) & (n < edges[1])),
        "count_high": strata_safe(y, pred, n >= edges[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out-tag", default="ann_diagnostics_v5")
    args = ap.parse_args()

    np.seterr(all="raise")
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_data_b5()
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    r2000 = z["r2000"].astype(np.int64)
    mm400 = make_memmap()
    mm2000 = np.memmap(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap",
                       dtype="uint8", mode="r", shape=(507365, 2000))
    pf = promoter_features(mm400, mm2000, r400, r2000)

    n_prom_f = n_prom.astype(np.float64)
    log1p_count = np.log1p(n_prom_f)

    train_gene_idx = np.flatnonzero(gene_split == "train")
    inner_tr, inner_val = inner_split_components(comp_dense, train_gene_idx, seed=args.seed)

    comp44_all = gene_mean(pf, starts, np.arange(n_genes))
    gc_gene = comp44_all[:, 20]

    # train-defined strata thresholds (explicit quantiles)
    gc_edges = [float(x) for x in np.quantile(gc_gene[train_gene_idx], [1.0 / 3.0, 2.0 / 3.0])]
    count_edges = [float(x) for x in np.quantile(n_prom_f[train_gene_idx], [1.0 / 3.0, 2.0 / 3.0])]

    results = {
        "alpha_grid": ALPHA_GRID,
        "inner_split": "group-preserving component fold (seed %d); components never split" % args.seed,
        "inner_realized": {
            "n_inner_train_genes": int(len(inner_tr)),
            "n_inner_val_genes": int(len(inner_val)),
            "inner_val_fraction": round(len(inner_val) / float(len(train_gene_idx)), 6),
            "n_inner_train_components": int(len(np.unique(comp_dense[inner_tr]))),
            "n_inner_val_components": int(len(np.unique(comp_dense[inner_val]))),
        },
        "selection_rule": "max inner-val gene E Spearman over the group-preserving fold",
        "scaling": "train-only (inner-train for selection, full train for final fit)",
        "strata_thresholds": {
            "gc_tertile_edges_train": gc_edges,
            "count_tertile_edges_train": count_edges,
            "fixed_n_strata": "n=1, n=2-3 (n<4), n>=4",
        },
        "note": "descriptive annotation baseline; NOT causal; strata thresholds derived from TRAIN only",
        "variants": {},
    }

    variants = {
        "count_only": log1p_count[:, None],
        "comp44": comp44_all,
        "comp44_count": np.concatenate([comp44_all, log1p_count[:, None]], axis=1),
    }

    train_rows = {}
    inner_rows = {}
    for vname, Xall in variants.items():
        inner_rows[vname] = (Xall[inner_tr], Xall[inner_val])
        train_rows[vname] = Xall[train_gene_idx]

    all_rows = []
    for vname, Xall in variants.items():
        Xtr_i, Xval_i = inner_rows[vname]
        best_a, mu_inner, sd_inner, grid = ridge_inner_select(
            Xtr_i, gene_score_e[inner_tr], Xval_i, gene_score_e[inner_val], ALPHA_GRID)

        Xf = train_rows[vname]
        mu_f = Xf.mean(axis=0)
        sd_f = Xf.std(axis=0)
        sd_f[sd_f == 0] = 1.0
        from sklearn.linear_model import Ridge
        mfinal = Ridge(alpha=best_a, fit_intercept=True).fit((Xf - mu_f) / sd_f, gene_score_e[train_gene_idx])

        vres = {"selected_alpha": best_a, "inner_grid": grid,
                "n_features": int(Xf.shape[1]),
                "inner_val_spearman_selected": grid[str(best_a)]["inner_val_spearman"]}
        for split_name in ["val", "test"]:
            idx = np.flatnonzero(gene_split == split_name)
            X = Xall[idx]
            pred = mfinal.predict((X - mu_f) / sd_f)
            y = gene_score_e[idx]
            comp = comp_dense[idx]
            nsub = n_prom[idx]
            gcsub = gc_gene[idx]
            vres[split_name] = {
                "full": metrics_dict(y, pred),
                "component_bootstrap_spearman": component_bootstrap_ci(
                    y, pred, comp, spearman, n=2000, seed=args.seed + 1000),
                "component_bootstrap_r2": component_bootstrap_ci(
                    y, pred, comp, r2, n=2000, seed=args.seed + 2000),
                "n_genes": int(len(y)),
                "n_components": int(len(np.unique(comp))),
                "fixed_n_strata": fixed_n_strata(y, pred, nsub),
                "gc_strata": gc_strata(y, pred, gcsub, gc_edges),
                "count_strata": count_strata(y, pred, nsub, count_edges),
            }
            if vname == "comp44_count" and split_name == "test":
                for gid, sp, yv, pv in zip(gene_gid[idx], gene_split[idx], y, pred):
                    all_rows.append({"gene_id": gid, "split": sp, "Score_E": float(yv),
                                     "pred_E_annridge": float(pv)})
        results["variants"][vname] = vres

    # best inner-selected annotation baseline (by inner-val Spearman among the 3)
    best_variant = max(results["variants"].keys(),
                       key=lambda v: results["variants"][v]["inner_val_spearman_selected"])
    results["best_inner_selected_baseline"] = {
        "variant": best_variant,
        "selected_alpha": results["variants"][best_variant]["selected_alpha"],
        "inner_val_spearman": results["variants"][best_variant]["inner_val_spearman_selected"],
    }

    additive = {}
    for split_name in ["val", "test"]:
        base = results["variants"]["comp44"][split_name]["full"]
        full = results["variants"]["comp44_count"][split_name]["full"]
        additive[split_name] = {
            "delta_spearman": full["spearman"] - base["spearman"],
            "delta_r2": full["r2"] - base["r2"],
            "delta_mae": full["mae"] - base["mae"],
        }
    results["additive_count_info"] = additive

    out_dir = REPORT / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pred_path = out_dir / ("pred_%s.tsv" % args.out_tag)
    pd.DataFrame(all_rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)
    atomic_write_json(REPORT / ("%s.json" % args.out_tag), results)
    print(json.dumps(results["additive_count_info"], indent=2))
    print(json.dumps(results["best_inner_selected_baseline"], indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
