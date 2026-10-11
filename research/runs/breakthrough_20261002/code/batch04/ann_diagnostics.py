#!/usr/bin/env python3
"""Batch04 annotation diagnostics (CPU, deterministic ridge baselines).

Three predeclared feature variants, each ridge-fit with the SAME alpha grid and
the SAME train-inner 90/10 gene fold as batch03 (seed 20261011), then final-fit
on FULL original train genes and evaluated once on frozen val/test:
  1. log1p(promoter_count)          (1 feature)
  2. 44 composition features        (mono+dinuc+GC+length, sym400+up2000)
  3. 44 composition + log1p(count)  (45 features)

Reports additive count information and train-defined count/GC strata metrics.
This is a descriptive baseline set, NOT a new NN arm, NOT causal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b4_common import atomic_write_json, metrics_dict, bootstrap_spearman_ci  # noqa: E402

BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))
from train_scratch_gene import load_data, make_memmap, inner_split  # noqa: E402
from baseline_seq_ridge import promoter_features  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

COUNT_STRATA = [(1, 1), (2, 4), (5, 10 ** 9)]
GC_EDGES = [0.4525, 0.57]


def gene_mean(pf, starts, idx):
    out = np.zeros((len(idx), pf.shape[1]), dtype=np.float64)
    for k, g in enumerate(idx):
        out[k] = pf[int(starts[g]):int(starts[g + 1])].mean(axis=0)
    return out


def ridge_select(X, y, alpha_grid):
    from sklearn.linear_model import Ridge
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    best_a, best_sp = None, -np.inf
    grid = {}
    for a in alpha_grid:
        m = Ridge(alpha=a, fit_intercept=True)
        m.fit(Xs, y)
        p = m.predict(Xs)
        sp = metrics_dict(y, p)["spearman"]
        grid[str(a)] = {"alpha": a, "spearman": sp}
        if sp > best_sp:
            best_sp, best_a = sp, a
    return best_a, mu, sd, grid


def fit_ridge(Xtr, ytr, alpha, mu, sd):
    from sklearn.linear_model import Ridge
    Xs = (Xtr - mu) / sd
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, ytr)
    return m


def strata_metrics(y, pred, counts, gc):
    out = {}
    spec = [("count=1", COUNT_STRATA[0]), ("count=2-4", COUNT_STRATA[1]), ("count>=5", COUNT_STRATA[2])]
    for name, (lo, hi) in spec:
        m = (counts >= lo) & (counts <= hi)
        out[name] = metrics_dict(y[m], pred[m])
    out["gc_low"] = metrics_dict(y[gc < GC_EDGES[0]], pred[gc < GC_EDGES[0]])
    out["gc_mid"] = metrics_dict(y[(gc >= GC_EDGES[0]) & (gc < GC_EDGES[1])], pred[(gc >= GC_EDGES[0]) & (gc < GC_EDGES[1])])
    out["gc_high"] = metrics_dict(y[gc >= GC_EDGES[1]], pred[gc >= GC_EDGES[1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out-tag", default="ann_diagnostics")
    args = ap.parse_args()

    np.seterr(all="raise")
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    z = np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)
    r2000 = z["r2000"].astype(np.int64)
    mm400 = make_memmap()
    mm2000 = np.memmap(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap",
                       dtype="uint8", mode="r", shape=(507365, 2000))
    pf = promoter_features(mm400, mm2000, r400, r2000)

    n_prom = np.diff(starts).astype(np.float64)
    log1p_count = np.log1p(n_prom)

    train_gene_idx = np.flatnonzero(gene_split == "train")
    tr_idx, val_idx = inner_split(args.seed, len(train_gene_idx))
    inner_tr = train_gene_idx[tr_idx]
    inner_val = train_gene_idx[val_idx]

    comp44_tr = gene_mean(pf, starts, inner_tr)
    comp44_iv = gene_mean(pf, starts, inner_val)

    results = {
        "alpha_grid": ALPHA_GRID,
        "inner_split": "identical to batch03 (seed %d)" % args.seed,
        "selection_rule": "max inner-val gene E Spearman",
        "count_strata": COUNT_STRATA,
        "gc_edges": GC_EDGES,
        "note": "descriptive annotation baseline; NOT causal; count/GC strata derived from TRAIN only",
        "variants": {},
    }

    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])
    gids = np.array([str(x) for x in (np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)["gene_id"][starts[:-1]])])

    variants = {
        "count_only": {"iv": log1p_count[inner_val][:, None],
                       "tr": log1p_count[inner_tr][:, None]},
        "comp44": {"iv": comp44_iv, "tr": comp44_tr},
        "comp44_count": {"iv": np.concatenate([comp44_iv, log1p_count[inner_val][:, None]], axis=1),
                         "tr": np.concatenate([comp44_tr, log1p_count[inner_tr][:, None]], axis=1)},
    }

    all_rows = []
    for vname, v in variants.items():
        best_a, mu, sd, grid = ridge_select(v["tr"], gene_score_e[inner_tr], ALPHA_GRID)
        Xf = gene_mean(pf, starts, train_gene_idx) if vname != "count_only" else log1p_count[train_gene_idx][:, None]
        if vname == "comp44_count":
            Xf = np.concatenate([gene_mean(pf, starts, train_gene_idx), log1p_count[train_gene_idx][:, None]], axis=1)
        mfinal = fit_ridge(Xf, gene_score_e[train_gene_idx], best_a, mu, sd)

        vres = {"selected_alpha": best_a, "inner_grid": grid, "n_features": int(Xf.shape[1])}
        for split_name in ["val", "test"]:
            idx = np.flatnonzero(gene_split == split_name)
            if vname == "count_only":
                X = log1p_count[idx][:, None]
            elif vname == "comp44":
                X = gene_mean(pf, starts, idx)
            else:
                X = np.concatenate([gene_mean(pf, starts, idx), log1p_count[idx][:, None]], axis=1)
            pred = mfinal.predict((X - mu) / sd)
            y = gene_score_e[idx]
            gg = gids[idx]
            unexp = ~np.isin(gg, list(exposed_val if split_name == "val" else exposed_test))
            vres[split_name] = {
                "full": metrics_dict(y, pred),
                "unexposed": metrics_dict(y[unexp], pred[unexp]),
                "n_unexposed": int(unexp.sum()),
                "bootstrap_full": bootstrap_spearman_ci(y, pred, seed=args.seed + 1000),
                "strata": strata_metrics(y, pred, np.diff(starts)[idx], gene_mean(pf, starts, idx)[:, 20]),
            }
            if vname == "comp44_count" and split_name == "test":
                for gid, sp, yv, pv in zip(gg, gene_split[idx], y, pred):
                    all_rows.append({"gene_id": gid, "split": sp, "Score_E": float(yv),
                                     "pred_E_annridge": float(pv)})
        results["variants"][vname] = vres

    additive = {}
    for split_name in ["val", "test"]:
        for scope in ["full", "unexposed"]:
            base = results["variants"]["comp44"][split_name][scope]
            full = results["variants"]["comp44_count"][split_name][scope]
            additive["%s_%s" % (split_name, scope)] = {
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
    print("DONE")


if __name__ == "__main__":
    main()
