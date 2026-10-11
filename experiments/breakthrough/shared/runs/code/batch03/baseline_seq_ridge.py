#!/usr/bin/env python3
"""Batch03 Track A baseline: sequence-feature ridge (gene-mean aggregation).

Features (per promoter, sym400 + up2000): mono counts(4), dinuc counts(16),
GC fraction(1), length(1) -> 22 each -> 44 features. Gene feature = mean over
the gene's promoters (gene-mean aggregation). Ridge alpha selected on the
train-inner 90/10 gene holdout (identical deterministic split as scratch
encoder, seed 20261011). Frozen original val/test evaluated once afterwards.

This baseline uses only raw sequence composition; no v6.3 weights, no labels,
no donor-ortholog transfer. Human original-train genes only for fitting.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import atomic_write_json, spearman, r2, mae, mse, bootstrap_spearman_ci  # noqa: E402
from train_scratch_gene import load_data, make_memmap, inner_split, gene_order  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch03"
L400 = 400
L2000 = 2000
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def rc_array(arr, length):
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[arr[:, ::-1]]


def promoter_features(mm400, mm2000, r400, r2000):
    """Per-promoter composition features over sym400 and up2000."""
    s4 = np.asarray(mm400[r400], dtype=np.int64)      # (n,400)
    s2 = np.asarray(mm2000[r2000], dtype=np.int64)    # (n,2000)
    feats = []
    for s, L in [(s4, L400), (s2, L2000)]:
        n = s.shape[0]
        mono = np.zeros((n, 4), dtype=np.float64)
        for c in range(4):
            mono[:, c] = (s == c).sum(axis=1)
        di = np.zeros((n, 16), dtype=np.float64)
        for a in range(4):
            for b in range(4):
                k = a * 4 + b
                di[:, k] = ((s[:, :-1] == a) & (s[:, 1:] == b)).sum(axis=1)
        gc = (mono[:, 1] + mono[:, 2]) / L
        length = np.full(n, float(L), dtype=np.float64)
        feats.append(np.concatenate([mono / L, di / (L - 1), gc[:, None], length[:, None]], axis=1))
    return np.concatenate(feats, axis=1)  # (n, 44)


def gene_mean_features(pf, starts, gene_idx):
    out = np.zeros((len(gene_idx), pf.shape[1]), dtype=np.float64)
    for k, g in enumerate(gene_idx):
        lo, hi = int(starts[g]), int(starts[g + 1])
        out[k] = pf[lo:hi].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out-tag", default="baseline_seq_ridge")
    ap.add_argument("--alpha", type=float, default=None, help="override grid selection")
    args = ap.parse_args()

    np.seterr(all="raise")
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    r2000 = np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)["r2000"].astype(np.int64)
    mm400 = make_memmap()
    mm2000 = np.memmap(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap",
                       dtype="uint8", mode="r", shape=(507365, L2000))

    from train_scratch_gene import N_HUMAN  # noqa: F401
    pf = promoter_features(mm400, mm2000, r400, r2000)

    train_mask = gene_split == "train"
    train_gene_idx = np.flatnonzero(train_mask)
    tr_idx, val_idx = inner_split(args.seed, len(train_gene_idx))
    inner_tr = train_gene_idx[tr_idx]
    inner_val = train_gene_idx[val_idx]

    X_tr = gene_mean_features(pf, starts, inner_tr)
    y_tr = gene_score_e[inner_tr]
    X_iv = gene_mean_features(pf, starts, inner_val)
    y_iv = gene_score_e[inner_val]

    # standardize using inner-train only
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr_s = (X_tr - mu) / sd
    X_iv_s = (X_iv - mu) / sd

    from sklearn.linear_model import Ridge
    grid_results = {}
    best_alpha = None
    best_spearman = -np.inf
    for a in ALPHA_GRID:
        m = Ridge(alpha=a, fit_intercept=True)
        m.fit(X_tr_s, y_tr)
        p = m.predict(X_iv_s)
        sp = spearman(p, y_iv)
        grid_results[str(a)] = {"alpha": a, "inner_val_spearman": sp,
                                "inner_val_r2": r2(p, y_iv), "inner_val_mae": mae(p, y_iv)}
        if sp > best_spearman:
            best_spearman = sp
            best_alpha = a

    selected_alpha = args.alpha if args.alpha is not None else best_alpha
    results = {
        "baseline": "sequence-feature ridge (gene-mean aggregation)",
        "feature_contract": "per-promoter mono(4)+dinuc(16)+GC(1)+length(1) for sym400 and up2000 => 44; gene=mean over promoters; z-score on inner-train",
        "alpha_grid": ALPHA_GRID,
        "alpha_grid_inner_results": grid_results,
        "selected_alpha": selected_alpha,
        "selection_rule": "max inner-val E Spearman over deterministic 90/10 gene holdout (seed 20261011)",
        "inner_split": "identical to scratch encoder inner split (seed 20261011)",
        "seed": args.seed,
    }

    # final fit on FULL original train genes, standardize on full train
    X_full_tr = gene_mean_features(pf, starts, train_gene_idx)
    y_full_tr = gene_score_e[train_gene_idx]
    mu_f = X_full_tr.mean(axis=0)
    sd_f = X_full_tr.std(axis=0)
    sd_f[sd_f == 0] = 1.0
    X_full_tr_s = (X_full_tr - mu_f) / sd_f
    mfinal = Ridge(alpha=selected_alpha, fit_intercept=True)
    mfinal.fit(X_full_tr_s, y_full_tr)

    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])
    gids = np.array([str(x) for x in gene_id[starts[:-1]]])

    all_rows = []
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        X = gene_mean_features(pf, starts, idx)
        X_s = (X - mu_f) / sd_f
        pred = mfinal.predict(X_s)
        y = gene_score_e[idx]
        gg = gids[idx]
        unexp = ~np.isin(gg, list(exposed_val if split_name == "val" else exposed_test))

        def m(yy, pp):
            return {"n": int(len(yy)), "spearman": spearman(pp, yy), "r2": r2(pp, yy),
                    "mae": mae(pp, yy), "mse": mse(pp, yy)}
        results[split_name] = {
            "full": m(y, pred),
            "unexposed": m(y[unexp], pred[unexp]),
            "n_unexposed": int(unexp.sum()),
            "bootstrap_spearman_full": bootstrap_spearman_ci(y, pred, seed=args.seed + 1000),
            "bootstrap_spearman_unexposed": bootstrap_spearman_ci(y[unexp], pred[unexp], seed=args.seed + 2000),
        }
        for gid, sp, yv, pv in zip(gg, gene_split[idx], y, pred):
            all_rows.append({"gene_id": gid, "split": sp, "Score_E": float(yv), "pred_E": float(pv)})

    out_dir = REPORT / "trackA_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pred_path = out_dir / f"pred_{args.out_tag}.tsv"
    pd.DataFrame(all_rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)
    atomic_write_json(out_dir / f"metrics_{args.out_tag}.json", results)

    # persist fit (coef + standardization) for replay
    np.savez_compressed(MODEL_DIR / f"{args.out_tag}.npz",
                        coef=mfinal.coef_, intercept=mfinal.intercept_,
                        mu_f=mu_f, sd_f=sd_f)
    results["model_artifact"] = str(MODEL_DIR / f"{args.out_tag}.npz")
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
