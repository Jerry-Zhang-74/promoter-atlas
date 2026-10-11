#!/usr/bin/env python3
"""Batch06 CPU step: OOF ridge residuals + final ridge + HGB baseline."""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_common import (  # noqa: E402
    REPORT, MODEL_DIR, DATA, RUNS, atomic_write_json, sha256_file,
    ALPHA_GRID, N_FOLDS, SEEDS, select_alpha_max_sp_tie_larger,
    spearman, fast_spearman, r2, mae, mse, metrics_dict,
)
from b6_data import (  # noqa: E402
    load_b6, load_features, make_folds_components, inner_split_components, zscore,
)


def fit_scaler_ridge_select(Xtr, ytr, Xval, yval, alpha_grid):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtrs = (Xtr - mu) / sd
    Xvals = (Xval - mu) / sd
    grid = {}
    for a in alpha_grid:
        m = Ridge(alpha=a, fit_intercept=True).fit(Xtrs, ytr)
        p = m.predict(Xvals)
        grid[float(a)] = {"alpha": float(a), "inner_spearman": fast_spearman(p, yval),
                          "inner_r2": r2(p, yval), "inner_mae": mae(p, yval)}
    return grid


def fit_final(Xfit, yfit, alpha):
    mu = Xfit.mean(axis=0)
    sd = Xfit.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xfit - mu) / sd
    m = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, yfit)
    return m, mu, sd


def predict_with_scaler(X, m, mu, sd):
    Xs = (X - mu) / sd
    return m.predict(Xs)


def main():
    np.seterr(all="raise")
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()
    comp44_all, log1p_count, gc_gene, n_prom_f = load_features()
    X45 = np.concatenate([comp44_all, log1p_count[:, None]], axis=1)

    np.savez_compressed(DATA / "features45.npz", comp44=comp44_all, log1p_count=log1p_count)
    features_sha = sha256_file(DATA / "features45.npz")

    train_idx = np.flatnonzero(gene_split == "train")
    y_train = gene_score_e[train_idx]
    X_train = X45[train_idx]

    fold_id, fold_assign = make_folds_components(comp_dense, train_idx, n_folds=N_FOLDS, seed=20261011)

    oof_pred = np.full(len(train_idx), np.nan, dtype=np.float64)
    fold_records = {}
    for f in range(N_FOLDS):
        fold_mask = fold_id == f
        heldout_pos = np.flatnonzero(fold_mask)
        other4_pos = np.flatnonzero(~fold_mask)
        heldout_genes = train_idx[heldout_pos]
        other4_genes = train_idx[other4_pos]

        inner_tr, inner_val = inner_split_components(comp_dense, other4_genes, seed=20261011, target_val=0.10)

        grid = fit_scaler_ridge_select(
            X45[inner_tr], gene_score_e[inner_tr], X45[inner_val], gene_score_e[inner_val], ALPHA_GRID)
        sp_by_alpha = {a: grid[a]["inner_spearman"] for a in grid}
        best_a, best_sp = select_alpha_max_sp_tie_larger(list(grid.keys()), sp_by_alpha)

        m, mu, sd = fit_final(X45[other4_genes], gene_score_e[other4_genes], best_a)
        pred_fold = predict_with_scaler(X45[heldout_genes], m, mu, sd)
        oof_pred[heldout_pos] = pred_fold

        fold_records[int(f)] = {
            "n_heldout_genes": int(len(heldout_genes)),
            "n_other4_genes": int(len(other4_genes)),
            "n_inner_tr_genes": int(len(inner_tr)),
            "n_inner_val_genes": int(len(inner_val)),
            "inner_grid": {str(a): grid[a] for a in grid},
            "selected_alpha": float(best_a),
            "inner_val_spearman_selected": float(best_sp),
            "leakage_assertion": {
                "heldout_labels_used_in_scaler": False,
                "heldout_labels_used_in_alpha": False,
                "heldout_labels_used_in_fit": False,
                "heldout_genes_excluded_from_inner_and_fit": True,
                "component_preserving": True,
            },
        }

    assert np.isfinite(oof_pred).all(), "all OOF predictions must be finite"
    residual_train = y_train - oof_pred

    inner_tr, inner_val = inner_split_components(comp_dense, train_idx, seed=20261011, target_val=0.10)
    grid_final = fit_scaler_ridge_select(
        X45[inner_tr], gene_score_e[inner_tr], X45[inner_val], gene_score_e[inner_val], ALPHA_GRID)
    sp_by_alpha = {a: grid_final[a]["inner_spearman"] for a in grid_final}
    final_alpha, final_sp = select_alpha_max_sp_tie_larger(list(grid_final.keys()), sp_by_alpha)
    mfinal, mu_final, sd_final = fit_final(X_train, y_train, final_alpha)

    hgb = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
        l2_regularization=0.1, early_stopping=False, random_state=20261011)
    hgb.fit(X_train, y_train)

    preds = {"finalridge": {}, "hgb": {}}
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        Xs = X45[idx]
        preds["finalridge"][split_name] = {
            "gene_id": gene_gid[idx].tolist(),
            "pred": predict_with_scaler(Xs, mfinal, mu_final, sd_final),
            "y": gene_score_e[idx],
            "comp": comp_dense[idx],
            "n_prom": n_prom[idx],
        }
        preds["hgb"][split_name] = {
            "gene_id": gene_gid[idx].tolist(),
            "pred": hgb.predict(Xs),
            "y": gene_score_e[idx],
            "comp": comp_dense[idx],
            "n_prom": n_prom[idx],
        }

    oof_vs_final = {
        "oof_train_pred": {"mean": float(np.mean(oof_pred)), "std": float(np.std(oof_pred)),
                           "min": float(np.min(oof_pred)), "max": float(np.max(oof_pred))},
        "finalfit_val_pred": {"mean": float(np.mean(preds["finalridge"]["val"]["pred"])),
                              "std": float(np.std(preds["finalridge"]["val"]["pred"])),
                              "min": float(np.min(preds["finalridge"]["val"]["pred"])),
                              "max": float(np.max(preds["finalridge"]["val"]["pred"]))},
        "finalfit_test_pred": {"mean": float(np.mean(preds["finalridge"]["test"]["pred"])),
                               "std": float(np.std(preds["finalridge"]["test"]["pred"])),
                               "min": float(np.min(preds["finalridge"]["test"]["pred"])),
                               "max": float(np.max(preds["finalridge"]["test"]["pred"]))},
        "residual_train": {"mean": float(np.mean(residual_train)), "std": float(np.std(residual_train)),
                           "min": float(np.min(residual_train)), "max": float(np.max(residual_train))},
    }

    np.savez_compressed(
        DATA / "oof_residuals.npz",
        train_idx=train_idx, fold_id=fold_id, oof_pred=oof_pred,
        residual_train=residual_train, y_train=y_train)
    np.savez_compressed(
        MODEL_DIR / "finalridge.npz",
        coef=mfinal.coef_, intercept=mfinal.intercept_, mu=mu_final, sd=sd_final,
        alpha=final_alpha)
    import joblib
    joblib.dump(hgb, MODEL_DIR / "hgb_baseline.joblib")

    result = {
        "candidate_version": "v6.7.0-residualgene.rc1",
        "features_sha256": features_sha,
        "n_outer_train": int(len(train_idx)),
        "folds": fold_records,
        "final_ridge": {
            "selected_alpha": float(final_alpha),
            "inner_val_spearman_selected": float(final_sp),
            "inner_grid": {str(a): grid_final[a] for a in grid_final},
            "n_inner_tr_genes": int(len(inner_tr)),
            "n_inner_val_genes": int(len(inner_val)),
        },
        "hgb": {
            "config": "HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.1, early_stopping=False, random_state=20261011)",
            "n_features": int(X45.shape[1]),
        },
        "oof_vs_final_distribution": oof_vs_final,
        "artifacts": {
            "features": str(DATA / "features45.npz"),
            "oof_residuals": str(DATA / "oof_residuals.npz"),
            "finalridge": str(MODEL_DIR / "finalridge.npz"),
            "hgb": str(MODEL_DIR / "hgb_baseline.joblib"),
        },
        "note": "OOF fold labels never influence scaler/alpha/fit; residual target = Score_E - OOF prediction; finalridge/hgb fitted on outer train only; no label scaling",
    }
    atomic_write_json(REPORT / "oof_ridge_b6.json", result)

    print("fold alphas:", {k: fold_records[k]["selected_alpha"] for k in sorted(fold_records)})
    print("final alpha:", final_alpha, "inner sp:", final_sp)
    print("residual mean/std:", oof_vs_final["residual_train"]["mean"], oof_vs_final["residual_train"]["std"])
    print("features_sha:", features_sha)
    print("DONE")


if __name__ == "__main__":
    main()
