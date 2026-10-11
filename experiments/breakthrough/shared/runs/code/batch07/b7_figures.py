#!/usr/bin/env python3
"""Batch07 figures: selective-risk vs coverage, distance hist, random-mask baseline."""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import REPORT, MODEL_DIR  # noqa: E402

FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    evalj = json.loads((REPORT / "eval_covergate_b7.json").read_text(encoding="utf-8"))
    t = evalj["by_split"]["test"]
    diag = t["hgb_diagnostic_quantiles"]
    qs = sorted([float(k) for k in diag.keys()])
    cov = [diag[f"{q:.2f}"]["coverage"] for q in qs]
    ret_mae = [diag[f"{q:.2f}"]["retained"]["mae"] for q in qs]
    ret_sp = [diag[f"{q:.2f}"]["retained"]["spearman"] for q in qs]
    full_mae = t["hgb_primary_gate"]["full"]["mae"]
    full_sp = t["hgb_primary_gate"]["full"]["spearman"]

    # 1) selective risk vs coverage (primary point highlighted)
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(cov, ret_mae, "o-", color="tab:red", label="retained MAE (HGB)")
    ax1.axhline(full_mae, color="gray", ls="--", lw=1, label="full-test MAE baseline")
    ax1.scatter([t["hgb_primary_gate"]["coverage"]], [t["hgb_primary_gate"]["retained"]["mae"]],
                color="black", zorder=5, s=70, label="primary 90% gate")
    ax1.set_xlabel("coverage (retained fraction)")
    ax1.set_ylabel("retained MAE", color="tab:red")
    ax1.set_title("Selective-risk vs coverage (diagnostic; primary = 90% train quantile)")
    ax2 = ax1.twinx()
    ax2.plot(cov, ret_sp, "s--", color="tab:blue", label="retained Spearman (HGB)")
    ax2.axhline(full_sp, color="tab:blue", ls=":", lw=1, label="full-test Spearman")
    ax2.set_ylabel("retained Spearman", color="tab:blue")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_selective_risk.png", dpi=130)
    plt.close(fig)

    # 2) relative MAE reduction vs coverage
    rel = [(full_mae - m) / full_mae * 100.0 for m in ret_mae]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(cov, rel, "o-", color="tab:purple")
    ax.axhline(5.0, color="red", ls="--", lw=1, label="pre-registered 5% threshold")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("coverage (retained fraction)")
    ax.set_ylabel("relative MAE reduction (%)")
    ax.set_title("HGB relative MAE reduction vs coverage (all diagnostic points)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_rel_mae_reduction.png", dpi=130)
    plt.close(fig)

    # 3) Mahalanobis distance histogram (train vs test)
    geom = np.load(MODEL_DIR / "covergate_geometry.npz")
    train_dist = geom["train_dist"]
    thr = float(geom["threshold"])
    gene_split_loaded = np.load(ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz", allow_pickle=True)["gene_split"]
    # recompute test distances from eval raw tsv
    import pandas as pd
    df = pd.read_csv(REPORT / "predictions" / "pred_v6.8.0-covergate.rc1.tsv", sep="\t")
    test_dist = df[df["split"] == "test"]["mahalanobis_dist"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bins = np.linspace(0, 200, 120)
    ax.hist(train_dist[train_dist <= 200], bins=bins, alpha=0.55, label="train distances", density=True)
    ax.hist(test_dist[test_dist <= 200], bins=bins, alpha=0.55, label="test distances", density=True)
    ax.axvline(thr, color="red", ls="--", lw=1.2, label="90% train quantile threshold")
    ax.set_xlabel("squared Mahalanobis distance")
    ax.set_ylabel("density")
    ax.set_title("Feature-coverage score distribution (train vs test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "mahalanobis_distance.png", dpi=130)
    plt.close(fig)

    # 4) random-mask baseline
    rmb = t["random_mask_baseline"]
    rma = np.asarray(t["random_mask_maes"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(rma, bins=60, alpha=0.7, label="500 random masks (MAE)")
    ax.axvline(rmb["gate_mae"], color="red", lw=2, label="gate retained MAE")
    ax.axvline(rmb["random_mae_mean"], color="gray", ls="--", lw=1, label="random-mask mean")
    ax.set_xlabel("retained MAE")
    ax.set_ylabel("count")
    ax.set_title("Random-mask baseline (same retained count, seed 20261012)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "random_mask_baseline.png", dpi=130)
    plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    main()
