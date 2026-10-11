# -*- coding: utf-8 -*-
"""batch26 step 4: the three required held-out figures (no fits)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C
import b26_metrics as M

FIG = C.REPORT26 / "figures"  # in-scope: reports/breakthrough_20261002/batch26/figures
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["A0_ridgeC", "A1_cnn_true_seed20261026", "A2_cnn_true_seed20261027", "A3_cnn_shuffle_seed20261026"]
LAB = {"A0_ridgeC": "C ridge\n(composition)", "A1_cnn_true_seed20261026": "CNN true\nseed ...026",
       "A2_cnn_true_seed20261027": "CNN true\nseed ...027", "A3_cnn_shuffle_seed20261026": "CNN shuffled\n(null)"}
COL = {"A0_ridgeC": "#7f7f7f", "A1_cnn_true_seed20261026": "#1f77b4",
       "A2_cnn_true_seed20261027": "#17becf", "A3_cnn_shuffle_seed20261026": "#d62728"}


def load():
    tm = C.read_json(C.REPORT26 / "test_metrics.json")
    import csv
    P = {m: [] for m in MODELS}
    tru, loc, grp, var = [], [], [], []
    with open(C.DATA26 / "test_predictions.tsv", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            tru.append(float(r["true"])); loc.append(r["locus_id"]); grp.append(r["group_id"])
            var.append(r["variant"])
            for m in MODELS:
                P[m].append(float(r[m]))
    meta = {}
    with open(C.DATA26 / "str_variant_rows.tsv", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            meta[r["variant"]] = (int(r["vreg_len"]), r["motif"], r["allele"])
    return tm, np.array(tru), P, np.array(loc, object), np.array(grp, object), np.array(var, object), meta


def fig1_metrics(tm, y, P, grp):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    names = MODELS + ["train_mean"]
    sp = [tm["test_metrics"][m].get("spearman", np.nan) for m in names]
    gb = [tm["test_metrics"][m].get("group_balanced_mae", np.nan) for m in names]
    r2 = [tm["test_metrics"][m].get("r2", np.nan) for m in names]
    tp = [tm["test_metrics"][m].get("top_decile", {}).get("precision", np.nan) for m in names]
    cols = [COL.get(m, "#2ca02c") for m in names]
    for ax_, vals, ttl, base in ((ax[0], sp, "Test Spearman (pooled)", None),
                                 (ax[1], gb, "Test group-balanced MAE (lower better)", None),
                                 (ax[2], r2, "Test R^2 (variance of truth)", 0.0),
                                 (ax[3], tp, "Top-decile precision", 0.10)):
        ax_.bar(range(len(names)), vals, color=cols)
        ax_.set_xticks(range(len(names)))
        ax_.set_xticklabels([LAB.get(m, "train\nmean") for m in names], fontsize=7.5)
        ax_.set_title(ttl, fontsize=10)
        if base is not None:
            ax_.axhline(base, color="k", ls=":", lw=1)
    ax[0].annotate("undefined\n(constant pred)", xy=(3, 0.02), xytext=(3, 0.18), ha="center",
                   fontsize=7, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax[0].set_ylim(0, max(0.55, np.nanmax(sp) * 1.2))
    fig.suptitle("batch26 v6.19.0-strvariant.rc1 - held-out locus test (8800 rows / 2840 groups) - "
                 "internal retrospective, not external validation", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "heldout_metrics.png", dpi=160)
    plt.close(fig)


def fig2_true_vs_pred(tm, y, P, var, meta):
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    for k, m in enumerate(["A0_ridgeC", "A1_cnn_true_seed20261026", "A3_cnn_shuffle_seed20261026"]):
        axx = fig.add_subplot(gs[0, k])
        p = np.array(P[m])
        hb = axx.hexbin(y, p, gridsize=45, bins="log", cmap="viridis", mincnt=1)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        axx.plot(lim, lim, "r--", lw=1)
        rho = tm["test_metrics"][m].get("spearman", np.nan)
        axx.set_title("%s\nrho=%.3f  gbMAE=%.3f" % (LAB[m].replace("\n", " "), rho,
                      tm["test_metrics"][m]["group_balanced_mae"]), fontsize=9)
        axx.set_xlabel("observed log2(cDNA/gDNA)"); axx.set_ylabel("predicted")
        fig.colorbar(hb, ax=axx, shrink=0.85, label="count (log)")
    axx = fig.add_subplot(gs[0, 2])
    lens = np.array([meta[v][0] for v in var])
    p1 = np.array(P["A1_cnn_true_seed20261026"])
    bins = np.unique(np.percentile(lens, np.linspace(0, 100, 9)).astype(int))
    xs, ys, ns = [], [], []
    for i in range(len(bins) - 1):
        msk = (lens >= bins[i]) & (lens < bins[i + 1])
        if msk.sum() >= 30:
            xs.append(0.5 * (bins[i] + bins[i + 1])); ys.append(M.spearman(p1[msk], y[msk])); ns.append(int(msk.sum()))
    axx.plot(xs, ys, "o-", color="#1f77b4")
    axx.axhline(tm["test_metrics"]["A1_cnn_true_seed20261026"]["spearman"], color="k", ls=":", lw=1,
                label="pooled rho")
    axx.set_xlabel("variable insert length (nt)"); axx.set_ylabel("Spearman within length bin")
    axx.set_title("length-domain context\n(full native vreg, no padding)", fontsize=9)
    axx.legend(fontsize=8); axx.grid(alpha=0.3)
    fig.suptitle("batch26 true vs predicted activity (held-out) + length context", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "true_vs_pred.png", dpi=160)
    plt.close(fig)


def within_deltas(y, P, var, meta):
    """recompute mutant-minus-ref deltas from saved test predictions (no fits)."""
    ref_i = {}
    for i, v in enumerate(var):
        if meta[v][2] == "ref":
            ref_i[var[i].rsplit("_", 1)[0]] = i
    out = {}
    for m in MODELS:
        p = np.array(P[m]); dp, do, gg = [], [], []
        for i, v in enumerate(var):
            if meta[v][2] == "ref":
                continue
            L = v.rsplit("_", 1)[0]
            if L not in ref_i or i == ref_i[L]:
                continue
            j = ref_i[L]
            if y[j] >= 1e9:
                continue
            dp.append(p[i] - p[j]); do.append(y[i] - y[j]); gg.append(L)
        out[m] = (np.array(dp), np.array(do), np.array(gg, object))
    return out


def fig3_within(tm, y, P, var, meta):
    dl = within_deltas(y, P, var, meta)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, m in enumerate(["A0_ridgeC", "A1_cnn_true_seed20261026", "A3_cnn_shuffle_seed20261026"]):
        dp, do, _ = dl[m]
        axx = ax[k]
        axx.axhline(0, color="k", lw=0.6); axx.axvline(0, color="k", lw=0.6)
        axx.hexbin(do, dp, gridsize=45, bins="log", cmap="magma", mincnt=1)
        axx.plot([do.min(), do.max()], [do.min(), do.max()], "c--", lw=1)
        w = tm["within_locus"].get(m, {})
        rho = w.get("delta_spearman", np.nan)
        ci = w.get("delta_rho_ci95") or [np.nan, np.nan]
        sa = w.get("sign_agreement_nonzero", np.nan)
        axx.set_title("%s\nDelta rho=%.3f [%.3f, %.3f]  sign=%.3f" %
                      (LAB[m].replace("\n", " "), rho, ci[0], ci[1], sa), fontsize=9)
        axx.set_xlabel("observed mutant-ref (log2)"); axx.set_ylabel("predicted mutant-ref")
    fig.suptitle("batch26 within-locus variant effect: predicted vs observed mutant-ref deltas "
                 "(A1: %d pairs / %d loci / %d groups) - direction support NOT fully met" % (tm["within_locus"]["A1_cnn_true_seed20261026"]["n_pairs"], tm["within_locus"]["A1_cnn_true_seed20261026"]["n_loci"], tm["within_locus"]["A1_cnn_true_seed20261026"]["n_groups_bootstrapped"]), fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "within_locus_delta.png", dpi=160)
    plt.close(fig)


def main():
    tm, y, P, loc, grp, var, meta = load()
    fig1_metrics(tm, y, P, grp)
    fig2_true_vs_pred(tm, y, P, var, meta)
    fig3_within(tm, y, P, var, meta)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
