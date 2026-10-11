#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 figures. Reads saved results only; ZERO fits."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C

C.set_thread_limits(4)


def load_oof():
    rows = list(csv.DictReader(open(C.DATA_OUT / "oof_predictions_batch24.tsv",
                                     encoding="utf-8"), delimiter="\t"))
    return rows


def fig1(res):
    names = ["R1_cached", "H1", "H2"]
    labels = ["R1 (cached\nbaseline)", "H1 global\nk4+k5", "H2 global+position"]
    sp = [res["pooled_oof"][n]["spearman"] for n in names]
    lo95 = [res["pooled_oof"][n]["spearman_ci_group_bootstrap"]["lo"] for n in names]
    hi95 = [res["pooled_oof"][n]["spearman_ci_group_bootstrap"]["hi"] for n in names]
    mae = [res["pooled_oof"][n]["mae"] for n in names]
    prec = [res["pooled_oof"][n]["top_decile"]["precision"] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    x = np.arange(len(names))
    yerr = [np.array(sp) - np.array(lo95), np.array(hi95) - np.array(sp)]
    axes[0].bar(x, sp, yerr=yerr, capsize=5, color=["#888", "#2b7bba", "#59a14f"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("pooled OOF Spearman")
    axes[0].set_title("Ranking (95% group-bootstrap CI)")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].bar(x, mae, color=["#888", "#2b7bba", "#59a14f"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("pooled OOF MAE (log2 ratio)")
    axes[1].set_title("Absolute error (lower better)")
    axes[2].bar(x, prec, color=["#888", "#2b7bba", "#59a14f"])
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("top-decile precision (k=68)")
    axes[2].set_title("Practical ranking utility")
    fig.suptitle("batch24 v6.18.0-hek-kernel.rc1  -  retrospective reuse of batch22 "
                 "HEK293 folds (NOT prospective)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(C.FIG_OUT / "fig1_metrics_with_uncertainty.png", dpi=150)
    plt.close(fig)


def fig2(rows):
    y = np.array([float(r["hek_norm"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharex=True, sharey=True)
    for f in range(C.N_FOLDS):
        m = fold == f
        ax = axes[f]
        for key, col, lab in (("pred_R1_cached", "#888", "R1"), ("pred_H1", "#2b7bba", "H1"),
                              ("pred_H2", "#59a14f", "H2")):
            ax.scatter(y[m], np.array([float(r[key]) for r in rows])[m], s=6, alpha=0.55,
                       color=col, label=lab)
        lim = [min(y.min(), -1.5), max(y.max(), 2.5)]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_title("fold %d (n=%d)" % (f, int(m.sum())), fontsize=9)
        ax.set_xlabel("observed hek_norm")
        if f == 0:
            ax.set_ylabel("OOF predicted")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("batch24 true vs OOF predicted by fixed public batch22 folds", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(C.FIG_OUT / "fig2_true_vs_predicted.png", dpi=150)
    plt.close(fig)


def fig3(loc):
    recs = loc["records"]
    xs = [r["sequence_length_bp"] for r in recs]
    ys = [r["H1_exploratory_model_score_mean"] for r in recs]
    obs = [r["endpoint_target_value_log10"] for r in recs]
    fams = sorted(set(r["family"] for r in recs))
    cmap = {f: c for f, c in zip(fams, ["#2b7bba", "#e15759", "#59a14f"])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for f in fams:
        sel = [i for i, r in enumerate(recs) if r["family"] == f]
        axes[0].scatter([xs[i] for i in sel], [ys[i] for i in sel], label=f,
                        color=cmap[f], s=28)
        axes[1].scatter([obs[i] for i in sel], [ys[i] for i in sel], label=f,
                        color=cmap[f], s=28)
    for i, r in enumerate(recs):
        axes[0].annotate(r["construct_id"], (xs[i], ys[i]), fontsize=6)
        axes[1].annotate(r["construct_id"], (obs[i], ys[i]), fontsize=6)
    axes[0].set_xlabel("construct length (bp)"); axes[0].set_ylabel("H1 exploratory score")
    axes[0].set_title("Applicability: score vs length (validated len = 133)")
    axes[0].axvline(133, color="k", ls=":", lw=0.9)
    axes[1].set_xlabel("local measured log10(GFP)")
    axes[1].set_ylabel("H1 exploratory score")
    axes[1].set_title("Descriptive only (NOT calibration; unsupported activity prediction)")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.suptitle("batch24 local-14 diagnostic: H1 exploratory score, H2 unsupported "
                 "(non-133 bp)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(C.FIG_OUT / "fig3_local_applicability.png", dpi=150)
    plt.close(fig)


def main():
    C.FIG_OUT.mkdir(parents=True, exist_ok=True)
    res = json.load(open(C.REPORT_OUT / "results.json", encoding="utf-8"))
    loc = json.load(open(C.REPORT_OUT / "local14_exploratory.json", encoding="utf-8"))
    rows = load_oof()
    fig1(res); fig2(rows); fig3(loc)
    man = {}
    for n in ("fig1_metrics_with_uncertainty.png", "fig2_true_vs_predicted.png",
              "fig3_local_applicability.png"):
        man[n] = C.sha256_file(C.FIG_OUT / n)
    C.atomic_write_json(C.REPORT_OUT / "figures_manifest.json",
                        {"generated_at_local": C.now_local(), "figures": man,
                         "zero_fits": True})
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
