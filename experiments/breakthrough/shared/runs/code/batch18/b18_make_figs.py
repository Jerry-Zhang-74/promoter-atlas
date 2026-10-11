#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch18 deliverable 2: publication-quality figures for the release report.

Reads ONLY the corrected release metrics recomputed from saved batch16
predictions (no refits) and writes PNG + SVG pairs under
reports/breakthrough_20261002/batch18/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REP = ROOT / "reports" / "breakthrough_20261002" / "batch18"
FIG = REP / "figures"
M = json.loads((REP / "corrected_release_metrics.json").read_text(encoding="utf-8"))
TASKS = ("E", "S", "B")
REF = M["reference_shipped_seed"]["per_seed_metrics"]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})

C_NEURAL = "#1f5fa9"
C_HGB = "#c8791a"
C_A = "#1f5fa9"
C_B = "#2e8b57"
C_C = "#b03030"


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / ("%s.%s" % (name, ext)), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig1_deployed_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(len(TASKS))
    w = 0.38
    for ax, metric, ylab in ((axes[0], "spearman", "Spearman rho (test)"),
                             (axes[1], "r2", "R2 (test)")):
        nv = [REF["%s_test" % t][metric] for t in TASKS]
        hv = [M["same_row_neural_vs_hgb"][t]["test"]["hgb"][metric] for t in TASKS]
        b1 = ax.bar(x - w / 2, nv, w, color=C_NEURAL, label="Deployed A (seed 20261016)")
        b2 = ax.bar(x + w / 2, hv, w, color=C_HGB, label="Matched HGB baseline (same rows)")
        for b in list(b1) + list(b2):
            ax.annotate("%.3f" % b.get_height(), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["E", "S", "B"])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(0.85, 1.08 * max(nv + hv)))
        ax.set_title(ylab)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_ylim(0, 0.86)
    fig.suptitle("v6.15.1-release.rc2  |  deployed independent-arm metrics vs matched baseline",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.045, "E = Score_E rank expression [0,1] (test n=5851);  S = S_raw dispersion (test n=1880, defined subset);  B = detectable-tissue breadth [0,1] (test n=5851).\n"
             "Exact Y/P recomputation from saved batch16 prediction arrays; retrospective, no new fits.",
             ha="center", fontsize=7.6, color="#555555")
    save(fig, "fig1_deployed_vs_baseline")


def fig2_validation_rule():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(TASKS))
    w = 0.34
    for off, arm, col in ((-w / 2, "B", C_B), (w / 2, "C", C_C)):
        g = [M["validation_rule_applied"][arm]["per_target"][t]["gain"] for t in TASKS]
        b = ax.bar(x + off, g, w, color=col,
                   label="arm %s (%s)" % (arm, "shared backbone" if arm == "B" else "shared + 7 aux heads"))
        for bb, v in zip(b, g):
            ax.annotate("%+.4f" % v, (bb.get_x() + bb.get_width() / 2, v),
                        ha="center", va="top" if v < 0 else "bottom", fontsize=7.4)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.axhline(0.02, color="#2e8b57", ls="--", lw=1.1)
    ax.axhline(-0.02, color="#b03030", ls="--", lw=1.1)
    ax.text(len(TASKS) - 0.5, 0.0205, "frozen gain threshold +0.02", ha="right", va="bottom",
            fontsize=7.6, color="#2e8b57")
    ax.text(len(TASKS) - 0.5, -0.0215, "frozen no-drop threshold -0.02", ha="right", va="top",
            fontsize=7.6, color="#b03030")
    ax.set_xticks(x)
    ax.set_xticklabels(["E", "S", "B"])
    ax.set_ylabel("validation Spearman gain vs arm A (mean-of-seed)", fontsize=9)
    ax.set_title("Frozen validation rule: BOTH shared arms FAIL (mean gain < 0.02)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(-0.032, 0.040)
    fig.text(0.5, -0.035, "Rule decided on validation only; test scores were never used to select the arm, seed or ensemble.",
             ha="center", fontsize=7.6, color="#555555")
    save(fig, "fig2_validation_rule")


def fig3_bootstrap():
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ys, labels, cols = [], [], []
    k = 0
    for arm, col in (("B", C_B), ("C", C_C)):
        for t in TASKS:
            b = M["paired_component_bootstrap"][arm][t]
            ys.append(k)
            ax.plot([b["ci95_low"], b["ci95_high"]], [k, k], color=col, lw=2.0, solid_capstyle="round")
            ax.plot([b["base_point_estimate"]], [k], "o", color=col, ms=6)
            labels.append("%s-A  %s" % (arm, t))
            cols.append(col)
            k += 1
        k += 0.6
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("paired component-cluster bootstrap delta in test Spearman (95% CI)")
    ax.set_title("Ensemble B/C minus A: 95% CI mostly spans 0 (no shared-network win)")
    fig.text(0.5, -0.03, "Whole components resampled with replacement (2000 resamples, seed 20261012).",
             ha="center", fontsize=7.6, color="#555555")
    save(fig, "fig3_bootstrap_ci")


def fig4_estimands():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(TASKS))
    w = 0.26
    series = [
        ("per-seed mean (A)", [M["mean_of_seed_and_ensemble"]["A"][t]["test"]["mean_of_seed"]["spearman"] for t in TASKS], C_A, -w),
        ("ensemble (A)", [M["mean_of_seed_and_ensemble"]["A"][t]["test"]["ensemble"]["spearman"] for t in TASKS], "#7fb0dd", 0.0),
        ("matched HGB", [M["same_row_neural_vs_hgb"][t]["test"]["hgb"]["spearman"] for t in TASKS], C_HGB, w),
    ]
    for name, vals, col, off in series:
        b = ax.bar(x + off, vals, w, color=col, label=name)
        for bb in b:
            ax.annotate("%.3f" % bb.get_height(), (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                        ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["E", "S", "B"])
    ax.set_ylabel("test Spearman rho")
    ax.set_ylim(0, 0.86)
    ax.set_title("Estimands are NOT interchangeable (mean-of-seed vs ensemble vs baseline)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig4_estimands")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_deployed_metrics()
    fig2_validation_rule()
    fig3_bootstrap()
    fig4_estimands()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
