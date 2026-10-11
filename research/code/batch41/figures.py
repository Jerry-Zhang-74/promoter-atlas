#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 up to 3 concise result figures (no UI repackaging; no official assets redistributed)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402

MODELS = ("parm", "h1", "gc")
NAMES = {"parm": "PARM HEK293 (5fold mean)", "h1": "H1 spectrum-kernel (frozen)", "gc": "GC fraction"}


def main():
    p1 = json.load(open(C.DATA / "metrics_part1.json", encoding="utf-8"))
    p2 = json.load(open(C.DATA / "metrics_part2.json", encoding="utf-8"))
    with open(C.DATA / "frozen_input_table.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    bio = [r for r in rows if r["control"] == "0"]

    # fig1: activity ranking comparison (pooled + fixed strata rho)
    strata = {"pooled": {m: p1["pooled"][m]["spearman"] for m in MODELS},
              "chr17": {m: p1["strata"]["chr17"]["rho"][m] for m in MODELS},
              "non-chr17": {m: p1["strata"]["non_chr17"]["rho"][m] for m in MODELS}}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(strata)); w = 0.25
    for j, m in enumerate(MODELS):
        vals = [strata[s][m] for s in strata]
        ax.bar(x + (j - 1) * w, vals, w, label=NAMES[m])
    ax.axhline(0.30, ls="--", c="r", lw=1)
    ax.text(2.2, 0.305, "PARM support threshold 0.30", color="r", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(list(strata))
    ax.set_ylabel("Spearman rho vs alpha (activity)")
    ax.set_title("batch41 activity ranking: all bio + fixed strata")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIG / "fig1_activity_ranking.png", dpi=130); plt.close(fig)

    # fig2: top-decile precision + paired conditional bootstrap delta CI
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    prec = [p1["pooled"][m]["top_decile"]["precision"] for m in MODELS]
    base = p1["pooled"]["parm"]["top_decile"]["k"] / p1["pooled"]["parm"]["n"]
    ax.bar([NAMES[m] for m in MODELS], prec, color=["#4477aa", "#66aa55", "#cc6677"])
    ax.axhline(base, ls=":", c="k", lw=1)
    ax.text(0, base, " chance %.3f" % base, fontsize=8, va="bottom")
    ax.set_ylabel("top-decile precision"); ax.set_title("top 10%% precision (k=ceil(0.1n))")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    d = p2["bootstrap"]["PARM_minus_H1"]
    ax.errorbar([0], [p1["paired_rho_differences_common_set"]["PARM_minus_H1"]],
                yerr=[[p1["paired_rho_differences_common_set"]["PARM_minus_H1"] - d["lo"]],
                      [d["hi"] - p1["paired_rho_differences_common_set"]["PARM_minus_H1"]]],
                fmt="o", capsize=6, color="#aa3377")
    ax.axhline(0, ls="--", c="k", lw=1); ax.axhline(0.05, ls=":", c="r", lw=1)
    ax.text(-0.15, 0.052, "support threshold +0.05", color="r", fontsize=8)
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    ax.set_ylabel("PARM rho - H1 rho")
    ax.set_title("paired component bootstrap (conditional)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIG / "fig2_topdecile_uncertainty.png", dpi=130); plt.close(fig)

    # fig3: scope evidence
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axis("off")
    lines = [
        "batch41 v6.28.0-hekft-fragment-benchmark.rc1  (ZERO-FIT frozen transfer diagnostic)",
        "",
        "Target : MPRAnalyze getAlpha activity, public HEK293FT OligoMPRA (7537 bio + 85 ctrl, 270bp inserts)",
        "Models : PARM HEK293 5fold mean | batch24 H1 kernel readout (inference only) | fixed GC fraction",
        "Inputs : raw[15:285] 270bp insert only; no RC ensemble / window / adapter / reference-DNA fill",
        "",
        "VERDICT: %s" % p2["decision_rule"]["verdict"],
        "  PARM pooled rho = %.3f  (< 0.30 support)" % p1["pooled"]["parm"]["spearman"],
        "  PARM - H1 delta  = %+.3f  (CI [%+.3f, %+.3f])" % (
            p1["paired_rho_differences_common_set"]["PARM_minus_H1"],
            p2["bootstrap"]["PARM_minus_H1"]["lo"], p2["bootstrap"]["PARM_minus_H1"]["hi"]),
        "",
        "PRESERVED (not converted):",
        "  - local 293FT D28 promoter calibration: FAIL (unchanged)",
        "  - A20/G20 2000bp constructs: UNSUPPORTED by PARM (L_max=600), no window-averaging substituted",
        "  - batch40 saved 0 overlap pairs vs printed 8821/8821 -> corrected independently (8821/8821) in batch41",
        "  - public HEK293FT != local 293FT/D28; minP/vector/autonomy UNKNOWN; retrospective transfer",
    ]
    for i, t in enumerate(lines):
        ax.text(0.01, 0.98 - i * 0.052, t, transform=ax.transAxes, fontsize=8.4,
                color=("crimson" if t.startswith("VERDICT") else "black"))
    fig.tight_layout(); fig.savefig(C.FIG / "fig3_scope_evidence.png", dpi=130); plt.close(fig)

    print(json.dumps({"figures": ["fig1_activity_ranking.png", "fig2_topdecile_uncertainty.png",
                                  "fig3_scope_evidence.png"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
