# -*- coding: utf-8 -*-
"""Batch34 C: CORRECTED technical-tube figure.

Parent finding: batch33 fig2 drew symmetric +/- FULL-RANGE error bars (tube_range_log10_primary),
which overstates the interval and is not a valid interval. This corrected figure shows, per
construct per context, the INDIVIDUAL technical-tube points and a min..max vertical interval
(NEVER a symmetric +/- range). Cross-context gate labels and raw source endpoints are retained.
Technical tubes are NOT independent biological replicates.
"""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:/promoter_atlas")
SRC = ROOT / "data/breakthrough_20261002/batch33/frozen_eligibility_matrix.tsv"
FIG = ROOT / "reports/breakthrough_20261002/batch34/figures"
FIG.mkdir(parents=True, exist_ok=True)
fm = list(csv.DictReader(open(SRC, encoding="utf-8"), delimiter="\t"))

CTX = [("S004", "S004 early 5dps\n(gate: All Events)"),
       ("S001", "S001 early 7dps\n(gate: All Events)"),
       ("S002", "S002 D28 2nd-sort\n(gate: P2)")]
FAMS = {"ACTB": ["A1", "A2", "A3", "A4", "A5", "A20"],
        "GAPDH": ["G1", "G2", "G3", "G4", "G5", "G20"],
        "CMV": ["CMVcore", "CMVfull"]}

def cell(sid, cid):
    return next((r for r in fm if r["source_id"] == sid and r["construct_id"] == cid), None)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
for ci, (fam, members) in enumerate(FAMS.items()):
    ax = axes[0][ci]
    for k, c in enumerate(members):
        for xi, (sid, _) in enumerate(CTX):
            r = cell(sid, c)
            if not r:
                continue
            tub = [float(x) for x in r["v1r_median_by_tube"].split(";") if x != ""]
            wt = float(r["wt_negative_mean"])
            pts = [float(np.log10(t / wt)) for t in tub]
            x = xi + (k - (len(members) - 1) / 2.0) * 0.06
            ax.scatter([x] * len(pts), pts, s=14, alpha=0.85, zorder=3,
                       color=plt.cm.tab10(k))
            if len(pts) > 1:
                ax.plot([x, x], [min(pts), max(pts)], lw=1.1, color=plt.cm.tab10(k),
                        alpha=0.7, zorder=2)  # min..max interval (asymmetric, not +/-)
            ax.scatter([x], [float(r["PRIMARY_log10_relative_gfp"])], marker="D", s=26,
                       edgecolor="black", linewidth=0.5, color=plt.cm.tab10(k), zorder=4)
        lab = c + (" (2000bp model-unsupported)" if c in ("A20", "G20") else "")
        ax.scatter([], [], color=plt.cm.tab10(k), label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in CTX], fontsize=7.5)
    ax.set_title("%s primary: log10(V1R tube median / batch WT V1L median)" % fam, fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.3, loc="best")
    if ci == 0:
        ax.set_ylabel("primary log10 relative GFP")
    ax = axes[1][ci]
    for k, c in enumerate(members):
        xs, ys = [], []
        for xi, (sid, _) in enumerate(CTX):
            r = cell(sid, c)
            if r:
                xs.append(xi); ys.append(float(r["SECONDARY_positive_fraction_pct"]))
        ax.plot(xs, ys, marker="s", lw=1.2, ms=4, color=plt.cm.tab10(k), label=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(["early5", "early7", "D28"], fontsize=8)
    ax.set_title("%s SECONDARY positive-fraction %%\n(denominator early=All Events vs D28=P2 -> NOT comparable)" % fam, fontsize=7.6)
    ax.grid(alpha=0.3)
    if ci == 0:
        ax.set_ylabel("V1R %% of parent")
fig.suptitle("Batch34 CORRECTED technical-tube figure (same clone ROSA26-4; source: batch33 frozen_eligibility_matrix.tsv)\n"
             "individual tube points + min..max interval (NOT symmetric +/-range); technical tubes are NOT independent biological replicates",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = FIG / "fig_corrected_tube_intervals.png"
fig.savefig(out, dpi=140); plt.close(fig)
print(json.dumps({"wrote": str(out)}, ensure_ascii=False))
