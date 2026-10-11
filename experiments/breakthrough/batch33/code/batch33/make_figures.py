# -*- coding: utf-8 -*-
"""Batch33 C: <=3 concise figures (matplotlib Agg)."""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(r"E:/promoter_atlas")
OUT  = ROOT/"data/breakthrough_20261002/batch33"
FIG  = ROOT/"reports/breakthrough_20261002/batch33/figures"
FIG.mkdir(parents=True, exist_ok=True)

fm = list(csv.DictReader(open(OUT/"frozen_eligibility_matrix.tsv", encoding="utf-8"), delimiter="\t"))
ctx = list(csv.DictReader(open(OUT/"context_registry.tsv", encoding="utf-8"), delimiter="\t"))
grid = list(csv.DictReader(open(OUT/"model_direction_grid.tsv", encoding="utf-8"), delimiter="\t"))
summ = json.load(open(OUT/"comparison_summary.json", encoding="utf-8"))

# ---------------- Figure 1: availability / context ----------------
order = ["S005","S006","S004","S001","S003","S007","S002"]
labelmap = {"S005":"S005 ROSA26-10\n5dps(13d?)","S006":"S006 ROSA26-10\n7dps(15d?)",
            "S004":"S004 ROSA26-4\n5dps/12dpt","S001":"S001 ROSA26-4\n7dps/14dpt",
            "S003":"S003 ROSA26-10\n10dps/18dpt","S007":"S007 ROSA26-10\n14dps/22dpt",
            "S002":"S002 ROSA26-4 D28\n28dps/43dpt 2ndsort"}
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(order))
joined = []
for i, sid in enumerate(order):
    c = next(c for c in ctx if c["source_id"] == sid)
    n = sum(1 for r in fm if r["source_id"] == sid and r["clone_reported"] == "ROSA26-4")
    is_join = (sid in ["S004","S001","S002"])
    joined.append(is_join)
    ax.barh(i, 14 if is_join else max(1, n), color=("#2c7fb8" if is_join else "#c7c7c7"), height=0.6)
    ax.text(0.15, i, "clone=%s  parent=%s" % (c["clone_reported"], c["v1_gate_parent_inferred"]), va="center", fontsize=8, color="white" if is_join else "black")
ax.set_yticks(y); ax.set_yticklabels([labelmap[s] for s in order], fontsize=8)
ax.set_xlim(0, 15); ax.set_xlabel("constructs with rebuilt endpoint (blue = eligible same-clone longitudinal set)")
ax.set_title("Batch33 A availability: 7 exported measurement contexts (all cell_line_as_recorded=HEK293T; USER target=293FT)\n"
             "eligible longitudinal set = 3 x 14 = 42 (blue); ROSA26-10 exports = different clone / unmapped tube names (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"fig1_context_availability.png", dpi=140); plt.close(fig)

# ---------------- Figure 2: observed family trends (primary + secondary separate) ----------------
fams = {"ACTB":["A1","A2","A3","A4","A5","A20"], "GAPDH":["G1","G2","G3","G4","G5","G20"], "CMV":["CMVcore","CMVfull"]}
ctxs = [("S004","early 5dps"), ("S001","early 7dps"), ("S002","D28 2nd-sort")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ci, (fam, mem) in enumerate(fams.items()):
    ax = axes[0][ci]
    for c in mem:
        xs, ys, es = [], [], []
        for xi, (sid, _) in enumerate(ctxs):
            r = next((r for r in fm if r["source_id"]==sid and r["construct_id"]==c), None)
            if r:
                xs.append(xi); ys.append(float(r["PRIMARY_log10_relative_gfp"]))
                es.append(float(r["tube_range_log10_primary"] or 0))
        lab = c + (" (2000bp, model-unsupported)" if c in ("A20","G20") else "")
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.3, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _,n in ctxs], fontsize=8)
    ax.set_title("%s primary endpoint\nlog10(V1R median / batch WT V1L median)" % fam, fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    if ci == 0: ax.set_ylabel("primary log10 relative GFP")
    ax = axes[1][ci]
    for c in mem:
        xs, ys = [], []
        for xi, (sid, _) in enumerate(ctxs):
            r = next((r for r in fm if r["source_id"]==sid and r["construct_id"]==c), None)
            if r: xs.append(xi); ys.append(float(r["SECONDARY_positive_fraction_pct"]))
        ax.plot(xs, ys, marker="s", lw=1.2, label=c)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _,n in ctxs], fontsize=8)
    ax.set_title("%s SECONDARY positive-fraction %%\n(denominator: early=All Events, D28=P2 -> NOT comparable)" % fam, fontsize=8)
    ax.grid(alpha=0.3)
    if ci == 0: ax.set_ylabel("V1R % of parent")
fig.suptitle("Batch33 A observed contexts (same clone ROSA26-4; technical tubes only; cross-sort = cross-context, not temporal decay)", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"fig2_family_trends.png", dpi=140); plt.close(fig)
print("fig1,fig2 done")

# ---------------- Figure 3: frozen-model direction grid ----------------
models = ["PARM","H1","R1"]
contexts = [("agree_at_D28","D28"),("agree_at_early5","early 5dps"),("agree_at_early7","early 7dps")]
fam_of = {}
for g in grid: fam_of[(g["a"],g["b"])] = g["family"]
M = np.full((len(models), len(contexts)), np.nan)
for mi, m in enumerate(models):
    for cj, (key, _) in enumerate(contexts):
        vals = [int(g[key]) for g in grid if g["model"]==m and g[key] != ""]
        if vals: M[mi, cj] = sum(vals)/len(vals)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(contexts))); ax.set_xticklabels([n for _,n in contexts])
ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
for mi in range(len(models)):
    for cj in range(len(contexts)):
        if not np.isnan(M[mi,cj]):
            n = sum(1 for g in grid if g["model"]==models[mi] and g[contexts[cj][0]] != "")
            ax.text(cj, mi, "%.2f\n(n=%d pairs)" % (M[mi,cj], n), ha="center", va="center", fontsize=9)
ax.set_title("Batch33 frozen-model pair-direction agreement (pooled, <=600bp; UNCHANGED saved scores)\n"
             "no new calibration; 21 pairs are NOT independent n", fontsize=9)
fig.colorbar(im, ax=ax, label="direction agreement", shrink=0.8)
fig.tight_layout(); fig.savefig(FIG/"fig3_model_direction_grid.png", dpi=140); plt.close(fig)
print("fig3 done")
for _, n in [("", "")]:
    pass
print(json.dumps({k: summ[k] for k in ("n_observed_cells","n_pairs")}))
