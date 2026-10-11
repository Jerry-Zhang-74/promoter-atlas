# -*- coding: utf-8 -*-
"""batch46 Phase D: at most TWO meaningful figures.
 fig1: same-generation selection gain across fixed generations (per lineage + generation median), short/long.
 fig2: fixed-selector utility and errors (utility vs uniform, hit-rate, sign counts) with scope/labels.
"""
import sys, json, math, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T
import b46_phaseB_analysis as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C.FIG.mkdir(parents=True, exist_ok=True)


def load():
    frags = T.build_fragments()
    freeze = T.load_freezer()
    triplets = T.build_triplets(frags, freeze)
    recs = A.per_triplet_records(triplets)
    prim = json.loads((C.OUT / "primary_contrasts.json").read_text(encoding="utf-8"))
    sel = json.loads((C.OUT / "selector_utility.json").read_text(encoding="utf-8"))
    return recs, prim, sel


def fig1(recs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
    for ax, lib in zip(axes, ("short", "long")):
        gens = sorted({r["generation"] for r in recs if r["library"] == lib})
        for key, col, lab in (("dmm_K562_meas", "#1f77b4", "max - median"),
                              ("dmn_K562_meas", "#d62728", "max - min")):
            for lin in sorted({r["lineage"] for r in recs if r["library"] == lib}):
                pts = sorted((r["generation"], r[key]) for r in recs
                             if r["library"] == lib and r["lineage"] == lin
                             and r[key] is not None and math.isfinite(r[key]))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, alpha=0.18, lw=0.8)
            med = []
            for g in gens:
                vals = [r[key] for r in recs if r["library"] == lib and r["generation"] == g
                        and r[key] is not None and math.isfinite(r[key])]
                med.append(C.median(vals))
            ax.plot(gens, med, color=col, marker="o", lw=2.2, label=lab + " (per-generation median)")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title("%s library (n=%d triplets) K562 measured contrast" %
                     (lib, sum(1 for r in recs if r["library"] == lib)), fontsize=10)
        ax.set_xlabel("GA generation"); ax.set_ylabel("measured log2 RNA/DNA difference")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig1 same-generation selection gain within a lineage (K562 fixed summary).\n"
                 "Faint lines = 10 GASearch lineages (NOT independent n); bold = generation median.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(C.FIG / "fig1_within_generation_gain.png", dpi=130)
    plt.close(fig)


def fig2(sel):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    libs = ["short", "long"]
    xs = range(len(libs))
    w = 0.36
    # (a) mean utility vs uniform-among-these-three
    ax = axes[0]
    for off, name, col in ((-w / 2, "K562_score", "#1f77b4"), (w / 2, "GC_high", "#ff7f0e")):
        vals = [sel["primary_K562"][name][lib]["mean_utility_vs_uniform"] for lib in libs]
        ax.bar([x + off for x in xs], vals, width=w, color=col, label=name)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(list(xs)); ax.set_xticklabels(libs)
    ax.set_ylabel("mean utility vs uniform (log2 units)")
    ax.set_title("(a) fixed-selector utility, K562", fontsize=10); ax.legend(fontsize=8)
    # (b) hit rate (tie-inclusive) vs chance 1/3
    ax = axes[1]
    for off, name, col in ((-w / 2, "K562_score", "#1f77b4"), (w / 2, "GC_high", "#ff7f0e")):
        vals = [sel["primary_K562"][name][lib]["hit_rate_tie_inclusive"] for lib in libs]
        ax.bar([x + off for x in xs], vals, width=w, color=col, label=name)
    ax.axhline(1 / 3, color="r", lw=1.0, ls=":")
    ax.text(0.02, 1 / 3 + 0.01, "uniform 1/3", color="r", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(libs); ax.set_ylim(0, 1)
    ax.set_ylabel("measured winner hit-rate (tie-inclusive)")
    ax.set_title("(b) selector hit-rate, K562", fontsize=10); ax.legend(fontsize=8)
    # (c) sign counts of utility for K562-score selector
    ax = axes[2]
    for i, lib in enumerate(libs):
        s = sel["primary_K562"]["K562_score"][lib]["utility_sign_counts"]
        bottom = 0
        for k, col in (("positive", "#2ca02c"), ("zero", "#cccccc"), ("negative", "#d62728")):
            ax.bar(i, s[k], bottom=bottom, color=col, label=k if i == 0 else None)
            bottom += s[k]
    ax.set_xticks(list(xs)); ax.set_xticklabels(libs)
    ax.set_ylabel("triplets"); ax.set_title("(c) K562-score utility sign counts", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Fig2 fixed-selection utility and errors (K562 primary). Uniform = random choice among THESE 3 "
                 "author-selected candidates; regret-vs-oracle is hindsight only; no CI/p-value.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(C.FIG / "fig2_selector_utility.png", dpi=130)
    plt.close(fig)


def main():
    recs, prim, sel = load()
    fig1(recs)
    fig2(sel)
    C.write_json(C.OUT / "figures_manifest.json", {
        "batch": C.BATCH, "version": C.VERSION,
        "figures": ["figures/fig1_within_generation_gain.png", "figures/fig2_selector_utility.png"],
        "definitions": {
            "fig1": "per-lineage (faint) and per-generation-median (bold) K562 measured max-median / max-min contrasts; y=0 dashed.",
            "fig2": "(a) mean utility vs uniform; (b) measured winner hit-rate vs uniform 1/3; (c) sign counts. Titles match plotted metrics."}})
    print("figures written")


if __name__ == "__main__":
    main()
