# -*- coding: utf-8 -*-
"""batch49 Protocol D: ONE scientific figure. E_raw vs S_raw over the frozen catalog, with the Pareto
frontier highlighted and the B gate noted. Draws ONLY from the CACHED default selection
(data/breakthrough_20261002/batch49/out/gene_rank_default.json + all_eligible.tsv): NO new compute."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b49_common as C  # noqa: E402


def _read_eligible_tsv(p):
    xs, ys, bs = [], [], []
    with open(p, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                xs.append(float(r["E_raw"]))
                ys.append(float(r["S_raw"]))
                bs.append(float(r["B_raw"]))
            except (TypeError, ValueError):
                continue
    return xs, ys, bs


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default_json = C.B49_DATA / "out" / "gene_rank_default.json"
    all_eligible = C.B49_DATA / "out" / "all_eligible.tsv"
    res = json.loads(default_json.read_text(encoding="utf-8"))
    xs, ys, bs = _read_eligible_tsv(all_eligible)
    front = [r for r in res["selected"] if r["is_frontier"]]
    fx = [r["E_raw"] for r in front]
    fy = [r["S_raw"] for r in front]

    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=140)
    sc = ax.scatter(xs, ys, c=bs, cmap="viridis", s=10, alpha=0.35, linewidths=0,
                    label="eligible catalog genes (n=%d)" % len(xs))
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("B_raw (breadth gate; detectable fraction)")
    ax.plot(fx, fy, "-", color="crimson", lw=1.2, alpha=0.6, zorder=3)
    ax.scatter(fx, fy, s=120, facecolors="none", edgecolors="crimson", linewidths=1.8, zorder=4,
               label="Pareto frontier (n=%d)" % len(front))
    for r in front:
        ax.annotate(r["symbol"] or r["gene_id"], (r["E_raw"], r["S_raw"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, color="crimson")
    ax.axhline(0.0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("E_raw  (cross-tissue mean log2(1+TPM); higher = stronger)")
    ax.set_ylabel("S_raw  (-MAD over detectable tissues; higher = less cross-tissue dispersion)")
    ax.set_title("batch49 gene-rank: frozen catalog E_raw x S_raw Pareto frontier\n"
                 "(B_raw gate min_b=%.2f; both axes higher-better; catalog is cross-tissue, NOT 293FT)"
                 % res["params"]["min_b"], fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    out = C.B49_REPORT / "figures" / "gene_rank_frontier.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {"figure": str(out.relative_to(C.ROOT).as_posix()), "n_eligible": len(xs), "n_frontier": len(front),
            "source": "cached default selection (no new compute)",
            "existing_figures_preserved_by_link": [
                "reports/breakthrough_20261002/batch45/figures/fig1_native_trajectory.png",
                "reports/breakthrough_20261002/batch45/figures/fig2_ranking_replicates.png",
                "reports/breakthrough_20261002/batch45/figures/fig3_evidence_scope.png"],
            "no_packaging_loop": True}


if __name__ == "__main__":
    print(json.dumps(draw(), ensure_ascii=False))
