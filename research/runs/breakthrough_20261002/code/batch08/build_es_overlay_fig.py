#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small E/S overlay figure: legacy Score_S vs S_affine vs raw S_raw, on 18947 defined genes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
OUT = ROOT / "reports" / "breakthrough_20261002" / "batch08" / "fig_ES_conventions.png"

def main():
    d = pd.read_csv(AUDIT, sep="\t", low_memory=False)
    d = d[d["S_defined_main"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    a = axes[0]; b = axes[1]; c = axes[2]

    a.hexbin(d["Score_E"], d["Score_S"], gridsize=55, mincnt=1, cmap="viridis")
    a.set_title("A) Score_E vs legacy Score_S\n(S rank over 58988 genes)", fontsize=10)
    a.set_xlabel("Score_E"); a.set_ylabel("legacy Score_S (0.6788..1)")

    b.hexbin(d["Score_E"], d["S_affine"], gridsize=55, mincnt=1, cmap="viridis")
    b.set_title("B) Score_E vs S_affine\n(affine rescale to 0..1)", fontsize=10)
    b.set_xlabel("Score_E"); b.set_ylabel("S_affine (0..1)")

    c.hexbin(d["E_raw"], d["S_raw_6dec"], gridsize=55, mincnt=1, cmap="viridis")
    c.set_title("C) E_raw vs S_raw (raw trace)\n(S_raw = -MAD log2(1+TPM))", fontsize=10)
    c.set_xlabel("E_raw = mean log2(1+TPM)"); c.set_ylabel("S_raw")

    fig.suptitle("E x S design space, 18947 S-defined genes (v6.9.0-labelaudit.rc1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=150)
    print("saved", OUT, "genes", len(d))

if __name__ == "__main__":
    main()
