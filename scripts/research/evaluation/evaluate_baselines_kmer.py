#!/usr/bin/env python3
"""Simple baselines for the E/S/B gene-level ranking task (5-mer Ridge + GC content).

Trains on human train rows, evaluates on the frozen gene-level test split with
the SAME aggregation (mean per gene) and spearman metric as the neural model.
This gives the "would k-mer counts alone do it" reference.

Output: models/eval/baselines_kmer_gc.tsv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs" / "scripts"))
from train_promoter_atlas_v6 import spearman_np, r2_np, gene_level_metrics, rankdata_average

INPUT = ROOT / "labels" / "build" / "promoter_atlas_v6_input.tsv"
MM2000 = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"
OUT = ROOT / "models" / "eval" / "baselines_kmer_gc.tsv"
K = 5
N_HUMAN = 507365

def kmer_counts(rows, mm, k=K):
    """(n, 4^k) counts per row from the 2000bp memmap."""
    counts = np.zeros((len(rows), 4 ** k), dtype=np.float32)
    stride = 0
    for i, r in enumerate(rows):
        seq = mm[r].astype(np.int64)
        valid = seq < 4
        seq[~valid] = 0
        # rolling hash of k-mers
        code = 0
        mask = (4 ** k) - 1
        for pos in range(2000):
            code = ((code << 2) | seq[pos]) & mask
            if pos >= k - 1:
                counts[i, code] += 1.0
        if i % 50000 == 0:
            print("  rows", i, flush=True)
    counts /= 2000.0
    return counts

def main():
    df = pd.read_csv(INPUT, sep="\t", low_memory=False)
    df = df[df["species"] == "homo_sapiens"]
    df = df[df["label_mask"] == 1]
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print("train rows:", len(tr), "test rows:", len(te), flush=True)
    mm = np.memmap(MM2000, dtype="uint8", mode="r", shape=(N_HUMAN, 2000))
    print("train kmer...", flush=True)
    Xtr = kmer_counts(tr["r2000"].to_numpy(), mm)
    print("test kmer...", flush=True)
    Xte = kmer_counts(te["r2000"].to_numpy(), mm)
    gc_tr = (tr["r2000"].to_numpy() * 0)  # placeholder; GC from counts below
    # GC content feature
    gc_tr = np.zeros((len(tr), 1), dtype=np.float32)
    gc_te = np.zeros((len(te), 1), dtype=np.float32)
    for i, r in enumerate(tr["r2000"].to_numpy()):
        s = mm[r]
        gc_tr[i, 0] = float(((s == 1) | (s == 2)).mean())
    for i, r in enumerate(te["r2000"].to_numpy()):
        s = mm[r]
        gc_te[i, 0] = float(((s == 1) | (s == 2)).mean())

    rows = []
    for name, Xtr_f, Xte_f in [("kmer5_ridge", Xtr, Xte), ("gc_only", gc_tr, gc_te), ("kmer5+gc", np.hstack([Xtr, gc_tr]), np.hstack([Xte, gc_te]))]:
        preds = np.zeros((len(te), 3), dtype=np.float32)
        for j, c in enumerate(["Score_E", "Score_S", "Score_B"]):
            y = tr[c].to_numpy(dtype=np.float32)
            mask = ~np.isnan(y)
            model = Ridge(alpha=1.0)
            model.fit(Xtr_f[mask], y[mask])
            preds[:, j] = model.predict(Xte_f)
        yte = te[["Score_E", "Score_S", "Score_B"]].to_numpy(dtype=np.float32)
        mte = (~np.isnan(yte)).astype(np.float32)
        g = gene_level_metrics(te["gene_id"].to_numpy(), preds, yte, mte, n_boot=200)
        for axis in ["Score_E", "Score_S", "Score_B"]:
            rows.append({"baseline": name, "axis": axis,
                         "gene_sp": g[f"{axis}_gene_spearman"],
                         "gene_r2": g[f"{axis}_gene_r2"],
                         "sp_ci_lo": g[f"{axis}_gene_sp_ci_lo"],
                         "sp_ci_hi": g[f"{axis}_gene_sp_ci_hi"]})
        print(name, {k: round(g[k], 4) for k in ["Score_E_gene_spearman", "Score_S_gene_spearman", "Score_B_gene_spearman"]}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
