#!/usr/bin/env python3
"""Build the MPRA supervised table from Weingarten-Gabbay 2019 (GSE118242, K562).

Outputs:
    features/mpra/mpra_supervised.tsv   (oligo_index, sequence, activity, split, family_id)
    features/mpra/mpra_families.tsv     (test variant families for paired ranking eval)

Filters: Total_reads_bins >= 10, CV <= 1.0, Mean_log10 finite.
Families: near-duplicate groups (Hamming distance <= 2) found within a sampled
subset, used to evaluate per-promoter variant ranking (the design-task metric).
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/mpra/GSM3323461_oligos_measurements_processed_data.tab.gz"
OUT = ROOT / "features/mpra/mpra_supervised.tsv"
FAM = ROOT / "features/mpra/mpra_families.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
FAMILY_SAMPLE = 2500
MAX_HAM = 8


def encode(s):
    return np.array([BASE.get(c, 4) for c in s.upper()], dtype=np.uint8)


def find_families(seqs, max_ham=MAX_HAM):
    """Find connected components of near-duplicate sequences via pairwise Hamming."""
    n = len(seqs)
    codes = [encode(s) for s in seqs]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # chunked pairwise Hamming
    CH = 250
    L = len(codes[0])
    arr = np.stack(codes)
    for i in range(0, n, CH):
        blk = arr[i:i + CH]
        d = (blk[:, None, :] != arr[None, :, :]).sum(axis=2)
        for a in range(len(blk)):
            for b in range(n):
                if b != i + a and d[a, b] <= max_ham:
                    union(i + a, b)
    fam = np.array([find(i) for i in range(n)])
    return fam


def main():
    df = pd.read_csv(SRC, sep="\t")
    df = df[(df["Total_reads_bins"] >= 10)
            & (df["CV"] <= 1.0)
            & (df["Mean_log10"].notna())
            & (df["Oligo_sequence"].str.len() == 164)].reset_index(drop=True)
    print("after filter:", len(df))

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    n = len(df)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    split = np.full(n, "test", dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_va]] = "val"
    df["split"] = split

    # family detection on a sampled subset (for the paired ranking test set)
    fam_idx = np.sort(rng.choice(n, size=min(FAMILY_SAMPLE, n), replace=False))
    sub = df.iloc[fam_idx].reset_index(drop=True)
    labels = find_families(sub["Oligo_sequence"].tolist())
    sub["family_id"] = ["F%d" % x for x in labels]
    fams = sub[sub.groupby("family_id")["family_id"].transform("size") >= 5]
    fams = fams[fams["split"] == "test"].rename(
        columns={"Oligo_sequence": "sequence", "Mean_log10": "activity"})
    print("families (size>=5, test):", fams["family_id"].nunique(), "| seqs:", len(fams))

    # assign family ids (only meaningful for the sampled subset)
    df["family_id"] = ""
    df.loc[fam_idx, "family_id"] = sub["family_id"].to_numpy()

    out = df[["Oligo_index", "Oligo_sequence", "Mean_log10", "Weighted_Avg",
              "Total_reads_bins", "CV", "split", "family_id"]].rename(
        columns={"Oligo_sequence": "sequence", "Mean_log10": "activity"})
    out.to_csv(OUT, sep="\t", index=False)
    if len(fams) > 0:
        fams[["Oligo_index", "family_id", "sequence", "activity", "split"]].to_csv(FAM, sep="\t", index=False)
        print("wrote", FAM)
    else:
        print("NOTE: no near-duplicate families found (paired eval uses ISM-vs-oracle protocol)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
