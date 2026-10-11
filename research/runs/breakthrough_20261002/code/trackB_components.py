#!/usr/bin/env python3
"""Track B stage 1a: near-duplicate connected components (documented BEFORE scores).

Efficient pairwise Hamming via one-hot matmul: M = O2 @ O2.T counts matching
positions; Hamming = L - M. Components are connected components of the graph
where Hamming <= HAM. Also unions identical and reverse-complement pairs and the
known family_id relations. Reports component size distribution only; the split
manifest is written by build_trackB_split.py once the threshold policy is fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
OUT = ROOT / "data" / "breakthrough_20261002"
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(RC)[::-1]


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC, sep="\t").reset_index(drop=True)
    n = len(df)
    seqs = df["sequence"].astype(str).tolist()
    L = len(seqs[0])
    codes = np.stack([[BASE[c] for c in s] for s in seqs]).astype(np.int64)

    # one-hot matmul for matching counts
    O = np.eye(4, dtype=np.float32)[codes]  # (n, L, 4)
    O2 = O.reshape(n, L * 4)
    del O
    print("matmul", O2.shape, flush=True)
    M = (O2 @ O2.T).astype(np.float32)  # (n, n) match counts
    del O2
    H = (L - M).round().astype(np.int16)
    del M
    np.fill_diagonal(H, L + 1)
    print("ham computed", flush=True)

    dsu = DSU(n)
    # union near-duplicate pairs (H <= ham), vectorized
    rr, cc = np.nonzero(H <= args.ham)
    for a, b in zip(rr.tolist(), cc.tolist()):
        dsu.union(a, b)
    del H
    n_ham_pairs = int(len(rr))
    del rr, cc
    print("ham pairs:", n_ham_pairs, flush=True)

    # RC pairs
    rc_map = {}
    for i, s in enumerate(seqs):
        rc_map.setdefault(revcomp(s), []).append(i)
    n_rc = 0
    for i, s in enumerate(seqs):
        for j in rc_map.get(s, []):
            if j > i:
                dsu.union(i, j)
                n_rc += 1

    # known families
    n_fam = 0
    fam = {}
    for i, fid in enumerate(df["family_id"].fillna("")):
        if fid != "":
            fam.setdefault(fid, []).append(i)
    for members in fam.values():
        for k in range(1, len(members)):
            dsu.union(members[0], members[k])
            n_fam += 1

    comp = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    uniq, sizes = np.unique(comp, return_counts=True)
    sizes_sorted = -np.sort(-sizes)
    report = {
        "ham_threshold": args.ham,
        "n": n,
        "seq_length": L,
        "n_components": int(len(uniq)),
        "largest": int(sizes_sorted[0]),
        "largest_fraction": float(sizes_sorted[0] / n),
        "size_ge2": int((sizes >= 2).sum()),
        "size1": int((sizes == 1).sum()),
        "top20_sizes": [int(x) for x in sizes_sorted[:20]],
        "n_ham_pairs": n_ham_pairs,
        "n_rc_pairs": n_rc,
        "n_fam_links": n_fam,
    }
    (args.out / "trackB_components.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

