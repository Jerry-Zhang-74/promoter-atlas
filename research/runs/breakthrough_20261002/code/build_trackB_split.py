#!/usr/bin/env python3
"""Track B stage 1b: build the grouped split from documented components.

Uses the efficient pairwise-Hamming matmul (same as trackB_components.py) to
recover the connected components documented BEFORE any score, then assigns whole
components to train/val/test with a deterministic largest-first greedy policy so
no component crosses a fold. The old labels/oldsplit file is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
FAM = ROOT / "features" / "mpra" / "mpra_families.tsv"
OUT = ROOT / "data" / "breakthrough_20261002"
HAM = 8
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGT", "TGCA")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def revcomp(s):
    return s.translate(RC)[::-1]


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC, sep="\t").reset_index(drop=True)
    n = len(df)
    seqs = df["sequence"].astype(str).tolist()
    L = len(seqs[0])
    codes = np.stack([[BASE[c] for c in s] for s in seqs]).astype(np.int64)

    O = np.eye(4, dtype=np.float32)[codes].reshape(n, L * 4)
    M = O @ O.T
    del O
    H = (L - M).round().astype(np.int16)
    del M
    np.fill_diagonal(H, L + 1)

    dsu = DSU(n)
    rr, cc = np.nonzero(H <= HAM)
    for a, b in zip(rr.tolist(), cc.tolist()):
        dsu.union(a, b)
    del H, rr, cc

    # identical / reverse-complement
    by_seq = {}
    for i, s in enumerate(seqs):
        by_seq.setdefault(s, []).append(i)
    for group in by_seq.values():
        for k in range(1, len(group)):
            dsu.union(group[0], group[k])
    rc_map = {}
    for i, s in enumerate(seqs):
        rc_map.setdefault(revcomp(s), []).append(i)
    for i, s in enumerate(seqs):
        for j in rc_map.get(s, []):
            if j > i:
                dsu.union(i, j)

    # known family_id (non-blank)
    fam = {}
    for i, fid in enumerate(df["family_id"].fillna("")):
        if fid != "":
            fam.setdefault(fid, []).append(i)
    for members in fam.values():
        for k in range(1, len(members)):
            dsu.union(members[0], members[k])

    root = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    uniq = np.unique(root)
    sizes = np.array([dsu.size[r] for r in uniq], dtype=np.int64)
    n_comp = len(uniq)

    # deterministic largest-first greedy assignment (whole components kept)
    order = np.argsort(-sizes, kind="mergesort")  # size desc, stable
    total = int(sizes.sum())
    target_train = int(round(total * args.train_frac))
    target_val = int(round(total * args.val_frac))
    target_test = total - target_train - target_val
    remain = np.array([target_train, target_val, target_test], dtype=np.int64)
    assign = np.full(n_comp, 2, dtype=np.int8)  # 0 train, 1 val, 2 test
    for k in order:
        sz = int(sizes[k])
        # fold with largest remaining deficit wins (ties -> train first)
        deficit = remain.copy()
        choice = int(np.argmax(deficit))
        assign[k] = choice
        remain[choice] -= sz
    fold_names = np.array(["train", "val", "test"])
    seq_fold = fold_names[assign[np.searchsorted(uniq, root)]]

    df = df.rename(columns={"split": "old_split"})
    df["new_split"] = seq_fold
    df["component_id"] = ["C%d" % int(r) for r in root]
    counts = df["new_split"].value_counts().to_dict()
    print("split counts:", counts, flush=True)
    print("components:", n_comp, "largest:", int(sizes.max()), flush=True)

    # leakage sanity: no component spans folds
    span = {}
    for i in range(n):
        span.setdefault(int(root[i]), set()).add(seq_fold[i])
    n_span = sum(1 for v in span.values() if len(v) > 1)
    assert n_span == 0, f"{n_span} components span folds"

    manifest = {
        "source_sha256": sha256_file(SRC),
        "families_sha256": sha256_file(FAM),
        "n_sequences": n,
        "seq_length": L,
        "ham_threshold": HAM,
        "n_components": int(n_comp),
        "largest_component": int(sizes.max()),
        "largest_fraction": float(sizes.max() / total),
        "n_components_size_ge2": int((sizes >= 2).sum()),
        "n_components_size1": int((sizes == 1).sum()),
        "split_counts": {str(k): int(v) for k, v in counts.items()},
        "target_fracs": {"train": args.train_frac, "val": args.val_frac, "test": round(1 - args.train_frac - args.val_frac, 6)},
        "seed": args.seed,
        "assignment_policy": "deterministic largest-first greedy; whole components never cross folds; ties broken toward train",
        "old_split_note": "old random split (train 9442 / val 1180 / test 1181) preserved immutably in mpra_supervised.tsv; old mpra_scratch test Spearman 0.7077 came from that old split and is not directly comparable.",
        "component_span_folds": int(n_span),
    }
    (args.out / "trackB_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    df[["Oligo_index", "sequence", "activity", "component_id", "new_split", "family_id", "old_split"]].to_csv(
        args.out / "trackB_grouped.tsv", sep="\t", index=False)
    print(json.dumps(manifest, indent=2), flush=True)
    print("wrote", args.out / "trackB_grouped.tsv", flush=True)


if __name__ == "__main__":
    main()

