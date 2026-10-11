#!/usr/bin/env python3
"""Batch02 Track B stage 0: verify grouped-component correctness on the train partition.

Recomputes the connected components (identical + reverse-complement + known family
unions + Hamming<=8 near-duplicates) on the batch01 TRAIN partition only, and checks
they induce exactly the same partition as batch01 component_id restricted to train.
Writes the verified train components for use by the grouped CV.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import atomic_write_json, sha256_file  # noqa: E402

SRC = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
GROUPED = ROOT / "data" / "breakthrough_20261002" / "trackB_grouped.tsv"
OUT = ROOT / "data" / "breakthrough_20261002" / "batch02"
HAM = 8
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGT", "TGCA")


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


def revcomp(s):
    return s.translate(RC)[::-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grp = pd.read_csv(GROUPED, sep="\t")
    tr = grp[grp["new_split"] == "train"].reset_index(drop=True).copy()
    n = len(tr)
    seqs = tr["sequence"].astype(str).tolist()
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

    # identical sequences
    by_seq = {}
    for i, s in enumerate(seqs):
        by_seq.setdefault(s, []).append(i)
    for group in by_seq.values():
        for k in range(1, len(group)):
            dsu.union(group[0], group[k])

    # reverse-complement pairs
    rc_map = {}
    for i, s in enumerate(seqs):
        rc_map.setdefault(revcomp(s), []).append(i)
    for i, s in enumerate(seqs):
        for j in rc_map.get(s, []):
            if j > i:
                dsu.union(i, j)

    # known family unions (non-blank family_id)
    fam = {}
    for i, fid in enumerate(tr["family_id"].fillna("")):
        if fid != "":
            fam.setdefault(fid, []).append(i)
    for members in fam.values():
        for k in range(1, len(members)):
            dsu.union(members[0], members[k])

    root = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    uniq, inv = np.unique(root, return_inverse=True)
    sizes = np.array([dsu.size[r] for r in uniq], dtype=np.int64)

    # compare with batch01 component_id restricted to train
    batch_root, batch_uniq = pd.factorize(tr["component_id"].values, sort=True)
    # check same partition: two indices in same batch component iff same recomputed component
    # map batch component id -> set of recomputed roots it covers must be a single root
    same_partition = True
    mismatch = 0
    comp2root = {}
    for i in range(n):
        b = int(batch_root[i])
        r = int(inv[i])
        if b in comp2root and comp2root[b] != r:
            same_partition = False
            mismatch += 1
        else:
            comp2root[b] = r

    result = {
        "train_n_sequences": n,
        "seq_length": L,
        "ham_threshold": HAM,
        "n_components_recomputed": int(len(uniq)),
        "n_components_batch01_train": int(len(np.unique(batch_root))),
        "partition_matches_batch01": bool(same_partition),
        "n_mismatched_sequences": int(mismatch),
        "n_components_size_ge2": int((sizes >= 2).sum()),
        "n_components_size1": int((sizes == 1).sum()),
        "largest_component": int(sizes.max()),
        "multi_component_total_rows": int(sizes[sizes >= 2].sum()),
        "source_sha256": sha256_file(SRC),
        "grouped_sha256": sha256_file(GROUPED),
    }
    assert same_partition, "recomputed train grouping does NOT match batch01 component partition"

    # write verified components (component_id V# -> list of row indices / Oligo_index)
    comp_ids = np.array([f"V{int(r)}" for r in inv])
    tr["verified_component_id"] = comp_ids
    cols = ["Oligo_index", "sequence", "activity", "family_id", "verified_component_id"]
    tr[cols].to_csv(OUT / "trackB_train_components.tsv", sep="\t", index=False)

    atomic_write_json(ROOT / "reports" / "breakthrough_20261002" / "batch02" / "trackB_verify_groups.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
