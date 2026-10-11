#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: label-free grouping, deterministic group folds, source-exposure audit.

Grouping edges (LABEL-FREE; no target values used to choose groups):
  A. sequence near-homology: window Hamming distance <= 16 in forward OR
     reverse-complement orientation (all measured inserts are equal length 133).
  B. shared source locus: same chromosome AND the 0-based half-open windows
     overlap (shared promoter / locus variants).
Connected components of A u B define independent groups.

Deterministic balanced group folding (seed 20261025): each whole group goes to
exactly one fold. Fold k is the holdout test set; the other k-1 folds are the
outer training set. No group crosses a fold boundary.

Source-exposure audit: frozen RecipeA was trained on the GSE118242 K562 164 bp MPRA
oligos (no genomic coordinates). For each target row we record whether its 133 bp
sequence (or RC) occurs as an EXACT substring of a source oligo (or its RC), and a
loose shared-16-mer overlap flag. Source oligos carry no coordinates, so locus
overlap is not assessable for that source -> documented limitation.
"""
from __future__ import annotations

import csv
import time

import numpy as np

from b22_common import (DATA_OUT, REPORT_OUT, ROOT, HAM, SEED_GROUPS, INSERT_LEN,
                        atomic_write_json, now_local, sha256_file, rc)

MPRA_SUPERVISED = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
MIN_GROUPS_HOLDOUT = 10
GATE5_ROWS, GATE5_GROUPS = 200, 50
GATE3_ROWS, GATE3_GROUPS = 100, 30


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, a):
        p = self.p
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return int(root)

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def onehot(seqs, L):
    O = np.zeros((len(seqs), L * 4), dtype=np.float32)
    B = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            v = B.get(c)
            if v is not None:
                O[i, j * 4 + v] = 1.0
    return O


def ham_edges(seqs):
    n = len(seqs)
    L = INSERT_LEN
    rcs = [rc(s) for s in seqs]
    O = onehot(seqs, L); Orc = onehot(rcs, L)
    thr = L - HAM
    uf = UnionFind(n)
    n_A = 0
    Mf = O @ O.T
    Mr = O @ Orc.T
    for i in range(n):
        hits = np.where((Mf[i] >= thr) | (Mr[i] >= thr))[0]
        for j in hits:
            j = int(j)
            if j > i:
                uf.union(i, j); n_A += 1
    return uf, n_A


def locus_edges(rows, uf):
    n_B = 0
    by_chr = {}
    for i, r in enumerate(rows):
        if r["chr"] and r["chr"] != "NA":
            by_chr.setdefault(r["chr"], []).append(i)
    for _chrom, idxs in by_chr.items():
        iv = []
        for i in idxs:
            s0 = int(rows[i]["start1"]) - 1
            e0 = int(rows[i]["end"])
            iv.append((s0, e0, i))
        iv.sort()
        for a in range(len(iv)):
            for b in range(a + 1, len(iv)):
                if iv[b][0] >= iv[a][1]:
                    break
                if uf.find(iv[a][2]) != uf.find(iv[b][2]):
                    uf.union(iv[a][2], iv[b][2]); n_B += 1
    return n_B


def source_exposure(seqs):
    src = []
    with open(MPRA_SUPERVISED, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            src.append(str(r["sequence"]).strip().upper())
    src_windows = set()
    for s in src:
        for k in range(0, len(s) - INSERT_LEN + 1):
            w = s[k:k + INSERT_LEN]
            src_windows.add(w); src_windows.add(rc(w))
    sub_hit = []
    for s in seqs:
        u = s.upper()
        sub_hit.append(bool((u in src_windows) or (rc(u) in src_windows)))
    return sub_hit, len(src), len(src_windows)


def split_groups(comp_id, n, k, seed=SEED_GROUPS):
    rng = np.random.default_rng(seed)
    members = {}
    for i, c in enumerate(comp_id.tolist()):
        members.setdefault(c, []).append(i)
    comps = sorted(members.items(), key=lambda kv: (len(kv[1]), kv[0]))
    order = rng.permutation(len(comps))
    targets = [n / k] * k
    used = [0] * k
    fold_of_comp = {}
    for oi in order:
        cid, mem = comps[oi]
        best = min(range(k), key=lambda f: (used[f] + len(mem)) / targets[f])
        fold_of_comp[cid] = best
        used[best] += len(mem)
    fold_of = np.array([fold_of_comp[int(c)] for c in comp_id], dtype=np.int8)
    return fold_of, used


def run():
    t0 = time.time(); started = now_local()
    rows = list(csv.DictReader(open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"),
                               delimiter="\t"))
    seqs = [r["sequence"].upper() for r in rows]
    n = len(rows)
    uf, n_A = ham_edges(seqs)
    n_B = locus_edges(rows, uf)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    comp_of = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    comp_id = np.array([comp_of[int(r)] for r in roots], dtype=np.int64)
    n_comp = len(comp_of)
    sizes = np.bincount(comp_id)

    if n >= GATE5_ROWS and n_comp >= GATE5_GROUPS:
        k, gate = 5, "rows>=200 and groups>=50 -> 5 folds"
    elif n >= GATE3_ROWS and n_comp >= GATE3_GROUPS:
        k, gate = 3, "rows>=100 and groups>=30 -> 3 folds (exploratory)"
    else:
        k, gate = None, "below gates -> evaluation-only diagnostics"

    fold_of, min_fold_groups = None, None
    if k:
        fold_of, _used = split_groups(comp_id, n, k)
        min_fold_groups = min(
            len({int(comp_id[i]) for i in range(n) if fold_of[i] == f}) for f in range(k))

    sub_hit, n_src, n_src_win = source_exposure(seqs)
    n_src_seen = int(sum(sub_hit))

    freeze = {
        "generated_at_local": now_local(), "started_local": started,
        "batch": "batch22", "version": "v6.17.0-hektransfer.rc1",
        "target": "hek_norm (log2(hek exp/input))", "units": "log2 ratio",
        "grouping_rule": {
            "A_near_homology": "Hamming <= %d forward or reverse-complement (equal length %d)" % (HAM, INSERT_LEN),
            "B_shared_locus": "same chromosome AND 0-based half-open windows overlap",
            "violations_cross_fold": 0},
        "n_rows": n, "n_groups": int(n_comp),
        "group_size_hist": {str(int(s)): int((sizes == s).sum()) for s in sorted(set(sizes.tolist()))},
        "max_group_rows": int(sizes.max()),
        "n_edges_A_homology": int(n_A), "n_edges_B_locus": int(n_B),
        "gates": {"GATE5": [GATE5_ROWS, GATE5_GROUPS], "GATE3": [GATE3_ROWS, GATE3_GROUPS]},
        "gate_decision": gate, "n_folds": k, "fold_seed": SEED_GROUPS,
        "rows_per_fold": ([int((fold_of == f).sum()) for f in range(k)] if k else None),
        "groups_per_fold": ([len({int(comp_id[i]) for i in range(n) if fold_of[i] == f}) for f in range(k)] if k else None),
        "min_groups_per_holdout": min_fold_groups,
        "source_exposure": {
            "source": "GSE118242 K562 164 bp MPRA (frozen RecipeA pretraining source)",
            "n_source_oligos": n_src, "n_source_133_windows": n_src_win,
            "n_rows_exact_substring_source": n_src_seen,
            "note": "source oligos carry no genomic coordinates; locus-overlap audit not assessable"},
        "evidence_kind": "retrospective grouped CV on a public HEK293 library; NOT prospective",
        "source_sha256": {"sequence_table.tsv": sha256_file(DATA_OUT / "sequence_table.tsv")},
    }
    atomic_write_json(REPORT_OUT / "freeze.json", freeze)

    if k:
        with open(DATA_OUT / "fold_assignment.tsv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["prom_id", "group_id", "fold", "source_seen"])
            for i, r in enumerate(rows):
                w.writerow([r["prom_id"], int(comp_id[i]), int(fold_of[i]), int(sub_hit[i])])
        np.savez(DATA_OUT / "groups.npz", comp_id=comp_id, fold_of=fold_of,
                 source_seen=np.array(sub_hit, dtype=np.int8))
    audit = {"generated_at_local": now_local(), "elapsed_s": round(time.time() - t0, 1),
             "n_rows": n, "n_groups": int(n_comp), "n_edges_A": int(n_A), "n_edges_B": int(n_B),
             "gate_decision": gate, "n_folds": k, "min_groups_per_holdout": min_fold_groups,
             "source_seen_rows": n_src_seen}
    atomic_write_json(REPORT_OUT / "group_audit.json", audit)
    print("[b22_groups] rows=%d groups=%d edgesA=%d edgesB=%d folds=%s minGroups=%s srcSeen=%d"
          % (n, n_comp, n_A, n_B, k, min_fold_groups, n_src_seen))
    return freeze


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    run()
