#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 strict near-homology grouping + deterministic group split.

Connected components over the 11803 GSE118242 164 bp sequences where an edge
exists iff two sequences are within Hamming <= 16 in FORWARD or REVERSE-COMPLEMENT
orientation (fixed equal length, exact duplicates included). Deterministic 80/10/10
group split, seed 20261023. Components stay whole -> cross-split near-homology is
zero by construction and is re-audited by an independent bounded recount.

If the strict grouping collapses validation/test into fewer than 30 independent
groups each, the strict split is NOT weakened; instead the pre-existing batch02
grouped 5-fold CV is used ONLY as a qualified historical-development analysis and
the residual similarity is disclosed.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from b21_common import (MPRA_SUPERVISED, DATA_OUT, REPORT_OUT, MPRA_LEN, BASE,
                        atomic_write_json, now_local, sha256_file)

HAM = 16
SEED_SPLIT = 20261023
MIN_GROUPS = 30
BLOCK = 1000


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


def _onehot(seqs):
    L = MPRA_LEN
    O = np.zeros((len(seqs), L * 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        codes = [BASE.get(c, 0) for c in s]
        for j, c in enumerate(codes):
            O[i, j * 4 + c] = 1.0
    return O


def build_components(seqs, ham=HAM):
    n = len(seqs)
    rc = [s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1] for s in seqs]
    O = _onehot(seqs); Orc = _onehot(rc)
    uf = UnionFind(n)
    thr = MPRA_LEN - ham
    n_edges = 0
    for b0 in range(0, n, BLOCK):
        b1 = min(n, b0 + BLOCK)
        Mf = O[b0:b1] @ O.T
        Mr = O[b0:b1] @ Orc.T
        for li in range(b1 - b0):
            gi = b0 + li
            hits = np.where((Mf[li] >= thr) | (Mr[li] >= thr))[0]
            for gj in hits:
                gj = int(gj)
                if gj > gi:
                    uf.union(gi, gj); n_edges += 1
                elif gj < gi:
                    uf.union(gi, gj); n_edges += 1
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    comp_of = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    comp_id = np.array([comp_of[int(r)] for r in roots], dtype=np.int64)
    return comp_id, n_edges


def split_groups(comp_id, n, seed=SEED_SPLIT, fracs=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    members = {}
    for i, c in enumerate(comp_id.tolist()):
        members.setdefault(c, []).append(i)
    comps = sorted(members.items(), key=lambda kv: (len(kv[1]), kv[0]))
    order = rng.permutation(len(comps))
    targets = {0: fracs[0] * n, 1: fracs[1] * n, 2: fracs[2] * n}
    used = {0: 0, 1: 0, 2: 0}
    split_of_comp = {}
    for k in order:
        cid, mem = comps[k]
        # fill-ratio greedy: put each whole component into the split that is
        # currently most under-filled relative to its target, which keeps the
        # 80/10/10 proportions and never starves the train split.
        best = min((0, 1, 2), key=lambda s: (used[s] + len(mem)) / targets[s])
        split_of_comp[cid] = best
        used[best] += len(mem)
    split_of = np.array([split_of_comp[int(c)] for c in comp_id], dtype=np.int8)
    return split_of, split_of_comp, {int(k): int(v) for k, v in used.items()}


def audit_cross_split(seqs, split_of, ham=HAM):
    """Independent bounded recount of cross-split pairs with Hamming<=16 (fwd + RC)."""
    n = len(seqs)
    rc = [s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1] for s in seqs]
    O = _onehot(seqs); Orc = _onehot(rc)
    thr = MPRA_LEN - ham
    fwd = rc_hit = 0
    idx = {s: np.where(split_of == s)[0] for s in (0, 1, 2)}
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            if b <= a:
                continue
            ia, ib = idx[a], idx[b]
            Mf = O[ia] @ O[ib].T
            Mr = O[ia] @ Orc[ib].T
            fwd += int((Mf >= thr).sum())
            rc_hit += int((Mr >= thr).sum())
    return {"cross_split_pairs_ham_le%d_fwd" % ham: fwd,
            "cross_split_pairs_ham_le%d_rc" % ham: rc_hit,
            "cross_split_pairs_ham_le%d_total" % ham: fwd + rc_hit}


def run():
    t0 = time.time()
    start = now_local()
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    y = df["activity"].to_numpy(dtype=np.float64)
    oligo = df["Oligo_index"].astype(str).tolist()
    n = len(seqs)
    comp_id, n_edges = build_components(seqs)
    split_of, split_of_comp, used = split_groups(comp_id, n)
    n_comp = len(set(comp_id.tolist()))
    per_split_comp = {s: len({int(comp_id[i]) for i in range(n) if split_of[i] == s}) for s in (0, 1, 2)}
    per_split_rows = {s: int((split_of == s).sum()) for s in (0, 1, 2)}
    audit = audit_cross_split(seqs, split_of)
    strict_ok = (per_split_comp[1] >= MIN_GROUPS and per_split_comp[2] >= MIN_GROUPS)
    res = {
        "generated_at_local": now_local(), "started_local": start,
        "elapsed_s": round(time.time() - t0, 1),
        "grouping": {"rule": "connected components, Hamming<=%d forward or RC, exact dup included" % HAM,
                     "n_sequences": n, "n_components": n_comp, "n_union_attempts": n_edges,
                     "max_component_rows": int(max(np.bincount(comp_id)))},
        "split": {"seed": SEED_SPLIT, "fractions": [0.8, 0.1, 0.1],
                  "split_names": {0: "train", 1: "val", 2: "test"},
                  "rows_per_split": per_split_rows, "components_per_split": per_split_comp},
        "strict_ok": bool(strict_ok), "min_groups_required": MIN_GROUPS,
        "cross_split_audit": audit,
        "source_sha256": {"mpra_supervised.tsv": sha256_file(MPRA_SUPERVISED)},
    }
    if not strict_ok:
        res["fallback"] = ("strict Hamming<=%d grouping collapses val/test below %d groups; "
                           "NOT weakened. Use the pre-existing batch02 grouped 5-fold CV only as a "
                           "qualified historical-development analysis with residual similarity disclosed."
                           % (HAM, MIN_GROUPS))
    np.savez(DATA_OUT / "groups.npz", comp_id=comp_id, split_of=split_of, activity=y,
             oligo=np.array(oligo))
    out = pd.DataFrame({"Oligo_index": oligo, "sequence_sha256": [__import__("hashlib").sha256(s.encode()).hexdigest() for s in seqs],
                        "activity": y, "component_id": comp_id,
                        "split": [["train", "val", "test"][int(s)] for s in split_of]})
    out.to_csv(DATA_OUT / "split_assignment.tsv", sep="\t", index=False)
    atomic_write_json(REPORT_OUT / "group_audit.json", res)
    return res
