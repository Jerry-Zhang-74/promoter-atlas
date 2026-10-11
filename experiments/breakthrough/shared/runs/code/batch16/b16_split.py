#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 corrected outer split builder (deterministic, label-independent).

Per the binding batch16 protocol (parent amendment 2026-10-03 10:02):

  * Base = ALL original batch05 components over the full gene universe
    (expected 57827 feature-covered genes; asserted below).
  * Union confirmed batch15 RC gene edges by exact geneID, merging whole
    original components (never rebuilding only the 18769 eligible nodes).
  * New componentID = min sorted geneID within the full union group.
  * Deterministic fold allocation from the first 8 bytes (big-endian) of
    SHA256(UTF8 'batch16|20261016|' + newcomponentID), modulo 1000:
        0..799 -> train, 800..899 -> val, 900..999 -> test.
  * No target labels or score-dependent sorting; no re-draw.

Writes (all under data/breakthrough_20261002/batch16/split/):
  gene_fold_assignment.tsv   gene -> new component id / fold / old component / old split
  component_manifest.tsv     new component id -> fold / n_genes / n_promoters
  split_arrays.npz           gene_split (train/val/test), new_comp_dense, new_component_id
  split_manifest.json        counts, membership moves, SHAs, assertions
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
DATA15 = ROOT / "data" / "breakthrough_20261002" / "batch15"
DATA16 = ROOT / "data" / "breakthrough_20261002" / "batch16"
SPLIT_OUT = DATA16 / "split"

EXPECTED_N_GENES = 57827
KEY_PREFIX = "batch16|20261016|"


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def deterministic_fold(new_component_id: str) -> str:
    """Hash 'batch16|20261016|<newcomponentID>' -> first 8 bytes BE % 1000."""
    key = (KEY_PREFIX + new_component_id).encode("utf-8")
    x = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    bucket = x % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "val"
    return "test"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SPLIT_OUT)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- load original batch05 assignment (authoritative 57827 gene universe) ----
    sa = pd.read_csv(DATA05 / "split_assignment.tsv", sep="\t",
                     dtype={"gene_id": str, "component_id": np.int64,
                            "split": str, "n_promoters": np.int64})
    assert len(sa) == EXPECTED_N_GENES, "expected %d genes, got %d" % (EXPECTED_N_GENES, len(sa))
    assert sa.gene_id.is_unique, "gene_id not unique"
    n_components_old = int(sa.component_id.max()) + 1
    # component id must be contiguous 0..n_components_old-1 (from b5_split comp_dense)
    assert set(np.unique(sa.component_id.values)) == set(range(n_components_old)), "component ids not contiguous"

    # ---- load confirmed batch15 RC gene edges ----
    ed = pd.read_csv(DATA15 / "verified_gene_edges.tsv", sep="\t", dtype=str)
    assert len(ed) == 1039, "expected 1039 verified gene edges, got %d" % len(ed)

    # map gene -> old component id
    gene_to_comp = dict(zip(sa.gene_id.values, sa.component_id.values))

    # ---- union-find over original components, driven by confirmed gene edges ----
    uf = UnionFind(n_components_old)
    n_union_ops = 0
    edge_genes = set()
    for ga, gb in zip(ed.gene_a.values, ed.gene_b.values):
        assert ga in gene_to_comp and gb in gene_to_comp, "edge gene missing: %s/%s" % (ga, gb)
        edge_genes.add(ga)
        edge_genes.add(gb)
        if uf.union(gene_to_comp[ga], gene_to_comp[gb]):
            n_union_ops += 1

    # ---- new component id = min sorted geneID within union group ----
    comp_to_genes = {}
    for gid, cid in zip(sa.gene_id.values, sa.component_id.values):
        comp_to_genes.setdefault(uf.find(int(cid)), []).append(gid)

    root_to_newid = {}
    comp_members = {}
    for root, genes in comp_to_genes.items():
        genes_sorted = sorted(genes)          # lexicographic gene-id sort
        newid = genes_sorted[0]               # min sorted geneID
        root_to_newid[root] = newid
        comp_members[newid] = genes_sorted

    new_comp_ids = np.array([root_to_newid[uf.find(int(c))] for c in sa.component_id.values], dtype=object)
    folds = np.array([deterministic_fold(cid) for cid in new_comp_ids], dtype=object)

    # ---- assemble per-gene table ----
    out_genes = pd.DataFrame({
        "gene_id": sa.gene_id.values,
        "new_component_id": new_comp_ids,
        "new_fold": folds,
        "old_component_id": sa.component_id.values,
        "old_split": sa.split.values,
        "n_promoters": sa.n_promoters.values,
    })

    # ---- component manifest ----
    comp_rows = []
    for newid in sorted(comp_members):
        genes = comp_members[newid]
        sub = out_genes[out_genes.new_component_id == newid]
        fold = sub.new_fold.iloc[0]
        n_prom = int(sub.n_promoters.sum())
        comp_rows.append({
            "new_component_id": newid,
            "fold": fold,
            "n_genes": len(genes),
            "n_promoters": n_prom,
            "member_genes": ",".join(genes),
        })
    comp_df = pd.DataFrame(comp_rows)
    n_components_new = len(comp_df)

    # ---- old-to-new membership moves ----
    new_to_old_comps = {}
    for gid, ocid, ncid in zip(sa.gene_id.values, sa.component_id.values, new_comp_ids):
        new_to_old_comps.setdefault(ncid, set()).add(int(ocid))
    merged_new = {ncid: sorted(olds) for ncid, olds in new_to_old_comps.items() if len(olds) > 1}
    n_merged_new = len(merged_new)

    # map old component -> new component id (whole old component always maps to one new comp)
    old_to_new = {}
    for gid, ocid, ncid in zip(sa.gene_id.values, sa.component_id.values, new_comp_ids):
        prev = old_to_new.setdefault(int(ocid), ncid)
        assert prev == ncid, "old component %d split across new components" % ocid
    moves = [{"old_component_id": ocid, "new_component_id": old_to_new[ocid]}
             for ocid in range(n_components_old)]

    # ---- assertions: no confirmed edge crosses the new fold boundary ----
    crossing_edges = []
    ga_map = dict(zip(sa.gene_id.values, new_comp_ids))
    gf_map = dict(zip(sa.gene_id.values, folds))
    for ga, gb in zip(ed.gene_a.values, ed.gene_b.values):
        na, nb = ga_map[ga], ga_map[gb]
        fa, fb = gf_map[ga], gf_map[gb]
        if na != nb or fa != fb:
            crossing_edges.append({"gene_a": ga, "gene_b": gb, "new_comp_a": na,
                                   "new_comp_b": nb, "fold_a": fa, "fold_b": fb})
    assert len(crossing_edges) == 0, "confirmed RC edges cross new split: %d" % len(crossing_edges)

    # ---- write artifacts ----
    out_genes.to_csv(out / "gene_fold_assignment.tsv", sep="\t", index=False)
    comp_df.to_csv(out / "component_manifest.tsv", sep="\t", index=False)

    newid_to_dense = {cid: i for i, cid in enumerate(sorted(comp_members))}
    new_comp_dense = np.array([newid_to_dense[cid] for cid in new_comp_ids], dtype=np.int64)
    gene_split = folds.astype(str)
    np.savez_compressed(
        out / "split_arrays.npz",
        gene_split=gene_split,
        new_comp_dense=new_comp_dense,
        new_component_id=new_comp_ids,
        old_component_id=sa.component_id.values,
    )

    fold_counts = out_genes.new_fold.value_counts().to_dict()
    comp_fold_counts = comp_df.fold.value_counts().to_dict()

    manifest = {
        "candidate_version": "v6.15.0-multitask.rc1",
        "batch": "batch16",
        "split_name": "corrected component-union (batch05 exact/RC+forward-near-identity components + batch15 confirmed RC edges), hash fold allocation",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "protocol_ref": "runs/breakthrough_20261002/batch16/PROTOCOL.txt",
        "key_prefix": KEY_PREFIX,
        "allocation": "sha256(key)[:8] big-endian % 1000 -> train<800, val<900, test>=900",
        "counts": {
            "n_genes": int(len(out_genes)),
            "n_genes_expected": EXPECTED_N_GENES,
            "n_components_old": int(n_components_old),
            "n_components_new": int(n_components_new),
            "n_merged_new_components": int(n_merged_new),
            "n_union_ops": int(n_union_ops),
            "gene_fold_counts": {k: int(v) for k, v in fold_counts.items()},
            "component_fold_counts": {k: int(v) for k, v in comp_fold_counts.items()},
            "n_edge_genes": int(len(edge_genes)),
        },
        "assertions": {
            "expected_57827_genes_asserted": True,
            "contiguous_old_component_ids": True,
            "1039_verified_edges_loaded": True,
            "no_confirmed_rc_edge_crosses_new_split": True,
            "old_components_never_split": True,
        },
        "old_to_new_membership_moves": moves,
        "merged_new_components": {str(k): [int(x) for x in v] for k, v in merged_new.items()},
        "source_hashes": {
            "batch05_split_assignment.tsv": sha256_file(DATA05 / "split_assignment.tsv"),
            "batch05_components.tsv": sha256_file(DATA05 / "components.tsv"),
            "batch05_split_arrays.npz": sha256_file(DATA05 / "split_arrays.npz"),
            "batch15_verified_gene_edges.tsv": sha256_file(DATA15 / "verified_gene_edges.tsv"),
        },
        "output_hashes": {
            "gene_fold_assignment.tsv": sha256_file(out / "gene_fold_assignment.tsv"),
            "component_manifest.tsv": sha256_file(out / "component_manifest.tsv"),
            "split_arrays.npz": sha256_file(out / "split_arrays.npz"),
        },
    }
    atomic_write_json(out / "split_manifest.json", manifest)

    print(json.dumps({
        "n_genes": int(len(out_genes)),
        "n_components_old": int(n_components_old),
        "n_components_new": int(n_components_new),
        "n_merged_new_components": int(n_merged_new),
        "n_union_ops": int(n_union_ops),
        "gene_fold_counts": {k: int(v) for k, v in fold_counts.items()},
        "component_fold_counts": {k: int(v) for k, v in comp_fold_counts.items()},
        "no_crossing_edges": True,
        "manifest": str(out / "split_manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
