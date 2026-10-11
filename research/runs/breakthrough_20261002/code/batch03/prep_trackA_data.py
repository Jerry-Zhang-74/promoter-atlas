#!/usr/bin/env python3
"""Batch03 Track A data preparation (UTF-8 explicit, deterministic)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs" / "overnight_20261002" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import atomic_write_json, sha256_file  # noqa: E402
from finetune_v63_adaptation import donor_exposure_sets, ORTHO  # noqa: E402

CACHE = ROOT / "data" / "breakthrough_20261002"
OUT = ROOT / "data" / "breakthrough_20261002" / "batch03"
INPUT = ROOT / "labels" / "build" / "promoter_atlas_v6_input.tsv"
N_HUMAN = 507365
L400 = 400
L2000 = 2000


def load_rows():
    rows = pd.read_csv(CACHE / "trackA_rows.tsv", sep="\t")
    assert (rows["species"] == "homo_sapiens").all()
    return rows


def build_gene_tables(rows):
    per_gene_splits = rows.groupby("gene_id")["split"].nunique()
    bad = int((per_gene_splits > 1).sum())
    assert bad == 0, f"{bad} genes span multiple splits"
    g = rows.groupby("gene_id").agg(
        split=("split", "first"),
        Score_E=("Score_E", "first"),
        Score_S=("Score_S", "first"),
        Score_B=("Score_B", "first"),
        n_promoters=("record_id", "count"),
    ).reset_index()
    g = g.sort_values("gene_id", kind="mergesort").reset_index(drop=True)
    return g


def exposure_sets():
    df = pd.read_csv(INPUT, sep="\t", low_memory=False,
                     usecols=["record_id", "species", "gene_id", "split",
                              "label_mask", "Score_E", "ortholog_E"])
    df["split"] = df["split"].fillna("train")
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    exposed_test, exposed_val, exp_rows, htest, hval = donor_exposure_sets(df, ORTHO)
    return {
        "exposed_test_genes": sorted(exposed_test),
        "exposed_val_genes": sorted(exposed_val),
        "n_exposed_test": len(exposed_test),
        "n_exposed_val": len(exposed_val),
        "per_species_rows": exp_rows,
    }


def rc_bytes(b):
    # A(0)<->T(3), C(1)<->G(2), N(4)->N(4)
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[b[::-1]]


def homology_audit(rows):
    mm400 = np.memmap(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap",
                      dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    mm2000 = np.memmap(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap",
                       dtype="uint8", mode="r", shape=(N_HUMAN, L2000))
    split = rows["split"].values
    r400 = rows["r400"].to_numpy(dtype=np.int64)
    r2000 = rows["r2000"].to_numpy(dtype=np.int64)
    audit = {}
    for name, mm, ridx, L in [("sym400", mm400, r400, L400), ("up2000", mm2000, r2000, L2000)]:
        seen = {}
        dup_pairs = 0
        rc_pairs = 0
        cross_split = {"train_val": 0, "train_test": 0, "val_test": 0}
        for i in range(len(rows)):
            seq = mm[ridx[i]].copy()
            key = seq.tobytes()
            rkey = rc_bytes(seq).tobytes()
            s = split[i]
            if key in seen:
                dup_pairs += 1
                s0 = seen[key]
                if {s, s0} == {"train", "val"}: cross_split["train_val"] += 1
                if {s, s0} == {"train", "test"}: cross_split["train_test"] += 1
                if {s, s0} == {"val", "test"}: cross_split["val_test"] += 1
            if rkey in seen:
                rc_pairs += 1
                s0 = seen[rkey]
                if {s, s0} == {"train", "val"}: cross_split["train_val"] += 1
                if {s, s0} == {"train", "test"}: cross_split["train_test"] += 1
                if {s, s0} == {"val", "test"}: cross_split["val_test"] += 1
            if key not in seen:
                seen[key] = s
            if rkey not in seen:
                seen[rkey] = s
        audit[name] = {"n_rows": int(len(rows)), "exact_dup_rows": dup_pairs,
                       "rc_dup_rows": rc_pairs, "cross_split": cross_split}
    audit["near_identity_20mer_400bp"] = near_identity_400bp(rows, mm400, r400, split)
    return audit


def near_identity_400bp(rows, mm400, r400, split, k=20, bloom_bits=2 ** 28):
    n_hash = 3
    bloom = np.zeros(bloom_bits, dtype=bool)
    pw = (4 ** np.arange(k, dtype=np.int64)).astype(np.int64)
    train_mask = split == "train"
    tr_idx = np.flatnonzero(train_mask)
    for start in range(0, len(tr_idx), 8000):
        idx = tr_idx[start:start + 8000]
        arr = np.asarray(mm400[r400[idx]], dtype=np.int64)
        n = arr.shape[0]
        codes = np.zeros((n, L400 - k + 1), dtype=np.int64)
        for pos in range(0, L400 - k + 1):
            codes[:, pos] = (arr[:, pos:pos + k] * pw).sum(axis=1)
        for h in range(n_hash):
            hvals = (codes * (2654435761 + h * 40503) + h * 2246822519) % bloom_bits
            bloom[hvals] = True

    def frac_present(seqs):
        tot = np.zeros(len(seqs), dtype=np.int64)
        hit = np.zeros(len(seqs), dtype=np.int64)
        for start in range(0, len(seqs), 4000):
            idx2 = seqs[start:start + 4000]
            arr = np.asarray(mm400[r400[idx2]], dtype=np.int64)
            n = arr.shape[0]
            for pos in range(0, L400 - k + 1):
                codes = (arr[:, pos:pos + k] * pw).sum(axis=1)
                present = np.ones(n, dtype=bool)
                for h in range(n_hash):
                    hvals = (codes * (2654435761 + h * 40503) + h * 2246822519) % bloom_bits
                    present &= bloom[hvals]
                hit[start:start + n] += present
            tot[start:start + n] += (L400 - k + 1)
        return hit / np.maximum(tot, 1)

    out = {}
    for s in ["val", "test"]:
        m = split == s
        frac = frac_present(np.flatnonzero(m))
        out[s] = {
            "n_promoters": int(m.sum()),
            "n_gt50pct_kmer_overlap": int((frac > 0.5).sum()),
            "max_kmer_overlap_fraction": float(frac.max()) if len(frac) else 0.0,
            "mean_kmer_overlap_fraction": float(frac.mean()) if len(frac) else 0.0,
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    g = build_gene_tables(rows)
    exp = exposure_sets()
    audit = homology_audit(rows)

    prom = rows[["record_id", "gene_id", "split", "r400", "r2000", "Score_E", "Score_S", "Score_B"]].copy()
    prom = prom.sort_values(["gene_id", "record_id"], kind="mergesort").reset_index(drop=True)
    prom.to_csv(OUT / "trackA_promoters.tsv", sep="\t", index=False)
    g.to_csv(OUT / "trackA_genes.tsv", sep="\t", index=False)

    meta = {
        "input_rows_sha256": sha256_file(CACHE / "trackA_rows.tsv"),
        "n_rows": int(len(rows)),
        "n_genes": {"train": int((g.split == "train").sum()),
                    "val": int((g.split == "val").sum()),
                    "test": int((g.split == "test").sum())},
        "n_promoters": {"train": int((rows.split == "train").sum()),
                        "val": int((rows.split == "val").sum()),
                        "test": int((rows.split == "test").sum())},
        "promoters_per_gene": {
            "train": {int(k): int(v) for k, v in g[g.split == "train"].n_promoters.value_counts().sort_index().items()},
            "val": {int(k): int(v) for k, v in g[g.split == "val"].n_promoters.value_counts().sort_index().items()},
            "test": {int(k): int(v) for k, v in g[g.split == "test"].n_promoters.value_counts().sort_index().items()},
        },
        "exposure": {k: exp[k] for k in ["n_exposed_test", "n_exposed_val", "per_species_rows"]},
        "exposed_test_genes": exp["exposed_test_genes"],
        "exposed_val_genes": exp["exposed_val_genes"],
        "homology_audit": audit,
    }
    atomic_write_json(OUT / "trackA_prep_meta.json", meta)

    order = np.argsort(rows["gene_id"].values, kind="mergesort")
    rows_s = rows.iloc[order].reset_index(drop=True)
    codes = pd.factorize(rows_s["gene_id"].values, sort=True)[0]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(codes)) + 1, [len(codes)]])
    np.savez_compressed(
        OUT / "trackA_gene_promoters.npz",
        gene_id=rows_s["gene_id"].values,
        record_id=rows_s["record_id"].values,
        r400=rows_s["r400"].to_numpy(dtype=np.int64),
        r2000=rows_s["r2000"].to_numpy(dtype=np.int64),
        split=rows_s["split"].values,
        Score_E=rows_s["Score_E"].to_numpy(dtype=np.float32),
        starts=starts.astype(np.int64),
    )
    print(json.dumps({k: meta[k] for k in ["n_rows", "n_genes", "n_promoters"]}, indent=2))
    print("exposure n", meta["exposure"]["n_exposed_test"], meta["exposure"]["n_exposed_val"])
    print("homology", json.dumps(audit, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
