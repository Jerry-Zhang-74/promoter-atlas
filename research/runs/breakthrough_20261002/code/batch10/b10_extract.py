#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 deterministic feature extraction (1024 positional sequence features).

Reads the read-only sym400/up2000 memmaps via the frozen r400/r2000 row indices
from trackA_gene_promoters.npz, computes per-promoter block frequencies
(4 bins x 64 trinucleotides + 256 tetranucleotides per window), normalizes
within each promoter, and aggregates by arithmetic mean across ALL promoters of
each gene. Writes the 57827 x 1024 gene feature matrix plus feature dictionary,
QC/denominator provenance, manifest and code snapshot into a unique attempt dir.

Thread caps are set before numpy/sklearn imports. threadpoolctl limits are applied
around the numerical loop. CPU only, bounded chunks. The entry point FAILS if its
target attempt already contains a completed features1024.npz (no overwrite).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# set thread caps BEFORE numpy/sklearn import
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from b10_common import (ROOT, TRACKA_NPZ, MEM400, MEM2000, DATA_OUT, MODEL_OUT,
                        CANDIDATE_VERSION, BATCH, N_FEATURE_UNIVERSE, N_HUMAN,
                        L400, L2000, FEATURE_NAMES_1024, FEATURE_NAMES_45,
                        window_tri_tetra, sha256_file, sha256_bytes,
                        atomic_write_json, thread_caps_record)  # noqa: E402

CHUNK = 6000


def append_ledger(record):
    p = DATA_OUT / "attempts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def snapshot_code(attempt_dir):
    """Copy the immutable batch10 code into the attempt dir and record hashes."""
    src_dir = Path(__file__).resolve().parent
    dst = attempt_dir / "code"
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in sorted(src_dir.glob("b10_*.py")):
        shutil.copy2(f, dst / f.name)
        hashes[f.name] = sha256_file(f)
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    t0 = time.time()
    attempt_dir = DATA_OUT / ("attempt_%03d_extract" % args.attempt)
    final_npz = attempt_dir / "features1024.npz"
    # ---- non-overwrite guard ----
    if final_npz.exists():
        raise SystemExit("REFUSE: attempt output already exists: %s (no overwrite)" % final_npz)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    code_hashes = snapshot_code(attempt_dir)
    atomic_write_json(attempt_dir / "code_hashes.json", code_hashes)
    source_hashes = {
        "trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
        "promoter_sym400_uint8.memmap": sha256_file(MEM400),
        "promoter_up2000_uint8.memmap": sha256_file(MEM2000),
    }
    atomic_write_json(attempt_dir / "source_hashes.json", source_hashes)

    # ---- load indices / grouping ----
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    r400 = z["r400"].astype(np.int64)
    r2000 = z["r2000"].astype(np.int64)
    starts = z["starts"].astype(np.int64)
    n_promoters_total = len(r400)
    n_genes = len(starts) - 1
    assert n_genes == N_FEATURE_UNIVERSE
    # promoter -> gene index (contiguous groups by starts)
    gene_idx = np.repeat(np.arange(n_genes, dtype=np.int64), np.diff(starts))
    assert len(gene_idx) == n_promoters_total

    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    mm2000 = np.memmap(MEM2000, dtype="uint8", mode="r", shape=(N_HUMAN, L2000))

    accum = np.zeros((n_genes, 1024), dtype=np.float64)
    qc_tot = {k: 0 for k in ("bin%d_tri_zero_denom_promoters" % b for b in range(4))}
    qc_tot.update({
        "bin%d_tri_zero_denom_promoters" % b: 0 for b in range(4)
    })
    qc_tot = {
        "sym400_bin0_tri_zero_denom_promoters": 0,
        "sym400_bin1_tri_zero_denom_promoters": 0,
        "sym400_bin2_tri_zero_denom_promoters": 0,
        "sym400_bin3_tri_zero_denom_promoters": 0,
        "sym400_tetra_zero_denom_promoters": 0,
        "up2000_bin0_tri_zero_denom_promoters": 0,
        "up2000_bin1_tri_zero_denom_promoters": 0,
        "up2000_bin2_tri_zero_denom_promoters": 0,
        "up2000_bin3_tri_zero_denom_promoters": 0,
        "up2000_tetra_zero_denom_promoters": 0,
    }

    chunk_manifest = []
    n_chunks = int(np.ceil(n_promoters_total / args.chunk))
    t_loop = time.time()
    with threadpool_limits(limits=4):
        for ci in range(n_chunks):
            c0 = ci * args.chunk
            c1 = min(c0 + args.chunk, n_promoters_total)
            ridx400 = r400[c0:c1]
            ridx2000 = r2000[c0:c1]
            seq400 = np.asarray(mm400[ridx400], dtype=np.uint8)
            seq2000 = np.asarray(mm2000[ridx2000], dtype=np.uint8)
            tri_sym, tetra_sym, qc_sym = window_tri_tetra(seq400, L400)
            tri_up, tetra_up, qc_up = window_tri_tetra(seq2000, L2000)
            block = np.concatenate([tri_sym, tri_up, tetra_sym, tetra_up], axis=1)
            assert block.shape == (c1 - c0, 1024)
            np.add.at(accum, gene_idx[c0:c1], block)
            # QC aggregation
            qc_tot["sym400_bin0_tri_zero_denom_promoters"] += qc_sym["bin0_tri_zero_denom_promoters"]
            qc_tot["sym400_bin1_tri_zero_denom_promoters"] += qc_sym["bin1_tri_zero_denom_promoters"]
            qc_tot["sym400_bin2_tri_zero_denom_promoters"] += qc_sym["bin2_tri_zero_denom_promoters"]
            qc_tot["sym400_bin3_tri_zero_denom_promoters"] += qc_sym["bin3_tri_zero_denom_promoters"]
            qc_tot["sym400_tetra_zero_denom_promoters"] += qc_sym["tetra_zero_denom_promoters"]
            qc_tot["up2000_bin0_tri_zero_denom_promoters"] += qc_up["bin0_tri_zero_denom_promoters"]
            qc_tot["up2000_bin1_tri_zero_denom_promoters"] += qc_up["bin1_tri_zero_denom_promoters"]
            qc_tot["up2000_bin2_tri_zero_denom_promoters"] += qc_up["bin2_tri_zero_denom_promoters"]
            qc_tot["up2000_bin3_tri_zero_denom_promoters"] += qc_up["bin3_tri_zero_denom_promoters"]
            qc_tot["up2000_tetra_zero_denom_promoters"] += qc_up["tetra_zero_denom_promoters"]
            chunk_manifest.append({
                "chunk": ci, "c0": int(c0), "c1": int(c1),
                "block_sha256": sha256_bytes(block.tobytes()),
            })
            if ci % 5 == 0 or ci == n_chunks - 1:
                print("chunk %d/%d (promoters %d..%d)" % (ci + 1, n_chunks, c0, c1), flush=True)

    # divide by promoter count (arithmetic mean over all promoters)
    n_prom_gene = np.diff(starts).astype(np.float64)
    assert (n_prom_gene > 0).all()
    X1024 = accum / n_prom_gene[:, None]

    # ---- write final artifacts ----
    np.savez_compressed(final_npz, X1024=X1024)
    feature_dictionary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "n_genes": int(n_genes),
        "n_features": 1024,
        "feature_names_1024": FEATURE_NAMES_1024,
        "baseline_feature_names_45": FEATURE_NAMES_45,
        "total_features_with_baseline": 1069,
        "order_note": ("X1024 column order = sym400 tri (4 bins x 64), up2000 tri (4 bins x 64), "
                       "sym400 tetra (256), up2000 tetra (256); k-mers lexicographic A,C,G,T"),
        "aggregation": "per-promoter block frequencies (zero valid starts => all-zero block), then arithmetic mean over ALL promoters of the gene",
        "encoding": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
        "windows": {"sym400": "length 400, bins [0,100),[100,200),[200,300),[300,400)",
                    "up2000": "length 2000, bins [0,500),[500,1000),[1000,1500),[1500,2000)"},
        "strand_orientation": "preserved; no reverse-complement",
        "gene_identity": "gene_id[starts[:-1]]",
    }
    atomic_write_json(attempt_dir / "feature_dictionary.json", feature_dictionary)

    manifest = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "attempt": args.attempt,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_promoters_total": int(n_promoters_total),
        "n_genes": int(n_genes),
        "n_chunks": int(n_chunks),
        "chunk_size": args.chunk,
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "zero_denominator_promoter_counts": qc_tot,
        "thread_caps": thread_caps_record(),
        "elapsed_s": round(time.time() - t0, 2),
        "loop_elapsed_s": round(time.time() - t_loop, 2),
        "chunk_manifest": chunk_manifest,
    }
    atomic_write_json(attempt_dir / "manifest.json", manifest)

    # append-only attempt ledger
    append_ledger({
        "ts_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "extract",
        "attempt": args.attempt,
        "attempt_dir": str(attempt_dir.resolve()),
        "status": "complete",
        "n_genes": int(n_genes),
        "n_promoters": int(n_promoters_total),
        "features1024_sha256": sha256_file(final_npz),
        "zero_denominator_promoter_counts": qc_tot,
        "elapsed_s": round(time.time() - t0, 2),
    })

    print("WROTE", final_npz)
    print(json.dumps({"n_genes": n_genes, "n_features": 1024,
                      "zero_denominator_promoter_counts": qc_tot,
                      "elapsed_s": round(time.time() - t0, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
