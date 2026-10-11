#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 frozen data assembly: features, sequence slots, split, train-only scalers.

Builds the diverse 4-slot sequence row indices per gene, loads the 1069-feature
matrix, corrected split, and E/S/B targets/masks; fits train-only feature scaler
and per-target standardization stats; persists everything for training/replay.

Writes (data/breakthrough_20261002/batch16/):
  data.npz          features X (57827x1069), slot_r400 (57827x4), split, targets/masks
  scalers.npz       X scaler + per-target train mean/std
  data_manifest.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MEM400, N_FEATURE_UNIVERSE, N_HUMAN, L400,
                        N_SLOTS, TASKS, diverse_slots, load_features, load_split,
                        sha256_file, atomic_write_json)


def build_slot_r400(starts, r400):
    n_genes = N_FEATURE_UNIVERSE
    slot_r400 = np.zeros((n_genes, N_SLOTS), dtype=np.int64)
    for g in range(n_genes):
        n = int(starts[g + 1] - starts[g])
        assert n > 0
        for k, s in enumerate(diverse_slots(n, g)):
            slot_r400[g, k] = int(r400[starts[g] + s])
    return slot_r400


def main():
    t0 = time.time()
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    lab = np.load(DATA_OUT / "labels.npz", allow_pickle=True)
    gene_gid = lab["gene_gid"].astype(str)
    starts = lab["starts"].astype(np.int64)
    r400 = lab["r400"].astype(np.int64)
    gene_split = lab["gene_split"].astype(str)
    new_comp_dense = lab["new_comp_dense"].astype(np.int64)
    E = lab["E"].astype(np.float64)
    S = lab["S"].astype(np.float64)
    B = lab["B"].astype(np.float64)
    E_mask = lab["E_mask"].astype(bool)
    S_mask = lab["S_mask"].astype(bool)
    B_mask = lab["B_mask"].astype(bool)

    X = load_features()

    # ---- diverse 4-slot sequence row indices ----
    slot_r400 = build_slot_r400(starts, r400)

    # ---- split masks (gene-level) ----
    tr = gene_split == "train"
    va = gene_split == "val"
    te = gene_split == "test"

    # ---- train-only feature scaler (fit on ALL train genes, shared across arms) ----
    X_train = X[tr]
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    # ---- per-target train-only standardization stats ----
    target_stats = {}
    Y = {"E": E, "S": S, "B": B}
    M = {"E": E_mask, "S": S_mask, "B": B_mask}
    for task in TASKS:
        y = Y[task]
        m = M[task]
        tr_valid = tr & m
        assert tr_valid.sum() > 0, "no valid train rows for %s" % task
        mu = float(np.mean(y[tr_valid]))
        sd = float(np.std(y[tr_valid]))
        if sd == 0:
            sd = 1.0
        target_stats[task] = {"mean": mu, "std": sd, "n_train_valid": int(tr_valid.sum())}

    # ---- persist ----
    np.savez_compressed(
        DATA_OUT / "data.npz",
        gene_gid=gene_gid,
        starts=starts,
        slot_r400=slot_r400,
        gene_split=gene_split,
        new_comp_dense=new_comp_dense,
        X=X,
        E=E, S=S, B=B,
        E_mask=E_mask, S_mask=S_mask, B_mask=B_mask,
    )
    np.savez_compressed(
        DATA_OUT / "scalers.npz",
        X_mean=X_mean, X_std=X_std,
        E_mean=target_stats["E"]["mean"], E_std=target_stats["E"]["std"],
        S_mean=target_stats["S"]["mean"], S_std=target_stats["S"]["std"],
        B_mean=target_stats["B"]["mean"], B_std=target_stats["B"]["std"],
    )

    manifest = {
        "candidate_version": "v6.15.0-multitask.rc1",
        "batch": "batch16",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "n_genes": N_FEATURE_UNIVERSE,
        "n_features": 1069,
        "n_slots": N_SLOTS,
        "slot_policy": "diverse 4-slot: p0 = gene_idx % n, slots = [(p0+k) % n for k=0..3] (fixed, no epoch rotation)",
        "split_gene_counts": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "target_train_valid_counts": {t: target_stats[t]["n_train_valid"] for t in TASKS},
        "target_stats": target_stats,
        "feature_scaler": "mean/std computed on ALL train genes (shared across arms), zero-std -> 1",
        "target_standardization": "per-target z = (y - train_mean) / train_std, fit on valid train rows only",
        "source_hashes": {
            "promoter_sym400_uint8.memmap": sha256_file(MEM400),
            "labels.npz": sha256_file(DATA_OUT / "labels.npz"),
        },
        "output_hashes": {
            "data.npz": sha256_file(DATA_OUT / "data.npz"),
            "scalers.npz": sha256_file(DATA_OUT / "scalers.npz"),
        },
    }
    atomic_write_json(DATA_OUT / "data_manifest.json", manifest)

    print(json.dumps({
        "n_genes": N_FEATURE_UNIVERSE,
        "n_features": 1069,
        "split_gene_counts": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "target_train_valid_counts": {t: target_stats[t]["n_train_valid"] for t in TASKS},
        "data_npz": str(DATA_OUT / "data.npz"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
