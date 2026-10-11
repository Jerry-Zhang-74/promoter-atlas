#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene database mode: symbol/ID -> frozen promoter bag + EXACT 1069 features.

Reads only read-only upstream artifacts. No user must supply a 1069 vector.
Observed catalog labels and model predictions are kept strictly separate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from b20_common import (ANALYSIS_SET, B16_DATA_NPZ, B16_LABELS_NPZ, B20InputError,
                        LABEL_TABLE, L2000, L400, MEM2000, MEM400, N_FEATURE_UNIVERSE,
                        N_HUMAN, N_SLOTS, TRACKA_NPZ, sha256_file)

_CACHE = {}


def _base(eid: str) -> str:
    return (eid or "").split(".")[0]


def diverse_slots(n_prom: int, gene_idx: int):
    """EXACT batch16 slot rule (fixed diverse 4-slot, no rotation)."""
    n = int(n_prom)
    p0 = int(gene_idx) % n
    return [(p0 + k) % n for k in range(N_SLOTS)]


def load_catalog():
    if "cat" in _CACHE:
        return _CACHE["cat"]
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    record_id = z["record_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    r2000 = z["r2000"].astype(np.int64)
    starts = z["starts"].astype(np.int64)
    score_e_row = z["Score_E"].astype(np.float64)
    gene_gid = gene_id[starts[:-1]]
    gene_score_e = score_e_row[starts[:-1]]
    assert len(gene_gid) == N_FEATURE_UNIVERSE, len(gene_gid)

    d = np.load(B16_DATA_NPZ, allow_pickle=True)
    X = d["X"].astype(np.float64)
    assert X.shape == (N_FEATURE_UNIVERSE, 1069)

    lab = np.load(B16_LABELS_NPZ, allow_pickle=True)
    E = lab["E"].astype(np.float64)
    S = lab["S"].astype(np.float64)
    B = lab["B"].astype(np.float64)
    S_mask = lab["S_mask"].astype(bool)
    B_mask = lab["B_mask"].astype(bool)
    gene_split = lab["gene_split"].astype(str)

    # observed catalog labels from the traceable label report (no silent remap)
    import pandas as pd
    lt = pd.read_csv(LABEL_TABLE, sep="\t", low_memory=False)
    lt = lt.drop_duplicates("gene_id").set_index("gene_id")
    obs = lt.reindex(gene_gid)
    ObsE = obs["Score_E"].to_numpy(dtype=np.float64)          # model E target units
    ObsS_raw = obs["S_raw"].to_numpy(dtype=np.float64)        # model S target units
    ObsB_raw = obs["B_raw"].to_numpy(dtype=np.float64)        # model B target units
    ObsScoreS = obs["Score_S"].to_numpy(dtype=np.float64)     # informational rank-scaled
    ObsScoreB = obs["Score_B"].to_numpy(dtype=np.float64)     # informational rank-scaled
    S_defined = obs["S_defined_main"].to_numpy()

    # gene symbol map from promoter_analysis_set (record_id -> gene_name)
    pan = pd.read_csv(ANALYSIS_SET, sep="\t", usecols=["record_id", "gene_name"],
                      low_memory=False).drop_duplicates("record_id")
    rid2sym = dict(zip(pan["record_id"].tolist(), pan["gene_name"].tolist()))
    promoter_sym = np.array([rid2sym.get(r) for r in record_id], dtype=object)
    gene_sym = np.empty(len(gene_gid), dtype=object)
    gene_sym[:] = None
    for g in range(len(gene_gid)):
        lo, hi = int(starts[g]), int(starts[g + 1])
        vals = promoter_sym[lo:hi]
        vals = [v for v in vals if v is not None]
        gene_sym[g] = vals[0] if vals else None

    by_gid = {gene_gid[g]: g for g in range(len(gene_gid))}
    by_sym = {}
    for g in range(len(gene_gid)):
        s = gene_sym[g]
        if s is not None:
            by_sym.setdefault(str(s).upper(), []).append(g)

    cat = {
        "gene_id": gene_gid, "record_id": record_id, "r400": r400, "r2000": r2000,
        "starts": starts, "gene_score_e": gene_score_e, "X": X,
        "E": E, "S": S, "B": B, "S_mask": S_mask, "B_mask": B_mask,
        "gene_split": gene_split, "gene_sym": gene_sym,
        "obs_E": ObsE, "obs_S_raw": ObsS_raw, "obs_B_raw": ObsB_raw,
        "obs_Score_S": ObsScoreS, "obs_Score_B": ObsScoreB, "obs_S_defined": S_defined,
        "by_gid": by_gid, "by_sym": by_sym,
    }
    _CACHE["cat"] = cat
    return cat


def resolve_gene(query: str):
    """Resolve a symbol or ENSG id -> (gene_idx, canonical_ensg). Raises B20InputError."""
    cat = load_catalog()
    q = (query or "").strip()
    if not q:
        raise B20InputError("empty gene query")
    if q.upper().startswith("ENSG"):
        base = _base(q).upper()
        if base not in cat["by_gid"]:
            raise B20InputError("gene id not in catalog: %s" % q)
        return cat["by_gid"][base], base
    hits = cat["by_sym"].get(q.upper(), [])
    if not hits:
        raise B20InputError("gene symbol not found in catalog: %s" % q)
    if len(hits) > 1:
        raise B20InputError("ambiguous symbol %s maps to %d genes: %s"
                            % (q, len(hits), [cat["gene_id"][h] for h in hits]))
    return hits[0], cat["gene_id"][hits[0]]


def gene_bag_sequences(gene_idx: int):
    """Return (seq400 (n,400), seq2000 (n,2000)) int arrays from the read-only memmaps."""
    cat = load_catalog()
    st = cat["starts"]
    lo, hi = int(st[gene_idx]), int(st[gene_idx + 1])
    r400 = cat["r400"][lo:hi]
    r2000 = cat["r2000"][lo:hi]
    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    mm2000 = np.memmap(MEM2000, dtype="uint8", mode="r", shape=(N_HUMAN, L2000))
    return (np.asarray(mm400[r400], dtype=np.uint8),
            np.asarray(mm2000[r2000], dtype=np.uint8))


def slot_sequences(gene_idx: int):
    """EXACT 4-slot (4,400) input to the E/S/B model."""
    cat = load_catalog()
    st = cat["starts"]
    n_prom = int(st[gene_idx + 1] - st[gene_idx])
    seq400, _ = gene_bag_sequences(gene_idx)
    slots = diverse_slots(n_prom, gene_idx)
    return seq400[slots]
