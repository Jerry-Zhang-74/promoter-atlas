# -*- coding: utf-8 -*-
"""batch54 INDEPENDENT cohort reader (own TSV parser; no import of producer loaders).

Reads the frozen corrected cohort TSVs read-only and returns plain numpy/python objects
targets, group ids, splits, pair index lists). Memory-light: only targets/groups/pairs are
materialised; the large codes array is memory-mapped ON DEMAND by callers (GPU replay only),
never loaded for CPU metrics. No fit, no GPU, no network.
"""
from __future__ import annotations
import json, os
import numpy as np

TARGET_COLUMN = "Log2RPM_HEK293"
SPLITS = ("TRAIN", "DEV", "TEST")


def _rows(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                yield header, ln.split("\t")


def load(cohort_dir):
    ep = os.path.join(cohort_dir, "endpoints.tsv")
    coord_ids, groups, splits, lengths, targets = [], [], [], [], []
    with open(ep, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci, gi, si, li, ti = (header.index(c) for c in
                              ("coord_id", "union_id", "split", "length", TARGET_COLUMN))
        for ln in f:
            if not ln.strip():
                continue
            r = ln.rstrip("\n").split("\t")
            coord_ids.append(r[ci]); groups.append(r[gi]); splits.append(r[si])
            lengths.append(int(r[li])); targets.append(float(r[ti]))
    coord_ids = np.array(coord_ids, dtype=object)
    groups = np.array(groups, dtype=object)
    splits = np.array(splits, dtype=object)
    lengths = np.array(lengths, dtype=np.int64)
    targets = np.array(targets, dtype=np.float64)
    idx = {c: i for i, c in enumerate(coord_ids.tolist())}

    pairs = {s: [] for s in SPLITS}
    with open(os.path.join(cohort_dir, "pairs.tsv"), "r", encoding="utf-8") as f:
        ph = f.readline().rstrip("\n").split("\t")
        ui, sp, sc, lc = (ph.index(c) for c in ("union_id", "split", "short_coord_id", "long_coord_id"))
        for ln in f:
            if not ln.strip():
                continue
            r = ln.rstrip("\n").split("\t")
            i, j = idx[r[sc]], idx[r[lc]]
            assert groups[i] == r[ui] == groups[j], "pair group mismatch %s" % r[sc]
            assert splits[i] == r[sp] == splits[j], "pair split mismatch %s" % r[sc]
            pairs[r[sp]].append((i, j))
    pair_arr = {s: (np.array(v, dtype=np.int64) if v else np.zeros((0, 2), dtype=np.int64))
                for s, v in pairs.items()}

    return {"coord_ids": coord_ids, "groups": groups, "splits": splits, "lengths": lengths,
            "targets": targets, "pairs": pair_arr, "index": idx}


def split_idx(co, split):
    return np.where(co["splits"] == split)[0]


def load_preds_tsv(path):
    """coord_id -> float; tolerant of an optional header and 2/3 columns."""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n").split("\t")
        if not _looks_numeric(first[-1]):
            pass
        else:
            d[first[0]] = float(first[-1])
        for ln in f:
            if not ln.strip():
                continue
            a = ln.rstrip("\n").split("\t")
            d[a[0]] = float(a[-1])
    return d


def load_preds_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


def _looks_numeric(s):
    try:
        float(s); return True
    except Exception:
        return False


def codes_memmap(cohort_dir):
    return np.load(os.path.join(cohort_dir, "codes_int64.npy"), mmap_mode="r")
