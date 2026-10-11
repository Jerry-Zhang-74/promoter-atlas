#!/usr/bin/env python3
"""Batch03 homology audit recheck (corrected): recompute exact/RC duplicate and
cross-split counts counting OCCURRENCES (not distinct splits) and tracking all
split pairs. Read-only. Supersedes the earlier attempt which miscounted."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import atomic_write_json  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
N_HUMAN = 507365
L400 = 400


def rc_bytes(b):
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[b[::-1]]


def summarize(d):
    """d: key -> Counter(split) with occurrence counts."""
    n_rows = sum(sum(c.values()) for c in d.values())
    n_unique = len(d)
    dup_rows_beyond_first = n_rows - n_unique
    pairs = {"train_test": 0, "train_val": 0, "val_test": 0}
    for key, cnt in d.items():
        s = set(cnt.keys())
        if "train" in s and "test" in s:
            pairs["train_test"] += 1
        if "train" in s and "val" in s:
            pairs["train_val"] += 1
        if "val" in s and "test" in s:
            pairs["val_test"] += 1
    return {"n_rows": n_rows, "n_unique": n_unique,
            "dup_rows_beyond_first": dup_rows_beyond_first,
            "cross_split_unique_keys": pairs}


def main():
    prom = pd.read_csv(DATA / "trackA_promoters.tsv", sep="\t")
    mm = np.memmap(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap",
                   dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    r400 = prom["r400"].to_numpy(dtype=np.int64)
    split = prom["split"].values

    fwd = {}
    canon = {}
    for i in range(len(prom)):
        key = mm[r400[i]].tobytes()
        fwd.setdefault(key, Counter())[split[i]] += 1
        rkey = rc_bytes(np.frombuffer(key, dtype=np.uint8)).tobytes()
        c = key if key <= rkey else rkey
        canon.setdefault(c, Counter())[split[i]] += 1

    fwd_res = summarize(fwd)
    canon_res = summarize(canon)
    out = {
        "n_rows_total": int(len(prom)),
        "forward_exact": fwd_res,
        "rc_canonical_exact": canon_res,
        "definitions": {
            "forward_exact": "byte-identical forward sym400 sequence; cross_split_unique_keys = #unique sequences present in both named splits",
            "rc_canonical_exact": "forward OR reverse-complement treated as the same entity (canonical min(seq,rc)); cross_split likewise",
        },
        "original_prep_audit": {
            "sym400_exact_dup_rows": 91203,
            "sym400_rc_dup_rows": 91203,
            "sym400_cross_split": {"train_test": 302, "train_val": 276, "val_test": 132},
            "note": "original prep stored forward AND rc keys in one dict and only the first-occurrence split, so its dup counts mix forward+rc and its cross-split counts miss pairs among non-first splits (under-count)."
        },
    }
    atomic_write_json(REPORT / "homology_recheck.json", out)
    import json
    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
