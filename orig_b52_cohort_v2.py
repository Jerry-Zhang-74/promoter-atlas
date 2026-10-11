# -*- coding: utf-8 -*-
"""batch52 data-owner LEAN cohort loader for the batch53_model_core pipeline.

Why this exists: data_adapter.load_cohort builds an int64 (N,600) codes array AND then
copies it with .tobytes() for the datahash (peak ~2x3.6GB on the full 761k cohort). This
loader returns the SAME data_adapter.Cohort object but memory-maps codes_int64.npy and
computes the datahash in the exact same byte order with chunked hashing, so the resulting
datahash is byte-identical to data_adapter._content_hash while peak RAM stays well under the
6 GB worker budget. All numeric/training/eval code is still the reviewed batch53 code.

No fit, no GPU, no network. Read-only over the corrected cohort dir.
"""
from __future__ import annotations
import json, os, sys, warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*not writable.*")

# reviewed helper package
HELPER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
        "runs", "breakthrough_20261002", "code", "batch53_model_core"))
if HELPER not in sys.path:
    sys.path.insert(0, HELPER)

from data_adapter import Cohort, AdapterError  # noqa: E402
from spec import TARGET_COLUMN, SOURCE_SHA256, MAX_LEN, MIN_ELIGIBLE_LEN, MAX_ELIGIBLE_LEN  # noqa: E402

SPLITS = ("TRAIN", "DEV", "TEST")


def _read_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return header, rows


def content_hash(coord_ids, group_ids, splits, codes_np, targets_np32, pair_t):
    """Byte-identical to data_adapter._content_hash, chunked to avoid a giant copy."""
    import hashlib
    h = hashlib.sha256()
    h.update(b"b53-cohort-v1|")
    for c, g, s in zip(coord_ids, group_ids, splits):
        h.update(("%s\t%s\t%s\n" % (c, g, s)).encode("utf-8"))
    N = codes_np.shape[0]
    for a in range(0, N, 50000):
        h.update(np.ascontiguousarray(codes_np[a:a + 50000]).tobytes())
    h.update(np.asarray(targets_np32, dtype=np.float32).tobytes())
    for s in SPLITS:
        h.update(pair_t[s].numpy().tobytes())
    return h.hexdigest()


def load_cohort_v2(cohort_dir, source_sha256, mmap_codes=True):
    ap = os.path.join(cohort_dir, "endpoints.tsv")
    header, rows = _read_tsv(ap)
    for col in ("coord_id", "union_id", "split", "length", TARGET_COLUMN):
        if col not in header:
            raise AdapterError("endpoints.tsv missing column %r" % col)
    ci, gi, si, li, ti = (header.index(c) for c in ("coord_id", "union_id", "split", "length", TARGET_COLUMN))
    coord_ids, group_ids, splits, lengths, targets = [], [], [], [], []
    seen = set()
    for r in rows:
        cid = r[ci]
        if cid in seen:
            raise AdapterError("duplicate coord_id %r" % cid)
        seen.add(cid)
        if r[si] not in SPLITS:
            raise AdapterError("bad split %r" % r[si])
        L = int(r[li])
        if L < MIN_ELIGIBLE_LEN or L > MAX_ELIGIBLE_LEN:
            raise AdapterError("length %d out of range for %s" % (L, cid))
        coord_ids.append(cid); group_ids.append(r[gi]); splits.append(r[si]); lengths.append(L)
        targets.append(float(r[ti]))
    gsplit = {}
    for g, s in zip(group_ids, splits):
        if g in gsplit and gsplit[g] != s:
            raise AdapterError("group %r spans splits" % g)
        gsplit[g] = s
    sg = {s: set() for s in SPLITS}
    for g, s in gsplit.items():
        sg[s].add(g)
    if (sg["TRAIN"] & sg["DEV"]) or (sg["TRAIN"] & sg["TEST"]) or (sg["DEV"] & sg["TEST"]):
        raise AdapterError("split groups not disjoint")

    prov = json.load(open(os.path.join(cohort_dir, "reference_provenance.json"), "r", encoding="utf-8"))
    if str(prov.get("assembly", "")).lower() != "hg19" or not prov.get("proxy_not_donor_clone", False):
        raise AdapterError("reference provenance must be hg19 proxy_not_donor_clone")
    man = json.load(open(os.path.join(cohort_dir, "manifest.json"), "r", encoding="utf-8"))
    if source_sha256 is not None and man.get("source_sha256", "").lower() != source_sha256.lower():
        raise AdapterError("manifest source_sha256 mismatch")

    codes_path = os.path.join(cohort_dir, "codes_int64.npy")
    N = len(coord_ids)
    if mmap_codes:
        codes_np = np.load(codes_path, mmap_mode="r")
    else:
        codes_np = np.load(codes_path, mmap_mode=None)
    if codes_np.shape != (N, MAX_LEN):
        raise AdapterError("codes shape %r != (%d,%d)" % (codes_np.shape, N, MAX_LEN))

    pp = os.path.join(cohort_dir, "pairs.tsv")
    ph, pr = _read_tsv(pp)
    ui, sp, sc, lc = (ph.index(c) for c in ("union_id", "split", "short_coord_id", "long_coord_id"))
    idx = {c: i for i, c in enumerate(coord_ids)}
    pairs = {s: [] for s in SPLITS}
    for r in pr:
        i, j = idx[r[sc]], idx[r[lc]]
        if group_ids[i] != r[ui] or group_ids[j] != r[ui]:
            raise AdapterError("pair not in its group")
        if splits[i] != r[sp] or splits[j] != r[sp]:
            raise AdapterError("pair split mismatch")
        pairs[r[sp]].append((i, j))
    pair_t = {s: (torch.tensor(v, dtype=torch.long) if v else torch.zeros((0, 2), dtype=torch.long))
              for s, v in pairs.items()}

    targets_t = torch.tensor(targets, dtype=torch.float32)
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    codes_t = torch.from_numpy(codes_np)
    dh = content_hash(coord_ids, group_ids, splits, codes_np, targets_t.numpy(), pair_t)
    return Cohort(coord_ids, group_ids, splits, codes_t, lengths_t, targets_t, pair_t, dh,
                  TARGET_COLUMN, source_sha256)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort_dir")
    ap.add_argument("--source-sha", default=SOURCE_SHA256)
    a = ap.parse_args()
    c = load_cohort_v2(a.cohort_dir, a.source_sha, mmap_codes=False)
    print(json.dumps({"n": len(c.coord_ids), "datahash": c.datahash,
                      "counts": {s: sum(1 for x in c.splits if x == s) for s in SPLITS},
                      "pairs": {s: int(c.pairs[s].shape[0]) for s in SPLITS},
                      "groups": {s: len({g for g, ss in zip(c.group_ids, c.splits) if ss == s}) for s in SPLITS}},
                     indent=2))
