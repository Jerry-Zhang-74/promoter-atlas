# -*- coding: utf-8 -*-
"""batch30 DATA / INPUT layer.  THE single common native-length input builder used by
TRAIN, VAL, TEST and FASTA paths.  No padding, no code 4 at the network input.

Repair vs batch29: batch29 built codes (n, maxlen=2000) filled with 4 and passed full-width
rows to the network (unmasked pooling over padding).  Here every batch is sliced to its exact
recorded length L: codes[ids, :L].  Equal-length buckets guarantee a rectangular tensor with
width == L; code 4/unknown bases are rejected.
"""
from __future__ import annotations
import csv
import numpy as np

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGT", "TGCA")


def load_rows(rows_tsv):
    rows = []
    with open(rows_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def load_fasta(path):
    """Parse a FASTA whose headers are key_chrom_start-end_strand style; returns {key: seq}."""
    d = {}; k = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                parts = line[1:].split("_")
                k = "%s:%s:%s" % (parts[0], parts[1], parts[2])
            else:
                d[k] = line
    return d


def encode_native(seqs):
    """Return (codes int8 padded store, lens int32) for the given sequences.
    codes[i, :lens[i]] holds the valid bases; columns beyond lens[i] are storage-only
    and are NEVER read by native_batch."""
    n = len(seqs)
    maxlen = max((len(s) for s in seqs), default=0)
    codes = np.full((n, maxlen), 4, dtype=np.int8)
    lens = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(seqs):
        lens[i] = len(s)
        for j, ch in enumerate(s):
            v = BASE.get(ch, 4)
            if v == 4:
                raise ValueError("non-ACGT base in sequence %d pos %d" % (i, j))
            codes[i, j] = v
    return codes, lens


def length_buckets(lens, ids):
    """Group row indices by EXACT recorded length.  Deterministic (sorted ids)."""
    b = {}
    for i in ids:
        b.setdefault(int(lens[i]), []).append(int(i))
    return {L: np.array(sorted(v), dtype=np.int64) for L, v in b.items()}


def native_batch(codes, lens, ids, device, torch):
    """THE common native-length input function (TRAIN / VAL / TEST / FASTA).

    ids: 1-D index array sharing the SAME recorded length L.  Returns (tensor[B,L] on device, L).
    Asserts: all recorded lengths == L; tensor width == L; values in {0,1,2,3}.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("ids must be a non-empty 1-D array")
    Ls = np.unique(lens[ids])
    if Ls.size != 1:
        raise AssertionError("native_batch requires equal recorded length; got %r" % (Ls.tolist(),))
    L = int(Ls[0])
    if L <= 0:
        raise AssertionError("non-positive native length %d" % L)
    x = np.asarray(codes[ids, :L])
    if x.shape != (ids.size, L):
        raise AssertionError("width mismatch: %r != (%d,%d)" % (x.shape, ids.size, L))
    if x.size:
        lo, hi = int(x.min()), int(x.max())
        if lo < 0 or hi > 3:
            raise AssertionError("code-4/unknown base present in valid tensor (min=%d max=%d)" % (lo, hi))
    t = torch.from_numpy(x.astype(np.int64)).to(device)
    if int(t.shape[-1]) != L:
        raise AssertionError("tensor width %d != L %d" % (int(t.shape[-1]), L))
    return t, L


def iter_native_batches(codes, lens, ids, device, torch, batch=32):
    """Yield (tensor[B,L], L, bucket_ids) for every exact-length bucket, chunked to batch<=32.
    Deterministic order: lengths ascending, ids ascending within bucket."""
    assert batch <= 32, "batch must be <= 32 (protocol)"
    for L in sorted(length_buckets(lens, ids)):
        b_ids = length_buckets(lens, ids)[L]
        for i in range(0, len(b_ids), batch):
            chunk = b_ids[i:i + batch]
            t, LL = native_batch(codes, lens, chunk, device, torch)
            yield t, LL, chunk
