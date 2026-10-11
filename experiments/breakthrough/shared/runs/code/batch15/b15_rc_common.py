# -*- coding: utf-8 -*-
"""BATCH15 rc-retrieval-audit-20261003-r1 shared utilities.

Cross-orientation (near reverse-complement) retrieval audit. Read-only over all
prior data; writes only new batch15 artifacts. No labels used for retrieval or
graph construction. CPU<=4, no GPU, no fits.
"""
import os
import json
import time
import ctypes
import hashlib
import datetime
from pathlib import Path

import numpy as np

ROOT = Path("E:/promoter_atlas")
DATA13 = ROOT / "data" / "breakthrough_20261002" / "batch13"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
DATA15 = ROOT / "data" / "breakthrough_20261002" / "batch15"
REPORT15 = ROOT / "reports" / "breakthrough_20261002" / "batch15"
RUNS15 = ROOT / "runs" / "breakthrough_20261002" / "batch15"
CODE15 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch15"

MEMMAP = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
N_HUMAN = 507365
L400 = 400
N_GENES_UNIVERSE = 57827

# ---- frozen retrieval parameters (identical to batch05 sketch, cross-orientation) ----
K = 15
N_HASH = 48
N_BANDS = 12
ROWS = 4
BAND_CAP = 2000
PAIR_CAP = 5_000_000
MIN_BANDS = 2
MAX_SHIFT = 40
IDENTITY_MIN = 0.90
COVERAGE_MIN = 0.90
SEED = 20261011
P = 2 ** 61 - 1

COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)

def thread_caps_record():
    try:
        from threadpoolctl import threadpool_info
        tpi = [dict(t) for t in threadpool_info()]
    except Exception:
        tpi = []
    return {"env": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
            "threadpoolctl": tpi}

def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()

def sha256_bytes(b):
    return hashlib.sha256(bytes(b)).hexdigest()

def sha256_text(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

def _json_native(o):
    if isinstance(o, dict):
        return {str(k): _json_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_native(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return _json_native(o.tolist())
    return o

def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_json_native(obj), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)

def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)

def now_local():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def worker_identity():
    pid = os.getpid()
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]
    fc = FILETIME(); fe = FILETIME(); fk = FILETIME(); fu = FILETIME()
    ok = kernel32.GetProcessTimes(h, ctypes.byref(fc), ctypes.byref(fe), ctypes.byref(fk), ctypes.byref(fu))
    def ft_to_dt(ft):
        v = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        return datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=v / 10)
    created_utc = ft_to_dt(fc)
    created_local = created_utc.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    return {
        "pid": int(pid),
        "os_creation_utc": created_utc.isoformat(),
        "os_creation_local": created_local.isoformat(),
        "getprocesstimes_ok": bool(ok),
    }

# ---- sequence helpers ------------------------------------------------------
def rc_bytes(b):
    return COMP[b[::-1]]

def rc_batch(B):
    return COMP[B[:, ::-1]]

# ---- minhash (vectorized, identical to batch05 forward sketch) -------------
def kmer_codes(seq2, k):
    N, L = seq2.shape
    nwin = L - k + 1
    codes = np.zeros((N, nwin), dtype=np.uint32)
    c = np.zeros(N, dtype=np.uint32)
    for j in range(k):
        c = (c << np.uint32(2)) | seq2[:, j].astype(np.uint32)
    codes[:, 0] = c
    mask = (np.uint32(1) << np.uint32(2 * k)) - np.uint32(1)
    for p in range(1, nwin):
        c = ((c << np.uint32(2)) & mask) | seq2[:, p + k - 1].astype(np.uint32)
        codes[:, p] = c
    return codes

def window_distinct(seq, k):
    N, L = seq.shape
    nwin = L - k + 1
    cnt = np.zeros((N, nwin), dtype=np.uint8)
    for c in range(4):
        hc = (seq == c).astype(np.int32)
        cs = np.cumsum(hc, axis=1)
        end = cs[:, k - 1:]
        start = np.zeros_like(end)
        start[:, 1:] = cs[:, :nwin - 1]
        cnt += (end - start > 0).astype(np.uint8)
    return cnt

def minhash_sketch(s, codes, win_valid, distinct_ok, a, b):
    N = s.shape[0]
    codes64 = codes.astype(np.uint64)
    keep = win_valid & distinct_ok
    codes64[~keep] = 0
    sketch = np.zeros((N, N_HASH), dtype=np.uint64)
    for i in range(N_HASH):
        hv = (a[i] * codes64 + b[i]) % np.uint64(P)
        hv[~keep] = np.uint64(P - 1)
        sketch[:, i] = hv.min(axis=1)
    return sketch

def hash_params(seed=SEED):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 31, size=N_HASH, dtype=np.uint64)
    b = rng.integers(0, P, size=N_HASH, dtype=np.uint64)
    return a, b

def compute_sketch(s, a, b):
    seq2 = np.where(s == 4, 0, s).astype(np.uint8)
    codes = kmer_codes(seq2, K)
    hasN = (s == 4).astype(np.int32)
    csN = np.cumsum(hasN, axis=1)
    nwin = L400 - K + 1
    endN = csN[:, K - 1:]
    startN = np.zeros_like(endN)
    startN[:, 1:] = csN[:, :nwin - 1]
    win_valid = (endN - startN) == 0
    distinct = window_distinct(s, K)
    distinct_ok = distinct >= 3
    sketch = minhash_sketch(s, codes, win_valid, distinct_ok, a, b)
    return sketch

def band_keys(sketch):
    N = sketch.shape[0]
    bk = np.zeros((N, N_BANDS), dtype=np.uint64)
    for j in range(N_BANDS):
        bk[:, j] = np.bitwise_xor.reduce(sketch[:, j * ROWS:(j + 1) * ROWS], axis=1)
    return bk

# ---- exact deterministic gapless RC alignment verification -----------------
def verify_pairs_batch(A, B, max_shift=MAX_SHIFT, identity_min=IDENTITY_MIN,
                       coverage_min=COVERAGE_MIN):
    """Gapless diagonal RC-orientation alignment of batch (P, L400) A vs B.

    A = forward seq of promoter i, B = forward seq of promoter j. We align A
    against reverse_complement(B) over shifts [-max_shift, max_shift]. Unknown
    bases (4) never count as matches. Returns per-pair:
      (best_identity, best_coverage, best_shift, best_overlap, best_matches)
    best chosen among shifts with coverage >= coverage_min.
    """
    P_ = A.shape[0]
    best_ident = np.zeros(P_, dtype=np.float64)
    best_cov = np.zeros(P_, dtype=np.float64)
    best_shift = np.zeros(P_, dtype=np.int64)
    best_overlap = np.zeros(P_, dtype=np.int64)
    best_matches = np.zeros(P_, dtype=np.int64)
    Brc = rc_batch(B)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            L = L400 - s
            a = A[:, :L]
            b = Brc[:, s:]
        else:
            s2 = -s
            L = L400 - s2
            a = A[:, s2:]
            b = Brc[:, :L]
        valid = (a != 4) & (b != 4)
        matches = ((a == b) & valid).sum(axis=1).astype(np.int64)
        ident = matches / float(L)
        cov = L / float(L400)
        better = (cov >= coverage_min) & (ident > best_ident)
        best_ident[better] = ident[better]
        best_cov[better] = cov
        best_shift[better] = s
        best_overlap[better] = L
        best_matches[better] = matches[better]
    return best_ident, best_cov, best_shift, best_overlap, best_matches

# ---- LSH band matching (index side vs query side) --------------------------
def band_matches(index_sketch, query_sketch, band_cap=BAND_CAP, pair_cap=PAIR_CAP,
                 min_bands=MIN_BANDS):
    """Generic banded LSH: index band keys bucketed, queried by query band keys.

    A pair (i,j) with i!=j is counted on band b iff the ACTUAL 4-value minhash
    tuple of index_sketch[i] equals that of query_sketch[j] on band b (not merely
    the XOR key). Cross-orientation retrieval = band_matches(F_sketch, R_sketch).
    Returns (pair_bands, ledger): pair_bands maps unordered pair -> number of
    distinct tuple-verified bands (>= min_bands kept).
    """
    N = index_sketch.shape[0]
    idx_bk = band_keys(index_sketch)
    qry_bk = band_keys(query_sketch)
    pair_count = {}
    n_skipped_buckets = 0
    n_truncated = 0
    n_tuple_hits = 0
    for b in range(N_BANDS):
        fkeys = idx_bk[:, b]
        order = np.argsort(fkeys, kind="stable")
        keys_s = fkeys[order]
        bnd = np.flatnonzero(np.diff(keys_s)) + 1
        starts = np.concatenate([[0], bnd])
        ends = np.concatenate([bnd, [len(keys_s)]])
        buckets = {}
        for si, ei in zip(starts, ends):
            bsize = int(ei - si)
            if bsize < 1:
                continue
            if bsize > band_cap:
                n_skipped_buckets += 1
                continue
            buckets[int(keys_s[si])] = order[si:ei]
        for j in range(N):
            qkey = int(qry_bk[j, b])
            if qkey not in buckets:
                continue
            qtuple = query_sketch[j, b * ROWS:(b + 1) * ROWS]
            for i in buckets[qkey]:
                if int(i) == int(j):
                    continue
                ituple = index_sketch[int(i), b * ROWS:(b + 1) * ROWS]
                if np.array_equal(ituple, qtuple):
                    n_tuple_hits += 1
                    keyp = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
                    if keyp in pair_count:
                        pair_count[keyp].add(b)
                    elif len(pair_count) < pair_cap:
                        pair_count[keyp] = {b}
                    else:
                        n_truncated += 1
    out = {k: len(v) for k, v in pair_count.items() if len(v) >= min_bands}
    ledger = {"n_skipped_buckets": int(n_skipped_buckets),
              "n_truncated_pair_increments": int(n_truncated),
              "n_tuple_verified_band_hits": int(n_tuple_hits),
              "n_pairs_tracked": int(len(pair_count))}
    return out, ledger
