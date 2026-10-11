#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 v6.11.0-sfeatgap.rc1 shared utilities.

Positional sequence features (local-position trinucleotide bins + full-window
tetranucleotide frequencies) added to the frozen batch09 45-feature baseline,
for conditional human gene-level raw S. Two scientific fits (Ridge + HGB) only.
No GPU / neural / sweep / old-file edits. Writes confined to batch10 dirs.

Encoding contract (verified against source memmaps, not assumed):
  0=A, 1=C, 2=G, 3=T, 4=N(ambiguous). Strand orientation preserved (no RC).
  sym400 window = promoter_sym400_uint8.memmap row, length 400, bins [0,100),
  [100,200), [200,300), [300,400). up2000 window = promoter_up2000_uint8.memmap
  row, length 2000, bins [0,500), [500,1000), [1000,1500), [1500,2000).
  gene = arithmetic mean over ALL its promoters (gene_id[starts[:-1]] identities).
"""
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch10"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch10"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch10"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch10"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch10"

EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"
FEATURES_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"
MEM400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
MEM2000 = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"

B9_PRED = ROOT / "data" / "breakthrough_20261002" / "batch09" / "predictions.tsv"
B9_MODELS = ROOT / "models" / "breakthrough_20261002" / "batch09" / "models.joblib"
B9_SCALER = ROOT / "models" / "breakthrough_20261002" / "batch09" / "X_scaler.joblib"
B9_TRANSFORMS = ROOT / "models" / "breakthrough_20261002" / "batch09" / "transforms.json"
B9_RANKREF = ROOT / "models" / "breakthrough_20261002" / "batch09" / "rank_reference.npz"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3
N_CANONICAL = 58988
M_DEFINED = 18947
N_FEATURE_UNIVERSE = 57827
N_EXPECTED_ELIGIBLE = 18769
N_EXPECTED_EXCLUDED = 178
N_HUMAN = 507365
L400 = 400
L2000 = 2000
BINS = {400: [0, 100, 200, 300, 400], 2000: [0, 500, 1000, 1500, 2000]}

CANDIDATE_VERSION = "v6.11.0-sfeatgap.rc1"
TRACK = "sfeatgap"
BATCH = "batch10"
SEED = 20261011
N_BOOTSTRAP = 2000
BOOT_LO_PCT = 1.25
BOOT_HI_PCT = 98.75
DELTA_MIN = 0.03

RIDGE = {"alpha": 0.1, "fit_intercept": True}
HGB = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "early_stopping": False,
    "random_state": SEED,
}

BASES = "ACGT"
KMERS3 = [a + b + c for a in BASES for b in BASES for c in BASES]
KMERS4 = [a + b + c + d for a in BASES for b in BASES for c in BASES for d in BASES]
assert len(KMERS3) == 64 and len(KMERS4) == 256

FEATURE_NAMES_45 = []
for region in ("sym400", "up2000"):
    for base in BASES:
        FEATURE_NAMES_45.append("%s_mono_%s" % (region, base))
    for a in BASES:
        for b in BASES:
            FEATURE_NAMES_45.append("%s_dinuc_%s%s" % (region, a, b))
    FEATURE_NAMES_45.append("%s_gc" % region)
    FEATURE_NAMES_45.append("%s_len" % region)
FEATURE_NAMES_45.append("log1p_count")
assert len(FEATURE_NAMES_45) == 45

FEATURE_NAMES_1024 = []
for region in ("sym400", "up2000"):
    for b in range(4):
        for k in KMERS3:
            FEATURE_NAMES_1024.append("%s_bin%d_tri_%s" % (region, b, k))
for region in ("sym400", "up2000"):
    for k in KMERS4:
        FEATURE_NAMES_1024.append("%s_tetra_%s" % (region, k))
assert len(FEATURE_NAMES_1024) == 1024

FEATURE_NAMES_ALL = FEATURE_NAMES_45 + FEATURE_NAMES_1024
assert len(FEATURE_NAMES_ALL) == 1069


def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def thread_caps_record():
    from threadpoolctl import threadpool_info
    return {
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
        "threadpoolctl": [dict(t) for t in threadpool_info()],
    }


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fhash(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p.resolve()), "bytes": st.st_size, "sha256": sha256_file(p),
            "mtime_ns": st.st_mtime_ns}


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def base_gene_id(v: str) -> str:
    return (v or "").split(".")[0]


def median(xs):
    return statistics.median(xs) if xs else 0.0


def mad(xs):
    if not xs:
        return 0.0
    m = median(xs)
    return median([abs(x - m) for x in xs])


def collapse_duplicate(values):
    nz = [x for x in values if x > 0.0]
    zc = sum(1 for x in values if x == 0.0)
    if nz and zc > 0:
        return max(nz), "prefer_nonzero_max"
    return statistics.fmean(values), "mean"


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def emp_rank(sorted_train, x):
    n = len(sorted_train)
    x = np.asarray(x, dtype=np.float64)
    lt = np.searchsorted(sorted_train, x, side="left")
    eq = np.searchsorted(sorted_train, x, side="right") - lt
    return (lt.astype(np.float64) + 0.5 * eq.astype(np.float64)) / float(n)


def reconstruct_s_raw(expr_path: Path) -> dict:
    import pandas as pd
    expr = pd.read_csv(expr_path, sep="\t", usecols=["gene_id", "tissue", "mean_tpm"],
                       dtype={"gene_id": "str", "tissue": "str", "mean_tpm": np.float64},
                       low_memory=False)
    gt = defaultdict(lambda: defaultdict(list))
    for gi, tissue, mt in zip(expr["gene_id"], expr["tissue"], expr["mean_tpm"]):
        gt[base_gene_id(gi)][tissue].append(float(mt))
    del expr
    full_s = {}
    for gene in sorted(gt):
        tm = gt[gene]
        collapsed = []
        for tissue in sorted(tm):
            vals = tm[tissue]
            collapsed.append(collapse_duplicate(vals)[0] if len(vals) > 1 else vals[0])
        tc = len(collapsed)
        detectable = [x for x in collapsed if x >= DETECTABLE_TPM]
        dtc = len(detectable)
        b_raw = dtc / tc if tc else 0.0
        if b_raw >= B_MIN and dtc >= MIN_DET_TISSUES:
            full_s[gene] = -mad([math.log2(1.0 + x) for x in detectable])
    return full_s


def load_gene_universe():
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    starts = z["starts"].astype(np.int64)
    gene_gid = gene_id[starts[:-1]]
    s = np.load(SPLIT_NPZ, allow_pickle=True)
    gene_split = s["gene_split"]
    comp_dense = s["comp_dense"].astype(np.int64)
    n_prom = s["n_promoters"].astype(np.int64)
    assert len(gene_gid) == N_FEATURE_UNIVERSE
    assert len(gene_split) == N_FEATURE_UNIVERSE
    assert len(comp_dense) == N_FEATURE_UNIVERSE
    assert np.array_equal(n_prom, np.diff(starts)), "n_promoters != diff(starts)"
    return gene_gid, gene_split, comp_dense, n_prom, starts, gene_id


def load_features45():
    z = np.load(FEATURES_NPZ, allow_pickle=True)
    comp44 = z["comp44"].astype(np.float64)
    log1p_count = z["log1p_count"].astype(np.float64)
    assert comp44.shape == (N_FEATURE_UNIVERSE, 44)
    assert log1p_count.shape == (N_FEATURE_UNIVERSE,)
    X = np.concatenate([comp44, log1p_count[:, None]], axis=1)
    assert X.shape == (N_FEATURE_UNIVERSE, 45)
    return X


def kmer_counts(seg, k):
    """Count valid (all-ACGT) contiguous k-mers per row.

    seg: (N, W) int array with values 0..4 (4 = ambiguous N).
    Returns counts (N, 4**k) and denom (N,) = number of valid k-mer starts.
    A k-mer containing any ambiguous base is invalid (never counted). k-mer index
    is base-4 with A=0,C=1,G=2,T=3 (lexicographic A,C,G,T).
    """
    seg = np.asarray(seg, dtype=np.int64)
    N, W = seg.shape
    n = W - k + 1
    idx = np.zeros((N, n), dtype=np.int64)
    valid = np.ones((N, n), dtype=bool)
    for j in range(k):
        win = seg[:, j:j + n]
        valid &= (win < 4)
        idx = idx * 4 + win
    counts = np.zeros((N, 4 ** k), dtype=np.int64)
    if valid.any():
        v = idx[valid]
        ri = np.repeat(np.arange(N), valid.sum(axis=1))
        flat = ri * (4 ** k) + v
        counts = np.bincount(flat, minlength=N * (4 ** k)).reshape(N, 4 ** k)
    return counts, valid.sum(axis=1)


def window_tri_tetra(seq, L):
    """Per-promoter window features for one window (seq: (N, L) int 0..4).

    Returns (tri (N,256), tetra (N,256), qc dict):
      tri  = 4 bins x 64 trinucleotide frequencies, normalized per promoter
             per block (zero valid starts => all-zero block, frequencies 0).
      tetra = 256 full-window tetranucleotide frequencies, normalized per
              promoter (zero valid starts => all-zero block).
    qc records per-block zero-denominator promoter counts for provenance.
    """
    N = seq.shape[0]
    edges = BINS[L]
    tri = np.zeros((N, 256), dtype=np.float64)
    qc = {}
    col = 0
    for b in range(4):
        seg = seq[:, edges[b]:edges[b + 1]]
        cnt, denom = kmer_counts(seg, 3)
        freq = np.zeros_like(cnt, dtype=np.float64)
        pos = denom > 0
        freq[pos] = cnt[pos] / denom[pos, None]
        tri[:, col:col + 64] = freq
        qc["bin%d_tri_zero_denom_promoters" % b] = int((denom == 0).sum())
        col += 64
    cnt4, denom4 = kmer_counts(seq, 4)
    tetra = np.zeros_like(cnt4, dtype=np.float64)
    pos4 = denom4 > 0
    tetra[pos4] = cnt4[pos4] / denom4[pos4, None]
    qc["tetra_zero_denom_promoters"] = int((denom4 == 0).sum())
    return tri, tetra, qc
