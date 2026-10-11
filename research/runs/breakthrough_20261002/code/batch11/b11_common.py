#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 v6.12.0-sfeatablate.rc1 shared utilities.

Conditional raw-S feature ablation: decompose the batch10 1024-feature block into
(A) local-position trinucleotide bins (512) and (B) full-window tetranucleotide
frequencies (512). Four frozen fits: (45+A) and (45+B), each with Ridge and HGB.
The batch10 full 1069-feature models and the batch09 45-feature baselines are
LOADED/REPLAYED, never retrained. Four co-primary contrasts are
(full - A) and (full - B) test Spearman within each learner vs the common
full-precision raw S truth, with paired component bootstrap 2000 / seed 20261011
and 98.75% two-sided percentile intervals [0.625, 99.375].

No GPU / neural / sweep / old-file edits. Writes confined to batch11 dirs.
"""
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch11"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch11"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch11"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch11"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch11"

EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"
FEATURES_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"
MEM400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
MEM2000 = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"

B10_FEATURES1024 = ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "features1024.npz"
B10_FEATDICT = ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "feature_dictionary.json"
B10_PRED = ROOT / "data" / "breakthrough_20261002" / "batch10" / "predictions.tsv"
B10_MODELS = ROOT / "models" / "breakthrough_20261002" / "batch10" / "attempt_001_fit" / "models.joblib"
B10_CONFIG = ROOT / "models" / "breakthrough_20261002" / "batch10" / "config.json"

B9_PRED = ROOT / "data" / "breakthrough_20261002" / "batch09" / "predictions.tsv"
B9_MODELS = ROOT / "models" / "breakthrough_20261002" / "batch09" / "models.joblib"
B9_SCALER = ROOT / "models" / "breakthrough_20261002" / "batch09" / "X_scaler.joblib"
B9_TRANSFORMS = ROOT / "models" / "breakthrough_20261002" / "batch09" / "transforms.json"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3
N_CANONICAL = 58988
M_DEFINED = 18947
N_FEATURE_UNIVERSE = 57827
N_EXPECTED_ELIGIBLE = 18769
N_EXPECTED_EXCLUDED = 178
N_HUMAN = 507365
N_COMPONENTS_TOTAL = 18400

CANDIDATE_VERSION = "v6.12.0-sfeatablate.rc1"
TRACK = "sfeatablate"
BATCH = "batch11"
SEED = 20261011
N_BOOTSTRAP = 2000
BOOT_LO_PCT = 0.625
BOOT_HI_PCT = 99.375
BOOT_TWO_SIDED_PCT = 98.75
DELTA_MIN = 0.01

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

# A = all position3mer (tri bins), B = all global4mer (tetra), each 512 cols.
TRI_IDX = [i for i, n in enumerate(FEATURE_NAMES_1024) if "_tri_" in n]
TETRA_IDX = [i for i, n in enumerate(FEATURE_NAMES_1024) if "_tetra_" in n]
assert len(TRI_IDX) == 512 and len(TETRA_IDX) == 512
assert TRI_IDX == list(range(0, 512))
assert TETRA_IDX == list(range(512, 1024))


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


def load_features1024():
    z = np.load(B10_FEATURES1024, allow_pickle=False)
    X = z["X1024"].astype(np.float64)
    assert X.shape == (N_FEATURE_UNIVERSE, 1024)
    return X


def feature_arm_indices():
    """Return verified column indices for the 45 baseline, tri (A) and tetra (B) blocks.

    Verification is by NAME against the frozen batch10 feature dictionary, not by
    blindly assuming the numeric slices. Returns (idx45, idxA, idxB) where idxA
    are the 1024-block positions of all position3mer (tri) features and idxB the
    all global4mer (tetra) features; the final arm matrices are built by
    concatenating the frozen 45-block with those 1024-block slices.
    """
    fd = json.load(open(B10_FEATDICT, encoding="utf-8"))
    names = fd["feature_names_1024"]
    assert names == FEATURE_NAMES_1024, "frozen feature dictionary order mismatch"
    tri = [i for i, n in enumerate(names) if "_tri_" in n]
    tetra = [i for i, n in enumerate(names) if "_tetra_" in n]
    assert tri == list(range(512)) and tetra == list(range(512, 1024))
    assert fd["baseline_feature_names_45"] == FEATURE_NAMES_45
    return list(range(45)), tri, tetra


def windows_process_creation_time(pid: int):
    """Return process creation epoch-seconds via Windows GetProcessTimes (ctypes).

    Falls back to None only when OpenProcess/GetProcessTimes fails. Distinct from
    any controller started_at / status started_at timestamps.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD)]

    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    ct = FILETIME(); et = FILETIME(); kt = FILETIME(); ut = FILETIME()
    ok = kernel32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et),
                                  ctypes.byref(kt), ctypes.byref(ut))
    kernel32.CloseHandle(h)
    if not ok:
        return None
    t = (ct.dwHighDateTime << 32) | ct.dwLowDateTime
    return t / 1e7 - 11644473600.0


def fmt_local(epoch_s):
    if epoch_s is None:
        return None
    lt = time.localtime(epoch_s)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + (".%06d+08:00" % int((epoch_s % 1) * 1e6))


def now_local():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

