#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 v6.14.0-scv5.rc1 shared utilities.

Component five-fold cross-fitting retrospective robustness of the fixed
45 -> 1069 feature augmentation for conditional human gene-level raw S.
Baseline = the frozen batch06 45-feature composition/count matrix; full = 45
baseline + the frozen batch10 1024 positional sequence features (total 1069).
Exactly ten scientific fits (HGB base/full per fold, five folds). No GPU, no
sweep, no feature selection, no ensemble, no extra seeds. Raw S only.

Fold allocation (label-independent; raw S values never used for assignment):
  sort unique component IDs by canonical string representation; compute
  SHA256(BATCH13SALT + ID); sort components by decreasing eligible gene size
  then by that hash then by ID; greedy-assign to the fold with the smallest
  current eligible gene count (ties -> lowest fold index 0..4).
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

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch13"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch13"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch13"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch13"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch13"

EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"
FEATURES45_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"
FEATURES1024_NPZ = (ROOT / "data" / "breakthrough_20261002" / "batch10"
                    / "attempt_001_extract" / "features1024.npz")

CANDIDATE_VERSION = "v6.14.0-scv5.rc1"
TRACK = "scv5"
BATCH = "batch13"

# HGB configuration is frozen (identical to the batch10 augmented arm).
HGB = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "early_stopping": False,
    "random_state": 20261011,
}
FIT_SEED = 20261011
BOOT_SEED = 20261013
N_FOLDS = 5
N_BOOTSTRAP = 2000
BOOT_LO_PCT = 2.5
BOOT_HI_PCT = 97.5
DELTA_MIN = 0.03
MIN_POSITIVE_FOLDS = 4
FOLD_SALT = "batch13|20261013|"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3

N_CANONICAL = 58988
M_DEFINED = 18947
N_FEATURE_UNIVERSE = 57827
N_EXPECTED_ELIGIBLE = 18769
N_EXPECTED_EXCLUDED = 178
N_EXPECTED_COMPONENTS = 18400

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

EXPECTED_SHA = {
    "split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
    "features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
    "features1024.npz": "2dc7dc0eda7c60b3fa740261e39fb37832dc0e0faf81a7fa2f9cd37a11bc14f2",
    "trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
}


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


def sha256_text(s) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
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
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def base_gene_id(v: str) -> str:
    return (str(v) or "").split(".")[0]


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
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64)
                                - np.asarray(y_pred, dtype=np.float64))))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, dtype=np.float64)
                          - np.asarray(y_pred, dtype=np.float64)) ** 2))


def reconstruct_s_raw(expr_path=EXPR) -> dict:
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


def mad(xs):
    if not xs:
        return 0.0
    m = statistics.median(xs)
    return statistics.median([abs(x - m) for x in xs])


def load_gene_universe():
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    starts = z["starts"].astype(np.int64)
    gene_gid = gene_id[starts[:-1]]
    s = np.load(SPLIT_NPZ, allow_pickle=True)
    gene_split = s["gene_split"].astype(str)
    comp_dense = s["comp_dense"].astype(np.int64)
    n_prom = s["n_promoters"].astype(np.int64)
    assert len(gene_gid) == N_FEATURE_UNIVERSE
    assert len(gene_split) == N_FEATURE_UNIVERSE
    assert len(comp_dense) == N_FEATURE_UNIVERSE
    assert len(n_prom) == N_FEATURE_UNIVERSE
    assert np.array_equal(n_prom, np.diff(starts)), "n_promoters != diff(starts)"
    return gene_gid, gene_split, comp_dense, n_prom


def load_features45():
    z = np.load(FEATURES45_NPZ, allow_pickle=False)
    comp44 = z["comp44"].astype(np.float64)
    log1p_count = z["log1p_count"].astype(np.float64)
    assert comp44.shape == (N_FEATURE_UNIVERSE, 44)
    assert log1p_count.shape == (N_FEATURE_UNIVERSE,)
    X = np.concatenate([comp44, log1p_count[:, None]], axis=1)
    assert X.shape == (N_FEATURE_UNIVERSE, 45)
    return X


def load_features1024():
    z = np.load(FEATURES1024_NPZ, allow_pickle=False)
    X = z["X1024"].astype(np.float64)
    assert X.shape == (N_FEATURE_UNIVERSE, 1024)
    return X


def load_full_features():
    return np.concatenate([load_features45(), load_features1024()], axis=1)


def fold_hash(component_id) -> str:
    return sha256_text(FOLD_SALT + str(component_id))
