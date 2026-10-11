#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 v6.10.0-starget.rc1 shared utilities.

Target-representation honest baseline: continuous raw S vs train-reference
empirical rank S, fixed Ridge/HGB (four fits). No GPU / neural / sweep / old-file
edits. Writes are confined to the five batch09-owned directories.
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

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch09"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch09"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch09"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch09"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch09"

EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"
FEATURES_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3
N_CANONICAL = 58988
M_DEFINED = 18947
N_FEATURE_UNIVERSE = 57827
N_EXPECTED_ELIGIBLE = 18769
N_EXPECTED_EXCLUDED = 178

CANDIDATE_VERSION = "v6.10.0-starget.rc1"
TRACK = "starget"
BATCH = "batch09"
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

AFF_A = 0.5
AFF_B = 2.0

FEATURE_NAMES = []
for region in ("sym400", "up2000"):
    for base in ("A", "C", "G", "T"):
        FEATURE_NAMES.append("%s_mono_%s" % (region, base))
    for a in "ACGT":
        for b in "ACGT":
            FEATURE_NAMES.append("%s_dinuc_%s%s" % (region, a, b))
    FEATURE_NAMES.append("%s_gc" % region)
    FEATURE_NAMES.append("%s_len" % region)
FEATURE_NAMES.append("log1p_count")
assert len(FEATURE_NAMES) == 45


def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


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
    return gene_gid, gene_split, comp_dense, n_prom


def load_features():
    z = np.load(FEATURES_NPZ, allow_pickle=True)
    comp44 = z["comp44"].astype(np.float64)
    log1p_count = z["log1p_count"].astype(np.float64)
    assert comp44.shape == (N_FEATURE_UNIVERSE, 44)
    assert log1p_count.shape == (N_FEATURE_UNIVERSE,)
    X = np.concatenate([comp44, log1p_count[:, None]], axis=1)
    assert X.shape == (N_FEATURE_UNIVERSE, 45)
    return X
