#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 shared utilities (frozen gene-level S interface).

This batch performs NO scientific training. It wraps the already-reviewed
batch10 full-1069 HistGradientBoosting raw-S model as a frozen, read-only,
conditional gene-level S inference + label-trace interface.

Frozen candidate = batch10 hgb_aug (full 1069 features, raw S target), used
as-is with its saved scaler and target transforms. Eligibility is label-defined
(canonical S-defined AND present in the 57827-gene feature universe); the
interface is a conditional catalog lookup, NOT a sequence-only promoter scorer
and NOT an S-eligibility predictor.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

B10_MODELS = ROOT / "models" / "breakthrough_20261002" / "batch10" / "attempt_001_fit" / "models.joblib"
B10_SCALER = ROOT / "models" / "breakthrough_20261002" / "batch10" / "attempt_001_fit" / "X_scaler.joblib"
B10_TRANSFORMS = ROOT / "models" / "breakthrough_20261002" / "batch10" / "attempt_001_fit" / "transforms.json"
FEATURES45_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"
FEATURES1024_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "features1024.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
EXPR_TSV = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON_TSV = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
B10_PRED_TSV = ROOT / "data" / "breakthrough_20261002" / "batch10" / "predictions.tsv"

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch12"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch12"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch12"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch12"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch12"

CANDIDATE_VERSION = "v6.13.0-siface.rc1"
TRACK = "siface"
BATCH = "batch12"

N_CANONICAL = 58988
M_S_DEFINED = 18947
N_FEATURE_UNIVERSE = 57827
N_ELIGIBLE = 18769
N_EXCLUDED = 178
N_TRAIN = 15004
N_VAL = 1898
N_TEST = 1867

S_TARGET_DEF = {
    "name": "raw_S",
    "formula": "S_raw = -MAD(log2(1 + collapsed_mean_tpm)) over detectable (TPM>=1) tissues",
    "eligibility": "B_raw = detectable/54 >= 0.30 AND detectable_tissue_count >= 3",
    "units": "log2(1+TPM) MAD (dimensionless, negative; larger/less-negative = less dispersion)",
    "range_hint": "S_raw in (-inf, 0] in theory; regression output is unconstrained and may be positive",
    "not": "NOT post-transfection temporal stability, NOT synthetic promoter activity, NOT HEK293T construct activity",
}
S_THEORETICAL_MIN = None
S_THEORETICAL_MAX = 0.0

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3

TARGET_TRAIN_MEAN = -0.5336610223136261
TARGET_TRAIN_STD = 0.33092373224240557

BASES = "ACGT"
_KMERS3 = [a + b + c for a in BASES for b in BASES for c in BASES]
_KMERS4 = [a + b + c + d for a in BASES for b in BASES for c in BASES for d in BASES]
FEATURE_NAMES_45 = []
for _region in ("sym400", "up2000"):
    for _base in BASES:
        FEATURE_NAMES_45.append("%s_mono_%s" % (_region, _base))
    for _a in BASES:
        for _b in BASES:
            FEATURE_NAMES_45.append("%s_dinuc_%s%s" % (_region, _a, _b))
    FEATURE_NAMES_45.append("%s_gc" % _region)
    FEATURE_NAMES_45.append("%s_len" % _region)
FEATURE_NAMES_45.append("log1p_count")
assert len(FEATURE_NAMES_45) == 45

FEATURE_NAMES_1024 = []
for _region in ("sym400", "up2000"):
    for _b in range(4):
        for _k in _KMERS3:
            FEATURE_NAMES_1024.append("%s_bin%d_tri_%s" % (_region, _b, _k))
for _region in ("sym400", "up2000"):
    for _k in _KMERS4:
        FEATURE_NAMES_1024.append("%s_tetra_%s" % (_region, _k))
assert len(FEATURE_NAMES_1024) == 1024
FEATURE_NAMES_ALL = FEATURE_NAMES_45 + FEATURE_NAMES_1024
assert len(FEATURE_NAMES_ALL) == 1069

FROZEN_ARTIFACTS = [
    {"key": "models.joblib", "path": str(B10_MODELS),
     "sha256": "fa50823f976af134227c13f1e55b3b8fcd88f21c4001cac6fda44425dbe0ace2",
     "role": "frozen model dict {ridge_aug, hgb_aug}; ONLY hgb_aug used"},
    {"key": "X_scaler.joblib", "path": str(B10_SCALER),
     "sha256": "050e83c3690de51f301d56bd19a83a8061005608674341cfa15828514553c352",
     "role": "StandardScaler fit on eligible TRAIN rows (1069 features)"},
    {"key": "transforms.json", "path": str(B10_TRANSFORMS),
     "sha256": "b84a03bc5fd928840747bd57d237f6757b9e2969dbe4655f1c1a51dc684df63b",
     "role": "X_scaler mean_/scale_ + target train_mean/train_std"},
    {"key": "features45.npz", "path": str(FEATURES45_NPZ),
     "sha256": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
     "role": "45 baseline composition/count features (comp44 + log1p_count)"},
    {"key": "features1024.npz", "path": str(FEATURES1024_NPZ),
     "sha256": "2dc7dc0eda7c60b3fa740261e39fb37832dc0e0faf81a7fa2f9cd37a11bc14f2",
     "role": "1024 positional trinucleotide/tetranucleotide features"},
    {"key": "split_arrays.npz", "path": str(SPLIT_NPZ),
     "sha256": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
     "role": "corrected batch05 component split (gene_split, comp_dense, n_promoters)"},
    {"key": "trackA_gene_promoters.npz", "path": str(TRACKA_NPZ),
     "sha256": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
     "role": "gene_id + starts for the 57827-gene feature universe identity"},
    {"key": "gene_expression_table.tsv", "path": str(EXPR_TSV),
     "sha256": "9876abd64aa8f5dc46f09796d4dffec4f2385e553e281f32120a72a23c606ae4",
     "role": "raw tissue mean_tpm (full-precision S_raw reconstruction)"},
    {"key": "gene_expression_label_table.tsv", "path": str(CANON_TSV),
     "sha256": "34e20a961934c0ecc15a04b13f414b1fad803a4e13260f33134b536b8d7795b2",
     "role": "canonical E_raw/S_raw/B_raw + Score_E/S/B"},
    {"key": "label_audit_all_genes.tsv", "path": str(AUDIT_TSV),
     "sha256": "632f4b7d9dcf0fefc2b1be96319afa8b41f1db67fcbaeba07d9a6ecb78e73ab7",
     "role": "batch08 full-precision S audit + eligibility reasons + scores"},
    {"key": "batch10_predictions.tsv", "path": str(B10_PRED_TSV),
     "sha256": "d6845fb2d64630213996d5e8f938bb9e9df96af579401a94f66a03a34b93e5d3",
     "role": "batch10 stored per-gene predictions (replay reference)"},
]

HGB_CONFIG = {
    "model": "HistGradientBoostingRegressor",
    "max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15,
    "l2_regularization": 0.1, "early_stopping": False, "random_state": 20261011,
}
MODEL_KEY = "hgb_aug"


def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)
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


def windows_process_creation_time(pid):
    """Return process creation epoch-seconds via Windows GetProcessTimes (ctypes).

    Distinct from any controller started_at / status-writer timestamp.
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
    assert np.array_equal(n_prom, np.diff(starts))
    return gene_gid, gene_split, comp_dense, n_prom


def load_feature_matrix():
    z45 = np.load(FEATURES45_NPZ, allow_pickle=False)
    comp44 = z45["comp44"].astype(np.float64)
    log1p = z45["log1p_count"].astype(np.float64)
    X45 = np.concatenate([comp44, log1p[:, None]], axis=1)
    assert X45.shape == (N_FEATURE_UNIVERSE, 45)
    X1024 = np.load(FEATURES1024_NPZ)["X1024"].astype(np.float64)
    assert X1024.shape == (N_FEATURE_UNIVERSE, 1024)
    X = np.concatenate([X45, X1024], axis=1)
    assert X.shape == (N_FEATURE_UNIVERSE, 1069)
    return X


def base_gene_id(v):
    return (str(v) or "").split(".")[0]


def _collapse_duplicate(values):
    nz = [x for x in values if x > 0.0]
    zc = sum(1 for x in values if x == 0.0)
    if nz and zc > 0:
        return max(nz)
    return statistics.fmean(values)


def reconstruct_s_raw_full(expr_path=None):
    import pandas as pd
    expr_path = Path(expr_path) if expr_path else EXPR_TSV
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
            collapsed.append(_collapse_duplicate(vals) if len(vals) > 1 else vals[0])
        tc = len(collapsed)
        detectable = [x for x in collapsed if x >= DETECTABLE_TPM]
        dtc = len(detectable)
        b_raw = dtc / tc if tc else 0.0
        if b_raw >= B_MIN and dtc >= MIN_DET_TISSUES:
            m = statistics.median([np.log2(1.0 + x) for x in detectable])
            mad = statistics.median([abs(np.log2(1.0 + x) - m) for x in detectable])
            full_s[gene] = -mad
    return full_s


def load_label_catalog():
    import pandas as pd
    df = pd.read_csv(AUDIT_TSV, sep="\t", dtype={"gene_id": "str"}, low_memory=False)
    catalog = {}
    for row in df.itertuples(index=False):
        gid = base_gene_id(row.gene_id)
        catalog[gid] = {
            "gene_id": gid,
            "S_defined_main": bool(int(row.S_defined_main)) if pd.notna(row.S_defined_main) else False,
            "S_eligibility_reason": ("" if pd.isna(row.S_eligibility_reason) else str(row.S_eligibility_reason)),
            "E_raw": (None if pd.isna(row.E_raw) else float(row.E_raw)),
            "S_raw_6dec": (None if pd.isna(row.S_raw_6dec) else float(row.S_raw_6dec)),
            "S_raw_full": (None if pd.isna(row.S_raw_full) else float(row.S_raw_full)),
            "B_raw": (None if pd.isna(row.B_raw) else float(row.B_raw)),
            "Score_E": (None if pd.isna(row.Score_E) else float(row.Score_E)),
            "Score_S": (None if pd.isna(row.Score_S) else float(row.Score_S)),
            "Score_B": (None if pd.isna(row.Score_B) else float(row.Score_B)),
            "detectable_tissue_count": (None if pd.isna(row.detectable_tissue_count) else int(row.detectable_tissue_count)),
            "tissue_count": (None if pd.isna(row.tissue_count) else int(row.tissue_count)),
        }
    return catalog


def verify_artifacts(registry=None):
    registry = registry if registry is not None else FROZEN_ARTIFACTS
    problems = []
    for art in registry:
        p = Path(art["path"])
        expect = art.get("sha256")
        if expect is None:
            continue
        if not p.is_file():
            problems.append({"key": art["key"], "path": str(p), "error": "missing_file"})
            continue
        actual = sha256_file(p)
        if actual != expect:
            problems.append({"key": art["key"], "path": str(p),
                             "expected_sha256": expect, "actual_sha256": actual,
                             "error": "sha256_mismatch"})
    return (len(problems) == 0), problems

