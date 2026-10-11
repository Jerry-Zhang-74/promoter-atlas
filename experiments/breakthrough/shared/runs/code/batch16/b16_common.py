#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 v6.15.0-multitask.rc1 shared utilities.

Controlled neural task-sharing comparison + final-candidate delivery:
  A = independent E/S/B heads (3 models/seed)
  B = shared backbone with E/S/B heads (1 model/seed)
  C = shared backbone + observed epigenetic auxiliary supervision (1 model/seed)

Human gene-level natural-expression E/S/B is the primary benchmark. All neural
arms share an identical input representation (diverse 4-slot gene-bag sequence
CNN + frozen 1069-feature composition/position branch). Matched HGB baselines
(E45, S1069, B45) are fit from scratch on the SAME corrected outer train.

Targets (canonical definitions, reconstructed from original expression):
  E = Score_E (rank-scaled [0,1] gene-level expression strength).
  S = S_raw = -MAD(log2(1+detectable TPM)) over detectable tissues (S_defined mask).
  B = B_raw = detectable_tissue_count / tissue_count (cross-tissue breadth).

Corrected split: batch16 component-union (batch05 exact/RC+forward-near-identity
components + batch15 confirmed RC edges), deterministic SHA256 fold allocation.
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

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch16"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch16"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch16"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch16"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch16"

EXPR = ROOT / "labels" / "build" / "gene_expression_table.tsv"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
FEATURES45_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"
FEATURES1024_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "features1024.npz"
SPLIT_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch16" / "split" / "split_arrays.npz"
SPLIT_MANIFEST = ROOT / "data" / "breakthrough_20261002" / "batch16" / "split" / "split_manifest.json"
MEM400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
OMICS_TSV = ROOT / "features" / "omics" / "promoter_omics_targets_v1.tsv"

DETECTABLE_TPM = 1.0
B_MIN = 0.30
MIN_DET_TISSUES = 3
N_FEATURE_UNIVERSE = 57827
N_HUMAN = 507365
L400 = 400
N_SLOTS = 4

CANDIDATE_VERSION = "v6.15.0-multitask.rc1"
TRACK = "multitask"
BATCH = "batch16"
SEED_REF = 20261011
SEEDS = [20261016, 20261017]

# ---- fixed HGB baseline config (matched, no tuning) ----
HGB = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "early_stopping": False,
    "random_state": SEED_REF,
}

# ---- neural training config (frozen) ----
NEURAL = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "max_epochs": 12,
    "patience": 3,
    "emb_dim": 128,
    "feat_hidden": 128,
    "aux_loss_coef": 0.1,
    "cpu_threads": 4,
}

TASKS = ("E", "S", "B")

# aux targets: observed epigenetic/accessibility assays only (excludes CAGE/PRO-cap
# TSS-activity, which are transcription-initiation proxies of the E target).
AUX_FIELDS = ["h3k4me3", "h3k27ac", "h3k27me3", "h3k4me1", "h3k36me3", "dnase", "atac"]

BASES = "ACGT"
KMERS3 = [a + b + c for a in BASES for b in BASES for c in BASES]
KMERS4 = [a + b + c + d for a in BASES for b in BASES for c in BASES for d in BASES]

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
    i, n = 0, len(x)
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


def reconstruct_targets(expr_path: Path) -> dict:
    """Reconstruct B_raw (all genes) and S_raw (eligible genes) from original expr.

    Matches batch09/10 reconstruct_s_raw definitions exactly (full precision,
    not rounded label export).
    """
    import pandas as pd
    expr = pd.read_csv(expr_path, sep="\t", usecols=["gene_id", "tissue", "mean_tpm"],
                       dtype={"gene_id": "str", "tissue": "str", "mean_tpm": np.float64},
                       low_memory=False)
    gt = defaultdict(lambda: defaultdict(list))
    for gi, tissue, mt in zip(expr["gene_id"], expr["tissue"], expr["mean_tpm"]):
        gt[base_gene_id(gi)][tissue].append(float(mt))
    del expr
    b_raw, s_raw = {}, {}
    for gene in sorted(gt):
        tm = gt[gene]
        collapsed = []
        for tissue in sorted(tm):
            vals = tm[tissue]
            collapsed.append(collapse_duplicate(vals)[0] if len(vals) > 1 else vals[0])
        tc = len(collapsed)
        detectable = [x for x in collapsed if x >= DETECTABLE_TPM]
        dtc = len(detectable)
        br = dtc / tc if tc else 0.0
        b_raw[gene] = br
        if br >= B_MIN and dtc >= MIN_DET_TISSUES:
            s_raw[gene] = -mad([math.log2(1.0 + x) for x in detectable])
    return {"b_raw": b_raw, "s_raw": s_raw}


def load_gene_universe():
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    record_id = z["record_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    score_e = z["Score_E"].astype(np.float64)
    starts = z["starts"].astype(np.int64)
    gene_gid = gene_id[starts[:-1]]
    assert len(gene_gid) == N_FEATURE_UNIVERSE
    return gene_gid, record_id, r400, score_e, starts


def load_features():
    z45 = np.load(FEATURES45_NPZ, allow_pickle=True)
    comp44 = z45["comp44"].astype(np.float64)
    log1p_count = z45["log1p_count"].astype(np.float64)
    z1024 = np.load(FEATURES1024_NPZ, allow_pickle=True)
    x1024 = z1024["X1024"].astype(np.float64)
    assert comp44.shape == (N_FEATURE_UNIVERSE, 44)
    assert log1p_count.shape == (N_FEATURE_UNIVERSE,)
    assert x1024.shape == (N_FEATURE_UNIVERSE, 1024)
    X = np.concatenate([comp44, log1p_count[:, None], x1024], axis=1)
    assert X.shape == (N_FEATURE_UNIVERSE, 1069)
    return X


def load_split():
    s = np.load(SPLIT_NPZ, allow_pickle=True)
    gene_split = s["gene_split"].astype(str)
    new_comp_dense = s["new_comp_dense"].astype(np.int64)
    assert len(gene_split) == N_FEATURE_UNIVERSE
    assert len(new_comp_dense) == N_FEATURE_UNIVERSE
    return gene_split, new_comp_dense


def diverse_slots(n_prom, gene_idx):
    """Fixed diverse 4-slot selection: p0 = gene_idx % n, slots = [(p0+k) % n, k=0..3].

    Identical across all arms and seeds; no epoch-dependent rotation (documented
    deviation from batch05 per-epoch rotation, kept fixed for cross-arm identity).
    """
    n = int(n_prom)
    p0 = int(gene_idx) % n
    return [(p0 + k) % n for k in range(N_SLOTS)]
