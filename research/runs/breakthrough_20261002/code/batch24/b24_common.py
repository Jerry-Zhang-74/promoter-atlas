#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 v6.18.0-hek-kernel.rc1 shared paths + helpers.

Read-only reuse of batch22 metric definitions (identical Spearman/MAE/R2 and
top-decile tie rule). NO refits of the cached batch22 R1 baseline.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
B22_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch22"

BATCH = "batch24"
VERSION = "v6.18.0-hek-kernel.rc1"

RUN_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch24"
CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch24"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch24"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch24"
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch24"
FIG_OUT = REPORT_OUT / "figures"

# ---- read-only frozen inputs -------------------------------------------------
B22_DATA = ROOT / "data" / "breakthrough_20261002" / "batch22"
SEQ_TABLE = B22_DATA / "sequence_table.tsv"
FOLD_TABLE = B22_DATA / "fold_assignment.tsv"
OOF_TABLE = B22_DATA / "oof_predictions.tsv"
B23_DATA = ROOT / "data" / "breakthrough_20261002" / "batch23"
LOCAL_CONSTRUCTS = B23_DATA / "local_constructs.tsv"

INPUT_SHA = {
    "sequence_table.tsv": "e6ea09134202bf6008a326ffe38cf3fd3acc003ae899845861f42d8c2e88b548",
    "fold_assignment.tsv": "f58589bb16632da7fc89c49cc2d20f8af2ba265449e58c7406f58d3a679ee82a",
    "oof_predictions.tsv": "e7ee2323db4fbd87b2435c3e9d29049226f31dd98d90165db67cd89f9247f151",
    "local_constructs.tsv": "7b479b187eceda5dcc581e81f51def4795018eeb37927aacec319c158dce300e",
}

ALPHA = 1.0
TARGET_COL = "hek_norm"
N_BOOT = 2000
SEED_BOOT = 20261025
GAIN_RHO = 0.03
MAE_RED = 0.10
MIN_POS_FOLDS = 4
N_FOLDS = 5
RECIPES = ("H1", "H2")


def set_thread_limits(n: int = 4) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def sha256_file(p) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def now_local() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_rows():
    """Return (pids, seqs, y, fold, group_id, source_seen, k562_norm)."""
    seqs = {}
    y = {}
    k562 = {}
    order = []
    with open(SEQ_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            order.append(r["prom_id"])
            seqs[r["prom_id"]] = r["sequence"].strip().upper()
            y[r["prom_id"]] = float(r[TARGET_COL])
            k562[r["prom_id"]] = float(r["k562_norm"])
    fold = {}
    grp = {}
    seen = {}
    with open(FOLD_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            fold[r["prom_id"]] = int(r["fold"])
            grp[r["prom_id"]] = int(r["group_id"])
            seen[r["prom_id"]] = int(r["source_seen"])
    assert set(order) == set(fold), "sequence_table / fold_assignment id mismatch"
    return (order,
            np.array([seqs[p] for p in order], dtype=object),
            np.array([y[p] for p in order], dtype=np.float64),
            np.array([fold[p] for p in order], dtype=np.int64),
            np.array([grp[p] for p in order], dtype=np.int64),
            np.array([seen[p] for p in order], dtype=np.int64),
            np.array([k562[p] for p in order], dtype=np.float64))


def load_cached_r1(order):
    """Cached batch22 R1 OOF predictions aligned to `order` by exact prom_id."""
    got = {}
    with open(OOF_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            got[r["prom_id"]] = r
    missing = [p for p in order if p not in got]
    assert not missing, "cached OOF missing ids: %s" % missing[:3]
    return np.array([float(got[p]["pred_R1"]) for p in order], dtype=np.float64)


# ---- metrics: identical definitions to batch22 (read-only import) ------------
import sys
if str(B22_CODE) not in sys.path:
    sys.path.insert(0, str(B22_CODE))
from b22_common import (  # noqa: E402
    spearman, mae, r2, top_decile_precision, group_bootstrap,
)


def metrics_block(pred, obs, tie_keys):
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    return {"n": int(len(p)), "spearman": spearman(p, o), "mae": mae(p, o),
            "mse": float(np.mean((p - o) ** 2)) if len(p) else float("nan"),
            "r2": r2(p, o), "top_decile": top_decile_precision(p, o, tie_keys)}


def paired_group_bootstrap(pred_a, pred_b, obs, groups, n_boot=N_BOOT, seed=SEED_BOOT):
    """Bootstrap independent GROUPS (paired). Returns CIs for (rho_a - rho_b)
    and MAE reduction (mae_b - mae_a, positive => a better)."""
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    drho, dmae, dprec = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), size=len(ug))
        idx = np.concatenate([idx_by[ug[j]] for j in pick])
        ra, rb = spearman(pred_a[idx], obs[idx]), spearman(pred_b[idx], obs[idx])
        if np.isfinite(ra) and np.isfinite(rb):
            drho.append(ra - rb)
        dmae.append(mae(pred_b[idx], obs[idx]) - mae(pred_a[idx], obs[idx]))
    drho = np.asarray(drho, float)
    dmae = np.asarray(dmae, float)

    def ci(v, lo_p, hi_p):
        return {"point": float(np.mean(v)), "lo": float(np.percentile(v, lo_p)),
                "hi": float(np.percentile(v, hi_p)), "n_boot": int(len(v)),
                "n_groups": int(len(ug))}
    return {
        "delta_spearman": {
            "ci95": ci(drho, 2.5, 97.5),
            "ci97_5_two_comparison": ci(drho, 1.25, 98.75),
        },
        "mae_reduction": {
            "ci95": ci(dmae, 2.5, 97.5),
            "ci97_5_two_comparison": ci(dmae, 1.25, 98.75),
        },
    }
