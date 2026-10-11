#!/usr/bin/env python3
"""Batch04 shared utilities (UTF-8 explicit, deterministic).

Re-uses the reviewed batch03 metric/seed/atomic-write helpers read-only so the
metric definitions are identical across batches. Adds a paired gene-level
bootstrap for the B2-B1 difference.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas
BATCH = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch04"
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch04"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"

sys.path.insert(0, str(BATCH03))
from b3_common import (  # noqa: E402
    sha256_file,
    sha256_bytes,
    atomic_write_json,
    rankdata,
    spearman,
    r2,
    mae,
    mse,
    set_seed,
    bootstrap_spearman_ci,
)


def metrics_dict(y, p):
    return {
        "n": int(len(y)),
        "spearman": spearman(p, y),
        "r2": r2(p, y),
        "mae": mae(p, y),
        "mse": mse(p, y),
    }


def paired_bootstrap_diff(y, p2, p1, metric, n=10000, seed=20261012):
    """Paired gene-level bootstrap of metric(B2) - metric(B1).

    Both arms predict the SAME genes (paired); resampling resamples genes with
    replacement identically for both arms. Non-independence (residual cross-gene
    homology/exact dup) is NOT corrected: the CI is conditional on the paired
    design and should not be read as an independent-gene significance test.
    """
    y = np.asarray(y, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    rng = np.random.default_rng(seed)
    base = float(metric(p2, y) - metric(p1, y))
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        ii = rng.integers(0, len(y), size=len(y))
        vals[i] = metric(p2[ii], y[ii]) - metric(p1[ii], y[ii])
    return {
        "base": base,
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_genes": int(len(y)),
        "n_boot": int(n),
        "nonindependence_note": "paired genes; residual cross-gene homology/exact-dup not corrected; CI is design-conditional",
    }
