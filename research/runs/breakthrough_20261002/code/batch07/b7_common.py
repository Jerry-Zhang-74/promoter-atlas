#!/usr/bin/env python3
"""Batch07 shared utilities (UTF-8 explicit, deterministic).

Re-uses the reviewed batch03/batch05/batch06 metric/seed/atomic-write helpers
read-only so metric definitions stay identical across batches. Adds the
Mahalanobis coverage-score helper and the fixed-coverage-rule acceptance
constants for v6.8.0-covergate.rc1.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas
BATCH = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch07"
BATCH05 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch05"
BATCH06 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch06"
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch07"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch07"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch07"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch07"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
DATA06 = ROOT / "data" / "breakthrough_20261002" / "batch06"
DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
MODEL_DIR06 = ROOT / "models" / "breakthrough_20261002" / "batch06"

sys.path.insert(0, str(BATCH03))
sys.path.insert(0, str(BATCH05))
sys.path.insert(0, str(BATCH06))

from b3_common import (  # noqa: E402
    sha256_file, sha256_bytes, atomic_write_json, rankdata, spearman, r2, mae, mse,
)
from b5_common import metrics_dict, component_bootstrap_ci  # noqa: E402
from b6_common import fast_spearman, fast_component_bootstrap_ci  # noqa: E402

CANDIDATE = "v6.8.0-covergate.rc1"

# Frozen upstream artifact SHAs (verified read-only).
SPLIT_ARRAYS_SHA = "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5"
FEATURES45_SHA = "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800"
HGB_SHA = "994db28e75a42e9bef239e5f854d1352aeefa89ebaa66676c0c1e652d2e726c6"
FINALRIDGE_SHA = "0e0825d1b3a7982e811acd8b5a69b650c93c24b06d8a16a967e92d0a1f230fcc"
OOF_RESIDUALS_SHA = "27fb8f65507f9faab3a8643d1093c6304b8ae5f312e01b1e6ed40aa275827e15"

# Frozen coverage-rule configuration (single pre-registered rule; no threshold grid).
PRIMARY_QUANTILE = 0.90
DIAGNOSTIC_QUANTILES = [0.50, 0.70, 0.90, 1.00]
QUANTILE_METHOD = "linear"  # deterministic numpy linear-interpolation quantile

# Acceptance thresholds (pre-registered; primary HGB gate only).
MIN_TEST_COVERAGE = 0.80
MIN_REL_MAE_REDUCTION = 0.05
BOOT_N = 1000
BOOT_SEED = 20261011
RANDOM_MASK_N = 500
RANDOM_MASK_SEED = 20261012
HIGH_ERROR_THRESHOLD = 0.20

GEOMETRY_NPZ = MODEL_DIR / "covergate_geometry.npz"
GEOMETRY_META = MODEL_DIR / "covergate_geometry.json"


def mahalanobis_sq(Xs, location, precision):
    """Squared Mahalanobis distance of standardized rows Xs (n x d).

    d_i = (x_i - mu)' P (x_i - mu) where P is the precision matrix.
    Lower = more central in the fitted train feature geometry (NOT a
    calibrated probability and NOT a comprehensive density/OOD guarantee).
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    delta = Xs - np.asarray(location, dtype=np.float64)
    return np.einsum("ij,jk,ik->i", delta, np.asarray(precision, dtype=np.float64), delta)


def quantile_linear(x, q):
    """Deterministic numpy linear-interpolation quantile."""
    return float(np.quantile(np.asarray(x, dtype=np.float64), q, method="linear"))


def high_error_rate(y, p, thr=HIGH_ERROR_THRESHOLD):
    e = np.abs(np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return float(np.mean(e > thr))


def describe_range(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std()),
        "mean": float(x.mean()),
    }


def component_bootstrap_gated_mae_diff(y, p, comp, dist, threshold, n=BOOT_N, seed=BOOT_SEED):
    """Whole-component bootstrap of (MAE over full resample) - (MAE over retained resample).

    Both metrics are recomputed on the SAME resample with the gate (dist <= threshold)
    fixed. Whole components are resampled with replacement so within-component gene
    dependence is preserved. Degenerate resamples (zero retained rows) are counted and
    yield NaN for the retained metric / difference.

    This estimates selective-risk association on this dataset only; it is NOT causal
    model improvement and NOT a conformal/calibration guarantee.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    comp = np.asarray(comp)
    dist = np.asarray(dist, dtype=float)
    assert len(y) == len(p) == len(comp) == len(dist)

    order = np.argsort(comp, kind="mergesort")
    comp_sorted = comp[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(comp_sorted)]])
    counts = ends - starts
    n_comps = len(counts)

    rng = np.random.default_rng(seed)
    full_point = mae(p, y)
    retained_mask = dist <= threshold
    retained_point = mae(p[retained_mask], y[retained_mask]) if retained_mask.sum() >= 1 else float("nan")

    vals = np.empty(n, dtype=np.float64)
    n_degenerate = 0
    retained_maes = np.empty(n, dtype=np.float64)
    full_maes = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, n_comps, size=n_comps)
        chosen_counts = counts[chosen]
        tot = int(chosen_counts.sum())
        cum = np.concatenate([[0], np.cumsum(chosen_counts)[:-1]])
        j = np.arange(tot)
        k = np.searchsorted(cum, j, side="right") - 1
        off = j - cum[k]
        idx = order[starts[chosen[k]] + off]
        fm = mae(p[idx], y[idx])
        full_maes[i] = fm
        m = dist[idx] <= threshold
        if m.sum() == 0:
            n_degenerate += 1
            retained_maes[i] = np.nan
            vals[i] = np.nan
        else:
            rm = mae(p[idx[m]], y[idx[m]])
            retained_maes[i] = rm
            vals[i] = fm - rm

    finite = vals[np.isfinite(vals)]
    return {
        "point_full_mae": full_point,
        "point_retained_mae": retained_point,
        "point_diff": float(full_point - retained_point) if np.isfinite(retained_point) else float("nan"),
        "bootstrap_diff_mean": float(np.mean(finite)) if len(finite) else float("nan"),
        "bootstrap_diff_ci_low": float(np.percentile(finite, 2.5)) if len(finite) else float("nan"),
        "bootstrap_diff_ci_high": float(np.percentile(finite, 97.5)) if len(finite) else float("nan"),
        "bootstrap_full_mae_mean": float(np.mean(full_maes)),
        "bootstrap_retained_mae_mean": float(np.nanmean(retained_maes)) if np.isfinite(retained_maes).any() else float("nan"),
        "n_genes": int(len(y)),
        "n_components": int(n_comps),
        "n_boot": int(n),
        "n_degenerate_zero_retained": int(n_degenerate),
        "nonindependence_note": "whole components resampled; within-component gene dependence preserved; cross-component residual near-identity/exact-dup NOT corrected; CI is design-conditional; gate fixed at train threshold",
    }
