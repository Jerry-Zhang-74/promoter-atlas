#!/usr/bin/env python3
"""Batch06 shared utilities. UTF-8 explicit, deterministic.

Re-uses the reviewed batch05 metric/seed/atomic-write/component-bootstrap
helpers read-only so metric definitions are identical across batches.
Adds a tie-breaking (larger-alpha) ridge alpha selector.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas
BATCH = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch06"
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
BATCH05 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch05"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch06"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch06"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch06"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch06"

sys.path.insert(0, str(BATCH05))
from b5_common import (  # noqa: E402
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
    metrics_dict,
    component_bootstrap_diff,
    component_bootstrap_ci,
)

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_FOLDS = 5
SEEDS = [20261011, 20261012]
CANDIDATE = "v6.7.0-residualgene.rc1"


def select_alpha_max_sp_tie_larger(alpha_grid, spearman_by_alpha):
    """Deterministic alpha selection: max inner Spearman, ties prefer LARGER alpha."""
    best_a = None
    best_sp = -np.inf
    for a in alpha_grid:
        sp = float(spearman_by_alpha[a])
        if sp > best_sp:
            best_sp = sp
            best_a = a
        elif sp == best_sp:
            # ties prefer larger alpha
            if a > best_a:
                best_a = a
    return best_a, best_sp


def fast_component_bootstrap_diff(y, p2, p1, comp, metric, n=2000, seed=20261012):
    """Vectorized paired component-level bootstrap of metric(B2)-metric(B1).

    Whole components resampled with replacement (within-component dependence
    preserved). Genes assumed ordered the same for y/p2/p1/comp. Uses a
    vectorized flat-index construction (much faster than the batch05 loop).
    """
    y = np.asarray(y, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    comp = np.asarray(comp)
    assert len(y) == len(p2) == len(p1) == len(comp)
    order = np.argsort(comp, kind="mergesort")
    comp_sorted = comp[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(comp_sorted)]])
    counts = ends - starts
    n_comps = len(counts)
    rng = np.random.default_rng(seed)
    base = float(metric(p2, y) - metric(p1, y))
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, n_comps, size=n_comps)
        chosen_counts = counts[chosen]
        tot = int(chosen_counts.sum())
        cum = np.concatenate([[0], np.cumsum(chosen_counts)[:-1]])
        j = np.arange(tot)
        k = np.searchsorted(cum, j, side="right") - 1
        off = j - cum[k]
        idx = order[starts[chosen[k]] + off]
        vals[i] = metric(p2[idx], y[idx]) - metric(p1[idx], y[idx])
    return {
        "base": base,
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_genes": int(len(y)),
        "n_components": int(n_comps),
        "n_boot": int(n),
        "nonindependence_note": "whole components resampled; within-component gene dependence preserved; cross-component residual near-identity/exact-dup NOT corrected; CI is design-conditional",
    }


def fast_component_bootstrap_ci(y, p, comp, metric, n=2000, seed=20261012):
    """Vectorized component-level bootstrap CI of a single metric."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    comp = np.asarray(comp)
    assert len(y) == len(p) == len(comp)
    order = np.argsort(comp, kind="mergesort")
    comp_sorted = comp[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(comp_sorted)]])
    counts = ends - starts
    n_comps = len(counts)
    rng = np.random.default_rng(seed)
    point = float(metric(p, y))
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, n_comps, size=n_comps)
        chosen_counts = counts[chosen]
        tot = int(chosen_counts.sum())
        cum = np.concatenate([[0], np.cumsum(chosen_counts)[:-1]])
        j = np.arange(tot)
        k = np.searchsorted(cum, j, side="right") - 1
        off = j - cum[k]
        idx = order[starts[chosen[k]] + off]
        vals[i] = metric(p[idx], y[idx])
    return {
        "point": point,
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_genes": int(len(y)),
        "n_components": int(n_comps),
        "n_boot": int(n),
        "nonindependence_note": "whole components resampled; within-component gene dependence preserved; cross-component residual near-identity/exact-dup NOT corrected; CI is design-conditional",
    }


def fast_spearman(a, b):
    """Spearman identical to batch05 (average-rank ties) but using scipy rankdata.

    Verified bitwise-identical to b3_common.spearman for tie-heavy data.
    """
    from scipy.stats import rankdata as scipy_rankdata
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = scipy_rankdata(a, method="average")
    rb = scipy_rankdata(b, method="average")
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)
