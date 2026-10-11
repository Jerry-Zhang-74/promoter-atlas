#!/usr/bin/env python3
"""Batch05 shared utilities (UTF-8 explicit, deterministic).

Re-uses the reviewed batch03 metric/seed/atomic-write helpers read-only so the
metric definitions are identical across batches. Adds a component-level
bootstrap for paired B2-B1 difference (whole components resampled).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas
BATCH = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch05"
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
BATCH04 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch04"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch05"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"

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


def component_bootstrap_diff(y, p2, p1, comp2, comp1, metric, n=2000, seed=20261012):
    """Paired component-level bootstrap of metric(B2) - metric(B1).

    Whole components are resampled with replacement (not individual genes), so
    the within-component dependence is preserved. Genes are assumed ordered the
    same for y/p2/p1/comp2/comp1. Returns CI over resampled component bags.
    """
    y = np.asarray(y, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    comp2 = np.asarray(comp2)
    comp1 = np.asarray(comp1)
    assert len(y) == len(p2) == len(p1) == len(comp2) == len(comp1)
    assert np.array_equal(comp2, comp1), "component arrays must be identical"
    comps = np.unique(comp2)
    rng = np.random.default_rng(seed)
    base = float(metric(p2, y) - metric(p1, y))
    # group gene indices by component for fast resampling
    order = np.argsort(comp2, kind="mergesort")
    comp_sorted = comp2[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(comp_sorted)]])
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, len(comps), size=len(comps))
        idx = []
        for c in chosen:
            lo = int(starts[c])
            hi = int(ends[c])
            idx.append(order[lo:hi])
        idx = np.concatenate(idx)
        vals[i] = metric(p2[idx], y[idx]) - metric(p1[idx], y[idx])
    return {
        "base": base,
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_genes": int(len(y)),
        "n_components": int(len(comps)),
        "n_boot": int(n),
        "nonindependence_note": "whole components resampled; within-component gene dependence preserved; cross-component residual near-identity/exact-dup NOT corrected; CI is design-conditional",
    }


def component_bootstrap_ci(y, p, comp, metric, n=2000, seed=20261012):
    """Component-level bootstrap CI of a single metric (whole components resampled).

    Preserves within-component gene dependence. Genes assumed ordered the same
    for y/p/comp. Returns percentile CI over resampled component bags.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    comp = np.asarray(comp)
    assert len(y) == len(p) == len(comp)
    comps = np.unique(comp)
    rng = np.random.default_rng(seed)
    order = np.argsort(comp, kind="mergesort")
    comp_sorted = comp[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(comp_sorted)]])
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, len(comps), size=len(comps))
        idx = np.concatenate([order[int(starts[c]):int(ends[c])] for c in chosen])
        vals[i] = metric(p[idx], y[idx])
    return {
        "point": float(metric(p, y)),
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_genes": int(len(y)),
        "n_components": int(len(comps)),
        "n_boot": int(n),
        "nonindependence_note": "whole components resampled; within-component gene dependence preserved; cross-component residual near-identity/exact-dup NOT corrected; CI is design-conditional",
    }
