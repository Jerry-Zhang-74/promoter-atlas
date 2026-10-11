#!/usr/bin/env python3
"""Batch07 geometry fit + freeze (TRAIN FEATURES ONLY, no labels/val/test).

Fits the fixed coverage-rule geometry:
  StandardScaler fit on outer-train 45-feature matrix only, then
  sklearn LedoitWolf covariance on the standardized train features.
Coverage score = squared Mahalanobis distance (x-mu)^T precision (x-mu).
Primary gate = distance <= 90th percentile of TRAIN distances
(deterministic linear-interpolation quantile).

This script intentionally loads NO labels and NO heldout features for fitting.
It freezes the geometry artifact + threshold and records their SHA BEFORE any
heldout evaluation happens in b7_evaluate.py.
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    MODEL_DIR, DATA05, DATA06, DATA03, atomic_write_json, sha256_file,
    CANDIDATE, PRIMARY_QUANTILE, DIAGNOSTIC_QUANTILES, QUANTILE_METHOD,
    mahalanobis_sq, quantile_linear,
    SPLIT_ARRAYS_SHA, FEATURES45_SHA,
)
from b7_data import load_features  # noqa: E402


def main():
    # Load ONLY the split assignment (for train mask) and the 45-feature matrix.
    # NO labels, NO prediction errors, NO heldout features are read here: the
    # geometry/threshold is fit from outer-train features only.
    a = np.load(DATA05 / "split_arrays.npz", allow_pickle=True)
    gene_split = a["gene_split"]
    X45, comp44, log1p_count, gc_gene = load_features()

    train_idx = np.flatnonzero(gene_split == "train")
    X_train = X45[train_idx].astype(np.float64)
    assert X_train.shape == (46263, 45), X_train.shape
    assert np.isfinite(X_train).all()

    # --- fit StandardScaler on TRAIN only ---
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train).astype(np.float64)

    # --- fit LedoitWolf on standardized TRAIN only ---
    lw = LedoitWolf(store_precision=True).fit(Xs_train)
    location = lw.location_.astype(np.float64)
    precision = lw.precision_.astype(np.float64)
    covariance = lw.covariance_.astype(np.float64)

    # --- train distances (squared Mahalanobis) ---
    train_dist = mahalanobis_sq(Xs_train, location, precision)
    assert np.isfinite(train_dist).all()

    # --- thresholds (frozen BEFORE any heldout label inspection) ---
    thresholds = {q: quantile_linear(train_dist, q) for q in DIAGNOSTIC_QUANTILES}
    primary_threshold = thresholds[PRIMARY_QUANTILE]

    # --- apply frozen geometry to all genes (feature-centrality only) ---
    Xs_all = scaler.transform(X45.astype(np.float64)).astype(np.float64)
    dist_all = mahalanobis_sq(Xs_all, location, precision)

    # --- save frozen geometry artifact ---
    np.savez_compressed(
        MODEL_DIR / "covergate_geometry.npz",
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        location=location,
        covariance=covariance,
        precision=precision,
        threshold=primary_threshold,
        diagnostic_quantiles=np.asarray(DIAGNOSTIC_QUANTILES, dtype=np.float64),
        diagnostic_thresholds=np.asarray([thresholds[q] for q in DIAGNOSTIC_QUANTILES], dtype=np.float64),
        train_idx=train_idx.astype(np.int64),
        train_dist=train_dist.astype(np.float64),
        n_train=int(len(train_idx)),
    )

    geometry_sha = sha256_file(MODEL_DIR / "covergate_geometry.npz")

    meta = {
        "candidate_version": CANDIDATE,
        "geometry_file": "models/breakthrough_20261002/batch07/covergate_geometry.npz",
        "geometry_sha256": geometry_sha,
        "fit": {
            "scaler": "sklearn.preprocessing.StandardScaler (with_mean=True, with_std=True)",
            "covariance": "sklearn.covariance.LedoitWolf(store_precision=True)",
            "fit_on": "outer-TRAIN 45-feature rows only (n=46263)",
            "labels_used": False,
            "heldout_features_used": False,
            "prediction_errors_used": False,
        },
        "coverage_score": "squared Mahalanobis distance (x-mu)^T precision (x-mu); lower = more central in this particular feature geometry, NOT calibrated probability",
        "primary_gate": {
            "quantile": PRIMARY_QUANTILE,
            "quantile_method": QUANTILE_METHOD,
            "threshold": float(primary_threshold),
        },
        "diagnostic_thresholds": {f"{q:.2f}": float(thresholds[q]) for q in DIAGNOSTIC_QUANTILES},
        "train_distance": {
            "n": int(len(train_dist)),
            "min": float(train_dist.min()),
            "max": float(train_dist.max()),
            "mean": float(train_dist.mean()),
            "std": float(train_dist.std()),
        },
        "source_shas": {
            "split_arrays_npz": SPLIT_ARRAYS_SHA,
            "features45_npz": FEATURES45_SHA,
        },
        "limitations": [
            "feature centrality, not comprehensive density/OOD guarantee",
            "multimodality and data-internal annotation bias remain",
            "gate miss is not proof of biological out-of-domain",
            "retrospective/exploratory on reused split, never prospective",
        ],
        "frozen_at": datetime.datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(MODEL_DIR / "covergate_geometry.json", meta)

    print(json.dumps({
        "geometry_sha256": geometry_sha,
        "primary_threshold": float(primary_threshold),
        "diagnostic_thresholds": {f"{q:.2f}": float(thresholds[q]) for q in DIAGNOSTIC_QUANTILES},
        "train_distance_stats": meta["train_distance"],
        "n_train": int(len(train_idx)),
    }, indent=2, ensure_ascii=False))
    print("GEOMETRY FROZEN")


if __name__ == "__main__":
    main()
