#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K562 MPRA 164 bp inference from RAW sequence (previously validation-selected).

Architecture, encoder and forward pass are imported READ-ONLY from the frozen
batch02 source (trackB_grouped_cv) so the network cannot silently diverge.
The reference predictor is the LEGNET ensemble (5 grouped folds x 3 seeds),
which was selected earlier by the held-out grouped criterion (legnet > smallcnn);
it is NOT re-selected here. Each fold has its own train-only target
standardization (mean/std of its outer-train activities), used to invert.

Scope: K562 MPRA reporter activity only. NOT HEK293T, NOT gene E/S/B.
Robustness was NOT confirmed in batch02, so outputs are labelled UNCONFIRMED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from b20_common import (BATCH02_CODE, MPRA_LEN, MPRA_WEIGHTS, MPRA_SUPERVISED,
                        TRACKA_NPZ, B20InputError, encode_uint8, sha256_file)

sys.path.insert(0, str(BATCH02_CODE))
import trackB_grouped_cv as t  # noqa: E402

DATA = Path(t.DATA)
COMPONENTS = DATA / "trackB_train_components.tsv"
FOLDS = range(5)
SEEDS = (20261008, 20261009, 20261010)

_CACHE = {}


def _fold_stats():
    """Per-fold train-only (y_mu, y_sd) recomputed exactly as the training code did."""
    if "fold_stats" in _CACHE:
        return _CACHE["fold_stats"]
    import pandas as pd
    df = pd.read_csv(COMPONENTS, sep="\t")
    comp_ids = df["verified_component_id"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    fold_of, _, _, _, _ = t.assign_folds(comp_ids, len(df), t.SEED_FOLD)
    stats = {}
    for f in FOLDS:
        outer = np.where(fold_of != f)[0]
        stats[f] = (float(y[outer].mean()), float(y[outer].std()))
    _CACHE["fold_stats"] = stats
    return stats


def load_legnet_ensemble():
    if "models" in _CACHE:
        return _CACHE["models"]
    import torch
    models = []
    for f in FOLDS:
        for s in SEEDS:
            p = MPRA_WEIGHTS["legnet"][f * len(SEEDS) + (s - SEEDS[0])]
            m = t.build_cnn("legnet")
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
            m.load_state_dict(sd)
            m.to(t.DEVICE)
            m.eval()
            models.append((f, s, m, str(p), sha256_file(p)))
    _CACHE["models"] = models
    return models


def predict_legnet(seqs, per_model: bool = False):
    """seqs: list of 164 bp strings (or a single string).

    Returns dict with ensemble mean activity (raw log10 units) and the per-model
    spread. All inputs must be exactly 164 bp; anything else raises B20InputError.
    """
    if isinstance(seqs, str):
        seqs = [seqs]
    x = np.zeros((len(seqs), MPRA_LEN), dtype=np.uint8)
    for i, s in enumerate(seqs):
        e = encode_uint8(s, MPRA_LEN, name="MPRA input")
        x[i] = e
    stats = _fold_stats()
    models = load_legnet_ensemble()
    per = np.zeros((len(models), len(seqs)), dtype=np.float64)
    for k, (f, s, m, path, sha) in enumerate(models):
        mu, sd = stats[f]
        raw = t.predict_cnn(m, x)          # z-units (standardized)
        per[k] = raw * sd + mu
    out = {
        "ensemble_mean": per.mean(axis=0),
        "ensemble_std_over_models": per.std(axis=0),
        "n_models": len(models),
        "units": "K562 MPRA activity (Mean_log10)",
    }
    if per_model:
        out["per_model"] = per
        out["model_index"] = [{"fold": f, "seed": s, "path": p, "sha256": sh}
                              for (f, s, _, p, sh) in models]
    return out


def source_sha():
    return {
        "trackB_grouped_cv.py": sha256_file(Path(t.__file__)),
        "trackB_train_components.tsv": sha256_file(COMPONENTS),
        "mpra_supervised.tsv": sha256_file(MPRA_SUPERVISED),
    }
