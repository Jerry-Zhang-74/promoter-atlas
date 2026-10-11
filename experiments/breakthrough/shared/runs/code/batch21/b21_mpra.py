#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K562 MPRA 164 bp legnet ensemble inference (READ-ONLY reuse of the frozen
batch02 architecture). Nothing is refit or reselected here.

Honesty contract for new sequences (batch21 repair of the batch20 reporting gap):
  * A sequence that SHA-matches an entry of the prior batch02 grouped-CV cohort
    gets its HONEST fold-aware OOF value (in_source=True, oof_available=True).
  * A sequence that matches the broader GSE118242 catalog but is NOT in the prior
    CV cohort gets in_source=True with oof_available=False; we do NOT present the
    15-model mean as new-sequence accuracy for it.
  * A genuinely new sequence gets the 15-model mean, labelled out_of_source, and
    is NEVER described with the in-sample all-15 agreement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from b21_common import (BATCH02_CODE, MPRA_LEN, MPRA_WEIGHT_DIR, COMPONENTS,
                        FOLD_ASSIGN, MPRA_SUPERVISED, BASE, AMBIG, B21InputError,
                        sha256_file, N_FOLD, FOLD_SEEDS)

sys.path.insert(0, str(BATCH02_CODE))
import trackB_grouped_cv as t  # noqa: E402

_CACHE = {}


def _catalog_sequences():
    if "cat" in _CACHE:
        return _CACHE["cat"]
    import pandas as pd
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    oligo = df["Oligo_index"].astype(str).tolist()
    _CACHE["cat"] = {"seqs": seqs, "oligo": oligo}
    return _CACHE["cat"]


def _cohort_oof():
    """Honest fold-aware OOF for the PRIOR batch02 CV cohort (9442 sequences)."""
    if "oof" in _CACHE:
        return _CACHE["oof"]
    import pandas as pd
    comp = pd.read_csv(COMPONENTS, sep="\t")
    oof = pd.read_csv(Path(COMPONENTS).parents[3] / "reports" / "breakthrough_20261002"
                      / "batch02" / "trackB_oof_predictions.tsv", sep="\t")
    m = comp[["Oligo_index", "sequence"]].merge(oof, on="Oligo_index", how="inner")
    d = {}
    for r in m.itertuples(index=False):
        d[str(r.sequence).upper()] = {
            "oof_legnet": float(r.legnet_ens_pred), "fold": int(r.fold),
            "component_id": str(r.component_id), "oligo_index": str(r.Oligo_index),
            "activity": float(r.activity)}
    _CACHE["oof"] = d
    return d


def _fold_stats():
    if "fold_stats" in _CACHE:
        return _CACHE["fold_stats"]
    import pandas as pd
    df = pd.read_csv(COMPONENTS, sep="\t")
    comp_ids = df["verified_component_id"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    fold_of, _, _, _, _ = t.assign_folds(comp_ids, len(df), t.SEED_FOLD)
    stats = {}
    for f in range(N_FOLD):
        outer = np.where(fold_of != f)[0]
        stats[f] = (float(y[outer].mean()), float(y[outer].std()))
    _CACHE["fold_stats"] = stats
    return stats


def load_legnet_ensemble():
    if "models" in _CACHE:
        return _CACHE["models"]
    import torch
    models = []
    for f in range(N_FOLD):
        for s in FOLD_SEEDS:
            p = MPRA_WEIGHT_DIR / ("weights_legnet_fold%d_seed%d.pt" % (f, s))
            m = t.build_cnn("legnet")
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
            m.load_state_dict(sd)
            m.to(t.DEVICE)
            m.eval()
            models.append((f, s, m, str(p), sha256_file(p)))
    _CACHE["models"] = models
    return models


def predict_ensemble(seqs):
    """15-model mean for genuinely NEW sequences (out of the CV cohort)."""
    if isinstance(seqs, str):
        seqs = [seqs]
    x = np.zeros((len(seqs), MPRA_LEN), dtype=np.uint8)
    for i, s in enumerate(seqs):
        e = t.encode_sequences([s])[0]
        x[i] = e
    stats = _fold_stats()
    models = load_legnet_ensemble()
    per = np.zeros((len(models), len(seqs)), dtype=np.float64)
    for k, (f, s, m, path, sha) in enumerate(models):
        mu, sd = stats[f]
        per[k] = t.predict_cnn(m, x) * sd + mu
    return {"ensemble_mean": per.mean(axis=0),
            "ensemble_std_over_models": per.std(axis=0),
            "n_models": len(models),
            "units": "K562 MPRA activity (Mean_log10)"}


def source_sha():
    return {"trackB_grouped_cv.py": sha256_file(Path(t.__file__)),
            "trackB_train_components.tsv": sha256_file(COMPONENTS),
            "mpra_supervised.tsv": sha256_file(MPRA_SUPERVISED),
            "trackB_cv_fold_assignment.tsv": sha256_file(FOLD_ASSIGN)}
