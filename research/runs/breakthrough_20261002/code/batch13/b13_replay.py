#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 saved-model replay (no refit).

For each of the five folds, load the saved base/full HGB models, scalers and
transforms from the attempt checkpoint and recompute the held-out predictions
in raw S units, verifying they match oof_predictions.tsv exactly. Confirms the
pooled OOF predictions come from the five separately trained models and that no
model was applied to rows outside its own held-out fold.
"""
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import joblib  # noqa: E402

from b13_common import (  # noqa: E402
    DATA_OUT, MODEL_OUT, CANDIDATE_VERSION, BATCH, N_FOLDS,
    N_EXPECTED_ELIGIBLE, reconstruct_s_raw, load_gene_universe,
    load_features45, load_full_features, atomic_write_json)


def main():
    attempt_dir = MODEL_OUT / "attempt_001_fit"
    pred = pd.read_csv(DATA_OUT / "oof_predictions.tsv", sep="\t", low_memory=False)
    assert len(pred) == N_EXPECTED_ELIGIBLE

    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    X45 = load_features45()[eligible]
    X_full = load_full_features()[eligible]

    fold_of_gene = pred["fold"].values.astype(np.int64)
    s_raw = pred["S_raw_full"].values.astype(np.float64)

    fold_checks = {}
    all_base = np.empty(N_EXPECTED_ELIGIBLE, dtype=np.float64)
    all_full = np.empty(N_EXPECTED_ELIGIBLE, dtype=np.float64)

    with threadpool_limits(limits=4):
        for f in range(N_FOLDS):
            fold_dir = attempt_dir / ("fold%d" % f)
            tf = json.loads((fold_dir / "transforms.json").read_text(encoding="utf-8"))
            mu = tf["train_mean"]
            sd = tf["train_std"]
            base_model = joblib.load(fold_dir / "base_model.joblib")
            full_model = joblib.load(fold_dir / "full_model.joblib")
            base_scaler = joblib.load(fold_dir / "base_scaler.joblib")
            full_scaler = joblib.load(fold_dir / "full_scaler.joblib")

            heldout = fold_of_gene == f
            pb = base_model.predict(base_scaler.transform(X45)) * sd + mu
            pf = full_model.predict(full_scaler.transform(X_full)) * sd + mu
            all_base[heldout] = pb[heldout]
            all_full[heldout] = pf[heldout]

            # only held-out rows are meaningful OOF; verify them
            ok_b = np.allclose(pb[heldout], pred["base_pred"].values[heldout], atol=1e-8)
            ok_f = np.allclose(pf[heldout], pred["full_pred"].values[heldout], atol=1e-8)
            fold_checks["fold%d" % f] = {
                "n_heldout": int(heldout.sum()),
                "base_replay_match": bool(ok_b),
                "full_replay_match": bool(ok_f),
            }

    ok_all_b = np.allclose(all_base, pred["base_pred"].values, atol=1e-8)
    ok_all_f = np.allclose(all_full, pred["full_pred"].values, atol=1e-8)

    out = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "n_eligible": N_EXPECTED_ELIGIBLE,
        "fold_checks": fold_checks,
        "pooled_base_replay_match": bool(ok_all_b),
        "pooled_full_replay_match": bool(ok_all_f),
        "all_folds_all_arms_match": bool(ok_all_b and ok_all_f and
                                         all(fold_checks["fold%d" % f]["base_replay_match"]
                                             and fold_checks["fold%d" % f]["full_replay_match"]
                                             for f in range(N_FOLDS))),
    }
    atomic_write_json(DATA_OUT / "replay.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
