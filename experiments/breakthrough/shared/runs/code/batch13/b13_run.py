#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 main run: component five-fold cross-fitting (exactly ten HGB fits).

For each of five folds: fit a HistGradientBoostingRegressor on the other four
folds for BOTH the 45-feature baseline and the 1069-feature full representation.
X StandardScaler fits the train folds only (independent scaler per representation);
raw S target mean/std fit the train folds only and are shared within a paired fold.
Held-out predictions are inverse-transformed to raw S units. Exactly ten scientific
fits total (2 per fold). Pooled OOF metrics come from the five separately trained
models; the primary contrast is pooled OOF full-minus-base Spearman with a paired
component bootstrap (2000 resamples, seed 20261013, 95 percent [2.5, 97.5]).

Entry point FAILS if the completed fit destination already exists (no overwrite).
Resume only unfinished fold/arm after verifying inputs/config/manifest identity.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
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
    ROOT, DATA_OUT, MODEL_OUT, REPORT_OUT, CANDIDATE_VERSION, BATCH, HGB, BOOT_SEED,
    N_FOLDS, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT, DELTA_MIN, MIN_POSITIVE_FOLDS,
    N_EXPECTED_ELIGIBLE, N_EXPECTED_COMPONENTS, EXPECTED_SHA, sha256_file,
    atomic_write_json, reconstruct_s_raw, load_gene_universe, load_features45,
    load_full_features, spearman, r2_score, mae, mse, thread_caps_record, now_local)
from b13_fold_manifest import assign_folds  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ARMS = ("base", "full")


def append_ledger(record):
    p = DATA_OUT / "attempts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def snapshot_code(attempt_dir):
    src_dir = Path(__file__).resolve().parent
    dst = attempt_dir / "code"
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in sorted(src_dir.glob("b13_*.py")):
        shutil.copy2(f, dst / f.name)
        hashes[f.name] = sha256_file(f)
    return hashes


def verify_source_shas():
    sha = {
        "split_arrays.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"),
        "features45.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"),
        "features1024.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "features1024.npz"),
        "trackA_gene_promoters.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"),
    }
    ok = all(sha[k] == EXPECTED_SHA[k] for k in EXPECTED_SHA)
    return ok, sha


def metrics(y_true, y_pred):
    return {
        "spearman": spearman(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def fit_fold(fold_dir, arm, X_arm, s_raw, train_mask):
    """Fit one arm of one fold (or load existing checkpoint on resume)."""
    model_path = fold_dir / ("%s_model.joblib" % arm)
    scaler_path = fold_dir / ("%s_scaler.joblib" % arm)
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, False

    scaler = StandardScaler()
    scaler.fit(X_arm[train_mask])
    Xs = scaler.transform(X_arm)
    train_mean = float(s_raw[train_mask].mean())
    train_std = float(s_raw[train_mask].std(ddof=0))
    z = (s_raw - train_mean) / train_std
    model = HistGradientBoostingRegressor(**HGB)
    model.fit(Xs[train_mask], z[train_mask])

    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    attempt_dir = MODEL_OUT / ("attempt_%03d_fit" % args.attempt)
    complete_marker = attempt_dir / "complete.json"

    if complete_marker.exists():
        raise SystemExit("REFUSE: completed fit attempt already exists: %s (no overwrite)"
                         % complete_marker)

    ok_sha, sha = verify_source_shas()
    if not ok_sha:
        raise SystemExit("SOURCE SHA MISMATCH: %s" % json.dumps(sha))

    fold_manifest_path = DATA_OUT / "fold_manifest.json"
    if not fold_manifest_path.exists():
        raise SystemExit("REFUSE: fold manifest missing (must be written BEFORE any fit): %s"
                         % fold_manifest_path)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))

    attempt_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = snapshot_code(attempt_dir)
    atomic_write_json(attempt_dir / "code_hashes.json", code_hashes)

    with threadpool_limits(limits=4):
        gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
        X45 = load_features45()
        X_full = load_full_features()
        full_s = reconstruct_s_raw()
        eligible = np.array([g in full_s for g in gene_gid])
        assert int(eligible.sum()) == N_EXPECTED_ELIGIBLE
        comp_e = comp_dense[eligible]
        s_raw = np.array([full_s[g] for g in gene_gid[eligible]], dtype=np.float64)
        assert np.isfinite(s_raw).all()
        split_e = gene_split[eligible]
        n_prom_e = n_prom[eligible]
        gene_id_e = gene_gid[eligible]
        X45_e = X45[eligible]
        X_full_e = X_full[eligible]

        # recompute fold assignment deterministically and verify vs saved manifest
        fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_e)
        assert fold_counts == fold_manifest["fold_gene_counts"], "fold assignment redraw mismatch"
        fold_of_gene = np.array([fold_of_component[int(c)] for c in comp_e], dtype=np.int64)

        n_fits_executed = 0
        oof_base = np.empty(N_EXPECTED_ELIGIBLE, dtype=np.float64)
        oof_full = np.empty(N_EXPECTED_ELIGIBLE, dtype=np.float64)
        fold_meta_out = {}

        for f in range(N_FOLDS):
            heldout = fold_of_gene == f
            train = ~heldout
            fold_dir = attempt_dir / ("fold%d" % f)
            fold_dir.mkdir(parents=True, exist_ok=True)

            # shared target standardization within the paired fold
            train_mean = float(s_raw[train].mean())
            train_std = float(s_raw[train].std(ddof=0))
            transforms = {"train_mean": train_mean, "train_std": train_std,
                          "n_train": int(train.sum()), "n_heldout": int(heldout.sum())}
            atomic_write_json(fold_dir / "transforms.json", transforms)

            base_model, base_scaler, fitted_base = fit_fold(
                fold_dir, "base", X45_e, s_raw, train)
            full_model, full_scaler, fitted_full = fit_fold(
                fold_dir, "full", X_full_e, s_raw, train)

            if fitted_base:
                n_fits_executed += 1
            if fitted_full:
                n_fits_executed += 1

            Xs_base = base_scaler.transform(X45_e)
            Xs_full = full_scaler.transform(X_full_e)
            pred_base = base_model.predict(Xs_base) * train_std + train_mean
            pred_full = full_model.predict(Xs_full) * train_std + train_mean
            oof_base[heldout] = pred_base[heldout]
            oof_full[heldout] = pred_full[heldout]

            fold_meta_out["fold%d" % f] = {
                "n_train": int(train.sum()),
                "n_heldout": int(heldout.sum()),
                "n_components": int(np.unique(comp_e[heldout]).size),
                "train_mean": train_mean,
                "train_std": train_std,
                "fitted_base_this_run": bool(fitted_base),
                "fitted_full_this_run": bool(fitted_full),
                "base_model_sha256": sha256_file(fold_dir / "base_model.joblib"),
                "full_model_sha256": sha256_file(fold_dir / "full_model.joblib"),
                "base_scaler_sha256": sha256_file(fold_dir / "base_scaler.joblib"),
                "full_scaler_sha256": sha256_file(fold_dir / "full_scaler.joblib"),
            }

        assert np.isfinite(oof_base).all() and np.isfinite(oof_full).all()
        assert int(n_fits_executed) <= 10

        # ---- fold-level metrics (held-out only) ----
        fold_metrics = {}
        fold_deltas = []
        for f in range(N_FOLDS):
            heldout = fold_of_gene == f
            mb = metrics(s_raw[heldout], oof_base[heldout])
            mf = metrics(s_raw[heldout], oof_full[heldout])
            delta = mf["spearman"] - mb["spearman"]
            fold_deltas.append(delta)
            fold_metrics["fold%d" % f] = {
                "n": int(heldout.sum()),
                "base": mb,
                "full": mf,
                "spearman_delta_full_minus_base": delta,
                "mae_delta": mf["mae"] - mb["mae"],
                "mse_delta": mf["mse"] - mb["mse"],
                "r2_delta": mf["r2"] - mb["r2"],
            }

        # ---- pooled OOF metrics (five separately trained models) ----
        pooled_base = metrics(s_raw, oof_base)
        pooled_full = metrics(s_raw, oof_full)
        pooled_delta = pooled_full["spearman"] - pooled_base["spearman"]

        # ---- same-fold train-mean constant baseline (no heldout fit) ----
        const_pred = np.empty(N_EXPECTED_ELIGIBLE, dtype=np.float64)
        for f in range(N_FOLDS):
            heldout = fold_of_gene == f
            train = ~heldout
            const_pred[heldout] = float(s_raw[train].mean())
        const_pooled = metrics(s_raw, const_pred)

        # ---- paired component bootstrap (within fold, preserving component counts) ----
        comp_gene_idx = {}
        for c in np.unique(comp_e):
            comp_gene_idx[int(c)] = np.flatnonzero(comp_e == c)
        fold_comps = {}
        for f in range(N_FOLDS):
            heldout = fold_of_gene == f
            fold_comps[f] = [int(c) for c in np.unique(comp_e[heldout])]

        rng = np.random.default_rng(BOOT_SEED)
        deltas = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for b in range(N_BOOTSTRAP):
            idx = []
            for f in range(N_FOLDS):
                comps = fold_comps[f]
                sample = rng.choice(comps, size=len(comps), replace=True)
                for c in sample:
                    idx.append(comp_gene_idx[c])
            idx = np.concatenate(idx)
            deltas[b] = (spearman(oof_full[idx], s_raw[idx])
                         - spearman(oof_base[idx], s_raw[idx]))

        lo = float(np.percentile(deltas, BOOT_LO_PCT))
        hi = float(np.percentile(deltas, BOOT_HI_PCT))
        n_positive_folds = int(sum(1 for d in fold_deltas if d > 0))
        criterion_met = bool(pooled_delta >= DELTA_MIN and lo > 0
                             and n_positive_folds >= MIN_POSITIVE_FOLDS)

        primary = {
            "contrast": "pooled OOF full-minus-base Spearman",
            "pooled_base_spearman": pooled_base["spearman"],
            "pooled_full_spearman": pooled_full["spearman"],
            "pooled_delta": pooled_delta,
            "bootstrap": {
                "seed": BOOT_SEED,
                "n_resamples": N_BOOTSTRAP,
                "unit": "components resampled within fold, preserving per-fold component counts",
                "lo_pct": BOOT_LO_PCT,
                "hi_pct": BOOT_HI_PCT,
                "lo": lo,
                "hi": hi,
                "bootstrap_mean": float(deltas.mean()),
                "bootstrap_std": float(deltas.std(ddof=0)),
                "n_bootstrap_nan": int(np.isnan(deltas).sum()),
            },
            "fold_spearman_deltas": fold_deltas,
            "n_positive_folds": n_positive_folds,
            "min_positive_folds_required": MIN_POSITIVE_FOLDS,
            "delta_min": DELTA_MIN,
            "criterion_met": criterion_met,
            "criterion_components": {
                "pooled_delta_ge_003": bool(pooled_delta >= DELTA_MIN),
                "lower95_gt_0": bool(lo > 0),
                "positive_folds_ge_4": bool(n_positive_folds >= MIN_POSITIVE_FOLDS),
            },
        }

        # ---- across-fold descriptive statistics (folds overlap in training) ----
        def desc(xs):
            xs = [float(x) for x in xs]
            return {"mean": float(np.mean(xs)), "min": float(np.min(xs)),
                    "max": float(np.max(xs)), "std": float(np.std(xs, ddof=0))}

        across_fold = {
            "spearman_base": desc([fold_metrics["fold%d" % f]["base"]["spearman"] for f in range(N_FOLDS)]),
            "spearman_full": desc([fold_metrics["fold%d" % f]["full"]["spearman"] for f in range(N_FOLDS)]),
            "spearman_delta": desc(fold_deltas),
            "mae_base": desc([fold_metrics["fold%d" % f]["base"]["mae"] for f in range(N_FOLDS)]),
            "mae_full": desc([fold_metrics["fold%d" % f]["full"]["mae"] for f in range(N_FOLDS)]),
            "r2_base": desc([fold_metrics["fold%d" % f]["base"]["r2"] for f in range(N_FOLDS)]),
            "r2_full": desc([fold_metrics["fold%d" % f]["full"]["r2"] for f in range(N_FOLDS)]),
            "note": "folds are NOT independent replicates (training overlaps); descriptive only, no t-test",
        }

        # ---- write OOF predictions (universe-eligible order) ----
        pred_df = pd.DataFrame({
            "gene_id": gene_id_e,
            "fold": fold_of_gene,
            "component_id": comp_e,
            "original_split": split_e,
            "n_promoters": n_prom_e,
            "S_raw_full": s_raw,
            "base_pred": oof_base,
            "full_pred": oof_full,
            "base_error": oof_base - s_raw,
            "full_error": oof_full - s_raw,
            "const_train_mean_pred": const_pred,
        })
        pred_df.to_csv(DATA_OUT / "oof_predictions.tsv", sep="\t", index=False,
                       float_format="%.17g")

        metrics_out = {
            "candidate_version": CANDIDATE_VERSION,
            "batch": BATCH,
            "written_at": now_local(),
            "seed_fit": HGB["random_state"],
            "seed_bootstrap": BOOT_SEED,
            "n_eligible": N_EXPECTED_ELIGIBLE,
            "n_components": N_EXPECTED_COMPONENTS,
            "n_fits_total_expected": 10,
            "n_fits_executed_this_run": int(n_fits_executed),
            "fold_gene_counts": fold_counts,
            "fold_metrics": fold_metrics,
            "pooled_oof": {
                "base": pooled_base,
                "full": pooled_full,
                "full_minus_base": {
                    "spearman_delta": pooled_delta,
                    "mae_delta": pooled_full["mae"] - pooled_base["mae"],
                    "mse_delta": pooled_full["mse"] - pooled_base["mse"],
                    "r2_delta": pooled_full["r2"] - pooled_base["r2"],
                },
                "note": "pooled from five separately trained models; NOT one fixed deployable model",
            },
            "same_fold_train_mean_constant_baseline_pooled": const_pooled,
            "primary_contrast": primary,
            "across_fold_descriptive": across_fold,
            "elapsed_s": round(time.time() - t0, 2),
        }
        atomic_write_json(REPORT_OUT / "metrics.json", metrics_out)

        attempt_meta = {
            "candidate_version": CANDIDATE_VERSION,
            "batch": BATCH,
            "attempt": args.attempt,
            "created_at_local": now_local(),
            "source_sha": sha,
            "code_hashes": code_hashes,
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "n_fits_executed_this_run": int(n_fits_executed),
            "fold_meta": fold_meta_out,
            "thread_caps": thread_caps_record(),
            "elapsed_s": round(time.time() - t0, 2),
        }
        atomic_write_json(attempt_dir / "attempt_meta.json", attempt_meta)
        atomic_write_json(complete_marker, {
            "candidate_version": CANDIDATE_VERSION,
            "batch": BATCH,
            "attempt": args.attempt,
            "completed_at_local": now_local(),
            "n_fits_executed_this_run": int(n_fits_executed),
            "complete": True,
        })

        append_ledger({
            "ts_local": now_local(),
            "kind": "fit",
            "attempt": args.attempt,
            "attempt_dir": str(attempt_dir.resolve()),
            "status": "complete",
            "n_fits_executed_this_run": int(n_fits_executed),
            "fits": ["base/full x 5 folds"],
            "elapsed_s": round(time.time() - t0, 2),
        })

        print(json.dumps({
            "n_fits_executed_this_run": int(n_fits_executed),
            "pooled_delta": pooled_delta,
            "bootstrap_lo": lo,
            "bootstrap_hi": hi,
            "n_positive_folds": n_positive_folds,
            "criterion_met": criterion_met,
            "fold_spearman_deltas": fold_deltas,
            "elapsed_s": round(time.time() - t0, 2),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
