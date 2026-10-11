#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 matched HGB baselines (E45, S1069, B45), fit from scratch on corrected train.

Three fixed HistGradientBoostingRegressor fits (no tuning), separately counted
from the <=10 neural fits. Same corrected outer train, same per-target validity
mask and evaluation rows as the neural arms. Reported in original target units.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT, CANDIDATE_VERSION,
                        HGB, TASKS, set_thread_limits, spearman, r2_score, mae,
                        mse, atomic_write_json)

set_thread_limits(4)

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
import joblib  # noqa: E402


BASELINES = {
    "E45": {"task": "E", "feat": "45"},
    "S1069": {"task": "S", "feat": "1069"},
    "B45": {"task": "B", "feat": "45"},
}


def main():
    t0 = time.time()
    for d in (MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_OUT / "data.npz", allow_pickle=True)
    X = data["X"].astype(np.float64)
    gene_split = data["gene_split"].astype(str)
    Y = {"E": data["E"].astype(np.float64), "S": data["S"].astype(np.float64),
         "B": data["B"].astype(np.float64)}
    M = {"E": data["E_mask"].astype(bool), "S": data["S_mask"].astype(bool),
         "B": data["B_mask"].astype(bool)}

    tr = gene_split == "train"
    va = gene_split == "val"
    te = gene_split == "test"

    results = {}
    for key, spec in BASELINES.items():
        task = spec["task"]
        feat = X[:, :45] if spec["feat"] == "45" else X
        y = Y[task]
        m = M[task]
        model = HistGradientBoostingRegressor(**HGB)
        model.fit(feat[tr & m], y[tr & m])
        out = {}
        for split_name, split_mask in (("train", tr), ("val", va), ("test", te)):
            idx = split_mask & m
            pred = model.predict(feat[idx])
            out[split_name] = {
                "n": int(idx.sum()),
                "spearman": spearman(pred, y[idx]),
                "r2": r2_score(y[idx], pred),
                "mae": mae(y[idx], pred),
                "mse": mse(y[idx], pred),
            }
        results[key] = {"task": task, "features": spec["feat"], **out}
        joblib.dump(model, MODEL_OUT / ("hgb_%s.joblib" % key))

    # ---- HGB predictions TSV (val+test) for replay/package ----
    pred_rows = []
    for key, spec in BASELINES.items():
        task = spec["task"]
        feat = X[:, :45] if spec["feat"] == "45" else X
        model = joblib.load(MODEL_OUT / ("hgb_%s.joblib" % key))
        m = M[task]
        y = Y[task]
        gene_gid = data["gene_gid"].astype(str)
        for split_name, split_mask in (("val", va), ("test", te)):
            idx = np.where(split_mask & m)[0]
            pred = model.predict(feat[idx])
            for i, gi in enumerate(idx):
                pred_rows.append({
                    "baseline": key, "task": task, "split": split_name,
                    "gene_id": gene_gid[gi], "y": float(y[gi]), "yhat": float(pred[i]),
                })
    import pandas as pd
    pd.DataFrame(pred_rows).to_csv(MODEL_OUT / "hgb_predictions.tsv", sep="\t", index=False)

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": "batch16",
        "kind": "matched HGB baselines (from scratch on corrected train)",
        "hgb_config": HGB,
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
    }
    atomic_write_json(MODEL_OUT / "hgb_results.json", summary)

    print(json.dumps({"results": results, "elapsed_s": round(time.time() - t0, 2)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
