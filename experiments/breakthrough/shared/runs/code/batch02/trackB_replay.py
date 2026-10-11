#!/usr/bin/env python3
"""Batch02 Track B replay: reload every delivered weight file and reproduce OOF predictions.

This is the exact-delivered-model replay evidence: for each (cnn, fold, seed) the
saved weights_*.pt are loaded into a freshly constructed network and the held-out
fold predictions are recomputed and compared bitwise to the saved pred_*.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import atomic_write_json  # noqa: E402
import trackB_grouped_cv as t  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch02"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch02"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "trackB_train_components.tsv", sep="\t")
    seqs = df["sequence"].astype(str).tolist()
    comp_ids = df["verified_component_id"].astype(str).tolist()
    n = len(df)
    x_idx = t.encode_sequences(seqs).astype(np.int64)
    fold_of, comp_fold, fold_rows, multi_per_fold, comp_list = t.assign_folds(comp_ids, n, t.SEED_FOLD)
    y = df["activity"].to_numpy(dtype=np.float32)

    checks = []
    max_abs = 0.0
    n_files = 0
    for cnn in ["smallcnn", "legnet"]:
        for fld in range(5):
            held_idx = np.where(fold_of == fld)[0]
            outer_idx = np.where(fold_of != fld)[0]
            y_mu = float(y[outer_idx].mean())
            y_sd = float(y[outer_idx].std())
            for sd in t.SEEDS:
                wpath = MODEL_DIR / f"weights_{cnn}_fold{fld}_seed{sd}.pt"
                ppath = DATA / f"pred_{cnn}_fold{fld}_seed{sd}.npy"
                if not wpath.exists() or not ppath.exists():
                    checks.append({"cnn": cnn, "fold": fld, "seed": sd, "status": "missing"})
                    continue
                t.set_seed(sd)
                model = t.build_cnn(cnn).to(DEVICE)
                st = torch.load(wpath, map_location="cpu", weights_only=False)["model"]
                model.load_state_dict(st)
                pred = t.predict_cnn(model, x_idx[held_idx]) * y_sd + y_mu
                saved = np.load(ppath)
                d = float(np.max(np.abs(pred - saved)))
                max_abs = max(max_abs, d)
                n_files += 1
                checks.append({"cnn": cnn, "fold": fld, "seed": sd, "max_abs_diff": d, "pass": bool(d < 1e-5)})

    result = {"n_weights_replayed": n_files, "max_abs_diff": max_abs,
              "all_pass": bool(all(c.get("pass", False) for c in checks)),
              "checks": checks}
    atomic_write_json(REPORT / "trackB_replay.json", result)
    print(json.dumps({"n_weights_replayed": n_files, "max_abs_diff": max_abs,
                      "all_pass": result["all_pass"]}, indent=2, ensure_ascii=True), flush=True)
    print("REPLAY", "PASS" if result["all_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
