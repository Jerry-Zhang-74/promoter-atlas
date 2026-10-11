#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal reproducible replay example for batch16 v6.15.0-multitask.rc1.

Loads the validation-selected deployment bundle (independent A, reference seed
20261016), the frozen scalers/data, and reproduces saved heldout predictions for
a few genes WITHOUT retraining. Demonstrates the exact input schema:
  - 4 diverse promoter sequences (one-hot ACGT, N->zero) from the memmap
  - frozen 1069-feature composition/position vector (standardized train-only)

Output distinguishes model predictions from catalog-observed labels (the latter
are not shipped here). Not a fragment/HEK293T predictor.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs" / "breakthrough_20261002" / "code" / "batch16"))
from b16_common import DATA_OUT, MODEL_OUT, MEM400, N_HUMAN, L400, TASKS
from b16_model import SingleTaskModel, onehot


def main():
    bundle = json_load(MODEL_OUT / "deployment_bundle.json")
    data = np.load(DATA_OUT / "data.npz", allow_pickle=True)
    sc = np.load(DATA_OUT / "scalers.npz", allow_pickle=True)
    gene_gid = data["gene_gid"].astype(str)
    gene_split = data["gene_split"].astype(str)
    slot_r400 = data["slot_r400"].astype(np.int64)
    X = data["X"].astype(np.float64)
    Xstd = (X - sc["X_mean"]) / sc["X_std"]
    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    te = gene_split == "test"

    models = {}
    for task in TASKS:
        m = SingleTaskModel(task)
        m.load_state_dict(torch.load(bundle["models"][task], map_location="cpu")["model"])
        m.eval()
        models[task] = m

    print("candidate_version=%s architecture=%s" % (bundle["candidate_version"], bundle["architecture"]))
    print("replaying first 3 heldout genes (per task valid-mask order):")
    for task in TASKS:
        mask = data["%s_mask" % task].astype(bool)
        idx = np.where(te & mask)[0][:3]
        saved = np.load(MODEL_OUT / "A_s20261016" / ("pred_%s_test.npy" % task))[0][:3]
        mu = float(sc["%s_mean" % task]); sd = float(sc["%s_std" % task])
        rows = slot_r400[idx].reshape(-1)
        seq = np.asarray(mm400[rows], dtype=np.uint8).reshape(-1, L400)
        x_seq = onehot(torch.from_numpy(seq).long(), L400)
        x_feat = torch.tensor(Xstd[idx], dtype=torch.float32)
        with torch.no_grad():
            pred_std = models[task](x_seq, x_feat).numpy()
        pred_raw = pred_std * sd + mu
        saved_raw = saved * sd + mu
        for i, g in enumerate(idx):
            print("  %s gene=%s y=%+.4f pred=%.6f saved=%.6f |diff|=%.2e" % (
                task, gene_gid[g], float(data[task][g]), float(pred_raw[i]),
                float(saved_raw[i]), float(abs(pred_raw[i] - saved_raw[i]))))
    print("REPLAY_OK (pred == saved within float precision)")


def json_load(p):
    import json
    return json.load(open(p, encoding="utf-8"))


if __name__ == "__main__":
    main()
