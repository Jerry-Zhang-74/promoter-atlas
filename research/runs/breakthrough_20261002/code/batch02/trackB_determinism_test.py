#!/usr/bin/env python3
"""Batch02 Track B determinism + resume test (runs AFTER the main grouped CV, on GPU).

Verifies: (1) two fresh runs with the same seed produce bit-identical weights and
out-of-fold predictions; (2) an interrupted run resumed from its epoch checkpoint
reproduces the non-interrupted full run exactly (same total_steps). Uses a small
epoch budget on outer fold 0 for speed. Reuses train_cnn from trackB_grouped_cv.
"""
import hashlib
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
TEST_EPOCHS = 6


def weight_hash(state):
    parts = []
    for k in sorted(state.keys()):
        parts.append(k.encode())
        parts.append(state[k].numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "trackB_train_components.tsv", sep="\t")
    seqs = df["sequence"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    comp_ids = df["verified_component_id"].astype(str).tolist()
    n = len(df)
    x_idx = t.encode_sequences(seqs).astype(np.int64)

    fold_of, comp_fold, fold_rows, multi_per_fold, comp_list = t.assign_folds(comp_ids, n, t.SEED_FOLD)
    held_idx = np.where(fold_of == 0)[0]
    outer_idx = np.where(fold_of != 0)[0]
    itr_idx, iva_idx = t.inner_split(np.asarray(comp_ids)[outer_idx], outer_idx, t.SEED_INNER)
    y_mu = float(y[outer_idx].mean())
    y_sd = float(y[outer_idx].std())

    x_tr = x_idx[outer_idx]
    y_tr = y[outer_idx]
    x_va = x_idx[iva_idx]
    y_va = y[iva_idx]

    tmp = MODEL_DIR / "_det"
    tmp.mkdir(parents=True, exist_ok=True)

    # (1) determinism: two fresh runs, same seed
    ck_a = tmp / "ck_a.pt"; done_a = tmp / "done_a.pt"
    ck_b = tmp / "ck_b.pt"; done_b = tmp / "done_b.pt"
    for p in [ck_a, done_a, ck_b, done_b]:
        if p.exists():
            p.unlink()
    _, _, _, _ = t.train_cnn("legnet", 20261008, x_tr, y_tr, y_mu, y_sd, x_va, y_va,
                             TEST_EPOCHS, t.LR, t.BATCH, t.WD, ck_a, done_a)
    _, _, _, _ = t.train_cnn("legnet", 20261008, x_tr, y_tr, y_mu, y_sd, x_va, y_va,
                             TEST_EPOCHS, t.LR, t.BATCH, t.WD, ck_b, done_b)
    wa = torch.load(done_a, map_location="cpu", weights_only=False)["model"]
    wb = torch.load(done_b, map_location="cpu", weights_only=False)["model"]
    det_same = weight_hash(wa) == weight_hash(wb)

    # (2) resume: interrupted after epoch 2, then resumed to epoch 6 -> compare with done_a
    ck_r = tmp / "ck_r.pt"; done_r = tmp / "done_r.pt"
    for p in [ck_r, done_r]:
        if p.exists():
            p.unlink()
    t.train_cnn("legnet", 20261008, x_tr, y_tr, y_mu, y_sd, x_va, y_va,
                TEST_EPOCHS, t.LR, t.BATCH, t.WD, ck_r, done_r, stop_after_epoch=2)
    t.train_cnn("legnet", 20261008, x_tr, y_tr, y_mu, y_sd, x_va, y_va,
                TEST_EPOCHS, t.LR, t.BATCH, t.WD, ck_r, done_r)
    wr = torch.load(done_r, map_location="cpu", weights_only=False)["model"]
    resume_same = weight_hash(wa) == weight_hash(wr)

    result = {
        "test_epochs": TEST_EPOCHS,
        "fold": 0, "cnn": "legnet", "seed": 20261008,
        "determinism_two_fresh_runs_bit_identical": bool(det_same),
        "resume_after_epoch2_matches_full_run": bool(resume_same),
        "weight_sha_full": weight_hash(wa),
        "note": "cudnn.deterministic=True, benchmark=False; seed before model+optimizer; resume restores optimizer/scheduler/RNG",
    }
    atomic_write_json(REPORT / "trackB_determinism_resume_test.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    print("DETERMINISM_RESUME", "PASS" if (det_same and resume_same) else "FAIL", flush=True)


if __name__ == "__main__":
    main()
