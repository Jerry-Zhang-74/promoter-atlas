#!/usr/bin/env python3
"""Percentile-rank calibration analysis for v2_long_both.

The long-context model showed excellent spearman but collapsed r2 (compressed
prediction variance). This script re-evaluates v2_long_both on the test set and
applies percentile-rank calibration (map pred ranks to train label ranks) to
quantify how much r2 can be recovered by calibration alone.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models/checkpoints/v2_long_both.pt"
INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"
MM_PATH = ROOT / "promoters/memmap/promoter_up2000_uint8.memmap"
MM_INDEX = ROOT / "promoters/memmap/promoter_up2000_index.tsv"
ARCH_FEATURES = ROOT / "features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv"
MOTIF_FEATURES = ROOT / "features/motif/promoter_motif_reduced_v1.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def r2(a, b):
    return 1.0 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lc", ROOT / "runs/scripts/train_promoter_model_v2_longcontext.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = mod.load_dataframe("both")
    mm = mod.load_memmap()
    feat_cols = [c for c in df.columns if c not in mod.BASE_COLS]
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    y_tr, m_tr = mod.build_targets(train_df)
    y_te, m_te = mod.build_targets(test_df)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    f_tr, st = mod.standardize_features(train_df[feat_cols].to_numpy())
    f_te = mod.standardize_features(test_df[feat_cols].to_numpy(), st)[0]

    model = mod.LongContextNet(3, feat_dim=len(feat_cols), norm="bn").to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def predict(df_, f_):
        rows = df_["row_index"].to_numpy(dtype=np.int64)
        x = np.asarray(mm[rows], dtype=np.uint8)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                xb = torch.from_numpy(x[i:i + 256]).to(DEVICE)
                fb = torch.from_numpy(f_[i:i + 256]).to(DEVICE)
                preds.append(model(xb, fb).cpu().numpy())
        return np.concatenate(preds, axis=0)

    pred_tr = predict(train_df, f_tr)
    pred_te = predict(test_df, f_te)

    names = ["Score_E", "Score_S", "Score_B"]
    print("target | raw r2 | raw sp | calibrated r2 | calibrated sp")
    for j, name in enumerate(names):
        mask_tr = m_tr[:, j] > 0
        mask_te = m_te[:, j] > 0
        ytr = y_tr[mask_tr, j]
        yte = y_te[mask_te, j]
        ptr = pred_tr[mask_tr, j]
        pte = pred_te[mask_te, j]
        # percentile-rank calibration: for each test pred, find its quantile in
        # train preds, map to the same quantile of train labels
        q = np.interp(pte, np.sort(ptr), np.linspace(0, 1, len(ptr)))
        # map quantile to label quantile (interpolate on sorted labels)
        sorted_y = np.sort(ytr)
        calibrated = np.interp(q, np.linspace(0, 1, len(sorted_y)), sorted_y)
        print("%s | %.4f | %.4f | %.4f | %.4f"
              % (name, r2(pte, yte), spearman(pte, yte), r2(calibrated, yte), spearman(calibrated, yte)))


if __name__ == "__main__":
    main()
