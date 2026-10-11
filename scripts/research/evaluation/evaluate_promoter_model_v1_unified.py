"""
Unified evaluation for promoter_model_v1 in promoter_atlas.

Purpose
-------
Re-evaluate the existing v1 sequence baseline using the same metric schema
used by later pilot models, especially adding E-S correlation diagnostics.

Inputs
------
- labels/build/promoter_model_input_core.tsv
- labels/releases/promoter_split_core.tsv
- promoters/memmap/promoter_up200_uint8.memmap
- promoters/memmap/promoter_up200_index.tsv
- models/checkpoints/promoter_model_v1.pt

Outputs
-------
- models/eval/promoter_model_v1_metrics_unified.tsv
- models/cards/promoter_model_v1_unified_eval_card.txt
"""

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


ROOT = Path(".")

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"
MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"
CHECKPOINT_PATH = ROOT / "models/checkpoints/promoter_model_v1.pt"

EVAL_PATH = ROOT / "models/eval/promoter_model_v1_metrics_unified.tsv"
CARD_PATH = ROOT / "models/cards/promoter_model_v1_unified_eval_card.txt"

SEQ_LEN = 200
BATCH_SIZE = 2048
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=20260315):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score_torch(pred, target):
    target_mean = target.mean()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target_mean) ** 2).sum()
    if ss_tot.item() == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rankdata_average(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_np(x, y):
    if len(x) < 2:
        return float("nan")
    rx = rankdata_average(np.asarray(x))
    ry = rankdata_average(np.asarray(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def corr_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def load_main_dataframe():
    core = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)

    req_core = ["record_id", "gene_id_base", "Score_E", "Score_S", "Score_B"]
    req_split = ["record_id", "split"]
    req_mm = ["record_id", "row_index"]

    miss_core = [c for c in req_core if c not in core.columns]
    miss_split = [c for c in req_split if c not in split.columns]
    miss_mm = [c for c in req_mm if c not in mm_index.columns]

    if miss_core:
        raise KeyError(f"Missing core columns: {miss_core}")
    if miss_split:
        raise KeyError(f"Missing split columns: {miss_split}")
    if miss_mm:
        raise KeyError(f"Missing memmap index columns: {miss_mm}")

    df = core.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")

    if df["record_id"].duplicated().any():
        dup_n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    return df.sort_values(["split", "record_id"]).reset_index(drop=True)


def build_targets(df):
    y = np.zeros((len(df), 3), dtype=np.float32)
    m = np.ones((len(df), 3), dtype=np.float32)

    y[:, 0] = pd.to_numeric(df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    y[:, 1] = pd.to_numeric(df["Score_S"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y[:, 2] = pd.to_numeric(df["Score_B"], errors="coerce").to_numpy(dtype=np.float32)

    s_missing = pd.to_numeric(df["Score_S"], errors="coerce").isna().to_numpy()
    m[s_missing, 1] = 0.0

    return y, m


class SeqBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.backbone = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.head_e = nn.Linear(128, 1)
        self.head_s = nn.Linear(128, 1)
        self.head_b = nn.Linear(128, 1)

    def forward(self, seq_idx):
        seq = F.one_hot(seq_idx.long(), num_classes=5)[..., :4].float()
        seq = seq.permute(0, 2, 1).contiguous()
        seq = self.features(seq).squeeze(-1)
        seq = self.backbone(seq)
        e = self.head_e(seq)
        s = self.head_s(seq)
        b = self.head_b(seq)
        return torch.cat([e, s, b], dim=1)


def evaluate(model, loader, split_name):
    model.eval()

    pred_all = []
    y_all = []
    m_all = []

    with torch.no_grad():
        for seqb, yb, mb in loader:
            seqb = seqb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            pred = model(seqb)

            pred_all.append(pred.cpu())
            y_all.append(yb.cpu())
            m_all.append(mb.cpu())

    pred_all = torch.cat(pred_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    m_all = torch.cat(m_all, dim=0)

    metrics = {"split": split_name}
    target_names = ["Score_E", "Score_S", "Score_B"]

    for i, name in enumerate(target_names):
        mask = m_all[:, i] > 0
        yp = pred_all[mask, i]
        yt = y_all[mask, i]

        if len(yt) == 0:
            metrics[f"{name}_mse"] = float("nan")
            metrics[f"{name}_r2"] = float("nan")
            metrics[f"{name}_spearman"] = float("nan")
            continue

        mse = float(torch.mean((yp - yt) ** 2))
        r2 = r2_score_torch(yp, yt)
        sp = spearman_np(yp.numpy(), yt.numpy())

        metrics[f"{name}_mse"] = mse
        metrics[f"{name}_r2"] = r2
        metrics[f"{name}_spearman"] = sp

    pred_e = pred_all[:, 0].numpy()
    pred_s = pred_all[:, 1].numpy()

    s_mask = m_all[:, 1].numpy() > 0
    true_s = y_all[:, 1].numpy()[s_mask]
    true_e_masked = y_all[:, 0].numpy()[s_mask]
    pred_s_masked = pred_s[s_mask]
    pred_e_masked = pred_e[s_mask]

    metrics["pred_E_pred_S_corr_all"] = corr_np(pred_e, pred_s)
    metrics["true_E_true_S_corr_defined"] = corr_np(true_e_masked, true_s)
    metrics["pred_E_pred_S_corr_defined"] = corr_np(pred_e_masked, pred_s_masked)

    return metrics


def main():
    set_seed()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    df = load_main_dataframe()

    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    n_rows = len(mm_index)
    mm = np.memmap(
        MEMMAP_PATH,
        dtype="uint8",
        mode="r",
        shape=(n_rows, SEQ_LEN),
    )

    split_rows = []
    for split_name in ["val", "test"]:
        sub = df.loc[df["split"] == split_name].copy().reset_index(drop=True)
        row_idx = sub["row_index"].to_numpy(dtype=np.int64)
        seq_x = np.asarray(mm[row_idx], dtype=np.uint8)
        y, m = build_targets(sub)

        ds = TensorDataset(
            torch.from_numpy(seq_x),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )

        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
            drop_last=False,
        )

        model = SeqBaselineModel().to(DEVICE)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        metrics = evaluate(model, loader, split_name)
        split_rows.append(metrics)

    out_df = pd.DataFrame(split_rows)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(EVAL_PATH, sep="\t", index=False)

    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as handle:
        handle.write("model_name\tpromoter_model_v1\n")
        handle.write("evaluation_type\tunified_re_evaluation\n")
        handle.write("checkpoint\tmodels/checkpoints/promoter_model_v1.pt\n")
        handle.write("input\tsequence_memmap_only\n")
        handle.write("eval_table\tmodels/eval/promoter_model_v1_metrics_unified.tsv\n")
        handle.write("note\tAdded unified E-S correlation diagnostics for registry consistency.\n")

    print("v1 unified evaluation finished")
    print(f"eval: {EVAL_PATH}")
    print(f"card: {CARD_PATH}")


if __name__ == "__main__":
    main()
