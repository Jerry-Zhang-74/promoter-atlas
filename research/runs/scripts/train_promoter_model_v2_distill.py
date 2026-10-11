#!/usr/bin/env python3
"""promoter_atlas training v2: improved architectures for 3-head measured-label training.

Reproduces the v1 data pipeline (memmap 200 bp sequences + masked Score_E/S/B)
and adds a stronger dilated-residual CNN architecture. Trains three measured
heads (Score_E/Score_S/Score_B) with masked MSE loss.

The split is gene-level (labels/releases/promoter_split_core.tsv).

Usage:
    .venv/Scripts/python.exe runs/scripts/train_promoter_model_v2_distill.py \
        --name v2_dilated_base --arch dilated_resnet --quick
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"
MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"

SEQ_LEN = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def r2_np(pred, target):
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def load_dataframe():
    df = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    df = df.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")
    return df


def load_memmap():
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    mm = np.memmap(MEMMAP_PATH, dtype="uint8", mode="r", shape=(len(mm_index), SEQ_LEN))
    return mm


def build_targets(df):
    """Return (y, mask). y[:, 0:3] = Score_E/S/B (raw [0,1])."""
    n_t = 3
    y = np.zeros((len(df), n_t), dtype=np.float32)
    m = np.ones((len(df), n_t), dtype=np.float32)
    y[:, 0] = pd.to_numeric(df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    y[:, 1] = pd.to_numeric(df["Score_S"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y[:, 2] = pd.to_numeric(df["Score_B"], errors="coerce").to_numpy(dtype=np.float32)
    m[pd.to_numeric(df["Score_E"], errors="coerce").isna().to_numpy(), 0] = 0.0
    m[pd.to_numeric(df["Score_S"], errors="coerce").isna().to_numpy(), 1] = 0.0
    m[pd.to_numeric(df["Score_B"], errors="coerce").isna().to_numpy(), 2] = 0.0
    return y, m


def unzscore_cols(y, cols, stats):
    out = y.copy()
    for j, (mu, sd) in zip(cols, stats):
        out[:, j] = out[:, j] * sd + mu
    return out


class ConvBaseline(nn.Module):
    """Exact reproduction of the v1 baseline architecture."""

    def __init__(self, num_targets=3):
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
        self.head = nn.Linear(128, num_targets)

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.features(x).squeeze(-1)
        x = self.backbone(x)
        return self.head(x)


class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, k=7):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class DilatedResNet(nn.Module):
    """Dilated residual CNN: receptive field ~187 bp over the 200 bp promoter."""

    def __init__(self, num_targets=3, base_ch=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, base_ch, 7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DilatedBlock(base_ch, 1),
            DilatedBlock(base_ch, 2),
            DilatedBlock(base_ch, 4),
            DilatedBlock(base_ch, 8),
            DilatedBlock(base_ch, 16),
        )
        self.head = nn.Sequential(
            nn.Linear(base_ch * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_targets),
        )

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.stem(x)
        x = self.blocks(x)
        avg = x.mean(dim=2)
        mx = x.max(dim=2).values
        return self.head(torch.cat([avg, mx], dim=1))


def make_model(arch, num_targets):
    if arch == "conv_baseline":
        return ConvBaseline(num_targets)
    if arch == "dilated_resnet":
        return DilatedResNet(num_targets)
    raise ValueError(arch)


def evaluate(model, loader, aux_stats, n_targets):
    model.eval()
    pred_all, y_all, m_all = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)
            pred_all.append(model(xb).cpu())
            y_all.append(yb.cpu())
            m_all.append(mb.cpu())
    pred = torch.cat(pred_all, dim=0).numpy()
    y = torch.cat(y_all, dim=0).numpy()
    m = torch.cat(m_all, dim=0).numpy()
    names = ["Score_E", "Score_S", "Score_B"]
    metrics = {}
    for j, name in enumerate(names):
        mask = m[:, j] > 0
        yp, yt = pred[mask, j], y[mask, j]
        if len(yt) == 0:
            metrics[name + "_r2"] = float("nan")
            metrics[name + "_spearman"] = float("nan")
            metrics[name + "_n"] = 0
            continue
        metrics[name + "_r2"] = r2_np(yp, yt)
        metrics[name + "_spearman"] = spearman_np(yp, yt)
        metrics[name + "_n"] = int(mask.sum())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="dilated_resnet", choices=["conv_baseline", "dilated_resnet"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260315)
    ap.add_argument("--quick", action="store_true", help="use 20% of train rows for a smoke run")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop early after N minutes")
    args = ap.parse_args()

    set_seed(args.seed)
    n_targets = 3

    df = load_dataframe()
    mm = load_memmap()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if args.quick:
        train_df = train_df.sample(frac=0.2, random_state=0).reset_index(drop=True)

    y_train, m_train = build_targets(train_df)
    y_val, m_val = build_targets(val_df)
    y_test, m_test = build_targets(test_df)

    aux_stats = None

    def make_loader(df_, y_, m_, shuffle):
        rows = df_["row_index"].to_numpy(dtype=np.int64)
        x = np.asarray(mm[rows], dtype=np.uint8)
        ds = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y_), torch.from_numpy(m_)
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=0,
                          pin_memory=(DEVICE == "cuda"), drop_last=False)

    train_loader = make_loader(train_df, y_train, m_train, True)
    val_loader = make_loader(val_df, y_val, m_val, False)
    test_loader = make_loader(test_df, y_test, m_test, False)

    model = make_model(args.arch, n_targets).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.08
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    ckpt_dir = ROOT / "models/checkpoints"
    log_dir = ROOT / "models/logs"
    eval_dir = ROOT / "models/eval"
    card_dir = ROOT / "models/cards"
    for d in (ckpt_dir, log_dir, eval_dir, card_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / ("train_" + args.name + ".log")
    ckpt_path = ckpt_dir / (args.name + ".pt")
    eval_path = eval_dir / (args.name + "_metrics.tsv")
    card_path = card_dir / (args.name + "_card.txt")

    def log_line(fh, text):
        print(text, flush=True)
        fh.write(text + "\n")
        fh.flush()

    best_val = -np.inf
    best_ckpt = None
    t_start = time.time()

    with open(log_path, "w", encoding="utf-8") as fh:
        log_line(fh, "name\t" + args.name)
        log_line(fh, "arch\t" + args.arch)
        log_line(fh, "device\t" + DEVICE)
        log_line(fh, "params\t%d" % n_params)
        log_line(fh, "train_rows\t%d" % len(train_df))
        log_line(fh, "val_rows\t%d" % len(val_df))
        log_line(fh, "test_rows\t%d" % len(test_df))

        for epoch in range(args.epochs):
            model.train()
            total_loss, nb = 0.0, 0
            for xb, yb, mb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(xb)
                    loss = torch.zeros((), device=DEVICE)
                    for j in range(3):
                        diff = (pred[:, j] - yb[:, j]) ** 2 * mb[:, j]
                        loss = loss + diff.sum() / mb[:, j].sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach())
                nb += 1
            val_metrics = evaluate(model, val_loader, aux_stats, n_targets)
            combined = (
                val_metrics["Score_E_spearman"] * 0.4
                + val_metrics["Score_S_spearman"] * 0.35
                + val_metrics["Score_B_spearman"] * 0.25
            )
            if np.isfinite(combined) and combined > best_val:
                best_val = combined
                best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log_line(fh, "epoch %d | loss %.4f | val E r2=%.4f sp=%.4f S r2=%.4f sp=%.4f B r2=%.4f sp=%.4f | combined %.4f"
                     % (epoch, total_loss / max(nb, 1),
                        val_metrics["Score_E_r2"], val_metrics["Score_E_spearman"],
                        val_metrics["Score_S_r2"], val_metrics["Score_S_spearman"],
                        val_metrics["Score_B_r2"], val_metrics["Score_B_spearman"], combined))
            if args.max_minutes is not None and (time.time() - t_start) > args.max_minutes * 60:
                log_line(fh, "reached time budget, stopping")
                break

    if best_ckpt is None:
        best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": best_ckpt, "arch": args.arch, "n_targets": n_targets,
                "aux_stats": aux_stats, "name": args.name}, ckpt_path)
    model.load_state_dict(best_ckpt)
    test_metrics = evaluate(model, test_loader, aux_stats, n_targets)
    test_metrics["combined_spearman"] = (
        test_metrics["Score_E_spearman"] * 0.4
        + test_metrics["Score_S_spearman"] * 0.35
        + test_metrics["Score_B_spearman"] * 0.25
    )
    pd.DataFrame([test_metrics]).to_csv(eval_path, sep="\t", index=False)
    print("\n=== TEST metrics ===")
    for k, v in test_metrics.items():
        print("  %s = %s" % (k, ("%.4f" % v) if isinstance(v, float) else str(v)))

    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write("name\t%s\narch\t%s\ndevice\t%s\nparams\t%d\n"
                 % (args.name, args.arch, DEVICE, n_params))
        fh.write("test_metrics\t%s\n" % json.dumps({k: v for k, v in test_metrics.items()}))
        fh.write("checkpoint\t%s\n" % ckpt_path)
    print("saved", ckpt_path, "|", eval_path, "|", card_path)


if __name__ == "__main__":
    main()
