#!/usr/bin/env python3
"""promoter_atlas training v2-fusion: sequence + architecture/motif features for 3-head measured-label training.

Same pipeline as train_promoter_model_v2_distill.py, plus a feature branch that
concatenates promoter-level feature tables to the pooled sequence embedding.
Trains three measured heads (Score_E/Score_S/Score_B) with masked MSE loss.

Feature tables (keyed by record_id):
  - features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv (17 numeric)
  - features/motif/promoter_motif_reduced_v1.tsv (64 numeric)

Usage:
    .venv/Scripts/python.exe runs/scripts/train_promoter_model_v2_fusion.py \
        --name v2_fusion_arch --features arch --epochs 30
"""

import argparse
import json
import math
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
ARCH_FEATURES = ROOT / "features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv"
MOTIF_FEATURES = ROOT / "features/motif/promoter_motif_reduced_v1.tsv"

SEQ_LEN = 200
NEEDED_COLS = ["record_id", "gene_id_base", "Score_E", "Score_S", "Score_B"]
BASE_COLS = set(NEEDED_COLS + ["split", "row_index"])
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


def load_dataframe(features_mode: str):
    df = pd.read_csv(INPUT_TABLE, sep="\t", usecols=NEEDED_COLS, low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    df = df.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")
    if features_mode in ("arch", "both"):
        arch = pd.read_csv(ARCH_FEATURES, sep="\t", low_memory=False)
        arch = arch.drop(columns=[c for c in arch.columns if c in ("gene_id_base",)])
        df = df.merge(arch, on="record_id", how="left")
    if features_mode in ("motif", "both"):
        mot = pd.read_csv(MOTIF_FEATURES, sep="\t", low_memory=False)
        mot = mot.drop(columns=[c for c in mot.columns if c in ("gene_id_base",)])
        df = df.merge(mot, on="record_id", how="left")
    return df


def load_memmap():
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    return np.memmap(MEMMAP_PATH, dtype="uint8", mode="r", shape=(len(mm_index), SEQ_LEN))


def build_targets(df):
    n_t = 3
    y = np.zeros((len(df), n_t), dtype=np.float32)
    m = np.ones((len(df), n_t), dtype=np.float32)
    for j, col in enumerate(["Score_E", "Score_S", "Score_B"]):
        v = pd.to_numeric(df[col], errors="coerce")
        y[:, j] = v.fillna(0.0).to_numpy(dtype=np.float32)
        m[:, j] = (~v.isna()).to_numpy(dtype=np.float32)
    return y, m


def standardize_features(feat, stats=None):
    feat = np.nan_to_num(np.asarray(feat, dtype=np.float32), nan=0.0)
    if stats is None:
        mu = feat.mean(axis=0, keepdims=True)
        sd = feat.std(axis=0, keepdims=True) + 1e-6
        stats = (mu, sd)
    feat = (feat - stats[0]) / stats[1]
    return feat, stats


class ConvBaseline(nn.Module):
    """Exact reproduction of the v1 baseline architecture, with optional features."""

    def __init__(self, num_targets=3, base_ch=96, feat_dim=0):
        super().__init__()
        self.feat_dim = feat_dim
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
        self.head = nn.Sequential(
            nn.Linear(128 + feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_targets),
        )

    def forward(self, x_idx, feat=None):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.features(x).squeeze(-1)
        if feat is not None:
            x = torch.cat([x, feat], dim=1)
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
    def __init__(self, num_targets=3, base_ch=96, feat_dim=0):
        super().__init__()
        self.feat_dim = feat_dim
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
            nn.Linear(base_ch * 2 + feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_targets),
        )

    def forward(self, x_idx, feat=None):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.stem(x)
        x = self.blocks(x)
        avg = x.mean(dim=2)
        mx = x.max(dim=2).values
        pooled = torch.cat([avg, mx], dim=1)
        if feat is not None:
            pooled = torch.cat([pooled, feat], dim=1)
        return self.head(pooled)


def evaluate(model, loader, aux_stats, n_targets):
    model.eval()
    pred_all, y_all, m_all = [], [], []
    with torch.no_grad():
        for xb, fb, yb, mb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            fb = fb.to(DEVICE, non_blocking=True) if fb is not None else None
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)
            pred_all.append(model(xb, fb).cpu())
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
    ap.add_argument("--arch", default="conv_baseline", choices=["conv_baseline", "dilated_resnet"])
    ap.add_argument("--features", default="none", choices=["none", "arch", "motif", "both"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260315)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    n_targets = 3

    df = load_dataframe(args.features)
    mm = load_memmap()
    feat_cols = [c for c in df.columns if c not in BASE_COLS]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if args.quick:
        train_df = train_df.sample(frac=0.2, random_state=0).reset_index(drop=True)

    y_train, m_train = build_targets(train_df)
    y_val, m_val = build_targets(val_df)
    y_test, m_test = build_targets(test_df)

    aux_stats = None

    feat_dim = len(feat_cols)
    f_train, feat_stats = standardize_features(train_df[feat_cols].to_numpy()) if feat_dim else (None, None)
    f_val = standardize_features(val_df[feat_cols].to_numpy(), feat_stats)[0] if feat_dim else None
    f_test = standardize_features(test_df[feat_cols].to_numpy(), feat_stats)[0] if feat_dim else None

    def make_loader(df_, y_, m_, f_, shuffle):
        rows = df_["row_index"].to_numpy(dtype=np.int64)
        x = np.asarray(mm[rows], dtype=np.uint8)
        if f_ is not None:
            f_ = f_.astype(np.float32)
        ds = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(f_) if f_ is not None else torch.zeros(len(x), 0),
            torch.from_numpy(y_),
            torch.from_numpy(m_),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=0,
                          pin_memory=(DEVICE == "cuda"), drop_last=False)

    train_loader = make_loader(train_df, y_train, m_train, f_train, True)
    val_loader = make_loader(val_df, y_val, m_val, f_val, False)
    test_loader = make_loader(test_df, y_test, m_test, f_test, False)

    if args.arch == "conv_baseline":
        model = ConvBaseline(n_targets, feat_dim=feat_dim).to(DEVICE)
    else:
        model = DilatedResNet(n_targets, feat_dim=feat_dim).to(DEVICE)
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
        log_line(fh, "features\t" + args.features)
        log_line(fh, "feat_dim\t%d" % feat_dim)
        log_line(fh, "device\t" + DEVICE)
        log_line(fh, "params\t%d" % n_params)
        log_line(fh, "train_rows\t%d" % len(train_df))

        for epoch in range(args.epochs):
            model.train()
            total_loss, nb = 0.0, 0
            for xb, fb, yb, mb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                fb = fb.to(DEVICE, non_blocking=True) if fb.numel() else None
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(xb, fb)
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
    torch.save({"state_dict": best_ckpt, "n_targets": n_targets, "feat_dim": feat_dim,
                "feat_cols": feat_cols, "feat_stats": feat_stats, "aux_stats": aux_stats,
                "name": args.name}, ckpt_path)
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
        fh.write("name\t%s\nfeatures\t%s\nfeat_dim\t%d\ndevice\t%s\nparams\t%d\n"
                 % (args.name, args.features, feat_dim, DEVICE, n_params))
        fh.write("test_metrics\t%s\n" % json.dumps({k: v for k, v in test_metrics.items()}))
    print("saved", ckpt_path, "|", eval_path, "|", card_path)


if __name__ == "__main__":
    main()
