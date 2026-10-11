#!/usr/bin/env python3
"""Dual-stream promoter model: 200 bp local stem + 2000 bp context stem.

Hypothesis: the 200 bp core carries local TF-motif signal while the 2000 bp
flank carries context (GC domains, CpG islands, distal elements) that helps
Score_S. Fuses both streams into one head.

Usage:
    .venv/Scripts/python.exe runs/scripts/train_promoter_model_v2_dual.py \
        --name v2_dual_base --epochs 25 --batch 1024
"""

import argparse
import json
import math
import random
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
MM200_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MM200_INDEX = ROOT / "promoters/memmap/promoter_up200_index.tsv"
MM2000_PATH = ROOT / "promoters/memmap/promoter_up2000_uint8.memmap"
MM2000_INDEX = ROOT / "promoters/memmap/promoter_up2000_index.tsv"
ARCH_FEATURES = ROOT / "features/sequence/promoter_architecture_features_v4_arch2_reduced.tsv"
MOTIF_FEATURES = ROOT / "features/motif/promoter_motif_reduced_v1.tsv"

SEQ200, SEQ2000 = 200, 2000
NEEDED_COLS = ["record_id", "gene_id_base", "Score_E", "Score_S", "Score_B"]
BASE_COLS = set(NEEDED_COLS + ["split"])
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


def load_dataframe(features_mode):
    df = pd.read_csv(INPUT_TABLE, sep="\t", usecols=NEEDED_COLS, low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    i200 = pd.read_csv(MM200_INDEX, sep="\t", low_memory=False)
    i2000 = pd.read_csv(MM2000_INDEX, sep="\t", low_memory=False)
    df = df.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(i200[["record_id", "row_index"]].rename(columns={"row_index": "r200"}), on="record_id", how="inner")
    df = df.merge(i2000[["record_id", "row_index"]].rename(columns={"row_index": "r2000"}), on="record_id", how="inner")
    if features_mode in ("arch", "both"):
        arch = pd.read_csv(ARCH_FEATURES, sep="\t", low_memory=False).drop(columns=["gene_id_base"], errors="ignore")
        df = df.merge(arch, on="record_id", how="left")
    if features_mode in ("motif", "both"):
        mot = pd.read_csv(MOTIF_FEATURES, sep="\t", low_memory=False).drop(columns=["gene_id_base"], errors="ignore")
        df = df.merge(mot, on="record_id", how="left")
    return df


def build_targets(df):
    y = np.zeros((len(df), 3), dtype=np.float32)
    m = np.ones((len(df), 3), dtype=np.float32)
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
    return (feat - stats[0]) / stats[1], stats


class LocalStem(nn.Module):
    """200 bp local motif stem (ConvBaseline-style)."""

    def __init__(self, out_ch=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.features(x).squeeze(-1)


def make_norm(norm, ch):
    if norm == "bn":
        return nn.BatchNorm1d(ch)
    if norm == "ln":
        return nn.GroupNorm(8, ch)
    return nn.Identity()


class ContextDilatedBlock(nn.Module):
    def __init__(self, ch, dilation, k=5, norm="bn"):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.n1 = make_norm(norm, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n2 = make_norm(norm, ch)

    def forward(self, x):
        r = x
        x = F.relu(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return F.relu(x + r)


class ContextStem(nn.Module):
    """2000 bp context stem: stride-2 convs + dilated blocks -> pooled."""

    def __init__(self, base_ch=64, out_ch=128, norm="bn"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, base_ch, 15, stride=2, padding=7), make_norm(norm, base_ch), nn.ReLU(inplace=True),
            nn.Conv1d(base_ch, base_ch, 15, stride=2, padding=7), make_norm(norm, base_ch), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ContextDilatedBlock(base_ch, 1, norm=norm),
            ContextDilatedBlock(base_ch, 2, norm=norm),
            ContextDilatedBlock(base_ch, 4, norm=norm),
            ContextDilatedBlock(base_ch, 8, norm=norm),
            ContextDilatedBlock(base_ch, 16, norm=norm),
            ContextDilatedBlock(base_ch, 32, norm=norm),
            ContextDilatedBlock(base_ch, 64, norm=norm),
        )
        self.proj = nn.Conv1d(base_ch, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        avg = x.mean(dim=2)
        mx = x.max(dim=2).values
        return torch.cat([avg, mx], dim=1)  # (B, out_ch*2)


class DualStreamNet(nn.Module):
    def __init__(self, num_targets=3, feat_dim=0, local_ch=128, ctx_ch=96, norm="bn"):
        super().__init__()
        self.local = LocalStem(out_ch=local_ch)
        self.context = ContextStem(base_ch=64, out_ch=ctx_ch, norm=norm)
        dim = local_ch + ctx_ch * 2 + feat_dim
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, num_targets),
        )

    def forward(self, x200, x2000, feat=None):
        x200 = F.one_hot(x200.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        x2000 = F.one_hot(x2000.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        a = self.local(x200)
        b = self.context(x2000)
        z = torch.cat([a, b], dim=1)
        if feat is not None:
            z = torch.cat([z, feat], dim=1)
        return self.head(z)


def evaluate(model, loader, s_stats=None):
    model.eval()
    pred_all, y_all, m_all = [], [], []
    with torch.no_grad():
        for x2, x20, fb, yb, mb in loader:
            x2 = x2.to(DEVICE, non_blocking=True)
            x20 = x20.to(DEVICE, non_blocking=True)
            fb = fb.to(DEVICE, non_blocking=True) if fb.numel() else None
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)
            pred_all.append(model(x2, x20, fb).cpu())
            y_all.append(yb.cpu())
            m_all.append(mb.cpu())
    pred = torch.cat(pred_all, dim=0).numpy()
    y = torch.cat(y_all, dim=0).numpy()
    m = torch.cat(m_all, dim=0).numpy()
    metrics = {}
    for j, name in enumerate(["Score_E", "Score_S", "Score_B"]):
        mask = m[:, j] > 0
        yp, yt = pred[mask, j], y[mask, j]
        if j == 1 and s_stats is not None:
            yp = yp * s_stats[1] + s_stats[0]
            yt = yt * s_stats[1] + s_stats[0]
        metrics[name + "_r2"] = r2_np(yp, yt) if len(yt) else float("nan")
        metrics[name + "_spearman"] = spearman_np(yp, yt) if len(yt) else float("nan")
        metrics[name + "_n"] = int(mask.sum())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="none", choices=["none", "arch", "motif", "both"])
    ap.add_argument("--norm", default="bn", choices=["bn", "ln", "none"])
    ap.add_argument("--zscore-s", action="store_true")
    ap.add_argument("--s-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260315)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    df = load_dataframe(args.features)
    mm200 = np.memmap(MM200_PATH, dtype="uint8", mode="r", shape=(len(pd.read_csv(MM200_INDEX, sep="\t")), SEQ200))
    mm2000 = np.memmap(MM2000_PATH, dtype="uint8", mode="r", shape=(len(pd.read_csv(MM2000_INDEX, sep="\t")), SEQ2000))
    feat_cols = [c for c in df.columns if c not in BASE_COLS and c not in ("r200", "r2000")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if args.quick:
        train_df = train_df.sample(frac=0.2, random_state=0).reset_index(drop=True)

    y_train, m_train = build_targets(train_df)
    y_val, m_val = build_targets(val_df)
    y_test, m_test = build_targets(test_df)

    s_stats = None
    if args.zscore_s:
        v = y_train[m_train[:, 1] > 0, 1]
        s_stats = (float(v.mean()), float(v.std()) + 1e-6)
        for y_, m_ in ((y_train, m_train), (y_val, m_val), (y_test, m_test)):
            y_[:, 1] = (y_[:, 1] - s_stats[0]) / s_stats[1] * m_[:, 1]

    feat_dim = len(feat_cols)
    f_train, feat_stats = standardize_features(train_df[feat_cols].to_numpy()) if feat_dim else (None, None)
    f_val = standardize_features(val_df[feat_cols].to_numpy(), feat_stats)[0] if feat_dim else None
    f_test = standardize_features(test_df[feat_cols].to_numpy(), feat_stats)[0] if feat_dim else None

    def make_loader(df_, y_, m_, f_, shuffle):
        r2 = df_["r200"].to_numpy(dtype=np.int64)
        r20 = df_["r2000"].to_numpy(dtype=np.int64)
        x2 = np.asarray(mm200[r2], dtype=np.uint8)
        x20 = np.asarray(mm2000[r20], dtype=np.uint8)
        ds = TensorDataset(
            torch.from_numpy(x2), torch.from_numpy(x20),
            torch.from_numpy(f_.astype(np.float32)) if f_ is not None else torch.zeros(len(x2), 0),
            torch.from_numpy(y_), torch.from_numpy(m_),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=0,
                          pin_memory=(DEVICE == "cuda"), drop_last=False)

    train_loader = make_loader(train_df, y_train, m_train, f_train, True)
    val_loader = make_loader(val_df, y_val, m_val, f_val, False)
    test_loader = make_loader(test_df, y_test, m_test, f_test, False)

    model = DualStreamNet(3, feat_dim=feat_dim, norm=args.norm).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch, pct_start=0.08)
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
        log_line(fh, "arch\tdual_stream_200_2000")
        log_line(fh, "features\t" + args.features)
        log_line(fh, "feat_dim\t%d" % feat_dim)
        log_line(fh, "device\t" + DEVICE)
        log_line(fh, "params\t%d" % n_params)
        log_line(fh, "train_rows\t%d" % len(train_df))

        for epoch in range(args.epochs):
            model.train()
            total_loss, nb = 0.0, 0
            for x2, x20, fb, yb, mb in train_loader:
                x2 = x2.to(DEVICE, non_blocking=True)
                x20 = x20.to(DEVICE, non_blocking=True)
                fb = fb.to(DEVICE, non_blocking=True) if fb.numel() else None
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(x2, x20, fb)
                    loss = torch.zeros((), device=DEVICE)
                    for j in range(3):
                        w = args.s_weight if j == 1 else 1.0
                        diff = (pred[:, j] - yb[:, j]) ** 2 * mb[:, j]
                        loss = loss + w * diff.sum() / mb[:, j].sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach())
                nb += 1
            val_metrics = evaluate(model, val_loader, s_stats)
            combined = (val_metrics["Score_E_spearman"] * 0.4
                        + val_metrics["Score_S_spearman"] * 0.35
                        + val_metrics["Score_B_spearman"] * 0.25)
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
    torch.save({"state_dict": best_ckpt, "feat_dim": feat_dim, "feat_cols": feat_cols,
                "feat_stats": feat_stats, "name": args.name}, ckpt_path)
    model.load_state_dict(best_ckpt)
    test_metrics = evaluate(model, test_loader, s_stats)
    test_metrics["combined_spearman"] = (test_metrics["Score_E_spearman"] * 0.4
                                         + test_metrics["Score_S_spearman"] * 0.35
                                         + test_metrics["Score_B_spearman"] * 0.25)
    pd.DataFrame([test_metrics]).to_csv(eval_path, sep="\t", index=False)
    print("\n=== TEST metrics ===")
    for k, v in test_metrics.items():
        print("  %s = %s" % (k, ("%.4f" % v) if isinstance(v, float) else str(v)))
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write("name\t%s\narch\tdual_stream_200_2000\nfeatures\t%s\nfeat_dim\t%d\ndevice\t%s\nparams\t%d\n"
                 % (args.name, args.features, feat_dim, DEVICE, n_params))
        fh.write("test_metrics\t%s\n" % json.dumps({k: v for k, v in test_metrics.items()}))
    print("saved", ckpt_path, "|", eval_path, "|", card_path)


if __name__ == "__main__":
    main()
