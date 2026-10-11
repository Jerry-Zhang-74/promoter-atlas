#!/usr/bin/env python3
"""promoter_atlas v6: multi-species, multi-omics, multi-task promoter model.

Sequence (+species) -> measured molecular readouts.
  Heads:
    E / S / B        gene-level GTEx labels (human, MSE; S z-scored)
    ortholog_E       human E transferred to other species via 1:1 orthologs (MSE)
    omics scalars    CAGE tss, PRO-cap tss, H3K4me3, H3K27ac, DNase, ... (MSE, human)
    TF ChIP          multi-label occupancy over top TFs from ReMap 2022 (BCE, human)
    species          species classifier (CE, all rows)
    masked token     masked-nucleotide prediction on the 400 bp window (CE, all rows)

Usage:
    .venv/Scripts/python.exe runs/scripts/train_promoter_atlas_v6.py --name v6_base --epochs 8 --batch 128
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
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "labels" / "build" / "promoter_atlas_v6_input.tsv"
MM_H2000 = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"
MM_H400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
MM_SP = ROOT / "data" / "multispecies" / "memmap"
TF_MAT = ROOT / "features" / "omics" / "promoter_chip_tf_matrix_v1_uint8.memmap"
TF_MAN = ROOT / "features" / "omics" / "chip_tf_manifest_v1.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SPECIES = 19
OMICS_COLS = ["cage_tss", "procap_tss", "h3k4me3", "h3k27ac", "h3k27me3",
              "h3k4me1", "h3k36me3", "dnase", "atac"]


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
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
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


# ---------------------------------------------------------------- model

class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, k=5):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        r = x
        x = F.relu(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return F.relu(x + r)


class LocalStem(nn.Module):
    """400 bp local tower; also returns per-position features for the masked head."""

    def __init__(self, ch=160, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, ch, 9, padding=4),
                                  nn.GroupNorm(min(8, ch), ch), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([DilatedBlock(ch, 2 ** i) for i in range(blocks)])
        self.mask_head = nn.Linear(ch, 5)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        avg = h.mean(dim=2)
        mx = h.max(dim=2).values
        return torch.cat([avg, mx], dim=1), h  # (B, 2ch), (B, ch, L)


class ContextStem(nn.Module):
    """2000 bp context tower (dilated conv U-Net-style encoder)."""

    def __init__(self, ch=192, blocks=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch // 2, 15, stride=2, padding=7), nn.GroupNorm(min(8, ch // 2), ch // 2), nn.ReLU(inplace=True),
            nn.Conv1d(ch // 2, ch, 15, stride=2, padding=7), nn.GroupNorm(min(8, ch), ch), nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([DilatedBlock(ch, 2 ** i) for i in range(blocks)])

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        avg = h.mean(dim=2)
        mx = h.max(dim=2).values
        return torch.cat([avg, mx], dim=1)  # (B, 2ch)


class PromoterAtlasNet(nn.Module):
    def __init__(self, n_tf=0, n_omics=0, local_ch=160, ctx_ch=192, ctx_blocks=8,
                 local_blocks=6, tf_layers=0, d_model=None):
        super().__init__()
        self.local = LocalStem(ch=local_ch, blocks=local_blocks)
        self.context = ContextStem(ch=ctx_ch, blocks=ctx_blocks)
        d = d_model or ctx_ch
        self.tf_layers = tf_layers
        if tf_layers > 0:
            self.ctx_pos = nn.Parameter(torch.zeros(1, 500, ctx_ch))
            self.ctx_proj = nn.Linear(ctx_ch, d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=d * 4,
                                               dropout=0.1, batch_first=True, norm_first=True)
            self.ctx_transformer = nn.TransformerEncoder(layer, num_layers=tf_layers)
            ctx_dim = 2 * d
        else:
            ctx_dim = 2 * ctx_ch
        self.species_emb = nn.Embedding(N_SPECIES, 32)
        dim = local_ch * 2 + ctx_dim + 32
        self.fuse = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.head_esb = nn.Linear(512, 3)
        self.head_ortho = nn.Linear(512, 1)
        self.head_omics = nn.Linear(512, n_omics) if n_omics else None
        self.head_tf = nn.Linear(512, n_tf) if n_tf else None
        self.head_species = nn.Linear(512, N_SPECIES)

    def forward(self, x400, x2000, species):
        x400 = F.one_hot(x400.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        x2000 = F.one_hot(x2000.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        a, pos = self.local(x400)
        b = self.context(x2000)
        if self.tf_layers > 0:
            h = self.context.stem(x2000)
            for blk in self.context.blocks:
                h = blk(h)
            h = h.permute(0, 2, 1)  # (B, 500, ch)
            h = self.ctx_proj(h) + self.ctx_pos[:, : h.size(1)]
            h = self.ctx_transformer(h)
            b = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        s = self.species_emb(species)
        z = self.fuse(torch.cat([a, b, s], dim=1))
        out = {"esb": self.head_esb(z),
               "ortho": self.head_ortho(z).squeeze(-1),
               "species": self.head_species(z),
               "mask": self.local.mask_head(pos.permute(0, 2, 1))}
        if self.head_omics is not None:
            out["omics"] = self.head_omics(z)
        if self.head_tf is not None:
            out["tf"] = self.head_tf(z)
        return out


# ---------------------------------------------------------------- dataset

class AtlasDataset(Dataset):
    def __init__(self, df, mmaps, tf_matrix=None, masked_frac=0.15, rng=None):
        self.df = df.reset_index(drop=True)
        self.mmaps = mmaps  # key -> {2000: memmap, 400: memmap}
        self.tf_matrix = tf_matrix
        self.masked_frac = masked_frac
        self.rng = rng or np.random.default_rng(0)
        self.n_omics = sum(1 for c in OMICS_COLS if c in df.columns)
        self.omics_cols = [c for c in OMICS_COLS if c in df.columns]
        self.has_omics = self.n_omics > 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        key = row["memmap_key"]
        x2000 = torch.from_numpy(self.mmaps[key][2000][row["r2000"]].copy())
        x400 = torch.from_numpy(self.mmaps[key][400][row["r400"]].copy())
        species = torch.tensor(int(row["species_id"]), dtype=torch.long)
        # labels (per-axis masks: labeled AND not NaN)
        y_esb = np.zeros(3, dtype=np.float32)
        m_esb = np.zeros(3, dtype=np.float32)
        for j, c in enumerate(["Score_E", "Score_S", "Score_B"]):
            v = row.get(c)
            if row["label_mask"] == 1 and pd.notna(v):
                y_esb[j] = float(v)
                m_esb[j] = 1.0
        y_ortho = np.float32(row.get("ortholog_E") if not pd.isna(row.get("ortholog_E")) else 0.0)
        m_ortho = np.float32(1.0 if not pd.isna(row.get("ortholog_E")) else 0.0)
        # omics targets
        if self.has_omics:
            y_om = np.zeros(self.n_omics, dtype=np.float32)
            m_om = np.zeros(self.n_omics, dtype=np.float32)
            for k, c in enumerate(self.omics_cols):
                v = row.get(c)
                if pd.notna(v):
                    y_om[k] = float(v)
                    m_om[k] = 1.0
        else:
            y_om = np.zeros(0, dtype=np.float32)
            m_om = np.zeros(0, dtype=np.float32)
        # tf vector
        if self.tf_matrix is not None and pd.notna(row.get("tf_row")):
            tf = torch.from_numpy(self.tf_matrix[int(row["tf_row"])].astype(np.float32))
            m_tf = torch.tensor(1.0)
        else:
            tf = torch.zeros(self.tf_matrix.shape[1], dtype=torch.float32) if self.tf_matrix is not None else torch.zeros(0)
            m_tf = torch.tensor(0.0)
        # masked tokens on x400
        mask_pos = self.rng.random(400) < self.masked_frac
        y_mask = x400.clone().long()
        x400_masked = x400.clone()
        x400_masked[mask_pos] = 4  # N
        y_mask[~mask_pos] = -100
        return (x400_masked, x2000, species,
                torch.from_numpy(y_esb), torch.from_numpy(m_esb),
                torch.tensor(y_ortho), torch.tensor(m_ortho),
                torch.from_numpy(y_om), torch.from_numpy(m_om),
                tf, m_tf, y_mask, torch.from_numpy(mask_pos.astype(np.float32)))


# ---------------------------------------------------------------- losses / eval

def compute_loss(out, batch, args):
    x400, x2000, species, y_esb, m_esb, y_ortho, m_ortho, y_om, m_om, tf, m_tf, y_mask, _ = batch
    y_esb, m_esb = y_esb.to(DEVICE), m_esb.to(DEVICE)
    y_ortho, m_ortho = y_ortho.to(DEVICE), m_ortho.to(DEVICE)
    y_om, m_om = y_om.to(DEVICE), m_om.to(DEVICE)
    tf, m_tf = tf.to(DEVICE), m_tf.to(DEVICE)
    y_mask = y_mask.to(DEVICE)
    losses = {}
    # E/S/B
    diff = (out["esb"] - y_esb) ** 2 * m_esb
    n_esb = m_esb.sum().clamp_min(1.0)
    w = torch.tensor([1.0, args.s_weight, 1.0], device=DEVICE).unsqueeze(0)
    losses["esb"] = ((diff * w).sum() / n_esb)
    # ortholog E
    if m_ortho.sum() > 0:
        losses["ortho"] = (((out["ortho"] - y_ortho) ** 2 * m_ortho).sum() / m_ortho.sum().clamp_min(1.0))
    # omics
    if out.get("omics") is not None and m_om.shape[1] > 0 and m_om.sum() > 0:
        d = (out["omics"] - y_om) ** 2 * m_om
        losses["omics"] = (d.sum() / m_om.sum().clamp_min(1.0))
    # TF BCE
    if out.get("tf") is not None and m_tf.sum() > 0:
        tf_pred = out["tf"][m_tf > 0]
        tf_tgt = tf[m_tf > 0]
        losses["tf"] = F.binary_cross_entropy_with_logits(tf_pred, tf_tgt)
    # species
    losses["species"] = F.cross_entropy(out["species"], species.to(DEVICE))
    # masked tokens
    losses["mask"] = F.cross_entropy(out["mask"].reshape(-1, 5), y_mask.reshape(-1))
    total = losses["esb"]
    for k, coef in [("ortho", 0.5), ("omics", 0.5), ("tf", 0.3), ("species", 0.2), ("mask", 0.3)]:
        if k in losses:
            total = total + coef * losses[k]
    return total, {k: float(v.detach()) for k, v in losses.items()}


def evaluate(model, loader, s_stats=None, gene_level=True):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x400, x2000, species, y_esb, m_esb, y_ortho, m_ortho, y_om, m_om, tf, m_tf, y_mask, _ = [t.to(DEVICE) for t in batch]
            out = model(x400, x2000, species)
            rows.append({
                "esb": out["esb"].cpu().numpy(),
                "y_esb": y_esb.cpu().numpy(), "m_esb": m_esb.cpu().numpy(),
                "ortho": out["ortho"].cpu().numpy(), "y_ortho": y_ortho.cpu().numpy(), "m_ortho": m_ortho.cpu().numpy(),
                "omics": out.get("omics", torch.zeros(1, 0)).cpu().numpy(), "y_om": y_om.cpu().numpy(), "m_om": m_om.cpu().numpy(),
                "tf": out.get("tf", torch.zeros(1, 0)).cpu().numpy(), "tf_tgt": tf.cpu().numpy(), "m_tf": m_tf.cpu().numpy(),
                "species_pred": out["species"].cpu().numpy(), "species_tgt": species.cpu().numpy(),
            })
    pred_esb = np.concatenate([r["esb"] for r in rows])
    y_esb = np.concatenate([r["y_esb"] for r in rows])
    m_esb = np.concatenate([r["m_esb"] for r in rows])
    metrics = {}
    for j, name in enumerate(["Score_E", "Score_S", "Score_B"]):
        mask = m_esb[:, j] > 0
        yp, yt = pred_esb[mask, j], y_esb[mask, j]
        if j == 1 and s_stats is not None:
            yp = yp * s_stats[1] + s_stats[0]
            yt = yt * s_stats[1] + s_stats[0]
        metrics[name + "_r2"] = r2_np(yp, yt) if len(yt) else float("nan")
        metrics[name + "_spearman"] = spearman_np(yp, yt) if len(yt) else float("nan")
        metrics[name + "_n"] = int(mask.sum())
    pred_ortho = np.concatenate([r["ortho"] for r in rows])
    y_ortho = np.concatenate([r["y_ortho"] for r in rows])
    m_ortho = np.concatenate([r["m_ortho"] for r in rows]).astype(bool)
    if m_ortho.any():
        metrics["orthoE_r2"] = r2_np(pred_ortho[m_ortho], y_ortho[m_ortho])
        metrics["orthoE_spearman"] = spearman_np(pred_ortho[m_ortho], y_ortho[m_ortho])
        metrics["orthoE_n"] = int(m_ortho.sum())
    om_all = np.concatenate([r["omics"] for r in rows])
    yom_all = np.concatenate([r["y_om"] for r in rows])
    mom_all = np.concatenate([r["m_om"] for r in rows])
    if om_all.shape[1]:
        for k in range(om_all.shape[1]):
            mask = mom_all[:, k] > 0
            if mask.sum() > 10:
                metrics[f"omics{k}_r"] = float(np.corrcoef(om_all[mask, k], yom_all[mask, k])[0, 1])
    sp_pred = np.concatenate([r["species_pred"] for r in rows])
    sp_tgt = np.concatenate([r["species_tgt"] for r in rows])
    metrics["species_acc"] = float((sp_pred.argmax(1) == sp_tgt).mean())
    # TF AUC-lite (mean rank correlation on positives)
    tfp = np.concatenate([r["tf"] for r in rows])
    tft = np.concatenate([r["tf_tgt"] for r in rows])
    mtf = np.concatenate([r["m_tf"] for r in rows]).astype(bool)
    if tfp.shape[1] and mtf.any():
        pos = tfp[mtf]
        tpos = tft[mtf]
        aucs = []
        for k in range(pos.shape[1]):
            p, t = pos[:, k], tpos[:, k]
            if t.sum() > 20 and (t == 0).sum() > 20:
                order = np.argsort(p)
                ranks = np.empty(len(p))
                ranks[order] = np.arange(len(p))
                aucs.append((ranks[t > 0].mean() - (t.sum() - 1) / 2) / max(1, (t == 0).sum()))
        if aucs:
            metrics["tf_avg_auc"] = float(np.mean(aucs))
    return metrics


# ---------------------------------------------------------------- gene-level eval

def gene_level_metrics(df, pred_esb, y_esb, m_esb, s_stats=None, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    genes = df["gene_id"].to_numpy()
    out = {}
    for j, name in enumerate(["Score_E", "Score_S", "Score_B"]):
        mask = m_esb[:, j] > 0
        g = genes[mask]
        yp, yt = pred_esb[mask, j], y_esb[mask, j]
        if j == 1 and s_stats is not None:
            yp = yp * s_stats[1] + s_stats[0]
            yt = yt * s_stats[1] + s_stats[0]
        ug = np.unique(g)
        gmap = {x: k for k, x in enumerate(ug)}
        idx = np.array([gmap[x] for x in g])
        gpred = np.bincount(idx, weights=yp) / np.bincount(idx)
        gtarg = np.bincount(idx, weights=yt) / np.bincount(idx)
        out[f"{name}_gene_r2"] = r2_np(gpred, gtarg)
        out[f"{name}_gene_spearman"] = spearman_np(gpred, gtarg)
        out[f"{name}_gene_n"] = len(ug)
        # bootstrap CI of spearman
        boots = []
        for _ in range(n_boot):
            sel = rng.choice(len(ug), size=len(ug), replace=True)
            boots.append(spearman_np(gpred[sel], gtarg[sel]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        out[f"{name}_gene_sp_ci_lo"] = float(np.percentile(boots, 2.5)) if len(boots) else float("nan")
        out[f"{name}_gene_sp_ci_hi"] = float(np.percentile(boots, 97.5)) if len(boots) else float("nan")
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--s-weight", type=float, default=2.0)
    ap.add_argument("--local-ch", type=int, default=160)
    ap.add_argument("--ctx-ch", type=int, default=192)
    ap.add_argument("--ctx-blocks", type=int, default=8)
    ap.add_argument("--local-blocks", type=int, default=6)
    ap.add_argument("--tf-layers", type=int, default=0)
    ap.add_argument("--zscore-s", action="store_true")
    ap.add_argument("--species-per-epoch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(INPUT, sep="\t", low_memory=False)
    df["split"] = df["split"].fillna("train")
    # drop all-NaN omics columns (e.g. ATAC before its files land)
    bad = [c for c in OMICS_COLS if c in df.columns and df[c].notna().sum() < 1000]
    if bad:
        df = df.drop(columns=bad)
        print("dropped empty omics cols:", bad, flush=True)
    # cap multi-species rows for memory in quick mode
    if args.quick:
        df = pd.concat([
            df[df["split"] != "train"],
            df[df["split"] == "train"].sample(frac=0.08, random_state=0),
        ], ignore_index=True)
    print("rows:", len(df), "| human:", (df.species == "homo_sapiens").sum(), flush=True)

    if args.zscore_s:
        tr = df[(df["split"] == "train") & (df["label_mask"] == 1)]
        v = tr["Score_S"].dropna()
        s_stats = (float(v.mean()), float(v.std()) + 1e-6)
        df["Score_S"] = (df["Score_S"] - s_stats[0]) / s_stats[1]
    else:
        s_stats = None

    # mmaps
    mmaps = {"human": {
        2000: np.memmap(MM_H2000, dtype="uint8", mode="r", shape=(507365, 2000)),
        400: np.memmap(MM_H400, dtype="uint8", mode="r", shape=(507365, 400)),
    }}
    for sp in df["memmap_key"].unique():
        if sp == "human":
            continue
        p2 = MM_SP / f"{sp}_up2000_uint8.memmap"
        p4 = MM_SP / f"{sp}_sym400_uint8.memmap"
        if not p2.exists() or not p4.exists():
            df = df[df["memmap_key"] != sp]
            print("DROP species (memmap missing):", sp, flush=True)
            continue
        i2 = pd.read_csv(MM_SP / f"{sp}_up2000_index.tsv", sep="\t", usecols=["row_index"])
        i4 = pd.read_csv(MM_SP / f"{sp}_sym400_index.tsv", sep="\t", usecols=["row_index"])
        n2 = int(i2["row_index"].max()) + 1
        n4 = int(i4["row_index"].max()) + 1
        mmaps[sp] = {
            2000: np.memmap(p2, dtype="uint8", mode="r", shape=(n2, 2000)),
            400: np.memmap(p4, dtype="uint8", mode="r", shape=(n4, 400)),
        }
    n_tf = 0
    tf_matrix = None
    if TF_MAT.exists():
        n_tf = int(np.lib.format.open_memmap(str(TF_MAT), mode="r").shape[1])
        tf_matrix = np.memmap(TF_MAT, dtype="uint8", mode="r", shape=(507365, n_tf))
        print("TF matrix:", tf_matrix.shape, flush=True)

    human_df = df[df["species"] == "homo_sapiens"]
    sp_df = df[df["species"] != "homo_sapiens"]
    sp_split = {k: g for k, g in sp_df.groupby("split")}

    ds_human_train = AtlasDataset(human_df[human_df["split"] == "train"], mmaps, tf_matrix,
                                  rng=np.random.default_rng(args.seed))
    ds_human_val = AtlasDataset(human_df[human_df["split"] == "val"], mmaps, tf_matrix)
    ds_human_test = AtlasDataset(human_df[human_df["split"] == "test"], mmaps, tf_matrix)
    if len(sp_df):
        ds_sp_train = AtlasDataset(sp_split.get("train", sp_df.iloc[0:0]), mmaps, None, rng=np.random.default_rng(args.seed + 1))
        ds_sp_val = AtlasDataset(sp_split.get("val", sp_df.iloc[0:0]), mmaps, None)
        ds_sp_test = AtlasDataset(sp_split.get("test", sp_df.iloc[0:0]), mmaps, None)
    else:
        ds_sp_train = AtlasDataset(sp_df, mmaps, None)
        ds_sp_val = ds_sp_train
        ds_sp_test = ds_sp_train

    def collate(batch):
        return tuple(torch.stack([b[i] for b in batch]) for i in range(len(batch[0])))

    train_h_loader = DataLoader(ds_human_train, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                num_workers=0, pin_memory=(DEVICE == "cuda"), drop_last=True)
    val_h_loader = DataLoader(ds_human_val, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
    test_h_loader = DataLoader(ds_human_test, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
    if len(sp_df):
        train_sp_loader = DataLoader(ds_sp_train, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                     num_workers=0, pin_memory=(DEVICE == "cuda"), drop_last=True)
        val_sp_loader = DataLoader(ds_sp_val, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
        test_sp_loader = DataLoader(ds_sp_test, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
    else:
        train_sp_loader = val_sp_loader = test_sp_loader = None

    n_omics = len([c for c in OMICS_COLS if c in df.columns])
    model = PromoterAtlasNet(n_tf=n_tf, n_omics=n_omics, local_ch=args.local_ch, ctx_ch=args.ctx_ch,
                             ctx_blocks=args.ctx_blocks, local_blocks=args.local_blocks,
                             tf_layers=args.tf_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_epoch = max(1, len(train_h_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                total_steps=args.epochs * steps_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    ckpt_dir, log_dir, eval_dir, card_dir = (ROOT / "models" / d for d in
                                             ["checkpoints", "logs", "eval", "cards"])
    for d in (ckpt_dir, log_dir, eval_dir, card_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{args.name}.log"
    ckpt_path = ckpt_dir / f"{args.name}.pt"
    eval_path = eval_dir / f"{args.name}_metrics.tsv"
    card_path = card_dir / f"{args.name}_card.txt"
    # resume support
    start_epoch = 0
    resume_path = Path(str(ckpt_path) + ".resume")
    if args.resume and resume_path.exists():
        rs = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(rs["state_dict"])
        optimizer.load_state_dict(rs["optimizer"])
        sched.load_state_dict(rs["sched"])
        start_epoch = rs.get("epoch", 0) + 1
        print("resumed from epoch", start_epoch, flush=True)

    def log_line(fh, text):
        print(text, flush=True)
        fh.write(text + "\n")
        fh.flush()

    best_val = -np.inf
    best_ckpt = None
    t_start = time.time()
    sp_iter = iter(train_sp_loader) if train_sp_loader is not None else None

    with open(log_path, "w", encoding="utf-8") as fh:
        log_line(fh, "name	" + args.name)
        log_line(fh, "arch	promoter_atlas_v6_multiomics")
        log_line(fh, f"params	{n_params}")
        log_line(fh, f"n_tf	{n_tf}")
        log_line(fh, f"n_omics	{n_omics}")
        log_line(fh, f"rows	{len(df)}")
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total, nb = 0.0, 0
            for bi, batch in enumerate(train_h_loader):
                # mix in species batch
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    out_h = model(batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE))
                    loss, parts = compute_loss(out_h, batch, args)
                    if len(sp_df):
                        try:
                            sp_batch = next(sp_iter)
                        except StopIteration:
                            sp_iter = iter(train_sp_loader)
                            sp_batch = next(sp_iter)
                        out_s = model(sp_batch[0].to(DEVICE), sp_batch[1].to(DEVICE), sp_batch[2].to(DEVICE))
                        loss_s, parts_s = compute_loss(out_s, sp_batch, args)
                        loss = loss + loss_s
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                sched.step()
                total += float(loss.detach())
                nb += 1
                if bi % 500 == 0:
                    log_line(fh, f"epoch {epoch} step {bi} loss {total / nb:.4f} " +
                             " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
            val_metrics = evaluate(model, val_h_loader, s_stats)
            # gene-level combined on val
            df_val = human_df[human_df["split"] == "val"].reset_index(drop=True)
            preds, ys, ms = [], [], []
            with torch.no_grad():
                for batch in val_h_loader:
                    x400, x2000, species, y_esb, m_esb, _, _, _, _, _, _, _, _ = [t.to(DEVICE) for t in batch]
                    out = model(x400, x2000, species)
                    preds.append(out["esb"].cpu().numpy())
                    ys.append(y_esb.cpu().numpy())
                    ms.append(m_esb.cpu().numpy())
            g = gene_level_metrics(df_val, np.concatenate(preds), np.concatenate(ys), np.concatenate(ms), s_stats, n_boot=50)
            combined = (g["Score_E_gene_spearman"] * 0.4 + g["Score_S_gene_spearman"] * 0.35
                        + g["Score_B_gene_spearman"] * 0.25)
            if np.isfinite(combined) and combined > best_val:
                best_val = combined
                best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log_line(fh, "epoch %d | loss %.4f | val E r2=%.4f S r2=%.4f B r2=%.4f | gene sp E=%.4f S=%.4f B=%.4f | comb %.4f | species_acc %.4f"
                     % (epoch, total / max(nb, 1),
                        val_metrics["Score_E_r2"], val_metrics["Score_S_r2"], val_metrics["Score_B_r2"],
                        g["Score_E_gene_spearman"], g["Score_S_gene_spearman"], g["Score_B_gene_spearman"],
                        combined, val_metrics["species_acc"]))
            # save resume checkpoint every epoch (crash-safe)
            torch.save({"state_dict": best_ckpt or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "n_tf": n_tf, "n_omics": n_omics,
                        "s_stats": s_stats, "name": args.name}, resume_path)
            if args.max_minutes is not None and (time.time() - t_start) > args.max_minutes * 60:
                log_line(fh, "time budget reached")
                break

    if best_ckpt is None:
        best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": best_ckpt, "n_tf": n_tf, "n_omics": n_omics,
                "s_stats": s_stats, "name": args.name}, ckpt_path)
    model.load_state_dict(best_ckpt)

    # final eval
    test_metrics = evaluate(model, test_h_loader, s_stats)
    df_test = human_df[human_df["split"] == "test"].reset_index(drop=True)
    preds, ys, ms = [], [], []
    with torch.no_grad():
        for batch in test_h_loader:
            x400, x2000, species, y_esb, m_esb, _, _, _, _, _, _, _, _ = [t.to(DEVICE) for t in batch]
            out = model(x400, x2000, species)
            preds.append(out["esb"].cpu().numpy())
            ys.append(y_esb.cpu().numpy())
            ms.append(m_esb.cpu().numpy())
    g = gene_level_metrics(df_test, np.concatenate(preds), np.concatenate(ys), np.concatenate(ms), s_stats)
    test_metrics.update(g)
    if len(sp_df):
        sp_test = evaluate(model, test_sp_loader, None)
        for k, v in sp_test.items():
            test_metrics["sp_" + k] = v
    pd.DataFrame([test_metrics]).to_csv(eval_path, sep="	", index=False)
    print("\n=== TEST metrics ===")
    for k, v in test_metrics.items():
        print("  %s = %s" % (k, ("%.4f" % v) if isinstance(v, float) else str(v)))
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(f"name\t{args.name}\narch\tpromoter_atlas_v6_multiomics\nparams\t{n_params}\nn_tf\t{n_tf}\nn_omics\t{n_omics}\n")
        fh.write("test_metrics\t" + json.dumps({k: v for k, v in test_metrics.items()}) + "\n")
    print("saved", ckpt_path, "|", eval_path, "|", card_path)


if __name__ == "__main__":
    main()
