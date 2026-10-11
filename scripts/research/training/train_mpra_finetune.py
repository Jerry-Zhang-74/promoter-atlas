#!/usr/bin/env python3
"""Round 5 (S2.2): fine-tune the promoter model on MPRA reporter activity.

Supervision: Weingarten-Gabbay 2019 (GSE118242, K562) synthetic core promoters
(164 bp) with measured activity (Mean_log10 of the RNA/DNA ratio).

Three training modes:
  --mode finetune : ConvBaseline pretrained on GTEx (v2_baseline_repro), new
                    1-output head, whole net fine-tuned (backbone lr x0.1)
  --mode scratch  : same arch, random init (control: pre-training benefit)
  --mode frozen   : pretrained backbone frozen, head only (transfer check)

Eval: cross-sequence r2/spearman on test + per-family variant ranking where
families have >= 3 test members (the design-task metric).
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
SUP = ROOT / "features/mpra/mpra_supervised.tsv"
FAM = ROOT / "features/mpra/mpra_families.tsv"
PRETRAINED = ROOT / "models/checkpoints/v2_baseline_repro.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class ConvBaseline(nn.Module):
    def __init__(self, num_targets=3, feat_dim=0):
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
        self.head = nn.Sequential(
            nn.Linear(128 + feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, num_targets),
        )

    def forward(self, x_idx, feat=None):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.features(x).squeeze(-1)
        if feat is not None:
            x = torch.cat([x, feat], dim=1)
        return self.head(x)


def load_data():
    df = pd.read_csv(SUP, sep="\t")
    n = len(df)
    x = np.zeros((n, 164), dtype=np.uint8)
    for i, s in enumerate(df["sequence"]):
        s = s.upper()
        for j, c in enumerate(s):
            x[i, j] = BASE.get(c, 4)
    y = df["activity"].to_numpy(dtype=np.float32)
    split = df["split"].to_numpy()
    return x, y, split, df


def make_loader(x, y, shuffle, batch):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(DEVICE == "cuda"), drop_last=False)


def evaluate(model, x, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.from_numpy(x[i:i + 512]).to(DEVICE)
            preds.append(model(xb).cpu().numpy()[:, 0])
    p = np.concatenate(preds)
    return p


def family_eval(model, fam_df):
    rows = []
    for fid, grp in fam_df.groupby("family_id"):
        if len(grp) < 3:
            continue
        xs = []
        for s in grp["sequence"]:
            arr = np.zeros(164, dtype=np.uint8)
            for j, c in enumerate(s.upper()):
                arr[j] = BASE.get(c, 4)
            xs.append(arr)
        x = np.stack(xs)
        p = evaluate(model, x, grp["activity"].to_numpy())
        rows.append({"family_id": fid, "n": len(grp),
                     "rho": spearman(p, grp["activity"].to_numpy())})
    if not rows:
        return float("nan"), 0
    r = pd.DataFrame(rows)
    return float(r["rho"].mean()), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="finetune", choices=["finetune", "scratch", "frozen"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    set_seed(args.seed)
    x, y, split, df = load_data()
    x_tr = x[split == "train"]; y_tr = y[split == "train"]
    x_va = x[split == "val"]; y_va = y[split == "val"]
    x_te = x[split == "test"]; y_te = y[split == "test"]
    print("train/val/test:", len(x_tr), len(x_va), len(x_te))

    model = ConvBaseline(1, feat_dim=0).to(DEVICE)
    if args.mode in ("finetune", "frozen"):
        ck = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=False)
        print("loaded pretrained backbone (head replaced, strict=False)")
    if args.mode == "frozen":
        for p in model.features.parameters():
            p.requires_grad_(False)
    if args.mode == "finetune":
        head_params = [p for p in model.head.parameters()]
        backbone_params = [p for p in model.features.parameters()]
        optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": args.lr},
             {"params": backbone_params, "lr": args.lr * 0.1}],
            weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loader = make_loader(x_tr, y_tr, True, args.batch)
    n_steps = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                total_steps=args.epochs * n_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_va = 1e9
    best_ckpt = None
    log_path = ROOT / "models/logs" / ("train_" + args.name + ".log")
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(t):
            print(t, flush=True)
            fh.write(t + "\n")
        log("name\t%s\tmode\t%s" % (args.name, args.mode))
        for epoch in range(args.epochs):
            model.train()
            total, nb = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(xb)[:, 0]
                    loss = torch.mean((pred - yb) ** 2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach())
                nb += 1
            pva = evaluate(model, x_va, y_va)
            mse_va = float(np.mean((pva - y_va) ** 2))
            if mse_va < best_va:
                best_va = mse_va
                best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                log("epoch %d | loss %.4f | val mse %.4f r2 %.4f sp %.4f"
                    % (epoch, total / nb, mse_va, r2(pva, y_va), spearman(pva, y_va)))

    model.load_state_dict(best_ckpt)
    pte = evaluate(model, x_te, y_te)
    res = {
        "name": args.name, "mode": args.mode,
        "test_mse": float(np.mean((pte - y_te) ** 2)),
        "test_r2": r2(pte, y_te),
        "test_spearman": spearman(pte, y_te),
        "test_n": int(len(y_te)),
    }
    fam_df = pd.read_csv(FAM, sep="\t") if FAM.exists() else pd.DataFrame({"family_id": [], "sequence": [], "activity": []})
    fam_rho, fam_n = family_eval(model, fam_df)
    res["family_mean_rho"] = fam_rho
    res["family_n"] = fam_n
    print("\n=== TEST ===")
    for k, v in res.items():
        print("  %s = %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))

    ckpt_dir = ROOT / "models/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_ckpt, "mode": args.mode, "name": args.name,
                "arch": "ConvBaseline_164"}, ckpt_dir / (args.name + ".pt"))
    pd.DataFrame([res]).to_csv(ROOT / "models/eval" / (args.name + "_metrics.tsv"), sep="\t", index=False)
    with open(ROOT / "models/cards" / (args.name + "_card.txt"), "w", encoding="utf-8") as fh:
        fh.write("name\t%s\nmode\t%s\nsupervision\tGSE118242 K562 MPRA (Weingarten-Gabbay 2019)\n" % (args.name, args.mode))
        fh.write("test_metrics\t%s\n" % json.dumps(res))
    print("saved", ckpt_dir / (args.name + ".pt"))


if __name__ == "__main__":
    main()
