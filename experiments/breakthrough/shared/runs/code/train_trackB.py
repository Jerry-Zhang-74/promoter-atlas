#!/usr/bin/env python3
"""Track B stage 2: K562 MPRA small models from scratch on the NEW grouped split.

Candidates (same new grouped split, target = activity = Mean_log10):
  ridge        : GC + 3-mer count features -> Ridge (alpha via component-grouped CV)
  smallcnn     : reproduced old ConvBaseline architecture, trained from scratch
  legnet       : ONE compact residual/dilated CNN (LegNet-inspired; NOT exact)

Target standardization is fit on TRAIN only. CNN validation early stopping uses
raw-activity val MSE. Final test is evaluated only with the predeclared
best-val-epoch rule, identical across the 3 fixed seeds.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
GROUPED = ROOT / "data" / "breakthrough_20261002" / "trackB_grouped.tsv"
OUT = ROOT / "reports" / "breakthrough_20261002"
CKPT_DIR = ROOT / "models" / "breakthrough_20261002"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
KMER = 3
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEEDS = [20261005, 20261006, 20261007]


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
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def r2(a, b):
    return 1.0 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def encode_sequences(seqs):
    n = len(seqs)
    x = np.zeros((n, 164), dtype=np.uint8)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j, c in enumerate(s):
            x[i, j] = BASE.get(c, 4)
    return x


def kmer_features(seqs, k=KMER):
    n = len(seqs)
    n_k = 4 ** k
    X = np.zeros((n, n_k + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        gc = 0.0
        cnt = np.zeros(n_k, dtype=np.float32)
        for j, c in enumerate(s):
            if c in "GC":
                gc += 1.0
            if j <= len(s) - k:
                code = 0
                ok = True
                for t in range(k):
                    b = BASE.get(s[j + t], 4)
                    if b == 4:
                        ok = False
                        break
                    code = code * 4 + b
                if ok:
                    cnt[code] += 1.0
        X[i, 0] = gc / len(s)
        X[i, 1:] = cnt / max(1, len(s) - k + 1)
    return X


class ConvBaseline(nn.Module):
    """Reproduced old small CNN (train_mpra_finetune.ConvBaseline)."""

    def __init__(self, num_targets=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, num_targets),
        )

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        x = self.features(x).squeeze(-1)
        return self.head(x)


class DilatedResBlock(nn.Module):
    def __init__(self, ch, dilation, k=5):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class LegNetInspired(nn.Module):
    """Compact residual/dilated CNN (LegNet-inspired; NOT exact reproduction)."""

    def __init__(self, num_targets=1, ch=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([DilatedResBlock(ch, d) for d in [1, 2, 4, 8]])
        self.head = nn.Sequential(
            nn.Linear(ch * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(64, num_targets),
        )

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        h = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        return self.head(h)


def build_cnn(name):
    if name == "smallcnn":
        return ConvBaseline(1)
    if name == "legnet":
        return LegNetInspired(1)
    raise ValueError(name)


def predict_cnn(model, x):
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x), 512):
            xb = torch.from_numpy(x[i:i + 512]).long().to(DEVICE)
            preds.append(model(xb).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def train_cnn(model, x_tr, y_tr_z, x_va, y_va, epochs, lr, batch, wd, seed, y_mu, y_sd):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, math.ceil(len(x_tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)
    best_mse = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    log = []
    n = len(x_tr)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        total, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(x_tr[idx]).long().to(DEVICE)
            yb = torch.from_numpy(y_tr_z[idx]).float().to(DEVICE)
            pred = model(xb)[:, 0]
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()); nb += 1
        pva_z = predict_cnn(model, x_va)
        pva = pva_z * y_sd + y_mu
        mse = float(np.mean((pva - y_va) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        log.append({"epoch": epoch, "train_loss": total / max(nb, 1), "val_mse": mse,
                    "val_r2": r2(pva, y_va), "val_spearman": spearman(pva, y_va)})
    model.load_state_dict(best_state)
    return model, best_epoch, best_mse, log


def group_bootstrap_metric(y, p, comp, metric_fn, n_boot=1000, seed=20261006):
    rng = np.random.default_rng(seed)
    comps = np.unique(comp)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(comps), size=len(comps))
        mask = np.isin(comp, comps[pick])
        if mask.sum() < 2:
            continue
        vals.append(metric_fn(y[mask], p[mask]))
    vals = np.asarray(vals)
    return {"mean": float(np.mean(vals)), "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ridge,smallcnn,legnet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--report", type=Path, default=OUT)
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(GROUPED, sep="\t")
    seqs = df["sequence"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    split = df["new_split"].to_numpy()
    comp = df["component_id"].to_numpy()
    x = encode_sequences(seqs)

    tr = split == "train"; va = split == "val"; te = split == "test"
    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]
    x_te, y_te = x[te], y[te]
    comp_tr, comp_te = comp[tr], comp[te]
    print("train/val/test:", tr.sum(), va.sum(), te.sum(), flush=True)

    # target standardization fit on train only
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std())
    y_tr_z = (y_tr - y_mu) / y_sd

    results = {"target": "activity_Mean_log10", "target_scale_fit_train": {"mean": y_mu, "std": y_sd},
               "split_manifest": "data/breakthrough_20261002/trackB_split_manifest.json",
               "seeds": SEEDS, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "wd": args.wd,
               "candidates": {}}

    # ---- ridge baseline ----
    Xk = kmer_features(seqs)
    Xk_tr, Xk_va, Xk_te = Xk[tr], Xk[va], Xk[te]
    mu_f = Xk_tr.mean(axis=0); sc_f = Xk_tr.std(axis=0); sc_f[sc_f < 1e-12] = 1.0
    # alpha selection via component-grouped CV on train
    gkf = GroupKFold(n_splits=5)
    cv_mse = {a: [] for a in ALPHAS}
    for a in ALPHAS:
        for ti, vi in gkf.split(Xk_tr, y_tr, comp_tr):
            muf = Xk_tr[ti].mean(axis=0); scf = Xk_tr[ti].std(axis=0); scf[scf < 1e-12] = 1.0
            clf = Ridge(alpha=a).fit((Xk_tr[ti] - muf) / scf, y_tr[ti])
            pv = clf.predict((Xk_tr[vi] - muf) / scf)
            cv_mse[a].append(float(np.mean((pv - y_tr[vi]) ** 2)))
    best_a = min(cv_mse, key=lambda a: np.mean(cv_mse[a]))
    clf = Ridge(alpha=best_a).fit((Xk_tr - mu_f) / sc_f, y_tr)
    p_te = clf.predict((Xk_te - mu_f) / sc_f)
    results["candidates"]["ridge"] = {
        "best_alpha": best_a, "inner_cv_mse": {a: float(np.mean(v)) for a, v in cv_mse.items()},
        "test_spearman": spearman(p_te, y_te), "test_mae": mae(p_te, y_te),
        "test_r2": r2(p_te, y_te), "test_mse": float(np.mean((p_te - y_te) ** 2)),
        "test_spearman_boot": group_bootstrap_metric(y_te, p_te, comp_te, spearman),
        "test_mae_boot": group_bootstrap_metric(y_te, p_te, comp_te, mae),
    }
    print("ridge alpha", best_a, "test sp", round(results["candidates"]["ridge"]["test_spearman"], 4),
          "mae", round(results["candidates"]["ridge"]["test_mae"], 4), flush=True)

    # ---- CNN candidates ----
    for name in ["smallcnn", "legnet"]:
        if name not in args.candidates:
            continue
        seeds_out = {}
        test_preds = []
        for sd in SEEDS:
            model = build_cnn(name).to(DEVICE)
            model, best_ep, best_mse, log = train_cnn(model, x_tr, y_tr_z, x_va, y_va,
                                                      args.epochs, args.lr, args.batch, args.wd, sd, y_mu, y_sd)
            p_te = predict_cnn(model, x_te) * y_sd + y_mu
            test_preds.append(p_te)
            seeds_out[str(sd)] = {
                "best_val_epoch": best_ep, "best_val_mse": best_mse,
                "test_spearman": spearman(p_te, y_te), "test_mae": mae(p_te, y_te),
                "test_r2": r2(p_te, y_te), "val_log": log,
            }
            print(name, "seed", sd, "best_ep", best_ep, "test sp",
                  round(seeds_out[str(sd)]["test_spearman"], 4),
                  "mae", round(seeds_out[str(sd)]["test_mae"], 4), flush=True)
        ens = np.mean(test_preds, axis=0)
        results["candidates"][name] = {
            "per_seed": seeds_out,
            "ensemble_test_spearman": spearman(ens, y_te),
            "ensemble_test_mae": mae(ens, y_te),
            "ensemble_test_r2": r2(ens, y_te),
            "mean_test_spearman": float(np.mean([seeds_out[str(s)]["test_spearman"] for s in SEEDS])),
            "mean_test_mae": float(np.mean([seeds_out[str(s)]["test_mae"] for s in SEEDS])),
        }

    # ---- success check ----
    base_sp = max(results["candidates"]["ridge"]["test_spearman"],
                  results["candidates"]["smallcnn"]["ensemble_test_spearman"])
    base_mae = min(results["candidates"]["ridge"]["test_mae"],
                   results["candidates"]["smallcnn"]["ensemble_test_mae"])
    lg_sp = results["candidates"]["legnet"]["ensemble_test_spearman"]
    lg_mae = results["candidates"]["legnet"]["ensemble_test_mae"]
    results["success_check"] = {
        "strongest_baseline_spearman": base_sp, "strongest_baseline_mae": base_mae,
        "legnet_spearman": lg_sp, "legnet_mae": lg_mae,
        "spearman_gain": lg_sp - base_sp, "relative_mae_improvement": 1.0 - lg_mae / base_mae,
        "spearman_criterion": bool(lg_sp - base_sp >= 0.05),
        "mae_criterion": bool(1.0 - lg_mae / base_mae >= 0.10),
        "breakthrough_candidate": bool(lg_sp - base_sp >= 0.05 and 1.0 - lg_mae / base_mae >= 0.10),
    }
    (args.report / "trackB_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results["success_check"], indent=2), flush=True)
    print("saved", args.report / "trackB_results.json", flush=True)


if __name__ == "__main__":
    main()

