#!/usr/bin/env python3
"""Batch02 Track B: grouped 5-fold cross-validation on the batch01 TRAIN partition only.

Frozen batch01 architectures (smallcnn vs legnet) with matched budgets, batch=32,
max 20 epochs, 3 fixed seeds 20261008/09/10 per CNN, one GPU sequential. Early
stopping uses a grouped inner validation within each outer training partition.
Includes a GC/3-mer ridge baseline with inner-only alpha. All weights, fold/seed
predictions, splits, choices and standardization are saved.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    atomic_write_json, spearman, r2, mae, mse, set_seed,
)

DATA = ROOT / "data" / "breakthrough_20261002" / "batch02"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch02"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"
COMPONENTS = DATA / "trackB_train_components.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

SEEDS = [20261008, 20261009, 20261010]
EPOCHS = 20
BATCH = 32
LR = 3e-4
WD = 1e-4
SEED_FOLD = 20261008
SEED_INNER = 20261009
N_FOLDS = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
KMER = 3


# ---------------------------------------------------------------- models
class ConvBaseline(nn.Module):
    """Reproduced old small CNN (batch01 frozen architecture)."""

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
    """Compact residual/dilated CNN (batch01 frozen architecture)."""

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


# ---------------------------------------------------------------- features
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


def predict_cnn(model, x_idx, batch=256):
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x_idx), batch):
            xb = torch.from_numpy(x_idx[i:i + batch]).long().to(DEVICE)
            preds.append(model(xb).cpu().numpy()[:, 0])
    return np.concatenate(preds)


# ---------------------------------------------------------------- trainer
def _standardize_fit(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def train_cnn(name, seed, x_tr_idx, y_tr, y_mu, y_sd, x_va_idx, y_va,
              epochs, lr, batch, wd, ckpt_path, done_path, stop_after_epoch=None):
    """Train one CNN with early stopping on grouped inner validation (raw-activity MSE).

    Deterministic: seeds set BEFORE model+optimizer construction; cudnn deterministic.
    Resumable: saves optimizer/scheduler/RNG state at every epoch boundary.
    stop_after_epoch: if set, save checkpoint and return after that epoch (0-indexed),
    simulating a clean interrupt for resume-determinism testing (total_steps unchanged).
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    model = build_cnn(name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, math.ceil(len(x_tr_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * steps_per_epoch, pct_start=0.1)
    y_tr_z = (y_tr - y_mu) / y_sd
    best_mse = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    start_epoch = 0
    log = []

    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch = int(st["epoch"])
        best_mse = float(st["best_mse"])
        best_state = st["best_state"]
        best_epoch = int(st["best_epoch"])
        log = st["log"]
        random.setstate(st["rng_py"])
        np.random.set_state(st["rng_np"])
        torch.set_rng_state(st["rng_torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"  resume {name} fold ckpt from epoch {start_epoch}", flush=True)

    n = len(x_tr_idx)
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = np.random.permutation(n)
        total, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(x_tr_idx[idx]).long().to(DEVICE)
            yb = torch.from_numpy(y_tr_z[idx]).float().to(DEVICE)
            pred = model(xb)[:, 0]
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()); nb += 1
        pva_z = predict_cnn(model, x_va_idx)
        pva = pva_z * y_sd + y_mu
        val_mse = float(np.mean((pva - y_va) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        log.append({"epoch": epoch, "train_loss": total / max(nb, 1), "val_mse": val_mse,
                    "val_spearman": spearman(pva, y_va)})
        # resumable checkpoint every epoch (<=250 steps)
        torch.save({
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch + 1,
            "best_mse": best_mse,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "log": log,
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "seed": seed, "name": name,
        }, ckpt_path)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            print(f"  [determinism-test] stopped after epoch {epoch} (checkpoint saved)", flush=True)
            return model, best_epoch, best_mse, log

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "seed": seed, "name": name,
                "best_epoch": best_epoch, "best_val_mse": best_mse}, done_path)
    return model, best_epoch, best_mse, log


# ---------------------------------------------------------------- fold assignment
def assign_folds(comp_ids, n, seed):
    rng = np.random.default_rng(seed)
    comps = {}
    for i, c in enumerate(comp_ids):
        comps.setdefault(c, []).append(i)
    comp_list = sorted(comps.items(), key=lambda kv: (len(kv[1]), kv[0]))
    # randomize whole components, then balanced row assignment to the smallest fold
    order = rng.permutation(len(comp_list))
    fold_rows = np.zeros(N_FOLDS, dtype=np.int64)
    fold_of = np.full(n, -1, dtype=np.int8)
    comp_fold = {}
    for k in order:
        cid, members = comp_list[k]
        f = int(np.argmin(fold_rows))
        comp_fold[cid] = f
        for i in members:
            fold_of[i] = f
        fold_rows[f] += len(members)
    assert (fold_of >= 0).all()
    # multi-member spread report
    multi_per_fold = np.zeros(N_FOLDS, dtype=np.int64)
    for cid, members in comp_list:
        if len(members) >= 2:
            multi_per_fold[comp_fold[cid]] += 1
    return fold_of, comp_fold, fold_rows, multi_per_fold, comp_list


def inner_split(outer_comp_ids, outer_idx, seed):
    """Grouped 80/20 component split within one outer training partition."""
    rng = np.random.default_rng(seed)
    comps = {}
    for i, c in zip(outer_idx, outer_comp_ids):
        comps.setdefault(c, []).append(i)
    cids = sorted(comps.keys())
    order = rng.permutation(len(cids))
    total = sum(len(comps[cids[k]]) for k in range(len(cids)))
    target = int(round(total * 0.8))
    itr_idx, iva_idx = [], []
    acc = 0
    for k in order:
        members = comps[cids[k]]
        if acc < target or not itr_idx:
            itr_idx.extend(members); acc += len(members)
        else:
            iva_idx.extend(members)
    return np.asarray(itr_idx, dtype=np.int64), np.asarray(iva_idx, dtype=np.int64)


# ---------------------------------------------------------------- ridge
def ridge_fold(Xk, y, comp, outer_idx, held_idx):
    Xo, yo = Xk[outer_idx], y[outer_idx]
    comp_o = comp[outer_idx]
    gkf = GroupKFold(n_splits=5)
    cv = {a: [] for a in ALPHAS}
    for a in ALPHAS:
        for ti, vi in gkf.split(Xo, yo, comp_o):
            muf, scf = _standardize_fit(Xo[ti])
            clf = Ridge(alpha=a).fit((Xo[ti] - muf) / scf, yo[ti])
            pv = clf.predict((Xo[vi] - muf) / scf)
            cv[a].append(float(np.mean((pv - yo[vi]) ** 2)))
    best_a = min(cv, key=lambda a: float(np.mean(cv[a])))
    muf, scf = _standardize_fit(Xo)
    clf = Ridge(alpha=best_a).fit((Xo - muf) / scf, yo)
    p_held = clf.predict((Xk[held_idx] - muf) / scf)
    return p_held, best_a, {a: float(np.mean(v)) for a, v in cv.items()}


# ---------------------------------------------------------------- hamming audit
def hamming_audit(seqs, fold_of, L=164):
    BASE2 = {"A": 0, "C": 1, "G": 2, "T": 3}
    n = len(seqs)
    codes = np.stack([[BASE2[c] for c in s] for s in seqs]).astype(np.int64)
    O = np.eye(4, dtype=np.float32)[codes].reshape(n, L * 4)
    M = O @ O.T
    del O
    H = (L - M).round().astype(np.int16)
    del M
    np.fill_diagonal(H, L + 1)
    cross8 = cross16 = 0
    for a in range(n):
        for b in range(a + 1, n):
            if fold_of[a] == fold_of[b]:
                continue
            h = int(H[a, b])
            if h <= 8:
                cross8 += 1
            if h <= 16:
                cross16 += 1
    del H
    return {"cross_fold_pairs_ham_le8": cross8, "cross_fold_pairs_ham_le16": cross16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--cnns", default="smallcnn,legnet")
    ap.add_argument("--seeds", default="20261008,20261009,20261010")
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    global SEEDS
    SEEDS = [int(s) for s in args.seeds.split(",")]
    torch.set_num_threads(4)

    df = pd.read_csv(COMPONENTS, sep="\t")
    seqs = df["sequence"].astype(str).tolist()
    y = df["activity"].to_numpy(dtype=np.float32)
    comp_ids = df["verified_component_id"].astype(str).tolist()
    oligo = df["Oligo_index"].astype(str).tolist()
    n = len(df)
    print("train partition n =", n, flush=True)

    x_idx_all = encode_sequences(seqs).astype(np.int64)
    Xk = kmer_features(seqs)

    fold_of, comp_fold, fold_rows, multi_per_fold, comp_list = assign_folds(comp_ids, n, SEED_FOLD)
    comp_size = {cid: len(members) for cid, members in comp_list}
    fold_assignment = {
        "seed": SEED_FOLD,
        "n_sequences": n,
        "n_components": int(len(comp_list)),
        "fold_row_counts": [int(v) for v in fold_rows],
        "multi_components_per_fold": [int(v) for v in multi_per_fold],
        "multi_component_rows_per_fold": [
            int(sum(comp_size[cid] for cid, f in comp_fold.items() if f == fld and comp_size[cid] >= 2))
            for fld in range(N_FOLDS)],
        "algorithm": "randomize whole components (rng seed 20261008) then balanced greedy assignment to smallest fold (tie->smallest index); multi-member components spread",
    }
    pd.DataFrame({"Oligo_index": oligo, "fold": fold_of}).to_csv(args.data_dir / "trackB_cv_fold_assignment.tsv",
                                                                 sep="\t", index=False)
    atomic_write_json(REPORT / "trackB_cv_fold_assignment.json", fold_assignment)
    print("fold assignment:", json.dumps(fold_assignment, ensure_ascii=True), flush=True)

    # ---------------------------------------------------------------- loop
    all_preds = {"smallcnn": {}, "legnet": {}, "ridge": {}}
    held_oof = {}   # cnn/ridge -> list of (index, pred)
    ridge_info = {}
    scaler_info = {}

    for fld in range(N_FOLDS):
        held_idx = np.where(fold_of == fld)[0]
        outer_idx = np.where(fold_of != fld)[0]
        y_mu = float(y[outer_idx].mean())
        y_sd = float(y[outer_idx].std())
        scaler_info[fld] = {"y_mu": y_mu, "y_sd": y_sd, "n_outer": int(len(outer_idx)), "n_held": int(len(held_idx))}
        itr_idx, iva_idx = inner_split(np.asarray(comp_ids)[outer_idx], outer_idx, SEED_INNER)
        print(f"fold {fld}: outer {len(outer_idx)} held {len(held_idx)} inner_tr {len(itr_idx)} inner_va {len(iva_idx)}", flush=True)

        # ridge
        p_ridge, best_a, cv = ridge_fold(Xk, y, np.asarray(comp_ids), outer_idx, held_idx)
        ridge_info[fld] = {"best_alpha": best_a, "inner_cv_mse": cv}
        all_preds["ridge"][fld] = p_ridge
        held_oof.setdefault("ridge", []).extend(zip(held_idx.tolist(), p_ridge.tolist()))
        print(f"  ridge alpha {best_a}", flush=True)

        # CNNs
        for cnn_name in args.cnns.split(","):
            for sd in SEEDS:
                ckpt_path = args.model_dir / f"ckpt_{cnn_name}_fold{fld}_seed{sd}.pt"
                done_path = args.model_dir / f"weights_{cnn_name}_fold{fld}_seed{sd}.pt"
                pred_path = args.data_dir / f"pred_{cnn_name}_fold{fld}_seed{sd}.npy"
                if pred_path.exists():
                    p_held = np.load(pred_path)
                    print(f"  {cnn_name} fold{fld} seed{sd} cached", flush=True)
                else:
                    t0 = time.time()
                    model, best_ep, best_mse, log = train_cnn(
                        cnn_name, sd, x_idx_all[outer_idx], y[outer_idx], y_mu, y_sd,
                        x_idx_all[iva_idx], y[iva_idx], args.epochs, LR, BATCH, WD,
                        ckpt_path, done_path)
                    p_held = predict_cnn(model, x_idx_all[held_idx]) * y_sd + y_mu
                    np.save(pred_path, p_held.astype(np.float32))
                    print(f"  {cnn_name} fold{fld} seed{sd} best_ep {best_ep} "
                          f"val_mse {best_mse:.6f} t {time.time()-t0:.1f}s", flush=True)
                all_preds[cnn_name][(fld, sd)] = p_held

    atomic_write_json(REPORT / "trackB_ridge_info.json",
                      {"ridge_info": ridge_info, "scaler_info": scaler_info})

    # ---------------------------------------------------------------- assemble OOF
    def oof_vec(model_key):
        pairs = sorted(held_oof[model_key], key=lambda t: t[0])
        idx = [t[0] for t in pairs]
        pred = np.array([t[1] for t in pairs])
        assert idx == list(range(n)), "OOF must cover every sequence exactly once"
        return pred

    ridge_oof = oof_vec("ridge")
    cnns = args.cnns.split(",")
    cnn_seed_oof = {c: {sd: np.zeros(n, dtype=np.float64) for sd in SEEDS} for c in cnns}
    for fld in range(N_FOLDS):
        held_idx = np.where(fold_of == fld)[0]
        for c in cnns:
            for sd in SEEDS:
                cnn_seed_oof[c][sd][held_idx] = all_preds[c][(fld, sd)]
    cnn_oof = {c: np.mean([cnn_seed_oof[c][sd] for sd in SEEDS], axis=0) for c in cnns}

    # strata
    is_multi = np.array([comp_size[c] >= 2 for c in comp_ids])

    def strata_metrics(pred):
        out = {}
        for name, mask in [("overall", np.ones(n, dtype=bool)),
                           ("singleton", ~is_multi), ("multi_member", is_multi)]:
            out[name] = {"n": int(mask.sum()), "spearman": spearman(pred[mask], y[mask]),
                         "mae": mae(pred[mask], y[mask]), "mse": mse(pred[mask], y[mask]),
                         "r2": r2(pred[mask], y[mask])}
        return out

    per_fold = {}
    for fld in range(N_FOLDS):
        mask = fold_of == fld
        entry = {"n": int(mask.sum()),
                 "ridge": {"spearman": spearman(ridge_oof[mask], y[mask]), "mae": mae(ridge_oof[mask], y[mask])}}
        for c in cnns:
            entry[c] = {"spearman": spearman(cnn_oof[c][mask], y[mask]), "mae": mae(cnn_oof[c][mask], y[mask])}
        per_fold[fld] = entry

    results = {
        "target": "activity (K562 MPRA Mean_log10)",
        "split": "batch01 TRAIN partition only (9442 seq); 5-fold grouped CV",
        "seeds": SEEDS,
        "epochs": args.epochs, "batch": BATCH, "lr": LR, "wd": WD,
        "fold_assignment": fold_assignment,
        "overall": {"ridge": strata_metrics(ridge_oof)},
        "per_fold": per_fold,
        "per_seed_spearman": {},
    }
    for c in cnns:
        results["overall"][c] = strata_metrics(cnn_oof[c])
        results["per_seed_spearman"][c] = {str(sd): spearman(cnn_seed_oof[c][sd], y) for sd in SEEDS}

    # save full OOF predictions TSV (always)
    oof_df = pd.DataFrame({
        "Oligo_index": oligo, "activity": y, "fold": fold_of,
        "is_multi_member": is_multi.astype(int), "component_id": comp_ids,
        "ridge_pred": ridge_oof,
    })
    for c in cnns:
        oof_df[f"{c}_ens_pred"] = cnn_oof[c]
        for sd in SEEDS:
            oof_df[f"{c}_seed{sd}_pred"] = cnn_seed_oof[c][sd]
    oof_df.to_csv(args.report / "trackB_oof_predictions.tsv", sep="\t", index=False)

    if "legnet" in cnns and "smallcnn" in cnns:
        # strongest baseline = max(ridge, smallcnn) per metric (predeclared)
        base_sp = max(results["overall"]["ridge"]["overall"]["spearman"],
                      results["overall"]["smallcnn"]["overall"]["spearman"])
        base_mae = min(results["overall"]["ridge"]["overall"]["mae"],
                       results["overall"]["smallcnn"]["overall"]["mae"])
        lg_sp = results["overall"]["legnet"]["overall"]["spearman"]
        lg_mae = results["overall"]["legnet"]["overall"]["mae"]
        results["criterion"] = {
            "strongest_baseline_spearman": base_sp,
            "strongest_baseline_mae": base_mae,
            "legnet_spearman": lg_sp,
            "legnet_mae": lg_mae,
            "spearman_gain": lg_sp - base_sp,
            "relative_mae_improvement": 1.0 - lg_mae / base_mae,
            "spearman_criterion": bool(lg_sp - base_sp >= 0.05),
            "mae_criterion": bool(1.0 - lg_mae / base_mae >= 0.10),
            "positive_mae_folds": sum(1 for fld in range(N_FOLDS)
                                      if per_fold[fld]["legnet"]["mae"] < per_fold[fld]["smallcnn"]["mae"]),
            "multi_member_legnet_spearman": results["overall"]["legnet"]["multi_member"]["spearman"],
            "multi_member_smallcnn_spearman": results["overall"]["smallcnn"]["multi_member"]["spearman"],
            "multi_member_legnet_mae": results["overall"]["legnet"]["multi_member"]["mae"],
            "multi_member_smallcnn_mae": results["overall"]["smallcnn"]["multi_member"]["mae"],
        }
        atomic_write_json(args.report / "trackB_results.json", results)
        print(json.dumps(results["criterion"], indent=2, ensure_ascii=True), flush=True)

        # group-bootstrap paired gains (legnet vs strongest baseline)
        rng = np.random.default_rng(20261010)
        uniq_comp = np.unique(comp_ids)
        sp_diffs, mae_diffs = [], []
        for _ in range(2000):
            pick = rng.integers(0, len(uniq_comp), size=len(uniq_comp))
            mask = np.isin(comp_ids, uniq_comp[pick])
            if mask.sum() < 2:
                continue
            base_sp_b = max(spearman(ridge_oof[mask], y[mask]), spearman(cnn_oof["smallcnn"][mask], y[mask]))
            base_mae_b = min(mae(ridge_oof[mask], y[mask]), mae(cnn_oof["smallcnn"][mask], y[mask]))
            sp_diffs.append(spearman(cnn_oof["legnet"][mask], y[mask]) - base_sp_b)
            mae_diffs.append(base_mae_b - mae(cnn_oof["legnet"][mask], y[mask]))
        sp_diffs = np.asarray(sp_diffs); mae_diffs = np.asarray(mae_diffs)
        results["group_bootstrap_paired_gain"] = {
            "n_components": int(len(uniq_comp)), "n_resamples": 2000,
            "spearman_gain_mean": float(np.mean(sp_diffs)),
            "spearman_gain_ci_low": float(np.percentile(sp_diffs, 2.5)),
            "spearman_gain_ci_high": float(np.percentile(sp_diffs, 97.5)),
            "spearman_gain_pct_positive": float(np.mean(sp_diffs > 0)),
            "mae_reduction_mean": float(np.mean(mae_diffs)),
            "mae_reduction_pct_positive": float(np.mean(mae_diffs > 0)),
        }
        atomic_write_json(args.report / "trackB_results.json", results)
        print(json.dumps(results["group_bootstrap_paired_gain"], indent=2, ensure_ascii=True), flush=True)

    # hamming sensitivity audit (no retraining)
    audit = hamming_audit(seqs, fold_of)
    results["hamming_audit"] = audit
    atomic_write_json(args.report / "trackB_results.json", results)
    print("hamming audit", json.dumps(audit, ensure_ascii=True), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
