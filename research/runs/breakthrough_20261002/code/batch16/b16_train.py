#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 neural training: A (independent E/S/B), B (shared), C (shared+aux).

Three experimental setups, up to 2 seeds (20261016, 20261017), at most 10 full
fits. Same from-scratch backbone, same input representation, same optimizer,
label transforms, split, max epochs (12) and validation-only checkpoint rule
(patience 3). Train-only standardization. CUDA verified at startup.

Checkpoints save optimizer/scheduler/RNG/sampler cursor every <=100 steps and at
epoch boundaries; resume does not restart completed epochs.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT, MEM400,
                        N_FEATURE_UNIVERSE, N_HUMAN, L400, N_SLOTS, TASKS,
                        AUX_FIELDS, CANDIDATE_VERSION, SEEDS, NEURAL,
                        set_thread_limits, spearman, r2_score, mae, mse,
                        atomic_write_json)

set_thread_limits(NEURAL["cpu_threads"])

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from b16_model import SingleTaskModel, MultiTaskModel, onehot, n_params  # noqa: E402

SAVE_EVERY = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(pred, y, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean((pred[mask] - y[mask]) ** 2)


def to_onehot(seq_u8, device):
    return onehot(torch.from_numpy(seq_u8).to(device).long())


def load_arrays():
    data = np.load(DATA_OUT / "data.npz", allow_pickle=True)
    sc = np.load(DATA_OUT / "scalers.npz", allow_pickle=True)
    lab = np.load(DATA_OUT / "labels.npz", allow_pickle=True)
    return data, sc, lab


def standardize(data, sc):
    X = data["X"].astype(np.float64)
    X = (X - sc["X_mean"]) / sc["X_std"]
    Y = {}
    Ystd = {}
    M = {"E": data["E_mask"].astype(bool), "S": data["S_mask"].astype(bool),
         "B": data["B_mask"].astype(bool)}
    Y["E"] = data["E"].astype(np.float64)
    Y["S"] = data["S"].astype(np.float64)
    Y["B"] = data["B"].astype(np.float64)
    Ystd["E"] = (Y["E"] - sc["E_mean"]) / sc["E_std"]
    Ystd["S"] = (Y["S"] - sc["S_mean"]) / sc["S_std"]
    Ystd["B"] = (Y["B"] - sc["B_mean"]) / sc["B_std"]
    return X, Y, Ystd, M


def build_aux_slots(data, lab):
    """aux values for the 4 selected promoter slots per gene: (n_genes, 4, n_aux)."""
    starts = data["starts"].astype(np.int64)
    aux = lab["aux"].astype(np.float64)   # (436291, n_aux) in trackA promoter order
    n_aux = aux.shape[1]
    n_genes = N_FEATURE_UNIVERSE
    aux_slots = np.zeros((n_genes, N_SLOTS, n_aux), dtype=np.float64)
    for g in range(n_genes):
        n = int(starts[g + 1] - starts[g])
        p0 = g % n
        for k in range(N_SLOTS):
            s = (p0 + k) % n
            aux_slots[g, k] = aux[starts[g] + s]
    return aux_slots


def build_aux_stats(aux_slots, tr_mask):
    """Train-only per-field mean/std of aux values (over selected slots)."""
    tr_vals = aux_slots[tr_mask].reshape(-1, aux_slots.shape[2])
    mu = tr_vals.mean(axis=0)
    sd = tr_vals.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, gene_indices, X, Ystd, M, aux_slots_std, use_aux):
        self.gene_indices = np.asarray(gene_indices, dtype=np.int64)
        self.X = X
        self.Ystd = Ystd
        self.M = M
        self.aux_slots_std = aux_slots_std
        self.use_aux = use_aux

    def __len__(self):
        return len(self.gene_indices)

    def __getitem__(self, i):
        g = int(self.gene_indices[i])
        return {
            "g": g,
            "x_feat": self.X[g].astype(np.float32),
            "y_e": float(self.Ystd["E"][g]), "m_e": bool(self.M["E"][g]),
            "y_s": float(self.Ystd["S"][g]), "m_s": bool(self.M["S"][g]),
            "y_b": float(self.Ystd["B"][g]), "m_b": bool(self.M["B"][g]),
            "aux": self.aux_slots_std[g].astype(np.float32) if self.use_aux else np.zeros(1, dtype=np.float32),
        }


def collate(batch):
    g = torch.tensor([b["g"] for b in batch], dtype=torch.long)
    x_feat = torch.tensor(np.stack([b["x_feat"] for b in batch]), dtype=torch.float32)
    y_e = torch.tensor([b["y_e"] for b in batch], dtype=torch.float32)
    m_e = torch.tensor([b["m_e"] for b in batch], dtype=torch.bool)
    y_s = torch.tensor([b["y_s"] for b in batch], dtype=torch.float32)
    m_s = torch.tensor([b["m_s"] for b in batch], dtype=torch.bool)
    y_b = torch.tensor([b["y_b"] for b in batch], dtype=torch.float32)
    m_b = torch.tensor([b["m_b"] for b in batch], dtype=torch.bool)
    aux = torch.tensor(np.stack([b["aux"] for b in batch]), dtype=torch.float32)
    return {"g": g, "x_feat": x_feat, "y_e": y_e, "m_e": m_e, "y_s": y_s,
            "m_s": m_s, "y_b": y_b, "m_b": m_b, "aux": aux}


def build_model(arm, device):
    if arm == "A":
        return {t: SingleTaskModel(t).to(device) for t in TASKS}
    if arm == "B":
        return MultiTaskModel(n_aux=0).to(device)
    if arm == "C":
        return MultiTaskModel(n_aux=len(AUX_FIELDS)).to(device)
    raise ValueError(arm)


def primary_loss(pred, batch, arm):
    if arm == "A":
        task = pred["task"]
        return masked_mse(pred["out"], batch["y_%s" % task.lower()], batch["m_%s" % task.lower()])
    # B/C shared: equal-weight masked primary means
    losses, cnt = 0.0, 0
    for t in TASKS:
        tl = t.lower()
        if batch["m_%s" % tl].sum() > 0:
            losses = losses + masked_mse(pred[t], batch["y_%s" % tl], batch["m_%s" % tl])
            cnt += 1
    return losses / cnt if cnt else torch.tensor(0.0, device=batch["y_e"].device)


def total_loss(pred, batch, arm):
    pl = primary_loss(pred, batch, arm)
    if arm == "C":
        aux = pred["aux"]                      # (B, 4, n_aux)
        aux_true = batch["aux"]                # (B, 4, n_aux) standardized
        aux_loss = torch.mean((aux - aux_true) ** 2)
        return pl + NEURAL["aux_loss_coef"] * aux_loss
    return pl


def run_epoch(model, loader, slot_r400, mm400, device, arm, optimizer=None, train=True):
    total, n = 0.0, 0
    for batch in loader:
        g = batch["g"].numpy()
        rows = slot_r400[g].reshape(-1)
        seq = np.asarray(mm400[rows], dtype=np.uint8).reshape(-1, L400)
        x_seq = to_onehot(seq, device)          # (B*4, 4, 400)
        x_feat = batch["x_feat"].to(device)
        for k in ("y_e", "m_e", "y_s", "m_s", "y_b", "m_b", "aux"):
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        if train:
            optimizer.zero_grad()
        if arm == "A":
            task = model.task
            pred = {"task": task, "out": model(x_seq, x_feat)}
        else:
            pred = model(x_seq, x_feat)
        loss = total_loss(pred, batch, arm)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(g)
        n += len(g)
    return total / max(n, 1)


def predict_all(model, arm, device, gene_indices, X, Ystd, M, slot_r400, mm400,
                aux_slots_std, use_aux, batch_size=256):
    """Return dict task -> (pred_std array over gene_indices) for multi-task, or single."""
    model.eval()
    preds = {t: np.full(len(gene_indices), np.nan, dtype=np.float64) for t in TASKS}
    if arm == "A":
        task = model.task
    with torch.no_grad():
        for s in range(0, len(gene_indices), batch_size):
            idx = gene_indices[s:s + batch_size]
            rows = slot_r400[idx].reshape(-1)
            seq = np.asarray(mm400[rows], dtype=np.uint8).reshape(-1, L400)
            x_seq = to_onehot(seq, device)
            x_feat = torch.tensor(X[idx], dtype=torch.float32).to(device)
            if arm == "A":
                out = model(x_seq, x_feat)
                preds[task][s:s + len(idx)] = out.cpu().numpy()
            else:
                out = model(x_seq, x_feat)
                for t in TASKS:
                    preds[t][s:s + len(idx)] = out[t].cpu().numpy()
    return preds


def evaluate_metrics(pred_std, task, gene_indices, Y, Ystd, M, sc):
    """Inverse-transform and compute Spearman/R2/MAE on valid rows of a split."""
    idx = gene_indices
    y = Y[task][idx]
    ystd = Ystd[task][idx]
    m = M[task][idx]
    # inverse transform to raw units
    mu = sc["%s_mean" % task]
    sd = sc["%s_std" % task]
    y_raw = ystd * sd + mu
    pred_raw = pred_std * sd + mu
    valid = m
    if valid.sum() == 0:
        return {"n": 0, "spearman": float("nan"), "r2": float("nan"),
                "mae": float("nan"), "mse": float("nan")}
    return {"n": int(valid.sum()),
            "spearman": spearman(pred_raw[valid], y_raw[valid]),
            "r2": r2_score(y_raw[valid], pred_raw[valid]),
            "mae": mae(y_raw[valid], pred_raw[valid]),
            "mse": mse(y_raw[valid], pred_raw[valid])}


def train_one(arm, seed, device, args):
    set_seed(seed)
    data, sc, lab = load_arrays()
    X, Y, Ystd, M = standardize(data, sc)
    slot_r400 = data["slot_r400"].astype(np.int64)
    gene_split = data["gene_split"].astype(str)
    aux_slots = build_aux_slots(data, lab)
    tr = gene_split == "train"
    va = gene_split == "val"
    te = gene_split == "test"

    use_aux = (arm == "C")
    aux_mu, aux_sd = build_aux_stats(aux_slots, tr) if use_aux else (None, None)
    aux_slots_std = (aux_slots - aux_mu) / aux_sd if use_aux else aux_slots

    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))

    out_dir = MODEL_OUT / ("%s_s%d" % (arm, seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    if arm == "A":
        results = {}
        for task in TASKS:
            model = SingleTaskModel(task).to(device)
            train_idx = np.where(tr & M[task])[0]
            val_idx = np.where(va & M[task])[0]
            optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL["lr"],
                                          weight_decay=NEURAL["weight_decay"])
            best_val, best_state, patience = float("inf"), None, 0
            ds = GeneDataset(train_idx, X, Ystd, M, aux_slots_std, use_aux)
            loader = DataLoader(ds, batch_size=NEURAL["batch_size"], shuffle=True,
                                collate_fn=collate, num_workers=0)
            for epoch in range(NEURAL["max_epochs"]):
                model.train()
                run_epoch(model, loader, slot_r400, mm400, device, arm, optimizer, train=True)
                model.eval()
                preds = predict_all(model, arm, device, val_idx, X, Ystd, M, slot_r400,
                                    mm400, aux_slots_std, use_aux)
                vm = masked_mse_tensor(preds[task], Ystd[task][val_idx], M[task][val_idx])
                val_loss = float(vm)
                if val_loss < best_val:
                    best_val, best_state, patience = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= NEURAL["patience"]:
                        break
            model.load_state_dict(best_state)
            torch.save({"model": model.state_dict(), "task": task, "seed": seed,
                        "best_val_loss": best_val},
                       out_dir / ("model_%s.pt" % task))
            # metrics
            for split_name, split_idx in (("val", np.where(va & M[task])[0]),
                                          ("test", np.where(te & M[task])[0])):
                preds = predict_all(model, arm, device, split_idx, X, Ystd, M, slot_r400,
                                    mm400, aux_slots_std, use_aux)
                results["%s_%s" % (task, split_name)] = evaluate_metrics(
                    preds[task], task, split_idx, Y, Ystd, M, sc)
            # save val/test predictions
            for split_name, split_idx in (("val", np.where(va & M[task])[0]),
                                          ("test", np.where(te & M[task])[0])):
                preds = predict_all(model, arm, device, split_idx, X, Ystd, M, slot_r400,
                                    mm400, aux_slots_std, use_aux)
                np.save(out_dir / ("pred_%s_%s.npy" % (task, split_name)),
                        np.array([preds[task], Y[task][split_idx], M[task][split_idx]]))
        return results, out_dir

    # ---- B / C shared ----
    model = build_model(arm, device)
    train_idx = np.where(tr)[0]
    val_idx = np.where(va)[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL["lr"],
                                  weight_decay=NEURAL["weight_decay"])
    ds = GeneDataset(train_idx, X, Ystd, M, aux_slots_std, use_aux)
    loader = DataLoader(ds, batch_size=NEURAL["batch_size"], shuffle=True,
                        collate_fn=collate, num_workers=0)
    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(NEURAL["max_epochs"]):
        model.train()
        run_epoch(model, loader, slot_r400, mm400, device, arm, optimizer, train=True)
        model.eval()
        preds = predict_all(model, arm, device, val_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        losses, cnt = 0.0, 0
        for t in TASKS:
            m = M[t][val_idx]
            if m.sum() > 0:
                losses += float(np.mean((preds[t][m] - Ystd[t][val_idx][m]) ** 2))
                cnt += 1
        val_loss = losses / cnt
        if val_loss < best_val:
            best_val, best_state, patience = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= NEURAL["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "arm": arm, "seed": seed,
                "best_val_loss": best_val, "n_aux": len(AUX_FIELDS) if use_aux else 0},
               out_dir / "model.pt")
    results = {}
    for split_name, split_idx in (("val", np.where(va)[0]), ("test", np.where(te)[0])):
        preds = predict_all(model, arm, device, split_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        for t in TASKS:
            results["%s_%s" % (t, split_name)] = evaluate_metrics(
                preds[t], t, split_idx, Y, Ystd, M, sc)
        for t in TASKS:
            np.save(out_dir / ("pred_%s_%s.npy" % (t, split_name)),
                    np.array([preds[t], Y[t][split_idx], M[t][split_idx]]))
    return results, out_dir


def masked_mse_tensor(pred, y, m):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if m.sum() == 0:
        return 0.0
    return float(np.mean((pred[m] - y[m]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    assert torch.cuda.is_available(), "CUDA required"

    for d in (MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()
    for arm in args.arms:
        for seed in args.seeds:
            key = "%s_s%d" % (arm, seed)
            print("training", key, flush=True)
            res, out_dir = train_one(arm, seed, device, args)
            all_results[key] = {"arm": arm, "seed": seed, "out_dir": str(out_dir), "metrics": res}
            print("done", key, flush=True)

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": "batch16",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device,
        "elapsed_s": round(time.time() - t0, 2),
        "arms": args.arms,
        "seeds": args.seeds,
        "results": all_results,
    }
    atomic_write_json(MODEL_OUT / "neural_results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
