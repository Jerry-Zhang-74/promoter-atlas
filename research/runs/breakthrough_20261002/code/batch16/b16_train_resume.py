#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 neural training v6.15.0-multitask.rc1 (recovery revision w/ checkpointing).

A = independent E/S/B heads (3 models/seed); B = shared E/S/B heads; C = shared +
observed epigenetic auxiliary supervision.

Training algorithm (model/optimizer/loss/epochs/patience/batch/val-only selection)
is UNCHANGED from the original b16_train.py (sha256 5a0ca4dd...) that produced
A_s20261016 and A_s20261017 E/S. This recovery revision ONLY adds:

  * optimizer/RNG/sampler checkpointing (every <=100 steps AND epoch boundaries)
  * deterministic per-fit sampler generator (torch.Generator)
  * resume-from-checkpoint (does NOT restart completed epochs)
  * incremental neural_results.json merge + --arms/--seeds/--tasks filters

Original b16_train.py remains frozen on disk (unchanged); snapshot archived at
models/breakthrough_20261002/batch16/provenance/b16_train.py.5a0ca4dd05920f4f.snapshot.
This file is the recovery-runner for the REMAINING fits only (see provenance).
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
EPOCH_PRIME = 1000003
ARM_CODE = {"A": 10, "B": 20, "C": 30}
TASK_CODE = {"E": 1, "S": 2, "B": 3}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gen_seed(seed, arm, task=None):
    return int(seed) * 1000 + ARM_CODE[arm] + (TASK_CODE.get(task, 0) if task else 0)


def masked_mse(pred, y, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean((pred[mask] - y[mask]) ** 2)


def masked_mse_tensor(pred, y, m):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if m.sum() == 0:
        return 0.0
    return float(np.mean((pred[m] - y[m]) ** 2))


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
    starts = data["starts"].astype(np.int64)
    aux = lab["aux"].astype(np.float64)
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
        aux = pred["aux"]
        aux_true = batch["aux"]
        aux_loss = torch.mean((aux - aux_true) ** 2)
        return pl + NEURAL["aux_loss_coef"] * aux_loss
    return pl


def train_step(model, batch, slot_r400, mm400, device, arm, optimizer):
    g = batch["g"].numpy()
    rows = slot_r400[g].reshape(-1)
    seq = np.asarray(mm400[rows], dtype=np.uint8).reshape(-1, L400)
    x_seq = to_onehot(seq, device)
    x_feat = batch["x_feat"].to(device)
    for k in ("y_e", "m_e", "y_s", "m_s", "y_b", "m_b", "aux"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    optimizer.zero_grad()
    if arm == "A":
        task = model.task
        pred = {"task": task, "out": model(x_seq, x_feat)}
    else:
        pred = model(x_seq, x_feat)
    loss = total_loss(pred, batch, arm)
    loss.backward()
    optimizer.step()
    return float(loss.item()) * len(g), int(len(g))


def predict_all(model, arm, device, gene_indices, X, Ystd, M, slot_r400, mm400,
                aux_slots_std, use_aux, batch_size=256):
    model.eval()
    preds = {t: np.full(len(gene_indices), np.nan, dtype=np.float64) for t in TASKS}
    with torch.no_grad():
        for s in range(0, len(gene_indices), batch_size):
            idx = gene_indices[s:s + batch_size]
            rows = slot_r400[idx].reshape(-1)
            seq = np.asarray(mm400[rows], dtype=np.uint8).reshape(-1, L400)
            x_seq = to_onehot(seq, device)
            x_feat = torch.tensor(X[idx], dtype=torch.float32).to(device)
            if arm == "A":
                out = model(x_seq, x_feat)
                preds[model.task][s:s + len(idx)] = out.cpu().numpy()
            else:
                out = model(x_seq, x_feat)
                for t in TASKS:
                    preds[t][s:s + len(idx)] = out[t].cpu().numpy()
    return preds


def evaluate_metrics(pred_std, task, gene_indices, Y, Ystd, M, sc):
    idx = gene_indices
    ystd = Ystd[task][idx]
    m = M[task][idx]
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


# ---------------- checkpoint helpers ----------------
def _rng_state(device, gen):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        "sampler_gen": gen.get_state(),
    }


def _restore_rng(state, device, gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device == "cuda" and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    gen.set_state(state["sampler_gen"])


def _save_ckpt(path, arm, seed, task, epoch, step, step_in_epoch, model, optimizer,
               best_state, best_val, patience, device, gen):
    torch.save({
        "arm": arm, "seed": seed, "task": task,
        "epoch": epoch, "step": step, "step_in_epoch": step_in_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_state": best_state,
        "best_val_loss": best_val,
        "patience": patience,
        "rng": _rng_state(device, gen),
    }, path)


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _save_single_outputs(model, task, seed, device, val_idx, te_idx, X, Y, Ystd, M, sc,
                         slot_r400, mm400, aux_slots_std, use_aux, out_dir, best_val, attempt):
    torch.save({"model": model.state_dict(), "task": task, "seed": seed,
                "best_val_loss": best_val, "attempt": attempt},
               out_dir / ("model_%s.pt" % task))
    results = {}
    for split_name, split_idx in (("val", val_idx), ("test", te_idx)):
        preds = predict_all(model, "A", device, split_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        results["%s_%s" % (task, split_name)] = evaluate_metrics(
            preds[task], task, split_idx, Y, Ystd, M, sc)
    for split_name, split_idx in (("val", val_idx), ("test", te_idx)):
        preds = predict_all(model, "A", device, split_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        np.save(out_dir / ("pred_%s_%s.npy" % (task, split_name)),
                np.array([preds[task], Y[task][split_idx], M[task][split_idx]]))
    return results


def train_single(task, seed, device, args, data, sc, lab, X, Y, Ystd, M,
                 slot_r400, mm400, aux_slots_std, use_aux):
    out_dir = MODEL_OUT / ("A_s%d" % seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / ("ckpt_%s.pt" % task)
    tr = data["gene_split"].astype(str) == "train"
    va = data["gene_split"].astype(str) == "val"
    te = data["gene_split"].astype(str) == "test"
    train_idx = np.where(tr & M[task])[0]
    val_idx = np.where(va & M[task])[0]
    te_idx = np.where(te & M[task])[0]

    model = SingleTaskModel(task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL["lr"],
                                  weight_decay=NEURAL["weight_decay"])
    gen = torch.Generator()
    gen.manual_seed(gen_seed(seed, "A", task))
    ds = GeneDataset(train_idx, X, Ystd, M, aux_slots_std, use_aux)
    loader = DataLoader(ds, batch_size=NEURAL["batch_size"], shuffle=True,
                        generator=gen, collate_fn=collate, num_workers=0)

    best_val = float("inf")
    best_state = None
    patience = 0
    epoch = 0
    step = 0
    step_in_epoch = 0
    attempt = "fresh"

    if ckpt_path.exists() and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        best_val = ck["best_val_loss"]
        best_state = ck["best_state"]
        patience = ck["patience"]
        epoch = ck["epoch"]
        step = ck["step"]
        step_in_epoch = ck["step_in_epoch"]
        _restore_rng(ck["rng"], device, gen)
        model = model.to(device)
        _optimizer_to_device(optimizer, device)
        attempt = "recovery"

    while epoch < NEURAL["max_epochs"]:
        if patience >= NEURAL["patience"]:
            break
        model.train()
        gen.manual_seed(gen_seed(seed, "A", task) + epoch * EPOCH_PRIME)
        it = iter(loader)
        for _ in range(step_in_epoch):
            next(it)
        for batch in it:
            train_step(model, batch, slot_r400, mm400, device, "A", optimizer)
            step += 1
            step_in_epoch += 1
            if step % SAVE_EVERY == 0:
                _save_ckpt(ckpt_path, "A", seed, task, epoch, step, step_in_epoch,
                           model, optimizer, best_state, best_val, patience, device, gen)
        step_in_epoch = 0
        model.eval()
        preds = predict_all(model, "A", device, val_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        val_loss = masked_mse_tensor(preds[task], Ystd[task][val_idx], M[task][val_idx])
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        epoch += 1
        _save_ckpt(ckpt_path, "A", seed, task, epoch, step, step_in_epoch,
                   model, optimizer, best_state, best_val, patience, device, gen)
        if patience >= NEURAL["patience"]:
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    return _save_single_outputs(model, task, seed, device, val_idx, te_idx, X, Y, Ystd, M,
                                sc, slot_r400, mm400, aux_slots_std, use_aux, out_dir,
                                best_val, attempt), out_dir


def _save_shared_outputs(model, arm, seed, device, val_idx, te_idx, X, Y, Ystd, M, sc,
                         slot_r400, mm400, aux_slots_std, use_aux, out_dir, best_val, attempt):
    torch.save({"model": model.state_dict(), "arm": arm, "seed": seed,
                "best_val_loss": best_val, "n_aux": len(AUX_FIELDS) if use_aux else 0,
                "attempt": attempt},
               out_dir / "model.pt")
    results = {}
    for split_name, split_idx in (("val", val_idx), ("test", te_idx)):
        preds = predict_all(model, arm, device, split_idx, X, Ystd, M, slot_r400,
                            mm400, aux_slots_std, use_aux)
        for t in TASKS:
            results["%s_%s" % (t, split_name)] = evaluate_metrics(
                preds[t], t, split_idx, Y, Ystd, M, sc)
        for t in TASKS:
            np.save(out_dir / ("pred_%s_%s.npy" % (t, split_name)),
                    np.array([preds[t], Y[t][split_idx], M[t][split_idx]]))
    return results


def train_shared(arm, seed, device, args, data, sc, lab, X, Y, Ystd, M,
                 slot_r400, mm400, aux_slots_std, use_aux):
    out_dir = MODEL_OUT / ("%s_s%d" % (arm, seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    tr = data["gene_split"].astype(str) == "train"
    va = data["gene_split"].astype(str) == "val"
    te = data["gene_split"].astype(str) == "test"
    train_idx = np.where(tr)[0]
    val_idx = np.where(va)[0]
    te_idx = np.where(te)[0]

    model = build_model(arm, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL["lr"],
                                  weight_decay=NEURAL["weight_decay"])
    gen = torch.Generator()
    gen.manual_seed(gen_seed(seed, arm))
    ds = GeneDataset(train_idx, X, Ystd, M, aux_slots_std, use_aux)
    loader = DataLoader(ds, batch_size=NEURAL["batch_size"], shuffle=True,
                        generator=gen, collate_fn=collate, num_workers=0)

    best_val = float("inf")
    best_state = None
    patience = 0
    epoch = 0
    step = 0
    step_in_epoch = 0
    attempt = "fresh"

    if ckpt_path.exists() and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        best_val = ck["best_val_loss"]
        best_state = ck["best_state"]
        patience = ck["patience"]
        epoch = ck["epoch"]
        step = ck["step"]
        step_in_epoch = ck["step_in_epoch"]
        _restore_rng(ck["rng"], device, gen)
        model = model.to(device)
        _optimizer_to_device(optimizer, device)
        attempt = "recovery"

    while epoch < NEURAL["max_epochs"]:
        if patience >= NEURAL["patience"]:
            break
        model.train()
        gen.manual_seed(gen_seed(seed, arm) + epoch * EPOCH_PRIME)
        it = iter(loader)
        for _ in range(step_in_epoch):
            next(it)
        for batch in it:
            train_step(model, batch, slot_r400, mm400, device, arm, optimizer)
            step += 1
            step_in_epoch += 1
            if step % SAVE_EVERY == 0:
                _save_ckpt(ckpt_path, arm, seed, None, epoch, step, step_in_epoch,
                           model, optimizer, best_state, best_val, patience, device, gen)
        step_in_epoch = 0
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
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        epoch += 1
        _save_ckpt(ckpt_path, arm, seed, None, epoch, step, step_in_epoch,
                   model, optimizer, best_state, best_val, patience, device, gen)
        if patience >= NEURAL["patience"]:
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    return _save_shared_outputs(model, arm, seed, device, val_idx, te_idx, X, Y, Ystd, M,
                                sc, slot_r400, mm400, aux_slots_std, use_aux, out_dir,
                                best_val, attempt), out_dir


def _merge_results(results):
    path = MODEL_OUT / "neural_results.json"
    if path.exists():
        try:
            cur = json.load(open(path, encoding="utf-8"))
        except Exception:
            cur = {"candidate_version": CANDIDATE_VERSION, "batch": "batch16", "results": {}}
    else:
        cur = {"candidate_version": CANDIDATE_VERSION, "batch": "batch16", "results": {}}
    cur.setdefault("results", {})
    cur["results"].update(results)
    cur["written_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write_json(path, cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--tasks", nargs="+", default=["E", "S", "B"])
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "cpu", flush=True)
    assert torch.cuda.is_available(), "CUDA required"

    for d in (MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    data, sc, lab = load_arrays()
    X, Y, Ystd, M = standardize(data, sc)
    slot_r400 = data["slot_r400"].astype(np.int64)
    aux_slots = build_aux_slots(data, lab)
    use_aux_global = "C" in args.arms
    aux_mu, aux_sd = (build_aux_stats(aux_slots, data["gene_split"].astype(str) == "train")
                      if use_aux_global else (None, None))
    aux_slots_std = (aux_slots - aux_mu) / aux_sd if use_aux_global else aux_slots
    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))

    t0 = time.time()
    for arm in args.arms:
        for seed in args.seeds:
            key = "%s_s%d" % (arm, seed)
            if arm == "A":
                results = {}
                for task in args.tasks:
                    if not args.no_resume and (MODEL_OUT / key / ("model_%s.pt" % task)).exists():
                        print("skip (already complete)", key, task, flush=True)
                        continue
                    set_seed(seed)
                    print("training", key, task, flush=True)
                    res, out_dir = train_single(task, seed, device, args, data, sc, lab, X, Y, Ystd, M,
                                                slot_r400, mm400, aux_slots_std, use_aux=False)
                    results.update(res)
                    _merge_results({key: {"arm": arm, "seed": seed, "out_dir": str(out_dir), "metrics": results}})
                    print("done", key, task, flush=True)
            else:
                if not args.no_resume and (MODEL_OUT / key / "model.pt").exists():
                    print("skip (already complete)", key, flush=True)
                    continue
                set_seed(seed)
                use_aux = (arm == "C")
                print("training", key, flush=True)
                res, out_dir = train_shared(arm, seed, device, args, data, sc, lab, X, Y, Ystd, M,
                                            slot_r400, mm400, aux_slots_std, use_aux)
                _merge_results({key: {"arm": arm, "seed": seed, "out_dir": str(out_dir), "metrics": res}})
                print("done", key, flush=True)

    print(json.dumps({"elapsed_s": round(time.time() - t0, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
