#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 bounded A/B loss comparison (4 full fits, single GPU, batch<=32).

Same LEGNET backbone, preprocessing, seed, initializer, optimizer and data order
for both arms; ONLY the loss differs:
  A: MSE of train-standardized measured activity.
  B: MSE + 0.1 * mean softplus(-sign(z_i - z_j) * (p_i - p_j)) over within-minibatch
     pairs with |z_i - z_j| >= 0.5 (standardized labels, TRAINING rows only); zero if
     no qualifying pairs.
Seeds 20261023 and 20261024 (2 seeds x 2 arms = 4 fits). AdamW lr 3e-4 wd 1e-4,
batch 32, max 20 epochs, val early stop patience 5 on Spearman (ties -> lower MAE
then earlier epoch). Per-fit cap 45 GPU-min; aggregate cap 3 GPU-h.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from b21_common import (BATCH02_CODE, DATA_OUT, MODEL_OUT, REPORT_OUT, atomic_write_json,
                        now_local, spearman, mae, sha256_file, set_thread_limits)

sys.path.insert(0, str(BATCH02_CODE))
import trackB_grouped_cv as t  # noqa: E402  (frozen architecture, READ-ONLY)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_evidence_note():
    """Explicit CUDA-execution evidence required by batch21/GPU_REQUIRED.txt."""
    import os
    from pathlib import Path
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to run the bounded fits on CPU")
    import subprocess
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    ev = {"torch": torch.__version__, "cuda_version": torch.version.cuda, "cuda_available": True,
          "device_index": dev, "device_name": name, "compute_capability": list(cap),
          "pid": os.getpid(), "train_pid": os.getpid(), "device": "cuda:0",
          "note": "all four neural fits run on cuda:0; no CPU/DirectML/integrated-GPU fallback"}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        ev["nvidia_smi_start"] = q.stdout.strip()
    except Exception as e:
        ev["nvidia_smi_start"] = "unavailable: %r" % (e,)
    return ev


def _smi():
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        return q.stdout.strip()
    except Exception as e:
        return "unavailable: %r" % (e,)
SEEDS = (20261023, 20261024)
ARMS = ("A", "B")
LR, WD, BATCH = 3e-4, 1e-4, 32
MAX_EPOCHS = 20
PATIENCE = 5
FIT_CAP_S = 45 * 60


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranking_loss(pred, z):
    """softplus(-sign(z_i-z_j)*(p_i-p_j)) over within-minibatch pairs |dz|>=0.5."""
    n = z.shape[0]
    if n < 2:
        return torch.zeros((), device=z.device)
    zi = z.unsqueeze(1); zj = z.unsqueeze(0)
    dz = zi - zj
    mask = dz.abs() >= 0.5
    if int(mask.sum()) == 0:
        return torch.zeros((), device=z.device)
    pi = pred.unsqueeze(1); pj = pred.unsqueeze(0)
    s = torch.sign(dz)
    term = F.softplus(-s * (pi - pj))
    return term[mask].mean()


def train_one(arm, seed, x_tr, z_tr, x_va, y_va, mu, sd, epochs=MAX_EPOCHS):
    """One fit. Returns (model_state, log, best_epoch, best_val, truncated)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _set_seed(seed)                      # SAME init for both arms at a given seed
    model = t.build_cnn("legnet").to(DEVICE)
    assert next(model.parameters()).is_cuda, "model parameters are not on CUDA"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, math.ceil(len(x_tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps,
                                                pct_start=0.1)
    n = len(x_tr)
    best = {"spearman": -np.inf, "mae": np.inf, "epoch": -1, "state": None}
    log = []
    t0 = time.time()
    truncated = False
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)   # same order for both arms at a given seed
        tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = torch.from_numpy(x_tr[idx]).long().to(DEVICE)
            zb = torch.from_numpy(z_tr[idx]).float().to(DEVICE)
            assert xb.is_cuda and zb.is_cuda, "batch tensors are not on CUDA"
            pred = model(xb)[:, 0]
            mse = torch.mean((pred - zb) ** 2)
            if arm == "A":
                loss = mse
            else:
                loss = mse + 0.1 * ranking_loss(pred, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        pva = t.predict_cnn(model, x_va) * sd + mu     # back to RAW activity units
        sp = spearman(pva, y_va); me = mae(pva, y_va)
        log.append({"epoch": epoch, "train_loss": tot / max(nb, 1), "val_spearman": sp,
                    "val_mae": me, "elapsed_s": round(time.time() - t0, 1)})
        better = sp > best["spearman"] + 1e-9 or (abs(sp - best["spearman"]) <= 1e-9 and me < best["mae"] - 1e-12)
        if better:
            best = {"spearman": sp, "mae": me, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if epoch - best["epoch"] >= PATIENCE:
            break
        if time.time() - t0 > FIT_CAP_S:
            truncated = True
            break
    return best, log, truncated


def run(epochs=MAX_EPOCHS, arms=ARMS, seeds=SEEDS):
    set_thread_limits(4)
    torch.set_num_threads(4)
    import pandas as pd
    from b21_common import MPRA_SUPERVISED
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    y = df["activity"].to_numpy(dtype=np.float64)
    g = np.load(DATA_OUT / "groups.npz", allow_pickle=True)
    split_of = g["split_of"]
    comp_id = g["comp_id"]
    x_all = t.encode_sequences(seqs).astype(np.int64)
    tr = np.where(split_of == 0)[0]; va = np.where(split_of == 1)[0]; te = np.where(split_of == 2)[0]
    mu = float(y[tr].mean()); sd = float(y[tr].std())
    z = (y - mu) / sd                      # SAME train-only normalization for both arms
    import os as _os
    gpu_ev = gpu_evidence_note()
    accounting = {"mu_train": mu, "sd_train": sd, "n_train": int(len(tr)),
                  "n_val": int(len(va)), "n_test": int(len(te)), "device": DEVICE,
                  "gpu": torch.cuda.get_device_name(0),
                  "train_pid": _os.getpid(), "gpu_evidence": gpu_ev,
                  "lr": LR, "wd": WD, "batch": BATCH, "max_epochs": epochs,
                  "patience": PATIENCE, "fit_cap_s": FIT_CAP_S, "fits": []}
    agg0 = time.time()
    for arm in arms:
        for seed in seeds:
            t0 = time.time(); started = now_local()
            smi_before = _smi()
            best, log, trunc = train_one(arm, seed, x_all[tr], z[tr].astype(np.float32),
                                         x_all[va], y[va], mu, sd, epochs=epochs)
            smi_after = _smi()
            param_device = "cuda" if torch.cuda.is_available() else "cpu"
            # recover best weights and predict -- convert z-units back to RAW activity
            model = t.build_cnn("legnet").to(DEVICE)
            model.load_state_dict(best["state"]); model.eval()
            pv = t.predict_cnn(model, x_all[va]) * sd + mu
            pt = t.predict_cnn(model, x_all[te]) * sd + mu
            np.savez(DATA_OUT / ("preds_%s_seed%d.npz" % (arm, seed)),
                     val_idx=va, test_idx=te, val_pred=pv, test_pred=pt,
                     val_true=y[va], test_true=y[te])
            torch.save({"model": best["state"], "arm": arm, "seed": seed,
                        "best_epoch": best["epoch"], "val_spearman": best["spearman"]},
                       MODEL_OUT / ("ckpt_%s_seed%d.pt" % (arm, seed)))
            rec = {"arm": arm, "seed": seed, "started_local": started, "elapsed_s": round(time.time() - t0, 1),
                   "epochs_run": len(log), "best_epoch": best["epoch"],
                   "best_val_spearman": best["spearman"], "best_val_mae": best["mae"],
                   "truncated_by_cap": bool(trunc), "fully_converged_20ep": bool(len(log) >= epochs),
                   "device": "cuda:0", "param_device": param_device,
                   "train_pid": _os.getpid(), "nvidia_smi_before": smi_before,
                   "nvidia_smi_after": smi_after,
                   "log": log}
            accounting["fits"].append(rec)
            atomic_write_json(MODEL_OUT / ("fit_%s_seed%d.json" % (arm, seed)), rec)
            print("[fit] arm=%s seed=%d best_ep=%d val_sp=%.4f mae=%.4f epochs=%d t=%.1fs%s"
                  % (arm, seed, best["epoch"], best["spearman"], best["mae"], len(log),
                     time.time() - t0, " TRUNC" if trunc else ""), flush=True)
    accounting["aggregate_elapsed_s"] = round(time.time() - agg0, 1)
    accounting["aggregate_gpu_min"] = round((time.time() - agg0) / 60.0, 2)
    accounting["n_full_fits"] = len(accounting["fits"])
    accounting["generated_at_local"] = now_local()
    accounting["source_sha256"] = {"trackB_grouped_cv.py": sha256_file(__import__("pathlib").Path(t.__file__)),
                                   "groups.npz": sha256_file(DATA_OUT / "groups.npz")}
    atomic_write_json(REPORT_OUT / "fit_accounting.json", accounting)
    atomic_write_json(REPORT_OUT / "training_gpu_evidence.json",
                      {"gpu_evidence": gpu_ev,
                       "fits": [{"arm": r["arm"], "seed": r["seed"], "device": r["device"],
                                 "param_device": r["param_device"], "train_pid": r["train_pid"],
                                 "started_local": r["started_local"], "epochs_run": r["epochs_run"],
                                 "best_epoch": r["best_epoch"],
                                 "nvidia_smi_before": r["nvidia_smi_before"],
                                 "nvidia_smi_after": r["nvidia_smi_after"]}
                                for r in accounting["fits"]],
                       "note": (">=300M-parameter-scale? no; compact legnet. All four fits executed on "
                                "cuda:0 with asserted CUDA parameter/batch placement; no CPU fallback.")})
    return accounting
