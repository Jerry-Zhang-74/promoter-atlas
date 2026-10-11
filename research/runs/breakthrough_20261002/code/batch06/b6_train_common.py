#!/usr/bin/env python3
"""Batch06 neural training helpers (residual gene-bag CNN, diverse_bag 4-slot)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_model import GeneBagModel, onehot, n_params, N_SLOTS, pool_meanmax_4slot  # noqa: E402
from b6_data import load_b6, make_memmap400  # noqa: E402

L400 = 400
ARM = "diverse_bag"


def build_slots(arm, g, epoch, starts):
    lo, hi = int(starts[g]), int(starts[g + 1])
    n = hi - lo
    p0 = (int(epoch) + int(g)) % n
    if arm == "repeat_bag":
        return [lo + p0] * N_SLOTS
    return [lo + ((p0 + k) % n) for k in range(N_SLOTS)]


def encode_batch(mm, ridx, device):
    x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.int64).copy()).to(device)
    return onehot(x, L400)


@torch.no_grad()
def predict_4slot(model, mm, r400, starts, gene_idx_list, device, arm=ARM, epoch=0, batch=64):
    model.eval()
    preds = np.full(len(gene_idx_list), np.nan, dtype=np.float64)
    for s in range(0, len(gene_idx_list), batch):
        gs = gene_idx_list[s:s + batch]
        flat = []
        for g in gs:
            sel = build_slots(arm, int(g), epoch, starts)
            for p in sel:
                flat.append(r400[p])
        if not flat:
            continue
        x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
        pred = pool_meanmax_4slot(model, x).cpu().numpy().astype(np.float64)
        preds[s:s + batch] = pred
    return preds


def snapshot_ckpt(model, opt, epoch, step, global_step, arm, lr, seed, device, path):
    ck = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": epoch, "step": step, "global_step": global_step,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": None,
                "cuda": torch.cuda.get_rng_state() if device.startswith("cuda") else None},
        "arm": arm, "lr": lr, "seed": seed, "n_params": n_params(model),
    }
    import random
    ck["rng"]["python"] = random.getstate()
    torch.save(ck, path)


def gene_order(seed, n_genes, epoch):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(epoch) * 100003)
    return torch.randperm(n_genes, generator=g).numpy()
