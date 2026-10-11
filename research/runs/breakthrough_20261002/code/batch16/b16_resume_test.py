#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interrupted-vs-uninterrupted resume test for b16_train_resume.py checkpointing.

Verifies that training interrupted by a mid-epoch checkpoint save + "process
restart" resume produces the SAME final model state as an uninterrupted run of
the same total steps. Reuses the ACTUAL _save_ckpt/_restore_rng/gen_seed/
set_seed helpers and mirrors the exact trainer loop (per-epoch sampler seed,
epoch iterator + step_in_epoch skip + step_in_epoch reset at epoch boundary).
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_train_resume import set_seed, gen_seed, _save_ckpt, _restore_rng, EPOCH_PRIME

DEVICE = "cpu"
N = 80          # 10 batches/epoch at batch_size 8
BATCH = 8
CKPT = Path(__file__).resolve().parent / "_resume_test_tmp.pt"


def epoch_seed(seed, epoch):
    return gen_seed(seed, "A", "E") + epoch * EPOCH_PRIME


def make_model():
    return torch.nn.Linear(1, 1)


def make_opt(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


def make_loader(gen):
    ds = torch.utils.data.TensorDataset(torch.arange(N))
    return DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen)


def step_fn(model, opt, batch):
    model.train()
    opt.zero_grad()
    x = batch[0].float().unsqueeze(1)
    ((model(x) - x) ** 2).mean().backward()
    opt.step()


def run_uninterrupted(n_epochs, seed):
    set_seed(seed)
    model = make_model()
    opt = make_opt(model)
    gen = torch.Generator()
    loader = make_loader(gen)
    step = 0
    for epoch in range(n_epochs):
        gen.manual_seed(epoch_seed(seed, epoch))
        it = iter(loader)
        for batch in it:
            step_fn(model, opt, batch)
            step += 1
    return {k: v.clone() for k, v in model.state_dict().items()}


def run_interrupted(n_epochs, seed, interrupt_step):
    # ---- phase 1: train epoch 0 up to interrupt_step, save ckpt, "crash" ----
    set_seed(seed)
    model = make_model()
    opt = make_opt(model)
    gen = torch.Generator()
    loader = make_loader(gen)
    epoch = 0
    step = 0
    step_in_epoch = 0
    gen.manual_seed(epoch_seed(seed, 0))
    it = iter(loader)
    for batch in it:
        step_fn(model, opt, batch)
        step += 1
        step_in_epoch += 1
        if step == interrupt_step:
            _save_ckpt(CKPT, "A", seed, "E", epoch, step, step_in_epoch,
                       model, opt, None, float("inf"), 0, DEVICE, gen)
            break
    assert step == interrupt_step and step_in_epoch == interrupt_step

    # ---- phase 2: fresh process, restore ckpt, continue same total steps ----
    model = make_model()
    opt = make_opt(model)
    gen = torch.Generator()
    loader = make_loader(gen)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    opt.load_state_dict(ck["optimizer_state"])
    _restore_rng(ck["rng"], DEVICE, gen)
    loader = make_loader(gen)          # rebuild after gen state restored
    epoch = ck["epoch"]
    step = ck["step"]
    step_in_epoch = ck["step_in_epoch"]

    while epoch < n_epochs:
        gen.manual_seed(epoch_seed(seed, epoch))
        it = iter(loader)
        for _ in range(step_in_epoch):
            next(it)
        for batch in it:
            step_fn(model, opt, batch)
            step += 1
            step_in_epoch += 1
        step_in_epoch = 0
        epoch += 1
    return {k: v.clone() for k, v in model.state_dict().items()}


def allclose(a, b):
    for k in sorted(set(a) | set(b)):
        if not torch.allclose(a[k], b[k], atol=1e-6):
            print("MISMATCH at", k, "unint:", a[k].item(), "interrupted:", b[k].item())
            return False
    return True


if __name__ == "__main__":
    seed = 20261016
    a = run_uninterrupted(2, seed)
    b = run_interrupted(2, seed, interrupt_step=4)
    ok = allclose(a, b)
    print("interrupted_vs_uninterrupted_resume:", "PASS" if ok else "FAIL")
    print("uninterrupted w/ b:",
          round(a["weight"].item(), 9), round(a["bias"].item(), 9),
          "| interrupted w/ b:",
          round(b["weight"].item(), 9), round(b["bias"].item(), 9))
    sys.exit(0 if ok else 1)
