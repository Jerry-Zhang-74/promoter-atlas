#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 deliverable 3: bounded PRODUCTION-PATH checkpoint/resume diagnostic.

Exercises the ACTUAL production training pieces (b16_train_resume.py, imported
read-only): real 1D CNN (SingleTaskModel, task E), AdamW, real data.npz +
promoter memmap sequence loading, target validity masks, the DataLoader sampler
with its torch.Generator, RNG capture/restore (_save_ckpt/_restore_rng), and the
production train_step. It compares a continuous run against a run that is
checkpointed mid-epoch and continued in a FRESH process.

Bounds (as declared at launch): the single CUDA job uses 12 production minibatches
(continuous 4 + continuous-repeat control 4 + interrupted 2 + resume 2) within the
<=16 bound; the separate CPU control adds 8 minibatches under a 4-thread cap.
batch_size 32. No full fits. Writes diagnostic provenance only; never touches
scientific weights. Interpretation is deliberately weak: one CUDA control pair
does not establish a band, and CPU equality supports only the tested CPU path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs" / "breakthrough_20261002" / "code" / "batch16"))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import b16_common as bc  # noqa: E402
import b16_train_resume as tr  # noqa: E402
from b16_model import SingleTaskModel  # noqa: E402

TASK = "E"
SEED = 20261016
N_STEPS = 4
INTERRUPT_AT = 2
DIAG_DIR = ROOT / "data" / "breakthrough_20261002" / "batch17" / "resume_diag"


def build_ctx(device):
    bc.set_thread_limits(4)
    tr.set_seed(SEED)
    data = np.load(bc.DATA_OUT / "data.npz", allow_pickle=True)
    sc = np.load(bc.DATA_OUT / "scalers.npz", allow_pickle=True)
    X = (data["X"].astype(np.float64) - sc["X_mean"]) / sc["X_std"]
    Ystd = {t: (data[t].astype(np.float64) - sc["%s_mean" % t]) / sc["%s_std" % t] for t in bc.TASKS}
    M = {t: data["%s_mask" % t].astype(bool) for t in bc.TASKS}
    gene_split = data["gene_split"].astype(str)
    slot_r400 = data["slot_r400"].astype(np.int64)
    mm = np.memmap(bc.MEM400, dtype="uint8", mode="r", shape=(bc.N_HUMAN, bc.L400))
    train_idx = np.where((gene_split == "train") & M[TASK])[0]

    model = SingleTaskModel(TASK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=bc.NEURAL["lr"], weight_decay=bc.NEURAL["weight_decay"])
    gen = torch.Generator()
    gen.manual_seed(tr.gen_seed(SEED, "A", TASK))
    ds = tr.GeneDataset(train_idx, X, Ystd, M, None, False)
    loader = DataLoader(ds, batch_size=bc.NEURAL["batch_size"], shuffle=True,
                        generator=gen, collate_fn=tr.collate, num_workers=0)
    return {"data": data, "sc": sc, "X": X, "Ystd": Ystd, "M": M, "slot_r400": slot_r400,
            "mm": mm, "train_idx": train_idx, "model": model, "opt": opt, "gen": gen,
            "loader": loader, "device": device}


def epoch_seed0():
    return tr.gen_seed(SEED, "A", TASK)


def run_steps(ctx, n_new, skip=0):
    model, opt, gen, loader = ctx["model"], ctx["opt"], ctx["gen"], ctx["loader"]
    model.train()
    gen.manual_seed(epoch_seed0())          # epoch 0 top, as in production train_single
    it = iter(loader)
    for _ in range(skip):
        next(it)
    for _ in range(n_new):
        batch = next(it)
        tr.train_step(model, batch, ctx["slot_r400"], ctx["mm"], ctx["device"], "A", opt)
    return model


def state_to_np(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def save_state(model, path):
    np.savez(path, **{k: v for k, v in state_to_np(model).items()})


def phase_continuous(device, out):
    ctx = build_ctx(device)
    run_steps(ctx, N_STEPS)
    save_state(ctx["model"], out)
    return {"phase": "continuous", "steps": N_STEPS, "device": device}


def phase_interrupted(device, out, ckpt):
    ctx = build_ctx(device)
    run_steps(ctx, INTERRUPT_AT)
    tr._save_ckpt(ckpt, "A", SEED, TASK, 0, INTERRUPT_AT, INTERRUPT_AT,
                  ctx["model"], ctx["opt"], None, float("inf"), 0, device, ctx["gen"])
    return {"phase": "interrupted-part1", "steps": INTERRUPT_AT, "ckpt": str(ckpt), "device": device}


def phase_resume(device, out, ckpt):
    ctx = build_ctx(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ctx["model"].load_state_dict(ck["model_state"])
    ctx["opt"].load_state_dict(ck["optimizer_state"])
    tr._restore_rng(ck["rng"], device, ctx["gen"])
    ctx["model"] = ctx["model"].to(device)
    tr._optimizer_to_device(ctx["opt"], device)
    run_steps(ctx, N_STEPS - INTERRUPT_AT, skip=ck["step_in_epoch"])
    save_state(ctx["model"], out)
    return {"phase": "interrupted-part2-resume", "carried_epoch": ck["epoch"],
            "carried_step": ck["step"], "carried_step_in_epoch": ck["step_in_epoch"], "device": device}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["continuous", "interrupted", "resume"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (DIAG_DIR / ("%s_%s.npz" % (args.phase, args.device)))
    ckpt = DIAG_DIR / ("ckpt_%s.pt" % args.device)
    if args.phase == "continuous":
        info = phase_continuous(args.device, out)
    elif args.phase == "interrupted":
        info = phase_interrupted(args.device, out, ckpt)
    else:
        info = phase_resume(args.device, out, ckpt)
    print(json.dumps({"info": info, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
