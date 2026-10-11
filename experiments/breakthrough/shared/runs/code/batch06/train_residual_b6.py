#!/usr/bin/env python3
"""Batch06 neural: from-scratch gene-bag CNN predicting OOF ridge residuals.

Identical architecture/input/update/inference to corrected batch05 B2
(diverse 4-slot), random init per seed, AdamW lr .001 wd .0001, batch 32,
12 epochs final only, MSE residual target. TWO full runs: seeds 20261011 and
20261012. Periodic checkpoint.pt every <=100 steps + epoch boundary for recovery.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_common import MODEL_DIR, DATA, atomic_write_json, set_seed  # noqa: E402
from b6_model import GeneBagModel, n_params, pool_meanmax_4slot  # noqa: E402
from b6_train_common import (  # noqa: E402
    build_slots, encode_batch, snapshot_ckpt, gene_order, ARM,
)
from b6_data import load_b6, make_memmap400  # noqa: E402

L400 = 400


def load_residual_targets(n_genes):
    z = np.load(DATA / "oof_residuals.npz")
    train_idx = z["train_idx"]
    residual_train = z["residual_train"]
    resid_by_gene = np.full(n_genes, np.nan, dtype=np.float64)
    resid_by_gene[train_idx] = residual_train
    return train_idx, resid_by_gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap (smoke tests)")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    torch.set_num_threads(4)
    np.seterr(all="raise")

    out = MODEL_DIR / args.attempt_dir
    out.mkdir(parents=True, exist_ok=True)

    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()
    train_idx, resid_by_gene = load_residual_targets(n_genes)
    mm = make_memmap400()

    device = args.device
    model = GeneBagModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch = 0
    step = 0
    global_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        epoch = int(ck["epoch"])
        step = int(ck["step"])
        global_step = int(ck.get("global_step", 0))
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if device.startswith("cuda") and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        print("resumed epoch=%d step=%d global_step=%d" % (epoch, step, global_step), flush=True)

    def run_epoch(ep, start_step=0):
        nonlocal step, global_step
        model.train()
        order = gene_order(args.seed, len(train_idx), ep)
        n_steps = int(np.ceil(len(train_idx) / args.batch))
        losses = []
        for bi in range(start_step, n_steps):
            if args.max_steps and global_step >= args.max_steps:
                break
            gs = train_idx[order[bi * args.batch:(bi + 1) * args.batch]]
            flat = []
            for g in gs:
                sel = build_slots(ARM, int(g), ep, starts)
                for p in sel:
                    flat.append(r400[p])
            x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
            y = torch.tensor(resid_by_gene[gs], dtype=torch.float32, device=device)
            pred = pool_meanmax_4slot(model, x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step = bi + 1
            global_step += 1
            if (global_step % args.ckpt_every) == 0:
                snapshot_ckpt(model, opt, ep, step, global_step, ARM, args.lr, args.seed, device,
                              out / "checkpoint.pt")
        return float(np.mean(losses)) if losses else float("nan")

    t0 = time.time()
    history = []
    for ep in range(epoch, args.epochs):
        start_step = step if ep == epoch else 0
        tr_loss = run_epoch(ep, start_step=start_step)
        m = {"epoch": ep, "train_loss": tr_loss, "global_step": global_step}
        history.append(m)
        print(json.dumps(m), flush=True)
        epoch = ep + 1
        step = 0
        snapshot_ckpt(model, opt, ep, step, global_step, ARM, args.lr, args.seed, device,
                      out / "checkpoint.pt")
        if args.max_steps and global_step >= args.max_steps:
            break
    snapshot_ckpt(model, opt, epoch, step, global_step, ARM, args.lr, args.seed, device,
                  out / "final.pt")

    result = {
        "seed": args.seed, "lr": args.lr, "epochs_run": epoch,
        "global_step": global_step, "n_params": n_params(model),
        "n_train_genes": int(len(train_idx)),
        "target": "OOF ridge residual (Score_E - OOF ridge prediction)",
        "elapsed_s": round(time.time() - t0, 1),
        "history": history,
    }
    atomic_write_json(out / "result.json", result)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
