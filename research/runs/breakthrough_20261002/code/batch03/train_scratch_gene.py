#!/usr/bin/env python3
"""Batch03 Track A: train scratch sequence encoder for gene-level Score_E.

Candidates (predeclared, human original-train genes only, random init):
  rc1  single-promoter : per-promoter scalar, gene-mean inference
  rc2  gene-bag        : per-gene concat(mean,max) embedding pooling -> scalar
Both share the identical Trunk capacity; only the readout differs.

Training is gene-level (one sample per gene per epoch):
  single : one promoter per gene per epoch (deterministic rotation), scalar loss
  bag    : up to 4 promoters per gene (deterministic rotation), mean+max pool
Inference always uses ALL promoters of a gene.

Resumable: checkpoints save model/optimizer/rng/epoch/step; epoch gene order is
re-derived deterministically from (seed, epoch). Save every <=250 steps.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import atomic_write_json, set_seed, spearman, r2, mae, mse  # noqa: E402
from scratch_gene_model import SinglePromoterModel, GeneBagModel, onehot, n_params  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch03"
N_HUMAN = 507365
L400 = 400
BAG_MAX = 4


def load_data():
    z = np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    split = z["split"].astype(str)
    score_e = z["Score_E"].astype(np.float64)
    starts = z["starts"].astype(np.int64)
    # gene-level (one entry per gene, first row of each gene carries the gene label)
    gene_split = split[starts[:-1]]
    gene_score_e = score_e[starts[:-1]]
    n_genes = len(starts) - 1
    return gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes


def make_memmap():
    return np.memmap(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap",
                     dtype="uint8", mode="r", shape=(N_HUMAN, L400))


def gene_order(seed, n_genes, epoch, device="cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(epoch) * 100003)
    return torch.randperm(n_genes, generator=g).numpy()


def inner_split(seed, n_train_genes):
    """Deterministic 90/10 gene holdout within original train genes."""
    order = gene_order(seed, n_train_genes, 0)
    n_val = int(round(n_train_genes * 0.1))
    val_idx = np.sort(order[:n_val])
    tr_idx = np.sort(order[n_val:])
    return tr_idx, val_idx


def build_model(mode, device):
    if mode == "single":
        m = SinglePromoterModel()
    elif mode == "bag":
        m = GeneBagModel()
    else:
        raise ValueError(mode)
    return m.to(device)


def encode_batch(mm, ridx, device):
    x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.int64).copy()).to(device)
    return onehot(x, L400)


@torch.no_grad()
def predict_genes(mode, model, mm, gene_r400, gene_starts, gene_idx_list, device, batch=64):
    """Predict gene-level Score_E over the given gene indices (inference = all promoters)."""
    model.eval()
    preds = np.full(len(gene_idx_list), np.nan, dtype=np.float64)
    for s in range(0, len(gene_idx_list), batch):
        gs = gene_idx_list[s:s + batch]
        flat, gi = [], []
        for k, g in enumerate(gs):
            lo, hi = int(gene_starts[g]), int(gene_starts[g + 1])
            for p in range(lo, hi):
                flat.append(gene_r400[p])
                gi.append(k)
        if not flat:
            continue
        x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
        if mode == "single":
            out = model(x).cpu().numpy()            # (T,)
            # mean per gene
            gi_np = np.asarray(gi, dtype=np.int64)
            sums = np.bincount(gi_np, weights=out, minlength=len(gs))
            cnts = np.bincount(gi_np, minlength=len(gs))
            pred = sums / np.maximum(cnts, 1)
        else:
            z = model.embed(x)                       # (T, emb)
            emb = z.shape[1]
            gi_t = torch.tensor(gi, dtype=torch.long, device=device)
            mean = torch.zeros(len(gs), emb, device=device)
            mean.index_add_(0, gi_t, z)
            cnts = torch.bincount(gi_t, minlength=len(gs)).clamp(min=1).unsqueeze(1)
            mean = mean / cnts
            maxv = torch.full((len(gs), emb), float("-inf"), device=device)
            maxv = maxv.scatter_reduce_(0, gi_t.unsqueeze(1).expand(-1, emb), z,
                                        reduce="amax", include_self=True)
            pred = model.head(torch.cat([mean, maxv], dim=1)).squeeze(-1).cpu().numpy()
        preds[s:s + batch] = pred
    return preds


def metrics_dict(y, p):
    return {"n": int(len(y)), "spearman": spearman(p, y), "r2": r2(p, y),
            "mae": mae(p, y), "mse": mse(p, y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single", "bag"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attempt-dir", required=True, help="unique attempt dir under models/batch03")
    ap.add_argument("--inner", action="store_true", help="train on inner-train, eval inner-val (LR selection)")
    ap.add_argument("--resume", default=None, help="path to checkpoint.pt to resume")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap (use for smoke tests)")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    torch.set_num_threads(4)
    np.seterr(all="raise")

    out = MODEL_DIR / args.attempt_dir
    out.mkdir(parents=True, exist_ok=True)

    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    train_mask = gene_split == "train"
    train_gene_idx = np.flatnonzero(train_mask)

    if args.inner:
        tr_idx, val_idx = inner_split(args.seed, len(train_gene_idx))
        inner_tr = train_gene_idx[tr_idx]
        inner_val = train_gene_idx[val_idx]
        train_genes = inner_tr
        eval_genes = inner_val
        eval_name = "inner_val"
    else:
        train_genes = train_gene_idx
        eval_genes = np.flatnonzero(gene_split == "val")
        eval_name = "val"

    mm = make_memmap()
    device = args.device
    model = build_model(args.mode, device)
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
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if device.startswith("cuda") and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"])
        print(f"resumed epoch={epoch} step={step} global_step={global_step}", flush=True)

    def snapshot_ckpt(path):
        ck = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state() if device.startswith("cuda") else None},
            "mode": args.mode, "lr": args.lr, "seed": args.seed,
            "n_params": n_params(model),
        }
        torch.save(ck, path)

    def run_epoch(epoch, train_genes, eval_genes):
        nonlocal step, global_step
        model.train()
        order = gene_order(args.seed, len(train_genes), epoch)
        n_steps = int(np.ceil(len(train_genes) / args.batch))
        losses = []
        for bi in range(n_steps):
            if args.max_steps and global_step >= args.max_steps:
                break
            gs = train_genes[order[bi * args.batch:(bi + 1) * args.batch]]
            if args.mode == "single":
                # one promoter per gene, deterministic rotation
                ridx = []
                for g in gs:
                    lo, hi = int(starts[g]), int(starts[g + 1])
                    n = hi - lo
                    p = lo + ((epoch + int(g)) % n)
                    ridx.append(r400[p])
                x = encode_batch(mm, np.asarray(ridx, dtype=np.int64), device)
                y = torch.tensor(gene_score_e[gs], dtype=torch.float32, device=device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
            else:
                flat, gi = [], []
                for k, g in enumerate(gs):
                    lo, hi = int(starts[g]), int(starts[g + 1])
                    n = hi - lo
                    if n <= BAG_MAX:
                        sel = list(range(lo, hi))
                    else:
                        s0 = (epoch * BAG_MAX + int(g)) % n
                        sel = [lo + ((s0 + t) % n) for t in range(BAG_MAX)]
                    for p in sel:
                        flat.append(r400[p])
                        gi.append(k)
                x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
                gi_t = torch.tensor(gi, dtype=torch.long, device=device)
                y = torch.tensor(gene_score_e[gs], dtype=torch.float32, device=device)
                pred = model(x, gi_t, len(gs))
                loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step = bi + 1
            global_step += 1
            if (global_step % args.ckpt_every) == 0:
                snapshot_ckpt(out / "checkpoint.pt")
        return float(np.mean(losses)) if losses else float("nan")

    t0 = time.time()
    best = None
    history = []
    for ep in range(epoch, args.epochs):
        epoch = ep
        tr_loss = run_epoch(ep, train_genes, eval_genes)
        pred = predict_genes(args.mode, model, mm, r400, starts, eval_genes, device)
        y = gene_score_e[eval_genes]
        m = metrics_dict(y, pred)
        m["epoch"] = ep
        m["train_loss"] = tr_loss
        history.append(m)
        print(json.dumps(m), flush=True)
        # best by eval spearman (train-inner selection metric)
        if best is None or m["spearman"] > best["spearman"]:
            best = m
            snapshot_ckpt(out / "best.pt")
        snapshot_ckpt(out / "checkpoint.pt")
        if args.max_steps and global_step >= args.max_steps:
            break
    snapshot_ckpt(out / "final.pt")

    result = {
        "mode": args.mode, "lr": args.lr, "seed": args.seed, "inner": args.inner,
        "epochs_run": epoch + 1, "global_step": global_step,
        "n_params": n_params(model), "eval_name": eval_name,
        "best": best, "history": history,
        "elapsed_s": round(time.time() - t0, 1),
        "n_train_genes": int(len(train_genes)), "n_eval_genes": int(len(eval_genes)),
    }
    atomic_write_json(out / "result.json", result)
    print("BEST", json.dumps(best), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
