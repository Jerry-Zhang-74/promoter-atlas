#!/usr/bin/env python3
"""Batch05 Track A v5: matched 4-slot gene-bag CNN on the frozen split (B1/B2).

Copied from verified batch04 train_bag_v2 with ONLY the data loader swapped to
the frozen sequence-component split (b5_data.load_data_b5). Identical
architecture/optimizer/seed/gene-order/4-slot inference/anchors as batch04:
  repeat_bag (B1): anchor p0=(epoch+g)%n repeated in ALL 4 slots.
  diverse_bag (B2): anchor matches B1; slots=(p0+k)%n for k=0..3 (cycle when n<4).

Fully deterministic (no dropout/stochastic sampling). Exactly 2 full runs
(seed 20261011, 12 epochs, batch 32, AdamW lr 1e-3 wd 1e-4). Evaluation uses the
FINAL checkpoint only (no epoch/heldout model selection). Periodic checkpoint.pt
is saved for crash recovery only.
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

from b5_common import atomic_write_json, set_seed, metrics_dict  # noqa: E402
from b5_model import GeneBagModel, onehot, n_params, N_SLOTS, pool_meanmax_4slot  # noqa: E402
from b5_data import load_data_b5, make_memmap, gene_order  # noqa: E402

MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
L400 = 400


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
def predict_4slot(model, mm, r400, starts, gene_idx_list, device, arm, epoch=0, batch=64):
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


@torch.no_grad()
def predict_all_promoters(model, mm, r400, starts, gene_idx_list, device, batch=64):
    model.eval()
    preds = np.full(len(gene_idx_list), np.nan, dtype=np.float64)
    for s in range(0, len(gene_idx_list), batch):
        gs = gene_idx_list[s:s + batch]
        flat, gi = [], []
        for k, g in enumerate(gs):
            lo, hi = int(starts[g]), int(starts[g + 1])
            for p in range(lo, hi):
                flat.append(r400[p])
                gi.append(k)
        if not flat:
            continue
        x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
        z = model.embed(x)
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


def initial_state_sha(model):
    import hashlib
    blob = json.dumps(
        {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()},
        sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["repeat_bag", "diverse_bag"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261011)
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
     n_genes, comp_dense, n_prom, gene_gid) = load_data_b5()
    train_mask = gene_split == "train"
    train_gene_idx = np.flatnonzero(train_mask)

    mm = make_memmap()
    device = args.device
    model = GeneBagModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch = 0
    step = 0
    global_step = 0
    init_sha = initial_state_sha(model)
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
        print("resumed epoch=%d step=%d global_step=%d" % (epoch, step, global_step), flush=True)

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
            "arm": args.arm, "lr": args.lr, "seed": args.seed,
            "n_params": n_params(model),
        }
        torch.save(ck, path)

    def run_epoch(ep, start_step=0):
        nonlocal step, global_step
        model.train()
        order = gene_order(args.seed, len(train_gene_idx), ep)
        n_steps = int(np.ceil(len(train_gene_idx) / args.batch))
        losses = []
        for bi in range(start_step, n_steps):
            if args.max_steps and global_step >= args.max_steps:
                break
            gs = train_gene_idx[order[bi * args.batch:(bi + 1) * args.batch]]
            flat = []
            for g in gs:
                sel = build_slots(args.arm, int(g), ep, starts)
                for p in sel:
                    flat.append(r400[p])
            x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
            y = torch.tensor(gene_score_e[gs], dtype=torch.float32, device=device)
            pred = pool_meanmax_4slot(model, x)
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
    history = []
    for ep in range(epoch, args.epochs):
        start_step = step if ep == epoch else 0
        tr_loss = run_epoch(ep, start_step=start_step)
        val_idx = np.flatnonzero(gene_split == "val")
        pred = predict_4slot(model, mm, r400, starts, val_idx, device, args.arm)
        y = gene_score_e[val_idx]
        m = metrics_dict(y, pred)
        m["epoch"] = ep
        m["train_loss"] = tr_loss
        history.append(m)
        print(json.dumps(m), flush=True)
        epoch = ep + 1
        step = 0
        snapshot_ckpt(out / "checkpoint.pt")
        if args.max_steps and global_step >= args.max_steps:
            break
    snapshot_ckpt(out / "final.pt")

    result = {
        "arm": args.arm, "lr": args.lr, "seed": args.seed,
        "epochs_run": epoch, "global_step": global_step,
        "n_params": n_params(model), "n_train_genes": int(len(train_gene_idx)),
        "n_val_genes": int((gene_split == "val").sum()),
        "n_test_genes": int((gene_split == "test").sum()),
        "n_train_components": int(len(np.unique(comp_dense[train_gene_idx]))),
        "n_val_components": int(len(np.unique(comp_dense[gene_split == "val"]))),
        "n_test_components": int(len(np.unique(comp_dense[gene_split == "test"]))),
        "initial_state_sha256": init_sha,
        "elapsed_s": round(time.time() - t0, 1),
        "history": history,
        "bag_policy": {
            "n_slots": N_SLOTS,
            "repeat_bag": "anchor p0=(epoch+g)%n repeated in all 4 slots",
            "diverse_bag": "anchor p0=(epoch+g)%n; slots=(p0+k)%n for k=0..3 (cycle when n<4)",
            "inference": "matched 4-slot policy, fixed epoch=0 rotation; descriptive all-promoter also reported",
            "n_lt4_note": "genes with n<4 promoters cycle so repeated promoters carry extra sampling weight in mean",
        },
    }
    atomic_write_json(out / "result.json", result)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
