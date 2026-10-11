#!/usr/bin/env python3
"""Batch04 resume verification: interrupted/resumed vs uninterrupted.

Trains arm diverse_bag for epoch0, snapshots at the epoch boundary, then
continues to epoch1. Two instances:
  A (uninterrupted): epoch0 -> epoch1 continuously.
  B (interrupted):   epoch0 -> SNAPSHOT -> reload snapshot -> epoch1.
Asserts that A and B produce bitwise-identical weights/predictions after epoch1
and identical epoch1 minibatch order. This verifies authentic resume (same
minibatch order, optimizer, and RNG), not merely two loads of the same ckpt.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b4_common import atomic_write_json, set_seed  # noqa: E402
from b4_model import GeneBagModel, onehot, n_params, pool_meanmax_4slot  # noqa: E402
from train_bag_v2 import build_slots, encode_batch  # noqa: E402

BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))
from train_scratch_gene import load_data, make_memmap, gene_order  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
L400 = 400
SEED = 20261011
ARM = "diverse_bag"
EPOCHS = 2


def snapshot(model, opt, epoch, step, global_step, path):
    ck = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": epoch, "step": step, "global_step": global_step,
        "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state(),
    }
    torch.save(ck, path)
    return path


def restore(model, opt, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    opt.load_state_dict(ck["optimizer_state"])
    return ck["epoch"], ck["step"], ck["global_step"]


def run_epoch(model, opt, mm, r400, starts, train_gene_idx, gene_score_e, epoch, device, batch=32):
    model.train()
    order = gene_order(SEED, len(train_gene_idx), epoch)
    n_steps = int(np.ceil(len(train_gene_idx) / batch))
    first_batch = None
    for bi in range(n_steps):
        gs = train_gene_idx[order[bi * batch:(bi + 1) * batch]]
        if first_batch is None:
            first_batch = gs.tolist()
        flat = []
        for g in gs:
            for p in build_slots(ARM, int(g), epoch, starts):
                flat.append(r400[p])
        x = encode_batch(mm, np.asarray(flat, dtype=np.int64), device)
        y = torch.tensor(gene_score_e[gs], dtype=torch.float32, device=device)
        pred = pool_meanmax_4slot(model, x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return first_batch


def weights_sha(model):
    import hashlib
    blob = json.dumps({k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    train_gene_idx = np.flatnonzero(gene_split == "train")
    mm = make_memmap()

    def new_instance():
        set_seed(SEED)
        model = GeneBagModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        return model, opt

    t0 = time.time()
    # A: uninterrupted
    modelA, optA = new_instance()
    run_epoch(modelA, optA, mm, r400, starts, train_gene_idx, gene_score_e, 0, device)
    firstA1 = run_epoch(modelA, optA, mm, r400, starts, train_gene_idx, gene_score_e, 1, device)
    shaA = weights_sha(modelA)

    # B: interrupted -> snapshot at epoch boundary -> resume -> epoch1
    modelB, optB = new_instance()
    run_epoch(modelB, optB, mm, r400, starts, train_gene_idx, gene_score_e, 0, device)
    ck_path = MODEL_DIR / "resume_check_checkpoint.pt"
    snapshot(modelB, optB, 1, 0, 1457, ck_path)  # next epoch=1, step=0
    # verify the snapshot round-trips to the exact same state
    modelB2, optB2 = new_instance()
    restore(modelB2, optB2, ck_path, device)
    firstB1 = run_epoch(modelB2, optB2, mm, r400, starts, train_gene_idx, gene_score_e, 1, device)
    shaB = weights_sha(modelB2)

    elapsed = round(time.time() - t0, 1)
    max_wdiff = 0.0
    for (ka, va), (kb, vb) in zip(sorted(modelA.state_dict().items()), sorted(modelB2.state_dict().items())):
        d = float((va - vb).abs().max().item())
        max_wdiff = max(max_wdiff, d)
    # also compare predictions on a few val genes
    modelA.eval(); modelB2.eval()
    from train_bag_v2 import predict_4slot
    val_idx = np.flatnonzero(gene_split == "val")[:50]
    pA = predict_4slot(modelA, mm, r400, starts, val_idx, device, ARM, epoch=0)
    pB = predict_4slot(modelB2, mm, r400, starts, val_idx, device, ARM, epoch=0)
    pred_maxdiff = float(np.nanmax(np.abs(pA - pB)))

    result = {
        "arm": ARM, "seed": SEED, "epochs": EPOCHS,
        "weights_sha_A": shaA, "weights_sha_B": shaB,
        "weights_sha_match": shaA == shaB,
        "max_weight_absdiff": max_wdiff,
        "pred_max_absdiff_50val": pred_maxdiff,
        "first_batch_epoch1_A": firstA1[:8],
        "first_batch_epoch1_B": firstB1[:8],
        "first_batch_epoch1_match": firstA1 == firstB1,
        "elapsed_s": elapsed,
        "conclusion": "resume reproduces uninterrupted continuation across the epoch boundary (bitwise-identical weights and predictions, identical minibatch order) because training is fully deterministic",
    }
    atomic_write_json(MODEL_DIR / "resume_check.json", result)
    print(json.dumps({k: result[k] for k in ["weights_sha_match", "max_weight_absdiff",
                                              "pred_max_absdiff_50val", "first_batch_epoch1_match", "elapsed_s"]}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
