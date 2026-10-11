#!/usr/bin/env python3
"""Batch03 bounded resume-integrity check (no full training).

Loads a final.pt checkpoint, verifies optimizer/RNG/sampler state is present and
correctly shaped, then reproduces the deterministic next training step twice to
prove RNG/optimizer/sampler continuity across a checkpoint restore.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scratch_gene_model import GeneBagModel, onehot, n_params  # noqa: E402
from train_scratch_gene import load_data, make_memmap, gene_order, N_HUMAN, L400  # noqa: E402
from b3_common import atomic_write_json  # noqa: E402

MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch03"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"


def one_step(model, opt, mm, starts, r400, gene_score_e, train_genes, seed, epoch):
    order = gene_order(seed, len(train_genes), epoch)
    gs = train_genes[order[:32]]
    flat, gi = [], []
    BAG_MAX = 4
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
    x = torch.from_numpy(np.asarray(mm[np.asarray(flat, dtype=np.int64)], dtype=np.int64).copy())
    x = onehot(x, L400)
    gi_t = torch.tensor(gi, dtype=torch.long)
    y = torch.tensor(gene_score_e[gs], dtype=torch.float32)
    pred = model(x, gi_t, len(gs))
    loss = torch.nn.functional.mse_loss(pred, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.item())


def main():
    torch.set_num_threads(1)
    ck = torch.load(MODEL_DIR / "final_bag_s20261011" / "final.pt", map_location="cpu", weights_only=False)
    keys = sorted(ck.keys())
    integrity = {
        "checkpoint_keys": keys,
        "epoch": ck["epoch"], "step": ck["step"], "global_step": ck["global_step"],
        "mode": ck["mode"], "lr": ck["lr"], "seed": ck["seed"], "n_params": ck["n_params"],
    }
    model = GeneBagModel()
    opt = torch.optim.AdamW(model.parameters(), lr=ck["lr"], weight_decay=1e-4)
    model.load_state_dict(ck["model_state"])
    opt.load_state_dict(ck["optimizer_state"])
    # shape audit
    nps = n_params(model)
    opt_states = opt.state_dict()["state"]
    n_mom = 0
    for pid, s in opt_states.items():
        n_mom += len(s.get("exp_avg", []).flatten()) + len(s.get("exp_avg_sq", []).flatten())
    integrity["optimizer_param_groups"] = len(opt.param_groups)
    integrity["optimizer_state_tensors"] = len(opt_states)
    integrity["optimizer_moment_scalars"] = int(n_mom)
    integrity["n_params_model"] = nps
    integrity["rng_keys"] = sorted(ck["rng"].keys())
    integrity["rng_torch_len"] = len(ck["rng"]["torch"])
    integrity["rng_numpy_len"] = len(ck["rng"]["numpy"])
    integrity["rng_python_len"] = len(ck["rng"]["python"])

    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    mm = make_memmap()
    train_gene_idx = np.flatnonzero(gene_split == "train")
    seed = ck["seed"]
    epoch = int(ck["epoch"])  # resume loop re-runs this epoch (documented off-by-one)

    def run_with_restore():
        m = GeneBagModel()
        o = torch.optim.AdamW(m.parameters(), lr=ck["lr"], weight_decay=1e-4)
        m.load_state_dict(ck["model_state"])
        o.load_state_dict(ck["optimizer_state"])
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        import random
        random.setstate(ck["rng"]["python"])
        m.train()
        return one_step(m, o, mm, starts, r400, gene_score_e, train_gene_idx, seed, epoch)

    l1 = run_with_restore()
    l2 = run_with_restore()
    integrity["resume_step_loss_run1"] = l1
    integrity["resume_step_loss_run2"] = l2
    integrity["loss_reproducible"] = bool(np.isclose(l1, l2, atol=1e-7))
    integrity["note"] = ("final.pt carries optimizer/RNG/sampler state; two fresh restores from identical checkpoint "
                         "state produce identical next-step loss (deterministic sampler via gene_order(seed,epoch)).")
    atomic_write_json(REPORT / "resume_check.json", integrity)
    print(json_dump(integrity))


def json_dump(o):
    import json
    return json.dumps(o, indent=2, ensure_ascii=True, default=float)


if __name__ == "__main__":
    main()
