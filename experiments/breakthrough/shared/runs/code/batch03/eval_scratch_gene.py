#!/usr/bin/env python3
"""Batch03 Track A: retrospective evaluation + inference-contract export.

Loads a trained scratch-gene checkpoint and evaluates on the FROZEN original
val/test (retrospective; never prospective). Reports gene E Spearman/R2/MAE on
full subsets and known-donor-unexposed subsets, plus gene-level bootstrap CIs.
Exports per-gene predictions and a serialized inference contract.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import atomic_write_json, set_seed, spearman, r2, mae, mse, bootstrap_spearman_ci, sha256_file  # noqa: E402
from scratch_gene_model import SinglePromoterModel, GeneBagModel, onehot, n_params  # noqa: E402
from train_scratch_gene import load_data, make_memmap, predict_genes  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch03"


def load_model(mode, device):
    return SinglePromoterModel().to(device) if mode == "single" else GeneBagModel().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", required=True, choices=["single", "bag"])
    ap.add_argument("--candidate", required=True, help="e.g. v6.6.0-scratchgene.rc2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    device = args.device
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    mm = make_memmap()

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_model(args.mode, device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])
    gene_ids = np.array([str(x) for x in (np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)["gene_id"][starts[:-1]])])

    results = {"candidate": args.candidate, "mode": args.mode, "seed": args.seed, "lr": args.lr,
               "checkpoint_sha256": sha256_file(args.checkpoint), "n_params": n_params(model)}
    all_rows = []
    for split_name in ["val", "test"]:
        split_gene_idx = np.flatnonzero(gene_split == split_name)
        pred = predict_genes(args.mode, model, mm, r400, starts, split_gene_idx, device)
        y = gene_score_e[split_gene_idx]
        gids = gene_ids[split_gene_idx]
        n_prom = np.diff(starts)[split_gene_idx]

        if split_name == "val":
            unexp = ~np.isin(gids, list(exposed_val))
        else:
            unexp = ~np.isin(gids, list(exposed_test))

        def m(yy, pp):
            return {"n": int(len(yy)), "spearman": spearman(pp, yy), "r2": r2(pp, yy),
                    "mae": mae(pp, yy), "mse": mse(pp, yy)}

        results[split_name] = {
            "full": m(y, pred),
            "unexposed": m(y[unexp], pred[unexp]),
            "n_unexposed": int(unexp.sum()),
            "bootstrap_spearman_full": bootstrap_spearman_ci(y, pred, seed=args.seed + 1000),
            "bootstrap_spearman_unexposed": bootstrap_spearman_ci(y[unexp], pred[unexp], seed=args.seed + 2000),
        }
        for gid, sp, yv, pv, npv in zip(gids, gene_split[split_gene_idx], y, pred, n_prom):
            all_rows.append({"gene_id": gid, "split": sp, "n_promoters": int(npv),
                             "Score_E": float(yv), "pred_E": float(pv)})

    tag = args.out_tag or f"{args.candidate}_s{args.seed}"
    out_dir = REPORT / "trackA_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"pred_{tag}.tsv"
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)

    # inference contract
    contract = {
        "candidate": args.candidate,
        "mode": args.mode,
        "seed": args.seed,
        "lr": args.lr,
        "trunk": "Trunk(out_dim=128) conv1(4->64,k7) conv2(64->64,k5) conv3(64->128,k5) BN+ReLU",
        "readout": ("per-promoter global-mean -> Linear(128,1); gene=mean of promoter scalars"
                    if args.mode == "single" else "per-gene concat(mean,max) embeddings -> Linear(256,1)"),
        "input": "sym400 uint8 ACGTN(0..4) -> one-hot ACGT (N=zero)",
        "inference": "predict gene E = mean over ALL promoters (single) or mean+max over ALL promoters (bag)",
        "training_scope": "human original-train genes only, random init, no v6.3 weights/distillation/ortholog labels",
        "checkpoint_sha256": results["checkpoint_sha256"],
        "n_params": results["n_params"],
    }
    contract_path = REPORT / "trackA_predictions" / f"contract_{tag}.json"
    atomic_write_json(contract_path, contract)

    res_path = REPORT / "trackA_predictions" / f"metrics_{tag}.json"
    atomic_write_json(res_path, results)
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
