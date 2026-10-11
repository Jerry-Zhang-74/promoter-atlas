#!/usr/bin/env python3
"""Batch03 Track A replication eval (fast, no bootstrap).

Same prediction contract as eval_scratch_gene.py but omits the slow bootstrap
CIs (already reported for the pilot). Reports val/test full + unexposed
Spearman/R2/MAE/MSE and exports predictions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b3_common import atomic_write_json, spearman, r2, mae, mse, sha256_file  # noqa: E402
from scratch_gene_model import SinglePromoterModel, GeneBagModel, n_params  # noqa: E402
from train_scratch_gene import load_data, make_memmap, predict_genes  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", required=True, choices=["single", "bag"])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    mm = make_memmap()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SinglePromoterModel().to(device) if args.mode == "single" else GeneBagModel().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])
    gids = np.array([str(x) for x in (np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)["gene_id"][starts[:-1]])])

    results = {"candidate": args.candidate, "mode": args.mode, "seed": args.seed, "lr": args.lr,
               "checkpoint_sha256": sha256_file(args.checkpoint), "n_params": n_params(model)}
    rows = []
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        pred = predict_genes(args.mode, model, mm, r400, starts, idx, device)
        y = gene_score_e[idx]
        gg = gids[idx]
        unexp = ~np.isin(gg, list(exposed_val if split_name == "val" else exposed_test))

        def m(yy, pp):
            return {"n": int(len(yy)), "spearman": spearman(pp, yy), "r2": r2(pp, yy),
                    "mae": mae(pp, yy), "mse": mse(pp, yy)}
        results[split_name] = {"full": m(y, pred), "unexposed": m(y[unexp], pred[unexp]),
                               "n_unexposed": int(unexp.sum())}
        for gid, sp, yv, pv in zip(gg, gene_split[idx], y, pred):
            rows.append({"gene_id": gid, "split": sp, "Score_E": float(yv), "pred_E": float(pv)})

    import pandas as pd
    out_dir = REPORT / "trackA_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.candidate}_s{args.seed}"
    pred_path = out_dir / f"pred_{tag}.tsv"
    pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)
    atomic_write_json(out_dir / f"metrics_{tag}.json", results)
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
