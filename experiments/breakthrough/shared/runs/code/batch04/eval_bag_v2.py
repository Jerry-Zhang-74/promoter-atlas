#!/usr/bin/env python3
"""Batch04 Track A v2: frozen retrospective evaluation of arms B1/B2.

Primary metric = matched 4-slot inference (frozen epoch=0 rotation, same policy
for both arms). A descriptive all-promoter inference is also reported to relate
to batch03 rc2 (which used all-promoter inference); it is NOT the primary
contrast and is a train/inference mismatch for B1 by design.

Reports gene E Spearman/R2/MAE/MSE on full and historical-donor-unexposed
subsets plus gene-level bootstrap CIs. No test-driven selection ever happens.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b4_common import (  # noqa: E402
    atomic_write_json, metrics_dict, bootstrap_spearman_ci, sha256_file,
)
from b4_model import GeneBagModel, n_params  # noqa: E402
from train_bag_v2 import predict_4slot, predict_all_promoters  # noqa: E402

BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))
from train_scratch_gene import load_data, make_memmap  # noqa: E402

DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True, choices=["repeat_bag", "diverse_bag"])
    ap.add_argument("--candidate", required=True, help="e.g. v6.6.0-scratchgene.rc4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = args.device
    gene_id, r400, split, score_e, starts, gene_split, gene_score_e, n_genes = load_data()
    mm = make_memmap()

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GeneBagModel().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    prep = json.loads((DATA / "trackA_prep_meta.json").read_text(encoding="utf-8"))
    exposed_test = set(prep["exposed_test_genes"])
    exposed_val = set(prep["exposed_val_genes"])
    gids = np.array([str(x) for x in (np.load(DATA / "trackA_gene_promoters.npz", allow_pickle=True)["gene_id"][starts[:-1]])])

    results = {
        "candidate": args.candidate, "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "checkpoint_sha256": sha256_file(args.checkpoint), "n_params": n_params(model),
    }
    all_rows = []
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        y = gene_score_e[idx]
        gg = gids[idx]
        unexp = ~np.isin(gg, list(exposed_val if split_name == "val" else exposed_test))

        p4 = predict_4slot(model, mm, r400, starts, idx, device, args.arm, epoch=0)
        pall = predict_all_promoters(model, mm, r400, starts, idx, device)

        def m(yy, pp):
            return metrics_dict(yy, pp)

        results[split_name] = {
            "4slot_full": m(y, p4),
            "4slot_unexposed": m(y[unexp], p4[unexp]),
            "4slot_n_unexposed": int(unexp.sum()),
            "4slot_bootstrap_full": bootstrap_spearman_ci(y, p4, seed=args.seed + 1000),
            "4slot_bootstrap_unexposed": bootstrap_spearman_ci(y[unexp], p4[unexp], seed=args.seed + 2000),
            "allpromoter_full": m(y, pall),
            "allpromoter_unexposed": m(y[unexp], pall[unexp]),
        }
        for gid, sp, yv, pv in zip(gg, gene_split[idx], y, p4):
            all_rows.append({"gene_id": gid, "split": sp, "Score_E": float(yv),
                             "pred_E_4slot": float(pv)})

    out_dir = Path(args.out_dir) if args.out_dir else (REPORT / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "%s_s%d" % (args.candidate, args.seed)
    import pandas as pd
    pred_path = out_dir / ("pred_%s.tsv" % tag)
    pd.DataFrame(all_rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)

    contract = {
        "candidate": args.candidate,
        "arm": args.arm,
        "seed": args.seed,
        "lr": args.lr,
        "architecture": "batch03 GeneBagModel: Trunk(conv1 4->64 k7, conv2 64->64 k5, conv3 64->128 k5, BN+ReLU); head Linear(256,1) on concat(mean,max)",
        "input": "sym400 uint8 ACGTN(0..4) one-hot ACGT (N=zero); ALWAYS 4 slots per gene",
        "bag_policy": "repeat_bag: anchor p0=(epoch+g)%n x4 | diverse_bag: (p0+k)%n k=0..3",
        "inference_primary": "matched 4-slot, fixed epoch=0 rotation, mean+max pooling",
        "inference_descriptive": "all-promoter mean+max (batch03 rc2 style; train/inference mismatch for B1)",
        "training_scope": "human original-train genes only, random init, no v6.3 weights/distillation/ortholog labels",
        "checkpoint_sha256": results["checkpoint_sha256"],
        "n_params": results["n_params"],
    }
    cpath = out_dir / ("contract_%s.json" % tag)
    atomic_write_json(cpath, contract)

    rpath = out_dir / ("metrics_%s.json" % tag)
    atomic_write_json(rpath, results)
    print(json.dumps({k: results[k] for k in ["candidate", "arm", "seed", "val", "test"]}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
