#!/usr/bin/env python3
"""Batch05 eval: frozen-split evaluation of arms B1/B2 (single seed 20261011).

Reports gene E Spearman/R2/MAE/MSE/sample size on the IDENTICAL val/test gene
sets (same frozen split for both arms), plus component-level bootstrap CIs,
n=1/n<4/n>=4 strata, and train-defined GC/count strata with explicit quantiles.
No test-driven model selection ever happens (final.pt = epoch 12 only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b5_common import (  # noqa: E402
    atomic_write_json, metrics_dict, component_bootstrap_ci, sha256_file, spearman, r2,
)
from b5_model import GeneBagModel, n_params  # noqa: E402
from train_bag_v5 import predict_4slot  # noqa: E402
from b5_data import load_data_b5, make_memmap  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
L400 = 400


def gene_gc(mm, r400, starts):
    s = np.asarray(mm[r400], dtype=np.uint8)
    gc = ((s == 1) | (s == 2)).sum(axis=1).astype(np.float64) / float(L400)
    out = np.zeros(len(starts) - 1, dtype=np.float64)
    for g in range(len(starts) - 1):
        out[g] = gc[int(starts[g]):int(starts[g + 1])].mean()
    return out


def strata_safe(y, pred, mask):
    if int(mask.sum()) < 2:
        return {"n": int(mask.sum()), "spearman": None, "r2": None, "mae": None, "mse": None}
    return metrics_dict(y[mask], pred[mask])


def fixed_n_strata(y, pred, n):
    return {
        "n=1": strata_safe(y, pred, n == 1),
        "n=2-3": strata_safe(y, pred, (n >= 2) & (n < 4)),
        "n>=4": strata_safe(y, pred, n >= 4),
    }


def gc_strata(y, pred, gc, edges):
    return {
        "gc_low": strata_safe(y, pred, gc < edges[0]),
        "gc_mid": strata_safe(y, pred, (gc >= edges[0]) & (gc < edges[1])),
        "gc_high": strata_safe(y, pred, gc >= edges[1]),
    }


def count_strata(y, pred, n, edges):
    return {
        "count_low": strata_safe(y, pred, n < edges[0]),
        "count_mid": strata_safe(y, pred, (n >= edges[0]) & (n < edges[1])),
        "count_high": strata_safe(y, pred, n >= edges[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True, choices=["repeat_bag", "diverse_bag"])
    ap.add_argument("--candidate", required=True, help="e.g. v6.6.0-scratchgene.rc6")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = args.device
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_data_b5()
    mm = make_memmap()

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GeneBagModel().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    gc = gene_gc(mm, r400, starts)

    # train-defined strata thresholds (explicit quantiles)
    train_idx = np.flatnonzero(gene_split == "train")
    gc_edges = [float(x) for x in np.quantile(gc[train_idx], [1.0 / 3.0, 2.0 / 3.0])]
    count_edges = [float(x) for x in np.quantile(n_prom[train_idx], [1.0 / 3.0, 2.0 / 3.0])]

    results = {
        "candidate": args.candidate, "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "checkpoint_sha256": sha256_file(args.checkpoint), "n_params": n_params(model),
        "strata_thresholds": {
            "gc_tertile_edges_train": gc_edges,
            "count_tertile_edges_train": count_edges,
            "fixed_n_strata": "n=1, n=2-3 (n<4), n>=4",
            "derived_from": "TRAIN genes only",
        },
    }
    all_rows = []
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        y = gene_score_e[idx]
        gg = gene_gid[idx]
        comp = comp_dense[idx]
        nsub = n_prom[idx]
        gcsub = gc[idx]

        p4 = predict_4slot(model, mm, r400, starts, idx, device, args.arm, epoch=0)

        results[split_name] = {
            "full": metrics_dict(y, p4),
            "component_bootstrap_spearman": component_bootstrap_ci(
                y, p4, comp, spearman, n=2000, seed=args.seed + 1000),
            "component_bootstrap_r2": component_bootstrap_ci(
                y, p4, comp, r2, n=2000, seed=args.seed + 2000),
            "n_genes": int(len(y)),
            "n_components": int(len(np.unique(comp))),
            "fixed_n_strata": fixed_n_strata(y, p4, nsub),
            "gc_strata": gc_strata(y, p4, gcsub, gc_edges),
            "count_strata": count_strata(y, p4, nsub, count_edges),
        }
        for gid, sp, comp_id, nv, gcv, yv, pv in zip(gg, gene_split[idx], comp, nsub, gcsub, y, p4):
            all_rows.append({"gene_id": gid, "split": sp, "component_id": int(comp_id),
                             "n_promoters": int(nv), "gc_sym400": float(gcv),
                             "Score_E": float(yv), "pred_E_4slot": float(pv)})

    out_dir = Path(args.out_dir) if args.out_dir else (REPORT / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    tag = "%s_s%d" % (args.candidate, args.seed)
    pred_path = out_dir / ("pred_%s.tsv" % tag)
    pd.DataFrame(all_rows).to_csv(pred_path, sep="\t", index=False)
    results["predictions_file"] = str(pred_path)

    contract = {
        "candidate": args.candidate, "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "architecture": "batch03 GeneBagModel: Trunk(conv1 4->64 k7, conv2 64->64 k5, conv3 64->128 k5, BN+ReLU); head Linear(256,1) on concat(mean,max)",
        "input": "sym400 uint8 ACGTN(0..4) one-hot ACGT (N=zero); ALWAYS 4 slots per gene",
        "bag_policy": "repeat_bag: anchor p0=(epoch+g)%n x4 | diverse_bag: (p0+k)%n k=0..3",
        "inference_primary": "matched 4-slot, fixed epoch=0 rotation, mean+max pooling",
        "checkpoint_sha256": results["checkpoint_sha256"],
        "n_params": results["n_params"],
        "split": "exact/RC + detected-near-identity blocked (NOT homology-free)",
    }
    cpath = out_dir / ("contract_%s.json" % tag)
    atomic_write_json(cpath, contract)

    rpath = out_dir / ("metrics_%s.json" % tag)
    atomic_write_json(rpath, results)
    print(json.dumps({k: results[k] for k in ["candidate", "arm", "seed", "val", "test"]}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
