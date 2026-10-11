#!/usr/bin/env python3
"""Evaluate a finalized checkpoint on the frozen gene-level test split (clean eval).

Independent verifier that mirrors the trainer's final-eval block:
  * human test: E/S/B promoter + gene-level spearman with bootstrap 95% CI
    (targets z-scored with ck["zstats"], masked_frac=0)
  * species test: species head accuracy + ortholog_E metrics (never on the
    human-only set, where species_acc is trivially 1.0)
  * MPRA test: mpra head r2/spearman (only for checkpoints that have head_mpra)

Writes models/eval/<out>_independent_metrics.tsv so the trainer's own richer
<out>_metrics.tsv is never overwritten.

Usage: python evaluate_checkpoint.py --checkpoint models/checkpoints/<name>.pt --out <name>
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs" / "scripts"))
from train_promoter_atlas_v6 import (AtlasDataset, PromoterAtlasNet, infer_arch, evaluate,
                                     gene_level_metrics, DEVICE, N_HUMAN, OMICS_COLS, DIST_COLS,
                                     MM_H2000, MM_H400, MM_H10000, MM_SP, TF_MAT, INPUT,
                                     MpraDataset)
from v6_evaluation_utils import load_evaluation_model, standardize_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    ck = torch.load(ROOT / args.checkpoint, map_location="cpu")
    df = pd.read_csv(INPUT, sep="	", low_memory=False)
    df["split"] = df["split"].fillna("train")
    # replicate trainer preprocessing so column sets and rows match training exactly
    bad = [c for c in OMICS_COLS if c in df.columns and df[c].notna().sum() < 1000]
    if bad:
        df = df.drop(columns=bad)
    for sp in df["memmap_key"].unique():
        if sp == "human":
            continue
        p2 = MM_SP / f"{sp}_up2000_uint8.memmap"
        p4 = MM_SP / f"{sp}_sym400_uint8.memmap"
        if not p2.exists() or not p4.exists():
            df = df[df["memmap_key"] != sp]
    # z-score ALL regression targets with the SAME stats used at training
    # (E/S/B + omics + dist + ortholog_E; mirrors trainer main())
    zstats = ck.get("zstats") or {}
    df = standardize_targets(df, zstats)
    human_df = df[df["species"] == "homo_sapiens"]
    sp_df = df[df["species"] != "homo_sapiens"]
    tf_matrix = np.load(TF_MAT, mmap_mode="r") if TF_MAT.exists() else None
    mmaps = {"human": {
        2000: np.memmap(MM_H2000, dtype="uint8", mode="r", shape=(N_HUMAN, 2000)),
        400: np.memmap(MM_H400, dtype="uint8", mode="r", shape=(N_HUMAN, 400)),
    }}
    for sp in sp_df["memmap_key"].unique():
        p2 = MM_SP / f"{sp}_up2000_uint8.memmap"
        p4 = MM_SP / f"{sp}_sym400_uint8.memmap"
        if not p2.exists() or not p4.exists():
            continue
        i2 = pd.read_csv(MM_SP / f"{sp}_up2000_index.tsv", sep="	", usecols=["row_index"])
        i4 = pd.read_csv(MM_SP / f"{sp}_sym400_index.tsv", sep="	", usecols=["row_index"])
        mmaps[sp] = {
            2000: np.memmap(p2, dtype="uint8", mode="r", shape=(int(i2["row_index"].max()) + 1, 2000)),
            400: np.memmap(p4, dtype="uint8", mode="r", shape=(int(i4["row_index"].max()) + 1, 400)),
        }
    arch = infer_arch(ck)
    mm10000 = np.memmap(MM_H10000, dtype="uint8", mode="r",
                        shape=(N_HUMAN, 10000)) if arch.get("ctx10k") else None
    model = load_evaluation_model(ck, DEVICE)

    def collate(batch):
        return tuple(torch.stack([b[i] for b in batch]) for i in range(len(batch[0])))

    # ---- human test (clean, masked_frac=0) ----
    ds_test = AtlasDataset(human_df[human_df["split"] == "test"], mmaps, tf_matrix,
                           masked_frac=0.0, mm10000=mm10000)
    loader = DataLoader(ds_test, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"Evaluating human test: {len(ds_test)} rows", flush=True)
    metrics, pred, y, m = evaluate(model, loader, zstats)
    g = gene_level_metrics(human_df[human_df["split"] == "test"]["gene_id"].to_numpy(),
                           pred, y, m, n_boot=200)
    metrics.update(g)

    # ---- species test: species head + ortholog_E (never on human-only set) ----
    if len(sp_df):
        sp_split = {k: gg for k, gg in sp_df.groupby("split")}
        ds_sp_test = AtlasDataset(sp_split.get("test", sp_df.iloc[0:0]), mmaps, None,
                                  masked_frac=0.0)
        sp_loader = DataLoader(ds_sp_test, batch_size=args.batch, shuffle=False,
                               collate_fn=collate, num_workers=0)
        print(f"Evaluating species test: {len(ds_sp_test)} rows", flush=True)
        sp_m, _, _, _ = evaluate(model, sp_loader, None)
        for k, v in sp_m.items():
            metrics["sp_" + k] = v

    # ---- MPRA test head (only for checkpoints actually trained with the MPRA task:
    # gate on mpra_activity zstats, which is only persisted by MPRA-enabled runs) ----
    mpra_path = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
    if mpra_path.exists() and "mpra_activity" in zstats and "head_mpra.weight" in ck["state_dict"]:
        _mpra_seed = int(ck.get("seed", 20260912))
        mpra_ds = MpraDataset(str(mpra_path), seed=_mpra_seed)
        mpra_stats = zstats["mpra_activity"]
        mpra_ds.y = (mpra_ds.y - mpra_stats[0]) / mpra_stats[1]
        mp_loader = DataLoader(mpra_ds.subset("test"), batch_size=args.batch, shuffle=False,
                               collate_fn=collate, num_workers=0)
        print(f"Evaluating MPRA test: {len(mpra_ds.subset('test'))} rows", flush=True)
        mp_m, _, _, _ = evaluate(model, mp_loader, None)
        for k, v in mp_m.items():
            metrics["mpra_" + k] = v

    out_path = ROOT / "models" / "eval" / (args.out + "_independent_metrics.tsv")
    pd.DataFrame([metrics]).to_csv(out_path, sep="	", index=False)
    print("saved", out_path, flush=True)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
                     indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
