#!/usr/bin/env python3
"""Compute per-promoter omics scalar targets from positional channel memmaps.

For every human up2000 promoter (record_id):
  cage_tss    = log1p(mean CAGE over sym400 positions [150, 250))   (TSS +/- 50 bp)
  procap_tss  = log1p(mean PROcap over sym400 positions [150, 250))
  k4me3,k27ac,k27me3,k4me1,k36me3,dnase,atac = log1p(mean over up2000 window)

Input : features/omics/channels/<track>_<win>_f16.memmap (+ promoter index)
Output: features/omics/promoter_omics_targets_v1.tsv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / "features" / "omics" / "channels"
IDX = ROOT / "promoters" / "qc" / "gencode_v49_transcript_promoter_index.tsv"
SYM_BED = ROOT / "promoters" / "bed" / "gencode_v49_transcript_promoters_sym400.bed"
OUT = ROOT / "features" / "omics" / "promoter_omics_targets_v1.tsv"

SYM_TRACKS = {"cage_tss": "CAGE", "procap_tss": "PROcap"}
WIN_TRACKS = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "DNase", "ATAC"]

def load_index(win):
    if win == "up2000":
        idx = pd.read_csv(IDX, sep="\t", low_memory=False)
        idx = idx[idx["window"] == "up2000"].drop_duplicates(subset="record_id", keep="first")
        idx = idx.sort_values(["chrom", "start0"]).reset_index(drop=True)
        return idx[["record_id"]]
    else:
        sym = pd.read_csv(SYM_BED, sep="\t", header=None,
                          names=["chrom", "start0", "end0", "record_id", "score", "strand"],
                          usecols=["chrom", "start0", "record_id"])
        sym = sym.sort_values(["chrom", "start0"]).reset_index(drop=True)
        return sym[["record_id"]]

def main():
    idx_sym = load_index("sym400")
    idx_win = load_index("up2000")
    sym_row = {r: k for k, r in enumerate(idx_sym["record_id"])}
    win_row = {r: k for k, r in enumerate(idx_win["record_id"])}
    common = sorted(set(sym_row) & set(win_row))
    print("common promoters:", len(common), flush=True)
    out = pd.DataFrame({"record_id": common})
    sym_rows = np.array([sym_row[r] for r in common], dtype=np.int64)
    win_rows = np.array([win_row[r] for r in common], dtype=np.int64)

    for name, track in SYM_TRACKS.items():
        p = CH / f"{track}_sym400_f16.memmap"
        if not p.exists():
            print("SKIP", name, flush=True)
            out[name] = np.nan
            continue
        mm = np.lib.format.open_memmap(str(p), mode="r", dtype=np.float16)
        v = mm[sym_rows][:, 150:250].mean(axis=1).astype(np.float32)
        out[name] = np.log1p(v)
        print(name, "done", flush=True)

    for track in WIN_TRACKS:
        p = CH / f"{track}_up2000_f16.memmap"
        name = track.lower()
        if not p.exists():
            print("SKIP", name, flush=True)
            out[name] = np.nan
            continue
        mm = np.lib.format.open_memmap(str(p), mode="r", dtype=np.float16)
        v = mm[win_rows].mean(axis=1).astype(np.float32)
        out[name] = np.log1p(v)
        print(name, "done", flush=True)

    out.to_csv(OUT, sep="\t", index=False)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
