#!/usr/bin/env python3
"""Build omics features for all human promoters (sym400 + up2000 windows).

1. Positional signal channels (float16 memmaps) from BED tracks:
   CAGE, H3K4me3, H3K27ac, H3K27me3, H3K4me1, H3K36me3, DNase, ATAC, PRO-cap
   -> features/omics/channels/<track>_<win>_f16.memmap  (values = log1p(score) per base)
2. (scalar targets are computed separately by build_omics_targets.py)

Coordinates: up2000 rows from the transcript index; sym400 rows from
promoters/bed/gencode_v49_transcript_promoters_sym400.bed (record_id, chrom, start0, end0).
"""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "omics" / "raw"
OUT = ROOT / "features" / "omics"
CH = OUT / "channels"
IDX = ROOT / "promoters" / "qc" / "gencode_v49_transcript_promoter_index.tsv"
SYM_BED = ROOT / "promoters" / "bed" / "gencode_v49_transcript_promoters_sym400.bed"

TRACK_FILES = [
    ("CAGE",    ["cage/fantom5_hg38_cage_peaks_phase1and2.bed.gz"]),
    ("H3K4me3", ["histone/K562_H3K4me3_*.bed.gz", "histone/GM12878_H3K4me3_*.bed.gz", "histone/HepG2_H3K4me3_*.bed.gz"]),
    ("H3K27ac", ["histone/K562_H3K27ac_*.bed.gz", "histone/GM12878_H3K27ac_*.bed.gz", "histone/HepG2_H3K27ac_*.bed.gz"]),
    ("H3K27me3", ["histone/K562_H3K27me3_*.bed.gz", "histone/GM12878_H3K27me3_*.bed.gz", "histone/HepG2_H3K27me3_*.bed.gz"]),
    ("H3K4me1", ["histone/K562_H3K4me1_*.bed.gz", "histone/GM12878_H3K4me1_*.bed.gz", "histone/HepG2_H3K4me1_*.bed.gz"]),
    ("H3K36me3", ["histone/K562_H3K36me3_*.bed.gz", "histone/GM12878_H3K36me3_*.bed.gz", "histone/HepG2_H3K36me3_*.bed.gz"]),
    ("DNase",   ["dnase/K562_*.bed.gz", "dnase/GM12878_*.bed.gz", "dnase/HepG2_*.bed.gz"]),
    ("ATAC",    ["atac/K562_*.bed.gz", "atac/GM12878_*.bed.gz", "atac/HepG2_*.bed.gz"]),
    ("PROcap",  ["procap/K562_*.bed.gz"]),
]
WINDOWS = {"up2000": 2000, "sym400": 400}

def bed_iter(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            chrom, s, e = p[0], int(p[1]), int(p[2])
            try:
                score = float(p[4])
            except (IndexError, ValueError):
                score = 1.0
            if np.isnan(score) or score <= 0:
                score = 1.0
            yield chrom, s, e, np.log1p(score)

def load_promoters():
    idx = pd.read_csv(IDX, sep="\t", low_memory=False)
    idx = idx[idx["window"] == "up2000"].drop_duplicates(subset="record_id", keep="first")
    idx = idx[["record_id", "chrom", "start0", "end0"]].sort_values(["chrom", "start0"]).reset_index(drop=True)
    sym = pd.read_csv(SYM_BED, sep="\t", header=None, names=["chrom", "start0", "end0", "record_id", "score", "strand"],
                      usecols=["chrom", "start0", "end0", "record_id"])
    sym = sym.sort_values(["chrom", "start0"]).reset_index(drop=True)
    return {"up2000": idx, "sym400": sym}

def build_track(track, files, proms):
    for win, L in WINDOWS.items():
        mm_path = CH / f"{track}_{win}_f16.memmap"
        if mm_path.exists():
            print("  (exists)", win, flush=True)
            continue
        sub = proms[win]
        n = len(sub)
        if n == 0:
            print("  (no promoters)", win, flush=True)
            continue
        mm = np.lib.format.open_memmap(str(mm_path), mode="w+", dtype=np.float16, shape=(n, L))
        by_chrom = {}
        for fpath in files:
            for chrom, s, e, v in bed_iter(fpath):
                by_chrom.setdefault(chrom, []).append((s, e, v))
        cur = 0
        for chrom, g in sub.groupby("chrom", sort=False):
            peaks = by_chrom.get(chrom)
            if peaks is None:
                mm[cur:cur + len(g)] = 0.0
                cur += len(g)
                continue
            max_end = max(int(g["end0"].max()), max(e for _, e, _ in peaks)) + 2
            arr = np.zeros(max_end, dtype=np.float64)
            for s, e, v in peaks:
                arr[s] += v
                arr[e] -= v
            np.cumsum(arr, out=arr)
            starts = g["start0"].to_numpy(dtype=np.int64)
            pos = starts[:, None] + np.arange(L)
            mm[cur:cur + len(g)] = arr[pos].astype(np.float16)
            cur += len(g)
            mm.flush()
        mm.flush()
        del mm
        print("  wrote", mm_path.name, (n, L), flush=True)

def main():
    CH.mkdir(parents=True, exist_ok=True)
    proms = load_promoters()
    for track, patterns in TRACK_FILES:
        files = []
        for pat in patterns:
            files.extend(sorted(RAW.glob(pat)))
        if not files:
            print("SKIP", track, "(no files)", flush=True)
            continue
        print("TRACK", track, len(files), "files", flush=True)
        build_track(track, files, proms)
    print("CHANNELS DONE", flush=True)

if __name__ == "__main__":
    main()
