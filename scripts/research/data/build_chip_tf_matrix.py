#!/usr/bin/env python3
"""Build promoter x TF ChIP occupancy matrix from ReMap 2022 non-redundant peaks (hg38).

Input : data/omics/raw/chip/remap2022_nr_macs2_hg38_v1_0.bed.gz
        promoters/qc/gencode_v49_transcript_promoter_index.tsv (up2000 rows)
Output: features/omics/promoter_chip_tf_matrix_v1_uint8.memmap   (n_promoters, n_tf)
        features/omics/chip_tf_manifest_v1.tsv  (tf, n_peaks, frac_promoters)
        features/omics/chip_tf_row_index_v1.tsv (record_id, row_index)

Overlap = a peak of TF t overlaps the promoter's up2000 window.
Matrix stores 0/1 overlap (uint8). TFs kept if they overlap >= 0.5% of promoters,
capped at the top 400 by coverage.
"""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMAP = ROOT / "data" / "omics" / "raw" / "chip" / "remap2022_nr_macs2_hg38_v1_0.bed.gz"
INDEX = ROOT / "promoters" / "qc" / "gencode_v49_transcript_promoter_index.tsv"
OUT = ROOT / "features" / "omics"
MM = OUT / "promoter_chip_tf_matrix_v1_uint8.memmap"
MAN = OUT / "chip_tf_manifest_v1.tsv"
ROWIDX = OUT / "chip_tf_row_index_v1.tsv"

def main():
    idx = pd.read_csv(INDEX, sep="\t", low_memory=False)
    idx = idx[idx["window"] == "up2000"].drop_duplicates(subset="record_id", keep="first").reset_index(drop=True)
    n = len(idx)
    print("promoters:", n, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # pass 1: collect per-TF peaks
    peaks = {}
    with gzip.open(REMAP, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            chrom, s, e, tf = p[0], int(p[1]), int(p[2]), p[3].split(":")[0]
            peaks.setdefault(tf, []).append((chrom, s, e))
            if i % 5000000 == 0:
                print("  read", i, "lines;", len(peaks), "TFs", flush=True)
    print("TFs total:", len(peaks), flush=True)

    # per-chrom interval arrays per TF
    tfs_all = sorted(peaks.keys())
    print("counting overlaps...", flush=True)
    counts = {}
    chroms_prom = idx["chrom"].to_numpy()
    starts_prom = idx["start0"].to_numpy(dtype=np.int64)
    ends_prom = idx["end0"].to_numpy(dtype=np.int64)
    for tf in tfs_all:
        pl = peaks[tf]
        by_chrom = {}
        for chrom, s, e in pl:
            by_chrom.setdefault(chrom, []).append((s, e))
        cnt = np.zeros(n, dtype=np.uint32)
        for chrom, ivs in by_chrom.items():
            m = chroms_prom == chrom
            if not m.any():
                continue
            ivs.sort()
            ss = np.array([s for s, _ in ivs], dtype=np.int64)
            ee = np.array([e for _, e in ivs], dtype=np.int64)
            ee_sorted = np.sort(np.array([e for _, e in ivs], dtype=np.int64))
            # overlapping = (start < p_end) - (end <= p_start)
            a = np.searchsorted(ss, ends_prom[m], side="left")   # starts < p_end
            b = np.searchsorted(ee_sorted, starts_prom[m], side="right")  # ends <= p_start
            cnt[m] = (a - b).astype(np.uint32)
        counts[tf] = int((cnt > 0).sum())
    frac = {tf: c / n for tf, c in counts.items()}
    kept = sorted([tf for tf in tfs_all if frac[tf] >= 0.005], key=lambda t: -frac[t])[:400]
    print("kept TFs:", len(kept), flush=True)
    kept_sorted = sorted(kept)
    tf_index = {tf: k for k, tf in enumerate(kept_sorted)}
    print("min frac:", min(frac[t] for t in kept), flush=True)

    # pass 2: fill matrix
    mm = np.lib.format.open_memmap(str(MM), mode="w+", dtype=np.uint8, shape=(n, len(kept_sorted)))
    mm[:] = 0
    for tf in kept_sorted:
        k = tf_index[tf]
        pl = peaks[tf]
        by_chrom = {}
        for chrom, s, e in pl:
            by_chrom.setdefault(chrom, []).append((s, e))
        for chrom, ivs in by_chrom.items():
            m = chroms_prom == chrom
            if not m.any():
                continue
            ivs.sort()
            ss = np.array([s for s, _ in ivs], dtype=np.int64)
            ee = np.array([e for _, e in ivs], dtype=np.int64)
            ee_sorted = np.sort(ee)
            a = np.searchsorted(ss, ends_prom[m], side="left")
            b = np.searchsorted(ee_sorted, starts_prom[m], side="right")
            rows = np.flatnonzero(m)[(a - b) > 0]
            mm[rows, k] = 1
        if k % 50 == 0:
            mm.flush()
            print("  tf", k, tf, flush=True)
    mm.flush()
    del mm

    man = pd.DataFrame({"tf": kept_sorted,
                        "n_peaks": [len(peaks[t]) for t in kept_sorted],
                        "frac_promoters": [frac[t] for t in kept_sorted]})
    man.to_csv(MAN, sep="\t", index=False)
    rowidx = pd.DataFrame({"record_id": idx["record_id"].to_numpy(), "row_index": np.arange(n)})
    rowidx.to_csv(ROWIDX, sep="\t", index=False)
    print("wrote", MM, MAN, ROWIDX, flush=True)

if __name__ == "__main__":
    main()
