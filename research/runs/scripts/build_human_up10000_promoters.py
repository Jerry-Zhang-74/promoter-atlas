#!/usr/bin/env python3
"""Build human up10000 promoter memmap (10 kb upstream of each transcript TSS).

Input : legacy_data/GRCh38.primary_assembly.genome.fa
        promoters/qc/gencode_v49_transcript_promoter_index.tsv
Output: promoters/memmap/promoter_up10000_uint8.memmap + promoter_up10000_index.tsv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta

ROOT = Path(__file__).resolve().parents[2]
GENOME = ROOT / "legacy_data" / "GRCh38.primary_assembly.genome.fa"
INDEX = ROOT / "promoters" / "qc" / "gencode_v49_transcript_promoter_index.tsv"
MM_OUT = ROOT / "promoters" / "memmap" / "promoter_up10000_uint8.memmap"
IDX_OUT = ROOT / "promoters" / "memmap" / "promoter_up10000_index.tsv"
L = 10000
LOOKUP = np.full(256, 4, dtype=np.uint8)
for b, i in zip(b"ACGT", range(4)):
    LOOKUP[b] = i
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")

def main():
    idx = pd.read_csv(INDEX, sep="\t", low_memory=False)
    idx = idx[idx["window"] == "up2000"].drop_duplicates(subset="record_id", keep="first").reset_index(drop=True)
    n = len(idx)
    print("transcripts:", n, flush=True)
    fa = Fasta(str(GENOME), as_raw=True, sequence_always_upper=True)
    mm = np.memmap(MM_OUT, dtype="uint8", mode="w+", shape=(n, L))
    for i, r in enumerate(idx.itertuples()):
        tss1 = r.tss_1based
        if r.strand == "+":
            s0 = max(0, tss1 - 1 - L)
            e0 = tss1 - 1
            seq = fa[r.chrom][s0:e0]
        else:
            s0 = tss1 - 1
            e0 = tss1 - 1 + L
            seq = fa[r.chrom][s0:e0].translate(COMP)[::-1]
        if len(seq) < L:
            seq = seq + "N" * (L - len(seq))
        seq = seq[:L]
        mm[i] = LOOKUP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        if i % 50000 == 0:
            mm.flush()
            print("rows:", i, flush=True)
    mm.flush()
    del mm
    out = pd.DataFrame({"record_id": idx["record_id"].to_numpy(), "row_index": np.arange(n)})
    out.to_csv(IDX_OUT, sep="\t", index=False)
    print("wrote", MM_OUT, IDX_OUT, flush=True)

if __name__ == "__main__":
    main()
