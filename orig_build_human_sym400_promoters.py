#!/usr/bin/env python3
"""Build human sym400 promoters (200 bp upstream + 200 bp downstream of each transcript TSS).

Input : legacy_data/GRCh38.primary_assembly.genome.fa (+ .fai)
        promoters/qc/gencode_v49_transcript_promoter_index.tsv (transcript TSS coords)
Output: promoters/bed/gencode_v49_transcript_promoters_sym400.bed
        promoters/fasta/gencode_v49_transcript_promoters_sym400.fa
        promoters/memmap/promoter_sym400_uint8.memmap + promoter_sym400_index.tsv

Window: [T-200, T+200) 0-based in the transcript orientation (reverse-complemented
on the minus strand), so the TSS base is always at 0-based index 200 of the 400 bp.
"""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta

ROOT = Path(__file__).resolve().parents[2]
GENOME = ROOT / "legacy_data" / "GRCh38.primary_assembly.genome.fa"
INDEX = ROOT / "promoters" / "qc" / "gencode_v49_transcript_promoter_index.tsv"
BED_OUT = ROOT / "promoters" / "bed" / "gencode_v49_transcript_promoters_sym400.bed"
FA_OUT = ROOT / "promoters" / "fasta" / "gencode_v49_transcript_promoters_sym400.fa"
MM_OUT = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
MM_INDEX = ROOT / "promoters" / "memmap" / "promoter_sym400_index.tsv"

UP, DOWN, L = 200, 200, 400
LOOKUP = np.full(256, 4, dtype=np.uint8)
for b, i in zip(b"ACGT", range(4)):
    LOOKUP[b] = i
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def main():
    idx = pd.read_csv(INDEX, sep="\t", low_memory=False)
    idx = idx.drop_duplicates(subset="record_id", keep="first")
    n = len(idx)
    print("transcripts:", n, flush=True)

    fa = Fasta(str(GENOME), as_raw=True, sequence_always_upper=True)
    rows = []
    BED_OUT.parent.mkdir(parents=True, exist_ok=True)
    FA_OUT.parent.mkdir(parents=True, exist_ok=True)
    MM_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(BED_OUT, "w") as bed, open(FA_OUT, "w") as fout:
        for i, r in enumerate(idx.itertuples()):
            tss1 = r.tss_1based
            chrom = r.chrom
            if r.strand == "+":
                start0 = tss1 - 1 - UP
                end0 = tss1 - 1 + DOWN
                strand_bed = "+"
            else:
                start0 = tss1 - 1 - DOWN
                end0 = tss1 - 1 + UP
                strand_bed = "-"
            start0 = max(0, start0)
            seq = fa[chrom][start0:end0]
            if r.strand == "-":
                seq = seq.translate(COMP)[::-1]
            if len(seq) < L:
                seq = seq + "N" * (L - len(seq))
            seq = seq[:L]
            rows.append((r.record_id, r.chrom, start0, end0, r.strand, seq))
            bed.write(f"{chrom}\t{start0}\t{end0}\t{r.record_id}\t0\t{strand_bed}\n")
            fout.write(f">{r.record_id}\n{seq}\n")
            if i % 50000 == 0:
                print("rows:", i, flush=True)

    df = pd.DataFrame(rows, columns=["record_id", "chrom", "start0", "end0", "strand", "seq"])
    mm = np.memmap(MM_OUT, dtype="uint8", mode="w+", shape=(n, L))
    for i, s in enumerate(df["seq"]):
        mm[i] = LOOKUP[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        if i % 50000 == 0:
            mm.flush()
    mm.flush()
    del mm
    out_idx = df[["record_id"]].copy()
    out_idx["row_index"] = np.arange(n)
    out_idx.to_csv(MM_INDEX, sep="\t", index=False)
    print("wrote", BED_OUT, FA_OUT, MM_OUT, MM_INDEX, flush=True)
    print("shape", (n, L), flush=True)


if __name__ == "__main__":
    main()
