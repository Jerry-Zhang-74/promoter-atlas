#!/usr/bin/env python3
"""Build uint8 memmap for the up2000 promoter sequences (vectorized).

Parses promoters/fasta/gencode_v49_transcript_promoters_up2000.fa (80-char
wrapped FASTA, one record per transcript) into a (n_records, 2000) uint8 array
(ACGTN -> 0..4) plus an index TSV (record_id, row_index).
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FASTA = ROOT / "promoters" / "fasta" / "gencode_v49_transcript_promoters_up2000.fa"
OUT = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"
INDEX = ROOT / "promoters" / "memmap" / "promoter_up2000_index.tsv"
SEQ_LEN = 2000

LOOKUP = np.full(256, 4, dtype=np.uint8)  # default -> N
LOOKUP[ord("A")] = 0
LOOKUP[ord("C")] = 1
LOOKUP[ord("G")] = 2
LOOKUP[ord("T")] = 3


def main():
    record_ids = []
    seq_bytes = []
    cur_id = None
    cur_parts = []

    def flush():
        nonlocal cur_id, cur_parts
        if cur_id is None:
            return
        s = "".join(cur_parts).upper().encode("ascii")
        if len(s) < SEQ_LEN:
            s = s + b"N" * (SEQ_LEN - len(s))
        record_ids.append(cur_id)
        seq_bytes.append(s[:SEQ_LEN])
        cur_id = None
        cur_parts = []

    with open(FASTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_id = line[1:].split()[0]
            else:
                cur_parts.append(line)
        flush()

    n = len(record_ids)
    print("records:", n, flush=True)
    mm = np.memmap(OUT, dtype="uint8", mode="w+", shape=(n, SEQ_LEN))
    for i, sb in enumerate(seq_bytes):
        row = np.frombuffer(sb, dtype=np.uint8)
        mm[i] = LOOKUP[row]
        if i % 50000 == 0:
            print("rows:", i, flush=True)
            mm.flush()
    mm.flush()
    del mm

    pd.DataFrame({"record_id": record_ids, "row_index": np.arange(n)}).to_csv(
        INDEX, sep="\t", index=False
    )
    print("wrote", OUT, "shape", (n, SEQ_LEN), flush=True)
    print("wrote", INDEX, flush=True)


if __name__ == "__main__":
    main()
