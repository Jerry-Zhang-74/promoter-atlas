#!/usr/bin/env python3
"""Build promoter sequences (up2000 + sym400) for one Ensembl species.

Usage:
    python build_multispecies_promoters.py --species mus_musculus
    python build_multispecies_promoters.py --species mus_musculus --window up2000

Inputs : data/genomes/<species>/<Species>.<assembly>.dna.toplevel.fa.gz
         data/genomes/<species>/<Species>.<assembly>.113.gtf.gz
Outputs: data/multispecies/promoters/<species>/<window>.bed
         data/multispecies/promoters/<species>/<window>.fa
         data/multispecies/memmap/<species>_<window>_uint8.memmap
         data/multispecies/memmap/<species>_<window>_index.tsv
         data/multispecies/promoter_master_index.tsv  (appended)
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

ROOT = Path(__file__).resolve().parents[2]
GENOMES = ROOT / "data" / "genomes"
OUT = ROOT / "data" / "multispecies"
MEMDIR = OUT / "memmap"
MASTER = OUT / "promoter_master_index.tsv"

LOOKUP = np.full(256, 4, dtype=np.uint8)
for b, i in zip(b"ACGT", range(4)):
    LOOKUP[b] = i
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")

BIOTYPE_WHITELIST = {
    "protein_coding", "lncRNA", "processed_transcript", "miRNA", "snRNA",
    "snoRNA", "antisense", "sense_intronic", "sense_overlapping",
    "transcribed_unprocessed_pseudogene", "transcribed_processed_pseudogene",
    "transcribed_unitary_pseudogene", "unprocessed_pseudogene",
    "processed_pseudogene", "unitary_pseudogene", "pseudogene",
    "misc_RNA", "rRNA", "Mt_rRNA", "Mt_tRNA", "lincRNA", "macro_lncRNA",
    "non_coding", "bidirectional_promoter_lncRNA",
}


def parse_gtf(path):
    """Yield (chrom, start0, end0, strand, gene_id, transcript_id, biotype) for transcripts."""
    pat = re.compile(r'gene_id "([^"]+)";|transcript_id "([^"]+)";|gene_biotype "([^"]+)";|transcript_biotype "([^"]+)";')
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "transcript":
                continue
            chrom, _, typ, start, end, _, strand, _, attrs = parts
            gid = tid = bio = None
            for m in re.finditer(r'(gene_id|transcript_id|gene_biotype|transcript_biotype) "([^"]+)"', attrs):
                if m.group(1) == "gene_id":
                    gid = m.group(2)
                elif m.group(1) == "transcript_id":
                    tid = m.group(2)
                elif m.group(1) == "gene_biotype":
                    bio = m.group(2)
                elif m.group(1) == "transcript_biotype" and bio is None:
                    bio = m.group(2)
            if not tid or not gid or bio not in BIOTYPE_WHITELIST:
                continue
            yield chrom, int(start), int(end), strand, gid, tid, bio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--window", default="both", choices=["both", "up2000", "sym400"])
    args = ap.parse_args()

    sp = args.species
    spdir = GENOMES / sp
    fa_files = sorted(spdir.glob("*.dna.toplevel.fa.gz"))
    gtf_files = sorted(spdir.glob("*.113.gtf.gz"))
    if not fa_files or not gtf_files:
        sys.exit(f"missing genome files for {sp}: {fa_files} {gtf_files}")
    fa_gz, gtf_gz = fa_files[0], gtf_files[0]
    fa = fa_gz.with_suffix("")  # .fa
    if not fa.exists():
        print("decompressing", fa_gz.name, flush=True)
        with gzip.open(fa_gz, "rb") as f_in, open(fa, "wb") as f_out:
            while True:
                chunk = f_in.read(8 << 20)
                if not chunk:
                    break
                f_out.write(chunk)
    print("fasta:", fa, flush=True)

    fasta = Fasta(str(fa), as_raw=True, sequence_always_upper=True)
    print("gtf:", gtf_gz, flush=True)

    windows = ["up2000", "sym400"] if args.window == "both" else [args.window]
    for win in windows:
        L = {"up2000": 2000, "sym400": 400}[win]
        UP = 2000 if win == "up2000" else 200
        DOWN = 0 if win == "up2000" else 200
        bed_out = OUT / "promoters" / sp / (win + ".bed")
        fa_out = OUT / "promoters" / sp / (win + ".fa")
        mm_out = MEMDIR / f"{sp}_{win}_uint8.memmap"
        idx_out = MEMDIR / f"{sp}_{win}_index.tsv"
        for p in (bed_out.parent, MEMDIR):
            p.mkdir(parents=True, exist_ok=True)

        rows = []
        n_seq = 0
        with open(bed_out, "w") as bed, open(fa_out, "w") as fout:
            for chrom, start, end, strand, gid, tid, bio in parse_gtf(gtf_gz):
                tss1 = start if strand == "+" else end  # 1-based TSS
                if strand == "+":
                    s0 = tss1 - 1 - UP
                    e0 = tss1 - 1 + DOWN
                    bstr = "+"
                else:
                    s0 = tss1 - 1 - DOWN
                    e0 = tss1 - 1 + UP
                    bstr = "-"
                s0c = max(0, s0)
                try:
                    seq = fasta[chrom][s0c:e0]
                except KeyError:
                    continue
                if strand == "-":
                    seq = seq.translate(COMP)[::-1]
                if len(seq) < L:
                    seq = seq + "N" * (L - len(seq))
                seq = seq[:L]
                rows.append((sp, chrom, s0, e0, strand, gid, tid, bio, seq))
                n_seq += 1
                bed.write(f"{chrom}\t{s0}\t{e0}\t{sp}|{tid}\t0\t{bstr}\n")
                fout.write(f">{sp}|{tid}\n{seq}\n")
        print(f"{sp} {win}: {n_seq} transcripts", flush=True)

        mm = np.memmap(mm_out, dtype="uint8", mode="w+", shape=(n_seq, L))
        for i, (_, _, _, _, _, _, _, _, seq) in enumerate(rows):
            mm[i] = LOOKUP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
            if i % 50000 == 0:
                mm.flush()
        mm.flush()
        del mm
        rec = pd.DataFrame({
            "record_id": [f"{sp}|{r[6]}" for r in rows],
            "species": sp,
            "window": win,
            "chrom": [r[1] for r in rows],
            "start0": [r[2] for r in rows],
            "end0": [r[3] for r in rows],
            "strand": [r[4] for r in rows],
            "gene_id": [r[5] for r in rows],
            "transcript_id": [r[6] for r in rows],
            "gene_biotype": [r[7] for r in rows],
            "row_index": np.arange(n_seq),
        })
        rec.to_csv(idx_out, sep="\t", index=False)
        hdr = not MASTER.exists()
        rec.to_csv(MASTER, sep="\t", index=False, mode="a", header=hdr)
        print("wrote", mm_out, idx_out, "| master rows now include", sp, win, flush=True)


if __name__ == "__main__":
    main()
