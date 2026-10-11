"""Map genomic (1-based) ISM alleles to every sequence window actually used."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASES = {base: i for i, base in enumerate("ACGT")}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def map_allele(position, ref, alt, start0, end0, strand):
    pos0 = int(position) - 1  # Variant position is 1-based.
    if not start0 <= pos0 < end0:
        raise ValueError(f"Variant {position} outside [{start0}, {end0})")
    if strand == "-":
        return end0 - 1 - pos0, BASES[ref.translate(COMPLEMENT)], BASES[alt.translate(COMPLEMENT)]
    if strand == "+":
        return pos0 - start0, BASES[ref], BASES[alt]
    raise ValueError(f"Unknown strand: {strand}")


class ReferenceWindows:
    def __init__(self, records, ctx10k=False):
        wanted = set(records)
        cols = ["window_name", "record_id", "chrom", "start0", "end0", "strand", "tss_1based"]
        qc = pd.read_csv(ROOT / "promoters/qc/gencode_v49_transcript_promoter_index.tsv", sep="\t", usecols=cols)
        qc = qc[qc.record_id.isin(wanted)]
        self.coords = {2000: qc[qc.window_name == "up2000"].drop_duplicates("record_id").set_index("record_id")}
        bed = pd.read_csv(ROOT / "promoters/bed/gencode_v49_transcript_promoters_sym400.bed",
                          sep="\t", header=None, names=["chrom", "start0", "end0", "record_id", "score", "strand"])
        self.coords[400] = bed[bed.record_id.isin(wanted)].set_index("record_id")
        windows = {400: "sym400", 2000: "up2000"}
        if ctx10k:
            ten = self.coords[2000].copy()
            tss0 = ten.tss_1based - 1
            ten["start0"] = np.where(ten.strand == "+", np.maximum(0, tss0 - 10000), tss0)
            ten["end0"] = np.where(ten.strand == "+", tss0, tss0 + 10000)
            self.coords[10000] = ten
            windows[10000] = "up10000"
        self.mmaps, self.rows = {}, {}
        for length, name in windows.items():
            path = ROOT / f"promoters/memmap/promoter_{name}_uint8.memmap"
            self.mmaps[length] = np.memmap(path, dtype="uint8", mode="r", shape=(path.stat().st_size // length, length))
            idx = pd.read_csv(ROOT / f"promoters/memmap/promoter_{name}_index.tsv", sep="\t")
            self.rows[length] = idx[idx.record_id.isin(wanted)].set_index("record_id").row_index.to_dict()

    def reference(self, record):
        return {length: mm[self.rows[length][record]].copy() for length, mm in self.mmaps.items()}

    def mutate(self, record, reference, variant):
        result = {}
        for length, sequence in reference.items():
            r = self.coords[length].loc[record]
            if variant.chrom != r.chrom:
                raise ValueError(f"Chromosome mismatch for {record}")
            idx, ref, alt = map_allele(variant.position, variant.ref_base, variant.alt_base,
                                       int(r.start0), int(r.end0), r.strand)
            if int(sequence[idx]) != ref:
                raise ValueError(f"Reference allele mismatch: {record}, {variant.position}, window={length}, index={idx}")
            if ref == alt:
                raise ValueError("ISM allele must change the sequence")
            mut = sequence.copy()
            mut[idx] = alt
            result[length] = mut
        return result
