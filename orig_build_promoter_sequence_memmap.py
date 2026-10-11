
"""
Build promoter sequence memmap.

Input
-----
promoters/fasta/gencode_v49_transcript_promoters_up200.fa

Output
------
promoters/memmap/promoter_up200_uint8.memmap
promoters/memmap/promoter_up200_index.tsv
"""

import numpy as np
import pandas as pd
from pathlib import Path


FASTA_PATH = Path("promoters/fasta/gencode_v49_transcript_promoters_up200.fa")

MEMMAP_PATH = Path("promoters/memmap/promoter_up200_uint8.memmap")
INDEX_PATH = Path("promoters/memmap/promoter_up200_index.tsv")


def fasta_iter(path):

    header = None
    seq = []

    with open(path) as f:
        for line in f:
            line=line.strip()

            if line.startswith(">"):

                if header is not None:
                    yield header,"".join(seq)

                header=line[1:].split()[0]
                seq=[]

            else:
                seq.append(line)

        if header is not None:
            yield header,"".join(seq)


def encode(seq):

    table = {
        "A":0,
        "C":1,
        "G":2,
        "T":3,
        "N":4
    }

    arr = np.zeros(len(seq),dtype=np.uint8)

    for i,c in enumerate(seq):
        arr[i]=table.get(c,4)

    return arr


records=[]
seqs=[]

for header,seq in fasta_iter(FASTA_PATH):

    records.append(header)
    seqs.append(seq)


N=len(seqs)
L=len(seqs[0])

Path("promoters/memmap").mkdir(parents=True,exist_ok=True)

mm = np.memmap(MEMMAP_PATH,dtype="uint8",mode="w+",shape=(N,L))

for i,seq in enumerate(seqs):

    mm[i]=encode(seq)


mm.flush()

df=pd.DataFrame({
    "record_id":records,
    "row_index":range(N)
})

df.to_csv(INDEX_PATH,sep="\t",index=False)

print("memmap built")
print("N promoters:",N)
print("length:",L)
