# -*- coding: utf-8 -*-
import gzip, collections, re
FA="data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
byh=collections.defaultdict(list)
with gzip.open(FA,"rt",encoding="utf-8") as f:
    cur=None;buf=[]
    def flush():
        if cur is not None: byh[cur].append("".join(buf))
    for ln in f:
        ln=ln.rstrip("\n")
        if ln.startswith(">"):
            flush(); cur=ln[1:].strip();buf=[]
        else: buf.append(ln)
    flush()
for p in range(44904680,44904700):
    h="SNV_chr17_%d_referenceseq"%p
    if h in byh:
        s=byh[h][0]
        print(p, "| idx120-131:", s[120:132], "| idx124:",s[124],"| idx125:",s[125],"| centers:", s[110:140])
print("--- with altseq present? ---")
for p in range(44904680,44904700):
    hs=[h for h in byh if re.match(r'^SNV_chr17_%d_altseq'%p,h)]
    print(p, hs)
