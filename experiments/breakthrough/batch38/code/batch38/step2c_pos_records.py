# -*- coding: utf-8 -*-
import gzip, collections, re
FA="data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
recs=[]
with gzip.open(FA,"rt",encoding="utf-8") as f:
    cur=None;buf=[]
    for ln in f:
        ln=ln.rstrip("\n")
        if ln.startswith(">"):
            if cur is not None: recs.append((cur,"".join(buf)))
            cur=ln[1:].strip();buf=[]
        else: buf.append(ln)
    recs.append((cur,"".join(buf)))
byh=collections.defaultdict(list)
for h,s in recs: byh[h].append(s)
for pos in ["44904682","44904684","44904693","44904697"]:
    print("=== pos",pos,"===")
    hs=[h for h in byh if re.match(r'^SNV_chr17_%s_'%pos,h)]
    for h in sorted(hs): print("  ",h, "n=",len(byh[h]), byh[h][0][:40],"...",byh[h][0][120:132])
