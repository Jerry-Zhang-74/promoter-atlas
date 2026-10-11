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
byh={h:s[0] for h,s in [(h,[x for x in [s] ]) for h,s in []]}
byh=collections.defaultdict(list)
for h,s in recs: byh[h].append(s)

def build(pat, idx):
    # returns dict pos->base using records whose header matches pat and base at idx
    m={}
    for h in byh:
        mm=re.match(pat,h)
        if mm: m[int(mm.group(1))]=byh[h][0][idx]
    return m

# referenceseq records across all chr (SNV_chrX_pos_referenceseq)
ref_recs={}
for h in byh:
    mm=re.match(r'^SNV_(chr[\w]+)_(\d+)_referenceseq$',h)
    if mm: ref_recs[(mm.group(1),int(mm.group(2)))]=byh[h][0]
print("n referenceseq records",len(ref_recs))
# consistency: for a record centered at p with index124=p, then rec[p][124+k]=genome[p+k]
# check overlap agreement for chr17 region 44904600-44904800
def check(ref_recs, idx, chrs=("chr17",), lo=44904600, hi=44904800):
    gg={}
    conflict=0; agree=0
    for (c,p),s in ref_recs.items():
        if c not in chrs or not (lo<=p<=hi): continue
        for k in range(len(s)):
            g=p+(k-idx)
            if g in gg:
                if gg[g]!=s[k]: conflict+=1
            else: gg[g]=s[k]
    return gg,conflict
gg,conf=check(ref_recs,124)
print("idx=124 conflicts in region:",conf,"positions:",len(gg))
gg2,conf2=check(ref_recs,125)
print("idx=125 conflicts in region:",conf2,"positions:",len(gg2))
# show genome around 44904682
s="".join(gg.get(44904682+d,"?") for d in range(-6,7))
print("genome(44904676..44904688) idx124:",s)
s2="".join(gg2.get(44904682+d,"?") for d in range(-6,7))
print("genome(44904676..44904688) idx125:",s2)
