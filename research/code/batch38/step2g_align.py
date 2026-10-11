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
        if ln.startswith(">"): flush();cur=ln[1:].strip();buf=[]
        else: buf.append(ln)
    flush()
def rc(s): return s.translate(str.maketrans("ACGTN","TGCAN"))[::-1]
ps=[44904682,44904684,44904693,44904697]
ss={p:byh["SNV_chr17_%d_referenceseq"%p][0] for p in ps}
for p in ps: print(p, ss[p])
# align consecutive by trying shifts
import difflib
for a,b in zip(ps,ps[1:]):
    d=b-a
    # expect s_b[k] == s_a[k+d]
    sA=ss[a]; sB=ss[b]
    for sh in (d,-d,2*d,-2*d):
        if 0<sh<len(sA):
            m=sum(1 for i in range(len(sB)) if 0<=i+sh<len(sA) and sB[i]==sA[i+sh])
            tot=sum(1 for i in range(len(sB)) if 0<=i+sh<len(sA))
            print("pair",a,b,"shift",sh,"match",m,"/",tot)
# try RC
for a,b in zip(ps,ps[1:]):
    d=b-a
    print("RC pair",a,b,"equal_rc?",ss[b]==rc(ss[a]))
