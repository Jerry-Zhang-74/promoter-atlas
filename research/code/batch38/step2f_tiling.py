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
ref={}
for h in byh:
    mm=re.match(r'^SNV_(chr[\w]+)_(\d+)_referenceseq$',h)
    if mm: ref[(mm.group(1),int(mm.group(2)))]=byh[h][0]
chrs=collections.Counter(c for c,p in ref)
print("refs per chr (top):",chrs.most_common(8))
# for chr17, sorted positions and gaps
ps=sorted(p for (c,p) in ref if c=="chr17")
print("chr17 n refs",len(ps),"min",ps[0],"max",ps[-1])
gaps=collections.Counter(ps[i+1]-ps[i] for i in range(len(ps)-1))
print("gap histogram",sorted(gaps.items())[:20])
# tiling test on chr17: for consecutive refs, test if rec2[i]==rec1[i+delta] for delta=p2-p1
def test(idx):
    okc=0; badc=0; ex=[]
    for i in range(len(ps)-1):
        p1,p2=ps[i],ps[i+1]; d=p2-p1
        s1=ref[("chr17",p1)]; s2=ref[("chr17",p2)]
        # expected: s2[k] == s1[k+d] for valid range using idx124=pos (or 125)
        # genome c = p1 + (k-idx)
        lo=max(0,d); hi=min(len(s1),len(s2)+d)
        good=True
        for k in range(lo,hi):
            if s2[k-d]!=s1[k]: good=False;break
        if good: okc+=1
        else:
            badc+=1
            if len(ex)<3: ex.append((p1,p2,d))
    return okc,badc,ex
for idx in (124,125):
    print("idx",idx,"test (s2[k]==s1[k+d]):",test(idx))
