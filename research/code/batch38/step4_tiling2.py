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
refs={int(re.match(r'^SNV_chr17_(\d+)_referenceseq$',h).group(1)):byh[h][0] for h in byh if re.match(r'^SNV_chr17_\d+_referenceseq$',h)}
ps=sorted(refs)
print("n",len(ps),"range",ps[0],ps[-1])
def tiletest(recs, ps, idx, middle_only=True):
    ok=bad=0; badlist=[]
    for a,b in zip(ps,ps[1:]):
        d=b-a
        if d>249: ok+=1; continue
        sA=recs[a]; sB=recs[b]
        if middle_only:
            lo=max(15,d); hi=min(234,234+d)
        else:
            lo=max(0,d); hi=min(249,249+d)
        good=all(sB[k-d]==sA[k] for k in range(lo,hi))
        if good: ok+=1
        else:
            bad+=1
            if len(badlist)<5: badlist.append((a,b,d))
    return ok,bad,badlist
for idx in (124,125):
    print("idx",idx,"middle-only tiling:",tiletest(refs,ps,idx,True))
# collapse runs: are the referenceseq records a set of DISTINCT windows or a tiling of a region?
# count unique variable-inserts
vset=collections.Counter(refs[p][15:234] for p in ps)
print("unique middles",len(vset),"max multiplicity",max(vset.values()))
# Compare referenceseq to the SNV ALT records: does alt[i]==ref[i] except at 125?
alts={}
for h in byh:
    m=re.match(r'^SNV_chr17_(\d+)_altseq_([ACGT])_to_([ACGT])$',h)
    if m: alts.setdefault(int(m.group(1)),[]).append((m.group(2),m.group(3),byh[h][0]))
print("n alt positions",len(alts))
# for positions with one alt: diff offset
offc=collections.Counter()
for p,lst in alts.items():
    for R,A,s in lst:
        rf=refs.get(p)
        if rf is None: continue
        d=[i for i in range(249) if s[i]!=rf[i]]
        offc[(len(d), tuple(d[:1]))]+=1
print("alt-vs-ref diff (n, first):",offc.most_common(8))
# check: does the alt base sit at 124 or 125?
b124=collections.Counter(); 
for p,lst in alts.items():
    for R,A,s in lst:
        b124[(s[124]==A, s[125]==A, s[124]==R, s[125]==R)]+=1
print("alt (s124==A,s125==A,s124==R,s125==R):",dict(b124))
