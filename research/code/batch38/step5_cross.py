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
# SNV positions with alt
snv_alt={}
for h in byh:
    m=re.match(r'^SNV_chr17_(\d+)_altseq_([ACGT])_to_([ACGT])$',h)
    if m: snv_alt[int(m.group(1))]=(m.group(2),m.group(3),byh[h][0])
# maptprom positions with alt
mp_alt={}
for h in byh:
    m=re.match(r'^maptprom_chr17_(\d+)_altseq(\d)_([ACGT])_to_([ACGT])$',h)
    if m: mp_alt.setdefault(int(m.group(1)),[]).append((m.group(3),m.group(4),byh[h][0]))
common=[p for p in snv_alt if p in mp_alt]
print("common positions",len(common))
# compare: for each common pos, find the maptprom alt with same R->A, compare sequences
cmpc=collections.Counter()
for p in common[:200]:
    R,A,s=snv_alt[p]
    for R2,A2,s2 in mp_alt[p]:
        if R2==R and A2==A:
            diffs=[i for i in range(249) if s[i]!=s2[i]]
            cmpc[(len(diffs),tuple(diffs[:2]))]+=1
print("SNV-alt vs maptprom-alt same R->A diff:",cmpc.most_common(8))
# show one example
for p in common[:400]:
    R,A,s=snv_alt[p]
    for R2,A2,s2 in mp_alt[p]:
        if R2==R and A2==A:
            print("pos",p,"R",R,"A",A)
            print(" SNV ",s[:40],"...",s[118:132],"...")
            print(" MP  ",s2[:40],"...",s2[118:132],"...")
            print("      idx124 SNV",s[124]," idx125 SNV",s[125]," idx124 MP",s2[124]," idx125 MP",s2[125])
            break
    else: continue
    break
