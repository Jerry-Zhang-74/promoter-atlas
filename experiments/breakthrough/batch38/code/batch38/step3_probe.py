# -*- coding: utf-8 -*-
import gzip, collections, re, csv
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

# (a) SAT arm convention
sat=list(csv.DictReader(open("data/breakthrough_20261002/batch36/mapt/cohort_SAT_HEK293FT.tsv",encoding="utf-8"),delimiter="\t"))
sat_snv=[r for r in sat if r["deletion"]=="0"][:400]
print("SAT SNV rows sampled",len(sat_snv))
offs=collections.Counter(); rf124=collections.Counter(); af124=collections.Counter()
for r in sat_snv:
    ah=r["design_header"]; pos=r["pos"]
    rh="maptprom_chr17_%s_referenceseq"%pos
    if rh not in byh: continue
    a=byh[ah][0]; rf=byh[rh][0]
    d=[i for i in range(len(a)) if a[i]!=rf[i]]
    offs[tuple(d[:1])]+=1
    rf124[(rf[124]==r["base_ref"])]+=1
    af124[(a[124]==r["base_alt"])]+=1
print("SAT diff offsets(first)",offs.most_common(5))
print("SAT ref[124]==base_ref",dict(rf124)," alt[124]==base_alt",dict(af124))

# (b) constant columns of SNV referenceseq records
snv_refs=[byh[h][0] for h in byh if re.match(r'^SNV_chr17_\d+_referenceseq$',h)]
print("n snv refs",len(snv_refs),"len set",set(len(s) for s in snv_refs))
ncol=len(snv_refs[0])
const=[i for i in range(ncol) if len({s[i] for s in snv_refs})==1]
varc=[i for i in range(ncol) if i not in set(const)]
print("constant cols count",len(const),"first var",min(varc),"last var",max(varc))
print("left flank",repr(snv_refs[0][:min(varc)]),"right flank",repr(snv_refs[0][max(varc)+1:]))

# (c) SNV positions overlapping MAPT region
snv_pos=sorted(int(re.match(r'^SNV_chr17_(\d+)_referenceseq$',h).group(1)) for h in byh if re.match(r'^SNV_chr17_\d+_referenceseq$',h))
mapt=[p for p in snv_pos if 45893536<=p<=45895535]
print("SNV refs in MAPT region:",len(mapt), mapt[:5], "..." , mapt[-3:] if mapt else "")
# compare same-position SNV vs maptprom ref
same=0; diff=0
for p in mapt[:50]:
    h1="SNV_chr17_%d_referenceseq"%p; h2="maptprom_chr17_%d_referenceseq"%p
    if h1 in byh and h2 in byh:
        same += (byh[h1][0]==byh[h2][0]); diff += (byh[h1][0]!=byh[h2][0])
print("SNV vs maptprom same-pos ref: same",same,"diff",diff)

# (d) tiling mismatch detail for (44904682,44904684)
sA=byh["SNV_chr17_44904682_referenceseq"][0]; sB=byh["SNV_chr17_44904684_referenceseq"][0]
mm=[]
for i in range(len(sB)):
    j=i+2
    if 0<=j<len(sA) and sB[i]!=sA[j]: mm.append((i,sB[i],sA[j]))
print("tiling mismatches (82,84,shift2):",len(mm),mm[:12])
