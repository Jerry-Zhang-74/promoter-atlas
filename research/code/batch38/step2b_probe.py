# -*- coding: utf-8 -*-
import gzip, collections, re, csv, json
FA="data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
KV="data/breakthrough_20261002/batch36/mapt/cohort_KV_SNV_HEK293FT.tsv"
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
rows=list(csv.DictReader(open(KV,encoding="utf-8"),delimiter="\t"))
# print sequences for first 6 rows
for r in rows[:6]:
    alt_h=r["design_header"]; ref_h=re.sub(r'_altseq_[ACGT]_to_[ACGT]$','_referenceseq',alt_h)
    a=byh[alt_h][0]; rf=byh[ref_h][0]
    print(r["original_id"], "hdr_ref",r["base_ref"],"hdr_alt",r["base_alt"],"logFC",r["logFC"])
    print(" ALT",a)
    print(" REF",rf)
    print("     ", "".join('^' if i==125 else ' ' for i in range(len(a))))
    print()
# cross-tab for single-diff rows
import itertools
ct=collections.Counter()
for r in rows:
    alt_h=r["design_header"]; ref_h=re.sub(r'_altseq_[ACGT]_to_[ACGT]$','_referenceseq',alt_h)
    a=byh[alt_h][0]; rf=byh[ref_h][0]
    diffs=[i for i in range(len(a)) if a[i]!=rf[i]]
    if len(diffs)==1 and diffs[0]==125:
        ct[(r["base_ref"], r["base_alt"], rf[125])]+=1
print("--- (hdr_ref,hdr_alt,ref@125) top 20 ---")
for k,v in ct.most_common(20): print(v,k)
print("distinct keys",len(ct))
# how often ref@125==hdr_ref vs hdr_alt vs other
t=collections.Counter()
for r in rows:
    alt_h=r["design_header"]; ref_h=re.sub(r'_altseq_[ACGT]_to_[ACGT]$','_referenceseq',alt_h)
    rf=byh[ref_h][0]
    if rf[125]==r["base_ref"]: t["ref@125==base_ref"]+=1
    elif rf[125]==r["base_alt"]: t["ref@125==base_alt"]+=1
    else: t["ref@125==other"]+=1
print("--- ref@125 vs table alleles (all 8513) ---", dict(t))
