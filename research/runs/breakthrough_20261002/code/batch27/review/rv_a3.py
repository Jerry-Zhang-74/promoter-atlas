# -*- coding: utf-8 -*-
"""REVIEWER independent reproduction of A3 variant-bin diagnostic. ZERO fits."""
import csv,json,os,collections,math
import numpy as np
D="data/breakthrough_20261002/batch26"
def rank(a):
    a=np.asarray(a,float);o=np.argsort(a,kind="mergesort");r=np.empty(len(a));sa=a[o];i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sa[j+1]==sa[i]: j+=1
        r[o[i:j+1]]=0.5*(i+j)+1.0;i=j+1
    return r
def spear(a,b):
    if len(a)<3: return float("nan")
    ra,rb=rank(a),rank(b);ra=ra-ra.mean();rb=rb-rb.mean();d=math.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d>0 else float("nan")
mrow={r["variant"]:(r["motif"],r["allele"]) for r in csv.DictReader(open(D+"/str_variant_rows.tsv",encoding="utf-8"),delimiter="\t")}
pred=list(csv.DictReader(open(D+"/test_predictions.tsv",encoding="utf-8"),delimiter="\t"))
MODELS=["A0_ridgeC","A1_cnn_true_seed20261026"]
byl=collections.defaultdict(list)
for p in pred: byl[p["locus_id"]].append(p)
def cm(m): L=len(m); return "1" if L==1 else "2" if L==2 else "3" if L==3 else "4-6" if L<=6 else ">6"
def cb(a):
    a=abs(a); return "zero" if a==0 else "[0,.1)" if a<0.1 else "[.1,.25)" if a<0.25 else "[.25,.5)" if a<0.5 else "[.5,inf)"
pairs=[]
for L,items in byl.items():
    ref=[p for p in items if mrow[p["variant"]][1]=="ref"]
    if not ref: continue
    ref=ref[0]
    for p in items:
        if mrow[p["variant"]][1]=="ref": continue
        rec={"locus":L,"motif":mrow[p["variant"]][0],"do":float(p["true"])-float(ref["true"])}
        for m in MODELS: rec[m]=float(p[m])-float(ref[m])
        pairs.append(rec)
print("n_pairs",len(pairs),"n_loci",len(set(x["locus"] for x in pairs)))
def block(sub):
    d={"n_pairs":len(sub),"n_loci":len(set(x["locus"] for x in sub))}
    do=np.array([x["do"] for x in sub])
    d["mean_abs_delta_obs"]=float(np.mean(np.abs(do)))
    for m in MODELS:
        dp=np.array([x[m] for x in sub]); nz=(do!=0)&(dp!=0)
        d[m]={"sign_agreement_nonzero":float(np.mean(np.sign(dp[nz])==np.sign(do[nz]))) if nz.sum() else float("nan"),
              "n_nonzero_both":int(nz.sum()),"delta_spearman":spear(dp,do)}
    return d
for k in ["1","2","3","4-6",">6"]:
    b=block([x for x in pairs if cm(x["motif"])==k]); print("motif",k,b["n_pairs"],b["n_loci"],"A1",b["A1_cnn_true_seed20261026"]["sign_agreement_nonzero"])
print("---abs---")
for k in ["zero","[0,.1)","[.1,.25)","[.25,.5)","[.5,inf)"]:
    b=block([x for x in pairs if cb(x["do"])==k]); print("abs",k,b["n_pairs"],b["n_loci"],"A1",b["A1_cnn_true_seed20261026"]["sign_agreement_nonzero"],"A1rho",b["A1_cnn_true_seed20261026"]["delta_spearman"])
