# -*- coding: utf-8 -*-
"""BATCH27 A3: descriptive mutant-ref delta stratification. NO fits, NO tuned thresholds."""
import csv, json, math, os, collections
import numpy as np
D="data/breakthrough_20261002/batch26"; R="reports/breakthrough_20261002/batch27"
os.makedirs(R+"/figures",exist_ok=True)
def _rank(a):
    a=np.asarray(a,float); o=np.argsort(a,kind="mergesort"); r=np.empty(len(a)); sa=a[o]; i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sa[j+1]==sa[i]: j+=1
        r[o[i:j+1]]=0.5*(i+j)+1.0; i=j+1
    return r
def spear(a,b):
    if len(a)<3: return float("nan")
    ra,rb=_rank(a),_rank(b); ra=ra-ra.mean(); rb=rb-rb.mean(); d=math.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d>0 else float("nan")
mrow={}
for r in csv.DictReader(open(D+"/str_variant_rows.tsv",encoding="utf-8"),delimiter="\t"):
    mrow[r["variant"]]=(r["motif"], r["allele"])
pred=list(csv.DictReader(open(D+"/test_predictions.tsv",encoding="utf-8"),delimiter="\t"))
MODELS=["A0_ridgeC","A1_cnn_true_seed20261026"]
byl=collections.defaultdict(list)
for p in pred: byl[p["locus_id"]].append(p)
def cat_motif(m):
    L=len(m); return "1" if L==1 else "2" if L==2 else "3" if L==3 else "4-6" if L<=6 else ">6"
def cat_bin(a):
    a=abs(a)
    return "zero" if a==0 else "[0,.1)" if a<0.1 else "[.1,.25)" if a<0.25 else "[.25,.5)" if a<0.5 else "[.5,inf)"
pairs=collections.defaultdict(list)  # model -> list of dict
for L,items in byl.items():
    ref=[p for p in items if mrow[p["variant"]][1]=="ref"]
    if not ref: continue
    ref=ref[0]
    for p in items:
        if mrow[p["variant"]][1]=="ref": continue
        rec={"locus":L,"motif":mrow[p["variant"]][0],
             "do":float(p["true"])-float(ref["true"])}
        for m in MODELS: rec[m]=float(p[m])-float(ref[m])
        pairs["all"].append(rec)
def block(sub):
    d={"n_pairs":len(sub),"n_loci":len(set(x["locus"] for x in sub))}
    if not sub: return d
    do=np.array([x["do"] for x in sub])
    d["n_delta_obs_zero"]=int((do==0).sum())
    d["mean_abs_delta_obs"]=float(np.mean(np.abs(do)))
    for m in MODELS:
        dp=np.array([x[m] for x in sub])
        nz=(do!=0)&(dp!=0)
        d[m]={"sign_agreement_nonzero":(float(np.mean(np.sign(dp[nz])==np.sign(do[nz]))) if nz.sum() else float("nan")),
              "n_nonzero_both":int(nz.sum()),"n_tie_pred":int((dp==0).sum()),
              "delta_spearman":spear(dp,do) if len(sub)>=3 else float("nan")}
    return d
out={"batch":"batch27","source":"batch26 saved test_predictions.tsv (6dp) + str_variant_rows.tsv; ZERO fits",
 "n_pairs":len(pairs["all"]),"n_loci":len(set(x["locus"] for x in pairs["all"])),
 "note":"descriptive outcome-stratification ONLY; not a deployable filter; no tuned thresholds; sign-ties excluded from denominators; 6-decimal rounding/tie impact disclosed.",
 "by_motif_length":collections.OrderedDict()}
for k in ["1","2","3","4-6",">6"]:
    out["by_motif_length"][k]=block([x for x in pairs["all"] if cat_motif(x["motif"])==k])
out["by_abs_delta_obs"]=collections.OrderedDict()
for k in ["zero","[0,.1)","[.1,.25)","[.25,.5)","[.5,inf)"]:
    out["by_abs_delta_obs"][k]=block([x for x in pairs["all"] if cat_bin(x["do"])==k])
json.dump(out,open(R+"/variant_bin_diagnostic.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("wrote variant_bin_diagnostic.json; pairs=%d loci=%d"%(out["n_pairs"],out["n_loci"]))
for k,v in out["by_abs_delta_obs"].items():
    print(" bin",k,"n=%d loci=%d A1_sign=%.4f"%(v["n_pairs"],v["n_loci"],v["A1_cnn_true_seed20261026"]["sign_agreement_nonzero"] if "A1_cnn_true_seed20261026" in v else float("nan")))
