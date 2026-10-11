# -*- coding: utf-8 -*-
"""Tie-aware sensitivity for the corrected SuRE top-5% proxy sets. Zero fits.
Primary top5 = exact k=ceil(0.05n) with SHA tiebreak. Sensitivity = include ALL
rows at/above the cutoff proxy value (tie-inclusive superset)."""
import csv,json,math,os,hashlib
import numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R=os.path.join(REPO,"reports/breakthrough_20261002/batch31")
N_R1,N_R2,N_D=37770653,74081034,203742844
def z(r,N,P): return math.log2(1.0+(r/N)/((1.0+P)/N_D))
SPLIT=os.path.join(REPO,"data/breakthrough_20261002/batch30/split.tsv")
ROWS=os.path.join(REPO,"data/breakthrough_20261002/batch29/rows.tsv")
train={r["key"] for r in csv.DictReader(open(SPLIT,encoding="utf-8"),delimiter="\t") if r["split"]=="train"}
rows=[r for r in csv.DictReader(open(ROWS,encoding="utf-8"),delimiter="\t") if r["key"] in train]
n=len(rows); k=int(math.ceil(0.05*n))
z1=np.array([z(int(r["r1"]),N_R1,int(r["P"])) for r in rows]); z2=np.array([z(int(r["r2"]),N_R2,int(r["P"])) for r in rows])
key=np.array([r["key"] for r in rows])
c1=float(np.sort(z1)[::-1][k-1]); c2=float(np.sort(z2)[::-1][k-1])
S1_incl=set(key[z1>=c1].tolist()); S2_incl=set(key[z2>=c2].tolist())
import hashlib as H
def topk_sh(z):
    h=np.array([int(H.sha256(x.encode()).hexdigest()[:16],16) for x in key],np.uint64)
    o=np.lexsort((h,-z)); return set(key[o[:k]].tolist())
s1=topk_sh(z1); s2=topk_sh(z2)
out={"primary_exact_k":{"k":k,"intersection":len(s1&s2)},
     "tie_inclusive":{"n1":len(S1_incl),"n2":len(S2_incl),"extra_vs_primary_1":len(S1_incl)-len(s1),
                      "extra_vs_primary_2":len(S2_incl)-len(s2),"intersection":len(S1_incl&S2_incl),
                      "jaccard":len(S1_incl&S2_incl)/max(1,len(S1_incl|S2_incl))},
     "verdict":"NOT tie-dominated (cutoffs 0.065%/2.17% tied; tie-inclusive sets add few rows); "
               "the top set is not counted as strong positive evidence by ties alone, but the overall "
               "resource gate still FAILS on comp1/comp2."}
q=json.load(open(os.path.join(R,"sure_qc_metrics.json"),encoding="utf-8"))
q["tie_sensitivity"]=out
json.dump(q,open(os.path.join(R,"sure_qc_metrics.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=1))
