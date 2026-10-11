#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent bootstrap verification from predictions.tsv only."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("E:/promoter_atlas")
DATA=ROOT/"data/breakthrough_20261002/batch09"
REPORT=ROOT/"reports/breakthrough_20261002/batch09"
SEED=20261011; N_BOOT=2000; LO=1.25; HI=98.75; DELTA_MIN=0.03

def rankdata(x):
    x=np.asarray(x,float); order=np.argsort(x,kind="mergesort"); r=np.empty(len(x))
    i=0; n=len(x)
    while i<n:
        j=i+1
        while j<n and x[order[j]]==x[order[i]]: j+=1
        r[order[i:j]]=0.5*(i+j-1)+1.0; i=j
    return r
def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rx=rankdata(x); ry=rankdata(y); rx-=rx.mean(); ry-=ry.mean()
    d=math.sqrt((rx**2).sum()*(ry**2).sum()); return float((rx*ry).sum()/d) if d else float("nan")

pred=pd.read_csv(DATA/"predictions.tsv",sep="\t",low_memory=False)
metrics=json.load(open(REPORT/"metrics.json",encoding="utf-8"))
mp=metrics["primary_target_representation"]
te=pred[pred["split"]=="test"].reset_index(drop=True)
comp_te=te["component_id"].values
uniq=np.unique(comp_te)
rng=np.random.default_rng(SEED)
clist={c: np.flatnonzero(comp_te==c) for c in uniq}
rr=te["ridge_rank_pred"].values; ra=te["ridge_raw_pred"].values
hr=te["hgb_rank_pred"].values; ha=te["hgb_raw_pred"].values
truth=te["S_raw_full"].values
d_r=np.empty(N_BOOT); d_h=np.empty(N_BOOT)
for b in range(N_BOOT):
    sample=rng.choice(uniq,size=len(uniq),replace=True)
    idx=np.concatenate([clist[c] for c in sample])
    tb=truth[idx]
    d_r[b]=spearman(rr[idx],tb)-spearman(ra[idx],tb)
    d_h[b]=spearman(hr[idx],tb)-spearman(ha[idx],tb)
pt_r=spearman(rr,truth)-spearman(ra,truth)
pt_h=spearman(hr,truth)-spearman(ha,truth)
r={}
r["ridge_point"]=float(pt_r); r["ridge_lo"]=float(np.percentile(d_r,LO)); r["ridge_hi"]=float(np.percentile(d_r,HI))
r["hgb_point"]=float(pt_h); r["hgb_lo"]=float(np.percentile(d_h,LO)); r["hgb_hi"]=float(np.percentile(d_h,HI))
r["ridge_nan"]=int(np.isnan(d_r).sum()); r["hgb_nan"]=int(np.isnan(d_h).sum())
r["n_test_components"]=int(len(uniq)); r["n_test_rows"]=int(len(te))
r["ridge_match"]=bool(abs(pt_r-mp["ridge"]["point_estimate"])<1e-12 and abs(np.percentile(d_r,LO)-mp["ridge"]["lo"])<1e-12 and abs(np.percentile(d_r,HI)-mp["ridge"]["hi"])<1e-12)
r["hgb_match"]=bool(abs(pt_h-mp["hgb"]["point_estimate"])<1e-12 and abs(np.percentile(d_h,LO)-mp["hgb"]["lo"])<1e-12 and abs(np.percentile(d_h,HI)-mp["hgb"]["hi"])<1e-12)
r["ridge_meets_rule"]=bool(pt_r>=DELTA_MIN and np.percentile(d_r,LO)>0)
r["hgb_meets_rule"]=bool(pt_h>=DELTA_MIN and np.percentile(d_h,LO)>0)
print(json.dumps(r,indent=2,default=str))
