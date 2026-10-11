#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent affine diagnostic + UTF8 + timestamp + GPU/thread checks."""
import json, math, statistics, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT=Path("E:/promoter_atlas")
EXPR=ROOT/"labels/build/gene_expression_table.tsv"
TRACKA=ROOT/"data/breakthrough_20261002/batch03/trackA_gene_promoters.npz"
SPLIT=ROOT/"data/breakthrough_20261002/batch05/split_arrays.npz"
FEAT=ROOT/"data/breakthrough_20261002/batch06/features45.npz"
MODEL=ROOT/"models/breakthrough_20261002/batch09"
REPORT=ROOT/"reports/breakthrough_20261002/batch09"
AFF_A, AFF_B = 0.5, 2.0

def base_id(v): return (v or "").split(".")[0]
def median(xs): return statistics.median(xs) if xs else 0.0
def mad(xs):
    m=median(xs); return median([abs(x-m) for x in xs])
def collapse_dup(vals):
    nz=[x for x in vals if x>0.0]; zc=sum(1 for x in vals if x==0.0)
    return max(nz) if (nz and zc>0) else statistics.fmean(vals)
df=pd.read_csv(EXPR,sep="\t",usecols=["gene_id","tissue","mean_tpm"],dtype={"gene_id":"str","tissue":"str","mean_tpm":np.float64},low_memory=False)
gt=defaultdict(lambda: defaultdict(list))
for gi,t,mt in zip(df["gene_id"],df["tissue"],df["mean_tpm"]): gt[base_id(gi)][t].append(float(mt))
del df
full_s={}
for g in sorted(gt):
    tm=gt[g]; coll=[collapse_dup(tm[t]) if len(tm[t])>1 else tm[t][0] for t in sorted(tm)]
    tc=len(coll); det=[x for x in coll if x>=1.0]; dtc=len(det); b=dtc/tc if tc else 0.0
    if b>=0.30 and dtc>=3: full_s[g]=-mad([math.log2(1.0+x) for x in det])
z=np.load(TRACKA,allow_pickle=True)
gene_gid=z["gene_id"].astype(str)[z["starts"].astype(np.int64)[:-1]]
s=np.load(SPLIT,allow_pickle=True); gene_split=s["gene_split"]
zf=np.load(FEAT,allow_pickle=True)
X=np.concatenate([zf["comp44"].astype(np.float64), zf["log1p_count"].astype(np.float64)[:,None]],axis=1)
em=np.array([g in full_s for g in gene_gid])
Xe=X[em]; split_e=gene_split[em]; tr=split_e=="train"; te=split_e=="test"
sraw=np.array([full_s[g] for g in gene_gid[em]])
scaler=joblib.load(MODEL/"X_scaler.joblib")
Xs=scaler.transform(Xe)
from sklearn.linear_model import Ridge
y1=sraw[tr]; y2=AFF_A+AFF_B*y1
m1=Ridge(alpha=0.1,fit_intercept=True).fit(Xs[tr],y1)
m2=Ridge(alpha=0.1,fit_intercept=True).fit(Xs[tr],y2)
p1=m1.predict(Xs); p2=m2.predict(Xs)
err=float(np.max(np.abs(p2-(AFF_A+AFF_B*p1))))
r={}
r["affine_max_err"]=err
r["affine_pass"]=bool(err<1e-8)

# UTF8 / mojibake check
def has_replacement(p):
    t=p.read_text(encoding="utf-8")
    return t.count("\ufffd")
r["report_utf8_ok"]={}
for f in ("report.txt","report.html","BATCH10_DRAFT.txt"):
    r["report_utf8_ok"][f]=bool(has_replacement(REPORT/f)==0)
# check key Chinese substrings present
txt=(REPORT/"report.txt").read_text(encoding="utf-8")
r["chinese_key_terms_present"]={k:(k in txt) for k in ["目标表示","跨组织离散度","训练参考经验秩","仿射诊断","未达标","负/中性"]}

# timestamp ordering (freeze before metrics)
cfg=json.load(open(MODEL/"config.json",encoding="utf-8"))
metrics=json.load(open(REPORT/"metrics.json",encoding="utf-8"))
preflight=json.load(open(REPORT/"preflight_summary.json",encoding="utf-8"))
r["freeze_order"]={
  "config_created": cfg["created_at"],
  "preflight_written": preflight["written_at"],
  "metrics_written": metrics["written_at"],
}
r["config_before_metrics"]=bool(cfg["created_at"] < metrics["written_at"])
r["config_before_preflight"]=bool(cfg["created_at"] < preflight["written_at"])

# GPU / thread checks on code
code=ROOT/"runs/breakthrough_20261002/code/batch09"
gpu_markers=set()
for f in code.glob("b9_*.py"):
    t=f.read_text(encoding="utf-8")
    for m in ("torch","cuda","gpu","device("):
        if m in t: gpu_markers.add((f.name,m))
r["gpu_markers_found"]=sorted(gpu_markers)
r["thread_limits_set"]=bool("set_thread_limits" in (code/"b9_run.py").read_text(encoding="utf-8"))

# status worker identity
status=json.load(open(REPORT/"status.json",encoding="utf-8"))
r["status_worker"]=status["worker"]
r["status_state"]=status["state"]
print(json.dumps(r,ensure_ascii=False,indent=2,default=str))
