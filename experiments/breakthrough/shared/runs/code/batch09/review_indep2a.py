#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path("E:/promoter_atlas")
EXPR = ROOT/"labels/build/gene_expression_table.tsv"
TRACKA = ROOT/"data/breakthrough_20261002/batch03/trackA_gene_promoters.npz"
SPLIT = ROOT/"data/breakthrough_20261002/batch05/split_arrays.npz"
FEAT = ROOT/"data/breakthrough_20261002/batch06/features45.npz"
MODEL = ROOT/"models/breakthrough_20261002/batch09"
DATA = ROOT/"data/breakthrough_20261002/batch09"
REPORT = ROOT/"reports/breakthrough_20261002/batch09"
SEED = 20261011

def base_id(v): return (v or "").split(".")[0]
def median(xs): return statistics.median(xs) if xs else 0.0
def mad(xs):
    m=median(xs); return median([abs(x-m) for x in xs])
def collapse_dup(vals):
    nz=[x for x in vals if x>0.0]; zc=sum(1 for x in vals if x==0.0)
    return max(nz) if (nz and zc>0) else statistics.fmean(vals)
def reconstruct_s(expr_path):
    df = pd.read_csv(expr_path, sep="\t", usecols=["gene_id","tissue","mean_tpm"],
                     dtype={"gene_id":"str","tissue":"str","mean_tpm":np.float64}, low_memory=False)
    gt = defaultdict(lambda: defaultdict(list))
    for gi,t,mt in zip(df["gene_id"],df["tissue"],df["mean_tpm"]): gt[base_id(gi)][t].append(float(mt))
    del df
    out={}
    for g in sorted(gt):
        tm=gt[g]; coll=[collapse_dup(tm[t]) if len(tm[t])>1 else tm[t][0] for t in sorted(tm)]
        tc=len(coll); det=[x for x in coll if x>=1.0]; dtc=len(det)
        b_raw=dtc/tc if tc else 0.0
        if b_raw>=0.30 and dtc>=3: out[g]=-mad([math.log2(1.0+x) for x in det])
    return out
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
    if len(x)<2: return float("nan")
    rx=rankdata(x); ry=rankdata(y); rx-=rx.mean(); ry-=ry.mean()
    d=math.sqrt((rx**2).sum()*(ry**2).sum()); return float((rx*ry).sum()/d) if d else float("nan")
def emp_rank(st,x):
    n=len(st); x=np.asarray(x,float)
    lt=np.searchsorted(st,x,side="left"); eq=np.searchsorted(st,x,side="right")-lt
    return (lt.astype(float)+0.5*eq.astype(float))/n

r={}
pred = pd.read_csv(DATA/"predictions.tsv", sep="\t", low_memory=False)
metrics = json.load(open(REPORT/"metrics.json", encoding="utf-8"))
cfg = json.load(open(MODEL/"config.json", encoding="utf-8"))
tf = json.load(open(MODEL/"transforms.json", encoding="utf-8"))
models = joblib.load(MODEL/"models.joblib")
scaler = joblib.load(MODEL/"X_scaler.joblib")
ref = np.load(MODEL/"rank_reference.npz")["sorted_train"]

def ridge_cfg(m): return {"alpha":float(m.alpha), "fit_intercept":bool(m.fit_intercept)}
def hgb_cfg(m): return {"max_iter":int(m.max_iter),"learning_rate":float(m.learning_rate),
    "max_leaf_nodes":int(m.max_leaf_nodes),"l2_regularization":float(m.l2_regularization),
    "early_stopping":bool(m.early_stopping),"random_state":int(m.random_state)}
r["ridge_raw_cfg"]=ridge_cfg(models["ridge_raw"]); r["ridge_rank_cfg"]=ridge_cfg(models["ridge_rank"])
r["hgb_raw_cfg"]=hgb_cfg(models["hgb_raw"]); r["hgb_rank_cfg"]=hgb_cfg(models["hgb_rank"])
r["ridge_cfg_ok"]=bool(ridge_cfg(models["ridge_raw"])=={"alpha":0.1,"fit_intercept":True} and ridge_cfg(models["ridge_rank"])=={"alpha":0.1,"fit_intercept":True})
r["hgb_cfg_ok"]=bool(hgb_cfg(models["hgb_raw"])=={"max_iter":200,"learning_rate":0.1,"max_leaf_nodes":15,"l2_regularization":0.1,"early_stopping":False,"random_state":SEED} and hgb_cfg(models["hgb_rank"])==hgb_cfg(models["hgb_raw"]))

fn=cfg["features"]["feature_names"]
r["n_features_cfg"]=len(fn)
r["all_composition_count"]=bool(all(("mono" in f) or ("dinuc" in f) or f.endswith("_gc") or f.endswith("_len") or f=="log1p_count" for f in fn))
r["no_label_features"]=bool(cfg["features"]["no_label_features"])

z=np.load(TRACKA,allow_pickle=True)
gene_gid=z["gene_id"].astype(str)[z["starts"].astype(np.int64)[:-1]]
s=np.load(SPLIT,allow_pickle=True)
gene_split=s["gene_split"]
zf=np.load(FEAT,allow_pickle=True)
X=np.concatenate([zf["comp44"].astype(np.float64), zf["log1p_count"].astype(np.float64)[:,None]],axis=1)
full_s=reconstruct_s(EXPR)
em=np.array([g in full_s for g in gene_gid])
Xe=X[em]; split_e=gene_split[em]
tr=split_e=="train"; te=split_e=="test"
mu_tr=Xe[tr].mean(axis=0); sd_tr=Xe[tr].std(axis=0)
r["scaler_mean_eq_train"]=bool(np.allclose(scaler.mean_, mu_tr))
r["scaler_scale_eq_train"]=bool(np.allclose(scaler.scale_, sd_tr))
r["scaler_not_full_data"]=bool(not np.allclose(scaler.mean_, Xe.mean(axis=0), atol=1e-6))

sraw=np.array([full_s[g] for g in gene_gid[em]])
F=emp_rank(np.sort(sraw[tr]), sraw)
r["train_mean_raw_match"]=bool(abs(tf["target_raw"]["train_mean"]-float(sraw[tr].mean()))<1e-12)
r["train_std_raw_match"]=bool(abs(tf["target_raw"]["train_std"]-float(sraw[tr].std(ddof=0)))<1e-12)
r["train_mean_rank_match"]=bool(abs(tf["target_rank"]["train_mean"]-float(F[tr].mean()))<1e-12)
r["train_std_rank_match"]=bool(abs(tf["target_rank"]["train_std"]-float(F[tr].std(ddof=0)))<1e-12)
r["F_reproduced"]=bool(np.allclose(F, pred["rank_target_F"].values, atol=1e-12))
st=np.sort(sraw[tr])
r["out_of_range"]={"below":int((sraw[~tr]<st[0]).sum()),"above":int((sraw[~tr]>st[-1]).sum())}
fnr=F[~tr]; _,cnt=np.unique(fnr,return_counts=True)
r["tied_F_values"]=int((cnt>1).sum()); r["tied_non_train_rows"]=int(cnt[cnt>1].sum())
r["rank_train_mean_is_0.5"]=bool(abs(float(F[tr].mean())-0.5)<1e-12)
r["rank_train_std"]=float(F[tr].std(ddof=0))

Xs=scaler.transform(Xe)
replay={
  "ridge_raw": models["ridge_raw"].predict(Xs)*tf["target_raw"]["train_std"]+tf["target_raw"]["train_mean"],
  "ridge_rank": models["ridge_rank"].predict(Xs)*tf["target_rank"]["train_std"]+tf["target_rank"]["train_mean"],
  "hgb_raw": models["hgb_raw"].predict(Xs)*tf["target_raw"]["train_std"]+tf["target_raw"]["train_mean"],
  "hgb_rank": models["hgb_rank"].predict(Xs)*tf["target_rank"]["train_std"]+tf["target_rank"]["train_mean"],
}
for k in ("ridge_raw","ridge_rank","hgb_raw","hgb_rank"):
    r["replay_"+k+"_match"]=bool(np.allclose(replay[k], pred[k+"_pred"].values, atol=1e-6))

sraw_te=sraw[te]
sp_re={k: spearman(replay[k][te], sraw_te) for k in replay}
r["recomputed_test_spearman"]=sp_re
m_arm=metrics["arms"]
r["spearman_match_metrics"]={k: bool(abs(sp_re[k]-m_arm[k]["spearman_vs_raw_truth"])<1e-10) for k in replay}
g=pred.groupby("component_id")["split"].nunique()
r["components_with_multi_split"]=int((g>1).sum())
r["same_rows_across_arms"]=bool(pred["gene_id"].nunique()==len(pred))

# inverse target units: verify predictions are in target units (rank preds in [0,1])
r["rank_preds_in_01"]=bool(pred["ridge_rank_pred"].min()>=-1e-9 and pred["ridge_rank_pred"].max()<=1+1e-9 and pred["hgb_rank_pred"].min()>=-1e-9 and pred["hgb_rank_pred"].max()<=1+1e-9)
r["raw_preds_negative"]=bool(pred["ridge_raw_pred"].max()<0 and pred["hgb_raw_pred"].max()<0)

print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
