#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent review computations for batch09 v6.10.0-starget.rc1 (part 1: provenance/population)."""
import hashlib, json, math, statistics
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("E:/promoter_atlas")
EXPR = ROOT/"labels/build/gene_expression_table.tsv"
CANON = ROOT/"labels/build/gene_expression_label_table.tsv"
AUDIT = ROOT/"data/breakthrough_20261002/batch08/label_audit_all_genes.tsv"
TRACKA = ROOT/"data/breakthrough_20261002/batch03/trackA_gene_promoters.npz"
SPLIT = ROOT/"data/breakthrough_20261002/batch05/split_arrays.npz"
FEAT = ROOT/"data/breakthrough_20261002/batch06/features45.npz"
EXPECT_SPLIT = "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5"
EXPECT_FEAT = "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800"

def sha(p):
    with open(p,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
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
        tm=gt[g]
        coll=[collapse_dup(tm[t]) if len(tm[t])>1 else tm[t][0] for t in sorted(tm)]
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

r={}
r["source_sha"]={p.name:sha(p) for p in (EXPR,CANON,AUDIT,TRACKA,SPLIT,FEAT)}
r["split_sha_ok"]=sha(SPLIT)==EXPECT_SPLIT
r["feature_sha_ok"]=sha(FEAT)==EXPECT_FEAT

z=np.load(TRACKA,allow_pickle=True)
gene_gid=z["gene_id"].astype(str)[z["starts"].astype(np.int64)[:-1]]
s=np.load(SPLIT,allow_pickle=True)
gene_split=s["gene_split"]; comp=s["comp_dense"].astype(np.int64); n_prom=s["n_promoters"].astype(np.int64)
r["universe_57827"]=int(len(gene_gid)==57827)

zf=np.load(FEAT,allow_pickle=True)
comp44=zf["comp44"].astype(np.float64); l1c=zf["log1p_count"].astype(np.float64)
X=np.concatenate([comp44,l1c[:,None]],axis=1)
r["features_shape"]=list(X.shape)
r["log1p_count_eq_n_prom"]=bool(np.allclose(X[:,44], np.log1p(n_prom.astype(float))))
r["len_cols"]=[float(X[0,21]), float(X[0,43])]
r["gc_in_01"]=bool(X[:,20].min()>=-1e-12 and X[:,20].max()<=1+1e-12 and X[:,42].min()>=-1e-12 and X[:,42].max()<=1+1e-12)
r["features_finite"]=bool(np.isfinite(X).all())

full_s=reconstruct_s(EXPR)
r["S_defined_full"]=int(len(full_s))
r["S_defined_18947"]=bool(len(full_s)==18947)

canon=pd.read_csv(CANON, sep="\t", low_memory=False)
r["n_canonical"]=int(len(canon))
audit=pd.read_csv(AUDIT, sep="\t", low_memory=False)
r["n_audit"]=int(len(audit))
defined=set(audit.loc[audit["S_defined_main"]==1,"gene_id"])
r["S_defined_audit"]=int(len(defined))

feat_ids=set(gene_gid)
eligible=defined & feat_ids
excluded=defined - feat_ids
r["eligible"]=int(len(eligible)); r["excluded"]=int(len(excluded))
r["eligible_18769"]=bool(len(eligible)==18769)
r["excluded_178"]=bool(len(excluded)==178)
em=np.array([g in eligible for g in gene_gid])
r["eligible_rows"]=int(em.sum())
r["eligible_rows_eq_gene_count"]=bool(em.sum()==len(eligible))

sc=Counter(gene_split[em])
r["split_counts"]={k:int(v) for k,v in sc.items()}
r["split_components"]={sp:int(np.unique(comp[em & (gene_split==sp)]).size) for sp in ("train","val","test")}
r["total_components"]=int(np.unique(comp[em]).size)

# excluded gene characterization
ex_rows=audit.loc[audit["gene_id"].isin(excluded)]
r["excluded_tissue_counts"]=sorted(ex_rows["tissue_count"].unique().tolist())
r["excluded_B_raw_min"]=float(ex_rows["B_raw"].min())
r["excluded_B_raw_max"]=float(ex_rows["B_raw"].max())

# tie ordering
tsv_s=dict(zip(audit.loc[audit["S_defined_main"]==1,"gene_id"], audit.loc[audit["S_defined_main"]==1,"S_raw_full"].astype(float)))
common=sorted(set(full_s)&set(tsv_s))
a=np.array([full_s[x] for x in common]); b=np.array([tsv_s[x] for x in common])
r["export_vs_mem_max_abs_diff"]=float(np.max(np.abs(a-b)))
r["export_vs_mem_spearman"]=spearman(a,b)
ra=rankdata(a); rb=rankdata(b); r["export_vs_mem_rank_differ"]=int((ra!=rb).sum())
tg=defaultdict(list)
for g,v in full_s.items(): tg[v].append(g)
r["true_tie_groups"]=sum(1 for v in tg.values() if len(v)>1)

# verify full_s == defined (same gene sets)
r["full_s_set_eq_defined_set"]=bool(set(full_s)==defined)

print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
