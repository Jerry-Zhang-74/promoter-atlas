#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31 section C CORRECTED: SuRE TRAIN-region-only repeat-vs-repeat QC on the
PREDECLARED FIXED-PROXY LABELS (not raw counts), zero model inference.
Declared proxy (batch29 selected_rows.json / target_freeze.json offsets):
  z1 = log2(1 + (r1/N_R1) / ((1+P)/N_D)), z2 likewise with r2/N_R2
N_R1=37770653  N_R2=74081034  N_D_1plusP=203742844.
Fixes vs provisional raw-count v1 (PARENT_QC_FINDINGS_2142.txt):
  * operate on float proxy labels (exact float ranking, no int() quantization)
  * top5 tie policy = exact float desc, then per-row SHA256 identity-hash asc
  * resource gate includes comp5 overlap-enrichment group-bootstrap lower>1
  * TRAIN rows selected by frozen split keys BEFORE parsing numeric labels
  * raw-count diagnostics kept and clearly labelled as secondary
Zero fits. Seed 20261031. Test never analysed/scored."""
import csv, json, hashlib, math, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

SEED = 20261031
N_R1, N_R2, N_D = 37770653, 74081034, 203742844
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS = os.path.join(REPO, "data/breakthrough_20261002/batch29/rows.tsv")
SPLIT = os.path.join(REPO, "data/breakthrough_20261002/batch30/split.tsv")
OUT = os.path.join(REPO, "reports/breakthrough_20261002/batch31")
P_BINS = [(1,1,"P1"),(2,3,"P2_3"),(4,8,"P4_8"),(9,10**9,"P9plus")]
LEN_BINS = [(200,399,"L200_399"),(400,799,"L400_799"),(800,1199,"L800_1199"),(1200,2000,"L1200_2000")]

def pbin(p):
    for lo,hi,nm in P_BINS:
        if lo<=p<=hi: return nm
    return "undef"
def lbin(l):
    for lo,hi,nm in LEN_BINS:
        if lo<=l<=hi: return nm
    return "undef"
def h64(key):
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],16)
def zproxy(r,P,N):
    return math.log2(1.0 + (r/N)/((1.0+P)/N_D))
def spearman(x,y):
    if len(x)<3: return None
    if np.all(np.asarray(x)==x[0]) or np.all(np.asarray(y)==y[0]): return None
    rho,_=stats.spearmanr(x,y)
    return None if math.isnan(rho) else float(rho)

def load():
    train=set()
    with open(SPLIT, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"]=="train": train.add(r["key"])
    rows=[]; scanned=0; matched=0; mism=0
    with open(ROWS, encoding="utf-8") as f:
        rd=csv.DictReader(f, delimiter="\t")
        for r in rd:
            scanned+=1
            if r["key"] in train:
                matched+=1
                if r["split"]!="train": mism+=1
                rows.append(r)
    return rows, {"rows_tsv_scanned":scanned,"frozen_train_keys":len(train),
                  "matched_train_rows":matched,"split_field_mismatch":mism,
                  "policy":"TRAIN rows selected by frozen SPLIT keys BEFORE parsing numeric labels; "
                           "rows.tsv fully scanned only to locate keys; NO test-row numeric label parsed/analysed/scored"}

def strat_report(rows):
    n=len(rows)
    if n==0: return {"n":0,"note":"empty"}
    x=[int(r["r1"]) for r in rows]; y=[int(r["r2"]) for r in rows]
    p=[int(r["P"]) for r in rows]
    z1=[zproxy(a,pi,N_R1) for a,pi in zip(x,p)]; z2=[zproxy(b,pi,N_R2) for b,pi in zip(y,p)]
    groups=len(set(r["group"] for r in rows))
    bp=sum(1 for a,b in zip(x,y) if a>0 and b>0); bz=sum(1 for a,b in zip(x,y) if a==0 and b==0)
    op=sum(1 for a,b in zip(x,y) if (a>0)!=(b>0))
    return {"n":n,"n_groups":groups,
            "proxy_spearman_z1_z2":spearman(z1,z2),
            "raw_count_spearman_r1_r2_secondary":spearman(x,y),
            "frac_both_positive":bp/n,"frac_both_zero":bz/n,"frac_one_positive":op/n,
            "const_r1":(max(x)-min(x)==0),"const_r2":(max(y)-min(y)==0),
            "const_z1":(max(z1)-min(z1)==0),"const_z2":(max(z2)-min(z2)==0)}

def main():
    rows, audit = load()
    n=len(rows)
    z1=np.array([zproxy(int(r["r1"]),int(r["P"]),N_R1) for r in rows],float)
    z2=np.array([zproxy(int(r["r2"]),int(r["P"]),N_R2) for r in rows],float)
    h=np.array([h64(r["key"]) for r in rows],np.uint64)
    st=np.array([P_BINS.index(next(b for b in P_BINS if b[0]<=int(r["P"])<=b[1]))*4
                 +LEN_BINS.index(next(b for b in LEN_BINS if b[0]<=int(r["len"])<=b[1])) for r in rows],int)
    gid={}
    for i,r in enumerate(rows): gid.setdefault(r["group"],[]).append(i)
    gvec=np.array([gid[r["group"]] for r in rows],dtype=object)
    gkeys=list(gid.keys())
    def topk(z,h,idx,k):
        zz=z[idx]; hh=h[idx]; o=np.lexsort((hh,-zz)); return idx[o[:k]]
    k=int(math.ceil(0.05*n)); allidx=np.arange(n)
    s1=set(topk(z1,h,allidx,k).tolist()); s2=set(topk(z2,h,allidx,k).tolist())
    inter=len(s1&s2); union=len(s1|s2)
    cut1=float(z1[sorted(s1)[0]]) if False else None
    # cutoff diagnostics on exact float values (no int quantization)
    o1=np.lexsort((h,-z1))[:k]; o2=np.lexsort((h,-z2))[:k]
    cutv1=float(z1[o1[-1]]); cutv2=float(z2[o2[-1]])
    nat1=int(np.sum(z1==cutv1)); nat2=int(np.sum(z2==cutv2))
    top5={"k":k,"cutoff_z1":cutv1,"cutoff_z2":cutv2,
          "frac_tied_at_cutoff_z1":nat1/n,"frac_tied_at_cutoff_z2":nat2/n,
          "cutoff_equals_min_z1":bool(cutv1==float(z1.min())),"cutoff_equals_min_z2":bool(cutv2==float(z2.min())),
          "intersection":inter,"union":union,"jaccard":(inter/union if union else None),
          "observed_overlap_frac_of_k":(inter/k if k else None),
          "tie_policy":"exact float proxy descending, then per-row SHA256 identity-hash ascending; NO int() quantization"}
    # within-stratum permutation of repeat2 proxy labels
    def perm_topk_intersection(z2,h,st,idx,k,base,rng):
        zz=z2[idx].copy(); sx=st[idx]
        for s in np.unique(sx):
            pos=np.flatnonzero(sx==s); zz[pos]=zz[pos][rng.permutation(pos.size)]
        o=np.lexsort((h[idx],-zz)); sel=set(idx[o[:k]].tolist()); return len(base&sel)
    rng=np.random.default_rng(SEED); R=1000; perm=[]
    for _ in range(R):
        perm.append(perm_topk_intersection(z2,h,st,allidx,k,s1,rng))
    perm=np.array(perm,float); nmean=float(perm.mean()); np95=float(np.percentile(perm,95))
    enr=(inter/nmean) if nmean>0 else None
    res={"labels":{"declared_proxy":"z = log2(1 + (r/N_R)/((1+P)/N_D))","N_R1":N_R1,"N_R2":N_R2,"N_D_1plusP":N_D,
                   "source":"data/breakthrough_20261002/batch29/selected_rows.json + target_freeze.json",
                   "primary_analysis":"predeclared fixed-proxy labels","raw_counts":"secondary diagnostic only"},
         "seed":SEED,"audit":audit,"n_train_rows_analysed":n}
    res["overall"]=strat_report(rows)
    res["by_P_bin"]={nm:strat_report([r for r in rows if pbin(int(r["P"]))==nm]) for _,_,nm in P_BINS}
    res["by_len_bin"]={nm:strat_report([r for r in rows if lbin(int(r["len"]))==nm]) for _,_,nm in LEN_BINS}
    res["grid_P_x_len"]={pn+"|"+ln:strat_report([r for r in rows if pbin(int(r["P"]))==pn and lbin(int(r["len"]))==ln])
                         for _,_,pn in P_BINS for _,_,ln in LEN_BINS}
    res["P1_focus"]=strat_report([r for r in rows if int(r["P"])==1])
    res["top5"]=top5
    res["permutation_reference"]={"R":R,"strata":"P_bin x len_bin","permuted_variable":"repeat2_PROXY_labels_within_stratum",
        "null_mean_intersection":nmean,"null_p95_intersection":np95,"observed_intersection":inter,
        "enrichment_obs_over_nullmean":enr,
        "scope":"conditional descriptive reference within depth/length strata; NOT formal significance or leakage proof"}

    gindices=[np.array(gid[gk]) for gk in gkeys]; ng=len(gkeys)
    def boot_rho(z1,z2,gindices,ng,B,seed):
        rng=np.random.default_rng(seed); out=[]
        for _ in range(B):
            pick=rng.integers(0,ng,ng)
            idx=np.concatenate([gindices[gi] for gi in pick])
            r=spearman(z1[idx],z2[idx])
            if r is not None: out.append(r)
        return np.array(out,float)
    rho_bs=boot_rho(z1,z2,gindices,ng,2000,SEED+1)
    def ci(a,q): return float(np.percentile(a,q)) if len(a) else None
    res["group_bootstrap"]={"n_groups":ng,"draws":2000,"rho_mean":(float(rho_bs.mean()) if len(rho_bs) else None),
        "rho_ci95_lower":ci(rho_bs,2.5),"rho_ci95_upper":ci(rho_bs,97.5),
        "scope":"group(frozen groupID)-level resampling on PROXY labels; respects multiple rows/group; not an independent-row CI"}
    p1mask=np.array([int(r["P"])==1 for r in rows]); p1idx=np.flatnonzero(p1mask)
    p1gid={}
    for i in p1idx: p1gid.setdefault(rows[i]["group"],[]).append(int(i))
    p1gind=[np.array(v) for v in p1gid.values()]; p1ng=len(p1gind)
    p1_bs=boot_rho(z1,z2,p1gind,p1ng,2000,SEED+2)
    p1_lower=ci(p1_bs,2.5)
    # comp5: group-bootstrap the top5 overlap ENRICHMENT (conditional within-stratum reference)
    R_INNER=20; rng5=np.random.default_rng(SEED+3); enr_bs=[]
    for _ in range(2000):
        pick=rng5.integers(0,ng,ng)
        idx=np.concatenate([gindices[gi] for gi in pick])
        kb=int(math.ceil(0.05*len(idx)))
        s1b=set(topk(z1,h,idx,kb).tolist()); s2b=set(topk(z2,h,idx,kb).tolist())
        inter_b=len(s1b&s2b)
        nulls=[perm_topk_intersection(z2,h,st,idx,kb,s1b,rng5) for _ in range(R_INNER)]
        nm=float(np.mean(nulls))
        if nm>0: enr_bs.append(inter_b/nm)
    enr_bs=np.array(enr_bs,float)
    res["enrichment_group_bootstrap"]={"draws":2000,"R_inner_within_stratum_perms":R_INNER,
        "enrichment_mean":(float(enr_bs.mean()) if len(enr_bs) else None),
        "enrichment_ci95_lower":ci(enr_bs,2.5),"enrichment_ci95_upper":ci(enr_bs,97.5),
        "observed_enrichment":enr,
        "scope":"group-level resampling of the top5-overlap enrichment vs a conditional within-stratum permutation reference; conditional descriptive only"}

    g=res["overall"]["proxy_spearman_z1_z2"]; rho_lower=res["group_bootstrap"]["rho_ci95_lower"]
    enr_lo=res["enrichment_group_bootstrap"]["enrichment_ci95_lower"]
    gate={"definition":"Astra resource gate for FUTURE budget priority only; NOT a reliability ceiling / promotion rule",
        "comp1_train_rho_ge_0.30":(g is not None and g>=0.30),
        "comp2_group_ci95_lower_gt_0.15":(rho_lower is not None and rho_lower>0.15),
        "comp3_P1_ci95_lower_gt_0":(p1_lower is not None and p1_lower>0),
        "comp4_top5_enrichment_ge_2":(enr is not None and enr>=2.0),
        "comp5_top5_enrichment_ci95_lower_gt_1":(enr_lo is not None and enr_lo>1.0),
        "observed":{"train_rho":g,"group_ci95_lower":rho_lower,"P1_rho":res["P1_focus"]["proxy_spearman_z1_z2"],
                    "P1_ci95_lower":p1_lower,"top5_enrichment":enr,"top5_enrichment_ci95_lower":enr_lo},
        "insufficient_evidence":((len(rho_bs)<1500) or (len(p1_bs)<1500) or (len(enr_bs)<1500)),
        "note":"all five components reported; a tie-dominated top set is not counted as strong positive evidence"}
    gate["gate_pass"]=all(gate[k] for k in ["comp1_train_rho_ge_0.30","comp2_group_ci95_lower_gt_0.15",
        "comp3_P1_ci95_lower_gt_0","comp4_top5_enrichment_ge_2","comp5_top5_enrichment_ci95_lower_gt_1"])
    res["resource_gate"]=gate
    res["top5"]["frac_tied_at_cutoff_r1"]=res["top5"]["frac_tied_at_cutoff_z1"]
    res["top5"]["frac_tied_at_cutoff_r2"]=res["top5"]["frac_tied_at_cutoff_z2"]
    v1=json.load(open(os.path.join(OUT,"sure_qc_raw_counts_v1/sure_qc_metrics_v1.json"),encoding="utf-8"))
    res["raw_counts_v1_reference"]={"provisional_overall_raw_count_rho":v1["overall"]["spearman_r1_r2"],
        "corrected_proxy_overall_rho":g,
        "note":"v1 raw-count rho is provisional/secondary; corrected primary analysis uses the predeclared proxy labels; "
               "neither is a hard ceiling"}
    with open(os.path.join(OUT,"sure_qc_metrics.json"),"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
    print(json.dumps({"proxy_overall_rho":g,"raw_v1_rho":v1["overall"]["spearman_r1_r2"],
        "group_ci_lower":rho_lower,"P1_rho":res["P1_focus"]["proxy_spearman_z1_z2"],"P1_ci_lower":p1_lower,
        "top5_inter":inter,"top5_enr":enr,"top5_enr_ci_lower":enr_lo,"gate":gate["observed"],
        "gate_pass":gate["gate_pass"],"audit":audit},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
