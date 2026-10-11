#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31 section C: SuRE TRAIN-region-only repeat-vs-repeat QC. ZERO fits."""
import csv, json, hashlib, math, os
from collections import defaultdict
import numpy as np
from scipy import stats

SEED = 20261031
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS = os.path.join(REPO, "data/breakthrough_20261002/batch29/rows.tsv")
SPLIT = os.path.join(REPO, "data/breakthrough_20261002/batch30/split.tsv")
OUT = os.path.join(REPO, "reports/breakthrough_20261002/batch31")
P_BINS = [(1,1,"P1"),(2,3,"P2_3"),(4,8,"P4_8"),(9,10**9,"P9plus")]
LEN_BINS = [(200,399,"L200_399"),(400,799,"L400_799"),(800,1199,"L800_1199"),(1200,2000,"L1200_2000")]
def pbin(p):
    for lo,hi,name in P_BINS:
        if lo<=p<=hi: return name
    return "undef"
def lbin(l):
    for lo,hi,name in LEN_BINS:
        if lo<=l<=hi: return name
    return "undef"
def hashrank(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
def spearman(x, y):
    if len(x)<3: return None
    if np.all(np.asarray(x)==x[0]) or np.all(np.asarray(y)==y[0]): return None
    rho,p = stats.spearmanr(x,y)
    if math.isnan(rho): return None
    return float(rho)

def top5(rows, value_key):
    n = len(rows)
    k = int(math.ceil(0.05*n))
    if k==0: return set(), None, 0.0, 0.0
    ranked = sorted(rows, key=lambda r:(-int(r[value_key]), hashrank(r["key"])))
    sel = ranked[:k]
    cutoff = int(sel[-1][value_key])
    n_at_cut = sum(1 for r in rows if int(r[value_key])==cutoff)
    return set(r["key"] for r in sel), cutoff, n_at_cut/n, (1.0 if cutoff==0 else 0.0)

def strat_report(rows, label):
    n=len(rows)
    if n==0: return {"n":0,"note":"empty"}
    x=[int(r["r1"]) for r in rows]; y=[int(r["r2"]) for r in rows]
    groups=len(set(r["group"] for r in rows))
    both_pos=sum(1 for a,b in zip(x,y) if a>0 and b>0)
    both_zero=sum(1 for a,b in zip(x,y) if a==0 and b==0)
    one_pos=sum(1 for a,b in zip(x,y) if (a>0) != (b>0))
    return {"n":n,"n_groups":groups,"spearman_r1_r2":spearman(x,y),
            "frac_both_positive":both_pos/n,"frac_both_zero":both_zero/n,
            "frac_one_positive":one_pos/n,"const_r1":(max(x)-min(x)==0),"const_r2":(max(y)-min(y)==0)}

def load():
    rows=[]
    with open(ROWS, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"): rows.append(r)
    splitmap={}
    with open(SPLIT, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"): splitmap[r["key"]]=r["split"]
    mism=sum(1 for r in rows if splitmap.get(r["key"])!=r["split"])
    return rows, mism

def main():
    rows_all, mism = load()
    tr=[r for r in rows_all if r["split"]=="train"]
    res={}
    res["split"]={"source_rows_tsv":ROWS,"split_tsv":SPLIT,"split_mismatches_vs_batch30":mism,
        "n_train":len(tr),"seed":SEED,"note":"TRAIN chr1-16 only; test never opened; no b29 model metrics used"}
    res["overall"]=strat_report(tr,"overall")
    res["by_P_bin"]={name:strat_report([r for r in tr if pbin(int(r["P"]))==name],"P") for _,_,name in P_BINS}
    res["by_len_bin"]={name:strat_report([r for r in tr if lbin(int(r["len"]))==name],"L") for _,_,name in LEN_BINS}
    grid={}
    for _,_,pn in P_BINS:
        for _,_,ln in LEN_BINS:
            grid[pn+"|"+ln]=strat_report([r for r in tr if pbin(int(r["P"]))==pn and lbin(int(r["len"]))==ln],"cell")
    res["grid_P_x_len"]=grid
    res["P1_focus"]=strat_report([r for r in tr if int(r["P"])==1],"P1")
    s1,c1,t1,z1 = top5(tr,"r1")
    s2,c2,t2,z2 = top5(tr,"r2")
    k=len(s1); inter=len(s1&s2); union=len(s1|s2)
    res["top5"]={"k":k,"cutoff_r1":c1,"cutoff_r2":c2,"frac_tied_at_cutoff_r1":t1,
        "frac_tied_at_cutoff_r2":t2,"cutoff_is_zero_r1":(c1==0),"cutoff_is_zero_r2":(c2==0),
        "intersection":inter,"union":union,"jaccard":(inter/union if union else None),
        "observed_overlap_frac_of_k":(inter/k if k else None)}

    rng=np.random.default_rng(SEED)
    strat=defaultdict(list)
    for r in tr: strat[(pbin(int(r["P"])),lbin(int(r["len"])))].append(r)
    groups_list=list(strat.values())
    perm_inter=[]; R=1000
    for it in range(R):
        y_perm={}
        for g in groups_list:
            vals=[int(r["r2"]) for r in g]
            rng.shuffle(vals)
            for r,v in zip(g,vals): y_perm[r["key"]]=v
        ranked=sorted(tr, key=lambda r:(-y_perm[r["key"]], hashrank(r["key"])))[:k]
        ii=len(s1&set(r["key"] for r in ranked)); perm_inter.append(ii)
    perm_inter=np.array(perm_inter,float)
    null_mean=float(perm_inter.mean()); null_p95=float(np.percentile(perm_inter,95))
    enr=(inter/null_mean) if null_mean>0 else None
    res["permutation_reference"]={"R":R,"strata":"P_bin x len_bin","permuted_variable":"repeat2_labels_within_stratum",
        "null_mean_intersection":null_mean,"null_p95_intersection":null_p95,"observed_intersection":inter,
        "enrichment_obs_over_nullmean":enr,
        "scope":"conditional descriptive reference within depth/length strata; NOT formal significance or leakage proof"}
    groups=defaultdict(list)
    for r in tr: groups[r["group"]].append(r)
    gkeys=list(groups.keys()); ng=len(gkeys)
    brng=np.random.default_rng(SEED+1)
    rho_bs=[]
    for it in range(2000):
        pick=brng.integers(0,ng,ng); samp=[]
        for gi in pick: samp.extend(groups[gkeys[gi]])
        rr=spearman([int(r["r1"]) for r in samp],[int(r["r2"]) for r in samp])
        if rr is not None: rho_bs.append(rr)
    rho_bs=np.array(rho_bs,float)
    def ci(a,q): return float(np.percentile(a,q)) if len(a) else None
    res["group_bootstrap"]={"n_groups":ng,"draws":2000,"rho_mean":(float(rho_bs.mean()) if len(rho_bs) else None),
        "rho_ci95_lower":ci(rho_bs,2.5),"rho_ci95_upper":ci(rho_bs,97.5),
        "scope":"group(frozen groupID)-level resampling; respects multiple rows/group; not independent-row CI"}

    g=res["overall"]["spearman_r1_r2"]; rho_lower=res["group_bootstrap"]["rho_ci95_lower"]
    p1=[r for r in tr if int(r["P"])==1]
    pg=defaultdict(list)
    for r in p1: pg[r["group"]].append(r)
    pgk=list(pg.keys()); png=len(pgk)
    prng=np.random.default_rng(SEED+2); p1_bs=[]
    for it in range(2000):
        pick=prng.integers(0,png,png); samp=[]
        for gi in pick: samp.extend(pg[pgk[gi]])
        rr=spearman([int(r["r1"]) for r in samp],[int(r["r2"]) for r in samp])
        if rr is not None: p1_bs.append(rr)
    p1_bs=np.array(p1_bs,float)
    p1_lower=float(np.percentile(p1_bs,2.5)) if len(p1_bs) else None
    gate={"definition":"Astra resource gate for FUTURE budget priority only; NOT reliability ceiling / promotion rule",
        "comp1_train_rho_ge_0.30":(g is not None and g>=0.30),
        "comp2_group_ci95_lower_gt_0.15":(rho_lower is not None and rho_lower>0.15),
        "comp3_P1_ci95_lower_gt_0":(p1_lower is not None and p1_lower>0),
        "comp4_top5_enrichment_ge_2":(enr is not None and enr>=2.0),
        "observed":{"train_rho":g,"group_ci95_lower":rho_lower,"P1_rho":res["P1_focus"]["spearman_r1_r2"],
                    "P1_ci95_lower":p1_lower,"top5_enrichment":enr},
        "insufficient_evidence":((len(rho_bs)<1500) or (len(p1_bs)<1500))}
    res["resource_gate"]=gate
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT,"sure_qc_metrics.json"),"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
    print(json.dumps({"overall":res["overall"],"top5":res["top5"],"perm":res["permutation_reference"],
        "boot":{"rho":res["group_bootstrap"]["rho_mean"],"lo":rho_lower},"P1":res["P1_focus"],
        "gate":gate["observed"]},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
