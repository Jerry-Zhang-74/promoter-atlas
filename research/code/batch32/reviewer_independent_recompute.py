#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH32 REVIEWER independent recompute. ZERO fits/GPU. Own implementation of
interval overlap + subgroup Spearman + group bootstrap; compares against worker outputs."""
import csv, gzip, json, math, os, bisect, hashlib
import numpy as np
from scipy import stats

HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.dirname(os.path.dirname(HERE))
B32=os.path.join(REPO,"data/breakthrough_20261002/batch32")
LIB=os.path.join(B32,"inputs/promoter_library")
SEQ=os.path.join(REPO,"data/breakthrough_20261002/batch22/sequence_table.tsv")
RANK=os.path.join(REPO,"reports/breakthrough_20261002/batch31/tables/hek673_ranking.tsv")
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch32")

def rho(x,y):
    if len(x)<3: return None
    r=stats.spearmanr(x,y)[0]
    return None if (r is None or math.isnan(r)) else float(r)
def rd(p):
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f,delimiter="\t"))

# ---- own library load (independent parse) ----
lib={}
for fn in os.listdir(LIB):
    if not fn.endswith(".txt.gz"): continue
    chrom=fn[:-7]
    S=[];E=[]
    with gzip.open(os.path.join(LIB,fn),"rt",encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<4: continue
            S.append(int(p[1]));E.append(int(p[2]))
    S=np.array(S);E=np.array(E)
    o=np.argsort(S,kind="mergesort")
    lib[chrom]=(S[o],E[o])
print("lib chroms:",len(lib),"total intervals:",sum(len(v[0]) for v in lib.values()))

seq=rd(SEQ)
flags={}
counts={"OVERLAP_any":0,"DISJOINT":0,"UNKNOWN":0}
for r in seq:
    ch=(r.get("chr") or "").strip()
    if ch not in lib:
        flags[r["prom_id"]]="UNKNOWN"; counts["UNKNOWN"]+=1; continue
    a0=int(r["start1"])-1; a1=int(r["end"])
    S,E=lib[ch]
    ml=int(np.max(E-S))
    lo=bisect.bisect_left(S,a0-ml); hi=bisect.bisect_left(S,a1)
    hit=any((S[i]<a1 and E[i]>a0) for i in range(lo,hi))
    flags[r["prom_id"]]="OVERLAP_any" if hit else "DISJOINT"
    counts["OVERLAP_any" if hit else "DISJOINT"]+=1
print("my counts:",counts)

# ---- compare against worker flags ----
wf={r["prom_id"]:r["subset"] for r in rd(os.path.join(OUT,"tables/hek673_overlap_flags.tsv"))}
wmap={"UNKNOWN_chrom_not_distributed":"UNKNOWN","UNKNOWN_no_coord":"UNKNOWN"}
mism=[]
for pid in flags:
    if flags[pid]!=wmap.get(wf.get(pid),wf.get(pid)): mism.append((pid,flags[pid],wf.get(pid)))
print("flag mismatches:",len(mism), mism[:5])
print("worker counts:", {k:v for k,v in rd(os.path.join(OUT,"../../runs/breakthrough_20261002/batch32/../../reports/breakthrough_20261002/batch32/status.json"))} if False else "")

# ---- ranking + subgroups ----
rank=rd(RANK)
ids=[r["prom_id"] for r in rank]
y=np.array([float(r["measured_hek_norm"]) for r in rank])
g=[r["group_id"] for r in rank]
P={"PARM":np.array([float(r["PARM_HEK293"]) for r in rank]),
   "R1":np.array([float(r["R1_oof"]) for r in rank]),
   "H1":np.array([float(r["H1_oof"]) for r in rank])}
print("n rows",len(ids),"distinct groups",len(set(g)))
print("full673 rho:",{m:rho(list(P[m]),list(y)) for m in P})
# no-coordinate check
drop=[pid for pid in ids if flags[pid]=="UNKNOWN"]
print("unknown ids:",drop)

def topk(vals,idlist,k):
    order=sorted(range(len(idlist)),key=lambda i:(-vals[i],hashlib.sha256(idlist[i].encode()).hexdigest()))
    return set(idlist[i] for i in order[:k])

def subgroup(name,idx):
    if not idx: return {"n":0}
    yy=[y[i] for i in idx]; gg=[g[i] for i in idx]
    kk=int(math.ceil(0.1*len(idx)))
    ot=topk(yy,[ids[i] for i in idx],kk)
    return {"n":len(idx),"n_groups":len(set(gg)),"k":kk,
            "rho":{m:rho([P[m][i] for i in idx],yy) for m in P},
            "top_prec":{m:len(topk([P[m][i] for i in idx],[ids[i] for i in idx],kk)&ot)/kk for m in P}}

ov=[i for i,pid in enumerate(ids) if flags[pid]=="OVERLAP_any"]
dj=[i for i,pid in enumerate(ids) if flags[pid]=="DISJOINT"]
print("OVERLAP:",subgroup("ov",ov))
print("DISJOINT:",subgroup("dj",dj))

# ---- group bootstrap for OVERLAP (multiplicities kept) ----
def boot(idx,draws=2000,seed=20261032):
    gg=[g[i] for i in idx]; ug=sorted(set(gg))
    if len(ug)<100: return {"note":"<100 groups"}
    byg={u:[i for i in idx if g[i]==u] for u in ug}
    rng=np.random.default_rng(seed); d=[];rP=[];rR=[]
    for _ in range(draws):
        pick=rng.integers(0,len(ug),len(ug))
        sel=np.concatenate([byg[ug[j]] for j in pick])
        yy=[y[i] for i in sel]
        rp=rho([P["PARM"][i] for i in sel],yy); r1=rho([P["R1"][i] for i in sel],yy)
        if rp is not None: rP.append(rp)
        if r1 is not None: rR.append(r1)
        if rp is not None and r1 is not None: d.append(rp-r1)
    def ci(a): a=np.array(a); return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5)),float(a.mean()),int(a.size)]
    return {"PARM":ci(rP),"R1":ci(rR),"PARM-R1":ci(d),"n_groups":len(ug)}
print("OVERLAP bootstrap:",boot(ov))

print("DONE")
