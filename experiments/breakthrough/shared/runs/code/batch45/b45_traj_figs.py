# -*- coding: utf-8 -*-
"""batch45 phase C: fixed secondary generation trajectories + 3 useful figures."""
import json, math, collections, sys, pickle, statistics
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=C.OUT; FIG=C.FIG; FIG.mkdir(parents=True, exist_ok=True)
per=pickle.loads((OUT/"_per.pkl").read_bytes())
by_info={v["info"]:v for v in per.values()}
def wj(n,o): (OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
import re
pat=re.compile(r"^K562_fitness_(?:232bp_)?replicate(?P<rep>\d+)_generation(?P<gen>\d+)_individual\d+_(?P<sel>max|median|min)$")
traj=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for v in per.values():
    m=pat.match(v["info"])
    if m: traj[v["lib"]][int(m["gen"])][m["sel"]].append(v)
gens=[0,4,8,12,16,20,24,28,32,36,40,44,48]
trajout={}
for L in ["short","long"]:
    trajout[L]={}
    for g in gens:
        rec={}
        for sel in ["min","median","max"]:
            vs=traj[L][g][sel]
            if vs:
                rec[sel]={"n_lineages":len(vs),
                  "mean_pred_K562":float(np.mean([x["pred_K562"] for x in vs])),
                  "mean_meas_K562":float(np.mean([x["K562_meas"] for x in vs if x["K562_meas"] is not None])),
                  "mean_pred_HepG2":float(np.mean([x["pred_HepG2"] for x in vs])),
                  "mean_meas_HepG2":float(np.mean([x["HepG2_meas"] for x in vs if x["HepG2_meas"] is not None]))}
        trajout[L][g]=rec
wj("trajectories_secondary.json",{"note":"fixed secondary trajectory; per-generation mean over the 10 GA search lineages (NOT independent replicates)","gens":gens,"data":trajout})
# ---------------- FIG 1: native design trajectory, measured vs score side-by-side
fig,axes=plt.subplots(2,2,figsize=(12,8))
for j,L in enumerate(["short","long"]):
    ax=axes[0][j]
    for sel,c in [("min","#888"),("median","#1f77b4"),("max","#d62728")]:
        xs=[];ys=[]
        for g in gens:
            r=trajout[L][g].get(sel)
            if r: xs.append(g); ys.append(r["mean_pred_K562"])
        ax.plot(xs,ys,"-o",color=c,label=sel)
    ax.set_title(f"{L} library ({'70 bp' if L=='short' else '230 bp synthetic / 232 bp control'})")
    ax.set_xlabel("GA generation"); ax.set_ylabel("PARM-predicted K562 score (published, source units)")
    ax.legend(title="selection"); ax.grid(alpha=.3)
    ax2=axes[1][j]
    for sel,c in [("min","#888"),("median","#1f77b4"),("max","#d62728")]:
        xs=[];ys=[]
        for g in gens:
            r=trajout[L][g].get(sel)
            if r: xs.append(g); ys.append(r["mean_meas_K562"])
        ax2.plot(xs,ys,"-o",color=c,label=sel)
    ax2.set_xlabel("GA generation"); ax2.set_ylabel("Measured K562 activity (mean log2 RNA/DNA)")
    ax2.legend(title="selection"); ax2.grid(alpha=.3)
fig.suptitle("Fig1  Native K562 design trajectory: published score (top) vs measured activity (bottom) - separate units, NOT comparable axes")
fig.tight_layout(); fig.savefig(FIG/"fig1_native_trajectory.png",dpi=130); plt.close(fig)
print("fig1 done")

# ---------------- FIG 2: ranking utility + measurement replicates
sec=json.loads((OUT/"secondary_reliability.json").read_text(encoding="utf-8"))
fig,axes=plt.subplots(1,3,figsize=(16,5))
ax=axes[0]
for L,mk in [("short","o"),("long","s")]:
    for sg,col in [("synthetic_designed","#1f77b4"),("natural_control","#ff7f0e")]:
        xs=[p["pred_K562"] for p in per.values() if p["lib"]==L and (p["category"] in {"design","best","top_individual","MUTATED"})==(sg=="synthetic_designed") and p["K562_meas"] is not None]
        ys=[p["K562_meas"] for p in per.values() if p["lib"]==L and (p["category"] in {"design","best","top_individual","MUTATED"})==(sg=="synthetic_designed") and p["K562_meas"] is not None]
        ax.scatter(xs,ys,s=8,marker=mk,color=col,alpha=.5,label=f"{L} {sg.split('_')[0]}")
ax.set_xlabel("PARM-predicted K562 score (source units)"); ax.set_ylabel("Measured K562 activity (log2 RNA/DNA)")
ax.set_title("(a) ranking utility (published pred vs measured)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax=axes[1]
for L,col in [("short","#2ca02c"),("long","#9467bd")]:
    xs=[p["K562_r1_mean"] for p in per.values() if p["lib"]==L and p["K562_r1_mean"] is not None and p["K562_r2_mean"] is not None]
    ys=[p["K562_r2_mean"] for p in per.values() if p["lib"]==L and p["K562_r1_mean"] is not None and p["K562_r2_mean"] is not None]
    ax.scatter(xs,ys,s=8,color=col,alpha=.5,label=L)
ax.set_xlabel("Assay summary replicate 1 (log2)"); ax.set_ylabel("Assay summary replicate 2 (log2)")
ax.set_title("(b) measurement replicate concordance (diagnostic, not ceiling)"); ax.legend(); ax.grid(alpha=.3)
ax=axes[2]
labels=[];vals=[];gcvals=[]
for k,v in sec["by_library_cell_subgroup"].items():
    if k.split(":")[2] in ("synthetic_designed","natural_control"):
        labels.append(k.replace("synthetic_designed","syn").replace("natural_control","nat"))
        vals.append(v["pred_vs_measured"]["rho"]); gcvals.append(v["gc_vs_measured_baseline"]["rho"])
x=np.arange(len(labels)); ax.bar(x-0.2,vals,0.4,label="source predictor",color="#1f77b4"); ax.bar(x+0.2,gcvals,0.4,label="GC baseline",color="#bbbbbb")
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=60,ha="right",fontsize=7); ax.set_ylabel("Spearman rho"); ax.set_title("(c) predictor vs GC baseline"); ax.legend(); ax.grid(alpha=.3,axis="y")
fig.suptitle("Fig2  Ranking utility, measurement replicates, and baseline (source predictions are analysis inputs, not new model output)")
fig.tight_layout(); fig.savefig(FIG/"fig2_ranking_replicates.png",dpi=130); plt.close(fig)
print("fig2 done")

# ---------------- FIG 3: evidence scope / denominators
fig,axes=plt.subplots(1,2,figsize=(13,5))
cats=["design","best","top_individual","MUTATED","control"]
short=[sum(1 for p in per.values() if p["lib"]=="short" and p["category"]==c) for c in cats]
long_=[sum(1 for p in per.values() if p["lib"]=="long" and p["category"]==c) for c in cats]
ax=axes[0]; x=np.arange(len(cats))
ax.bar(x-0.2,short,0.4,label="short.70bp",color="#2ca02c"); ax.bar(x+0.2,long_,0.4,label="long.232bp",color="#9467bd")
ax.set_xticks(x); ax.set_xticklabels(cats,rotation=30,ha="right"); ax.set_ylabel("unique query DNA"); ax.set_title("(a) evidence scope by category"); ax.legend(); ax.grid(alpha=.3,axis="y")
for i,(a,b) in enumerate(zip(short,long_)): ax.text(i-0.2,a+2,str(a),ha="center",fontsize=7); ax.text(i+0.2,b+2,str(b),ha="center",fontsize=7)
ax=axes[1]
t10=[];lab=[]
for k,v in sec["by_library_cell_subgroup"].items():
    if k.split(":")[2] in ("synthetic_designed","natural_control"):
        lab.append(k.replace("synthetic_designed","syn").replace("natural_control","nat")); t10.append(v["top10"].get("precision"))
x=np.arange(len(lab)); ax.bar(x,t10,color="#1f77b4"); ax.axhline(0.1,color="k",ls="--",lw=1,label="chance (0.1)")
ax.set_xticks(x); ax.set_xticklabels(lab,rotation=60,ha="right",fontsize=7); ax.set_ylabel("top-10% precision"); ax.set_title("(b) top-10% precision (k=ceil(0.1n))"); ax.legend(); ax.grid(alpha=.3,axis="y")
fig.suptitle("Fig3  Scope and denominators (889 unique DNA = 4 barcodes each; counts are NOT independent biological n)")
fig.tight_layout(); fig.savefig(FIG/"fig3_evidence_scope.png",dpi=130); plt.close(fig)
print("fig3 done")
