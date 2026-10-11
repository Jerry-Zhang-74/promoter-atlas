#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH32 Section C figures (decision-ready). Reads frozen metrics only."""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.dirname(os.path.dirname(HERE))
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch32")
FIG=os.path.join(OUT,"figures"); os.makedirs(FIG,exist_ok=True)
def rj(p): return json.load(open(p,encoding="utf-8"))
A=rj(os.path.join(OUT,"sectionA_metrics.json")); B=rj(os.path.join(OUT,"sectionB_metrics.json"))

# ---- Figure 1: HEK cohort by verified overlap status (+ explicit exposure-unresolved note) ----
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
cnt=A["frozen_counts"]
labels=["OVERLAP\n(coordinate)","DISJOINT\n(within dist. subset)","UNKNOWN\n(chrom not dist.)"]
vals=[cnt["OVERLAP_any"],cnt["DISJOINT"],cnt["UNKNOWN_chrom_not_distributed"]]
axes[0].bar(labels,vals,color=["#c44","#48a","#999"])
for i,v in enumerate(vals): axes[0].text(i,v+3,str(v),ha="center")
axes[0].set_title("HEK673 (n=673) by coordinate-overlap status vs\nofficial distributed PARM library subset")
axes[0].set_ylabel("n fragments"); axes[0].set_ylim(0,max(vals)*1.25)
subs=["OVERLAP_any","DISJOINT","UNKNOWN_chrom_not_distributed"]
x=range(len(subs)); w=0.25
for j,m in enumerate(("PARM","R1","H1")):
    ys=[A["subgroups"][s].get("rho",{}).get(m) for s in subs]
    axes[1].bar([i+(j-1)*w for i in x],[y if y is not None else 0 for y in ys],w,label=m)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(["OVERLAP","DISJOINT","UNKNOWN"])
axes[1].axhline(0,color="k",lw=0.8); axes[1].set_ylabel("Spearman rho (pred vs measured)")
axes[1].set_title("Frozen ranking rho by subgroup\n(CI only if subgroup >=100 groups)")
axes[1].legend()
for s in subs:
    b=A["subgroups"][s].get("bootstrap") or {}
    if b.get("note","").startswith("<100"):
        axes[1].text(subs.index(s),0.05,"<100 groups\nno CI",ha="center",fontsize=7,color="#555")
fig.suptitle("BATCH32 Fig1 | PARM HEK293 = CROSS-CELL; user target = 293FT. HEK293 TRAINING MEMBERSHIP = UNKNOWN "
             "(distributed subset carries HAP1/AGS columns only). DISJOINT != unseen families.",fontsize=8)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(os.path.join(FIG,"fig1_hek_overlap_status.png"),dpi=140); plt.close(fig)

# ---- Figure 2: local 293FT actual length/nesting + D28 observed vs frozen model direction ----
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
famcol={"ACTB":"#1f77b4","GAPDH":"#2ca02c","CMV":"#d62728"}
obs=B["observed_D28"]
# Panel A: observed vs actual length; 2000bp unsupported flagged
for fam,c in famcol.items():
    xs=[100,200,300,400,500] if fam!="CMV" else []
    ids=[f"{fam[0]}1",f"{fam[0]}2",f"{fam[0]}3",f"{fam[0]}4",f"{fam[0]}5"] if fam!="CMV" else []
    if fam!="CMV":
        ys=[obs[i] for i in ids]
        axes[0].plot(xs,ys,"o-",color=c,label=fam)
axes[0].plot([204],[obs["CMVcore"]],"s",color=famcol["CMV"]); axes[0].plot([584],[obs["CMVfull"]],"^",color=famcol["CMV"])
axes[0].annotate("CMVcore 204bp",(204,obs["CMVcore"]),fontsize=7,xytext=(210,obs["CMVcore"]-0.35))
axes[0].annotate("CMVfull 584bp",(584,obs["CMVfull"]),fontsize=7,xytext=(430,obs["CMVfull"]-0.35))
# unsupported 2000bp observed (no PARM row)
lo=[m for m in ("A20_2000bp_unsupported","G20_2000bp_unsupported")]
for k,c in zip(lo,["#1f77b4","#2ca02c"]):
    axes[0].plot([2000],[B["observed_D28"][k]],"x",color=c,ms=10,mew=2)
axes[0].text(2000,1.15,"2000bp observed\n(PARM unsupported\nL_max=600)",ha="center",fontsize=7,color="#666")
axes[0].set_xlabel("actual insert length (bp)"); axes[0].set_ylabel("D28 log10( GFP+ median / WT-neg median )")
axes[0].set_title("Local 293FT D28 endpoint vs actual length")
axes[0].legend(fontsize=8)
# Panel B: observed vs frozen model prediction, per family, direction agreement
for fam,c in famcol.items():
    ids=[k for k in obs if B["nesting_vs_2000bp"] and False]  # placeholder
mk={"PARM":"o","H1":"s","R1":"^"}
for fam,c in famcol.items():
    ids=([f"{fam[0]}{i}" for i in range(1,6)] if fam!="CMV" else ["CMVcore","CMVfull"])
    for m,mm in mk.items():
        axes[1].scatter([0]+[obs[i] for i in ids],[0]+[0]*len(ids),alpha=0.0)
axes[1].axis("off")
txt=["D28 observed vs frozen model direction (pairwise, <=584bp supported):",
     "  family       PARM   H1     R1"]
for fam in ("ACTB","GAPDH","CMV"):
    row=f"  {fam:10s} "
    for m in ("PARM","H1","R1"):
        v=B["direction_vs_frozen_models"][m][fam]["direction_accuracy"]
        row+=f"{(v if v is not None else float('nan')):5.2f}  "
    txt.append(row)
txt.append("  macro      " + "  ".join(f"{B['macro_family_direction'][m]:5.2f} " for m in ("PARM","H1","R1")))
txt.append("")
txt.append("Endpoint = log10( GFP-positive gated median /")
txt.append("           same-batch WT-negative median ), D43/D28, round2.")
txt.append("Pair direction only; n pairs not independent;")
txt.append("technical repeats only (no independent biological reps).")
txt.append("A20/G20 2000bp: observed only, PARM unsupported.")
axes[1].text(0.0,0.98,"\n".join(txt),va="top",ha="left",family="monospace",fontsize=8.5,transform=axes[1].transAxes)
axes[1].set_title("D28 observed-vs-frozen direction")
fig.suptitle("BATCH32 Fig2 | user target 293FT (historical label HEK293T preserved). Length-associated mismatch may reflect "
             "design window (A5 shifted), naming, orientation or normalization; absence of evidence is not proof of mechanism.",fontsize=7.5)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(os.path.join(FIG,"fig2_local_293ft_endpoint.png"),dpi=140); plt.close(fig)
print("figures written to",FIG)
# print nesting table for helper
print(json.dumps(B["nesting_vs_2000bp"],ensure_ascii=False))
