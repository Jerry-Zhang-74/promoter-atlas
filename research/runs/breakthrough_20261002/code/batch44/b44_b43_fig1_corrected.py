# -*- coding: utf-8 -*-
"""b44 Section E: NEW corrected copy of b43 fig1 with truthful titles. Old fig untouched."""
import json,os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rk=json.load(open('reports/breakthrough_20261002/batch43/rank_diagnostics.json',encoding='utf-8'))
m42=json.load(open('reports/breakthrough_20261002/batch42/metrics.json',encoding='utf-8'))
FIG=os.path.join('reports','breakthrough_20261002','batch44','figures'); os.makedirs(FIG,exist_ok=True)
names=["R0","R1","Rnull","parm","h1","gc"]; lab=["R0","R1","Rnull","parm(rank)","h1(rank)","gc(rank)"]
gm=[rk["TEST"][k]["group_mean_spearman"] for k in names]
pooled={"R0":m42["R0"]["pooled_spearman"],"R1":m42["R1"]["pooled_spearman"],"Rnull":m42["Rnull"]["pooled_spearman"],
        "parm":m42["cached_batch41_same_testids"]["cached_parm_mean"]["spearman"],
        "h1":m42["cached_batch41_same_testids"]["cached_h1_mean"]["spearman"],
        "gc":m42["cached_batch41_same_testids"]["gc"]["spearman"]}
pv=[pooled[k] for k in names]
COLORS={"R0":"#1f77b4","R1":"#d62728","Rnull":"#7f7f7f","parm":"#9467bd","h1":"#8c564b","gc":"#2ca02c"}
cols=[COLORS[k] for k in names]
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
ax[0].bar(range(len(names)),[np.nan if v is None else v for v in gm],color=cols)
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(lab,rotation=20,ha='right')
ax[0].set_title("TEST: GROUP-MEAN Spearman\n(per-group mean prediction vs per-group mean alpha)")
ax[0].set_ylabel("Spearman rho"); ax[0].axhline(0,color='k',lw=.8); ax[0].set_ylim(-0.3,0.3)
for i,v in enumerate(gm):
    if v is None: ax[0].text(i,0,"undefined\n(const)",ha='center',va='center',fontsize=7)
    else: ax[0].text(i,(v if v>=0 else v),"%.3f"%v,ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
ax[1].bar(range(len(names)),pv,color=cols)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(lab,rotation=20,ha='right')
ax[1].set_title("All 318 TEST rows: ORIGINAL POOLED Spearman\n(different quantity, NOT what b43 fig1 plotted)")
ax[1].set_ylabel("Spearman rho"); ax[1].axhline(0,color='k',lw=.8); ax[1].set_ylim(-0.3,0.3)
for i,v in enumerate(pv): ax[1].text(i,v if v>=0 else v,"%.3f"%v,ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
fig.suptitle("batch44 CORRECTED copy of batch43 fig1 — b43 mislabelled the group-mean bar chart as 'pooled Spearman'. Old file preserved; this is a NEW file.",fontsize=8)
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(os.path.join(FIG,"b43_fig1_CORRECTED_groupmean_vs_pooled.png"),dpi=130); plt.close(fig)
print("wrote corrected b43 fig1")
