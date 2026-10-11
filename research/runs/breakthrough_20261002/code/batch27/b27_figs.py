# -*- coding: utf-8 -*-
"""BATCH27 figures (no fits). A2 homology sensitivity, A3 variant bins, B cohort coverage."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R="reports/breakthrough_20261002/batch27"; FIG=R+"/figures"; os.makedirs(FIG,exist_ok=True)
M=[("A0_ridgeC","ridge C"),("A1_cnn_true_seed20261026","CNN seed1"),("A2_cnn_true_seed20261027","CNN seed2")]
# --- Fig 1: homology sensitivity ---
h=json.load(open(R+"/homology_sensitivity.json",encoding="utf-8"))
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
x=np.arange(3); w=0.36
full=[h["full_test_metrics"][k]["spearman"] for k,_ in M]
filt=[h["filtered_test_metrics"][k]["spearman"] for k,_ in M]
ax[0].bar(x-w/2,full,w,label="all test groups",color="#4C72B0")
ax[0].bar(x+w/2,filt,w,label="minus 2 known homolog test groups",color="#DD8452")
ax[0].set_xticks(x); ax[0].set_xticklabels([n for _,n in M])
ax[0].set_ylabel("Spearman rho (test)"); ax[0].set_ylim(0,0.55)
ax[0].set_title("Known-homology sensitivity\n2 test groups removed (7 rows)")
ax[0].legend(fontsize=8)
d=h["filtered_group_bootstrap_1000"]["delta_rho_ci95"]
for i,(k,_) in enumerate(M[:2]):
    pass
b=h["filtered_group_bootstrap_1000"]
means=[b["delta_rho_mean"]["A1_vs_A0"],b["delta_rho_mean"]["A2_vs_A0"]]
cis=[b["delta_rho_ci95"]["A1_cnn_true_seed20261026_vs_A0"],b["delta_rho_ci95"]["A2_cnn_true_seed20261027_vs_A0"]]
xs=np.arange(2)
ax[1].axhline(0,color="k",lw=0.8)
ax[1].errorbar(xs,means,yerr=[[m-c[0] for m,c in zip(means,cis)],[c[1]-m for m,c in zip(means,cis)]],fmt="o",capsize=6,color="#55A868")
ax[1].set_xticks(xs); ax[1].set_xticklabels(["CNN seed1\n-minus ridge","CNN seed2\n-minus ridge"])
ax[1].set_ylabel("delta rho (retained groups)"); ax[1].set_title("Group-bootstrap 1000 delta rho\n(retained groups, 95% CI)")
fig.tight_layout(); fig.savefig(FIG+"/homology_sensitivity.png",dpi=130); plt.close(fig)
# --- Fig 2: variant bins ---
v=json.load(open(R+"/variant_bin_diagnostic.json",encoding="utf-8"))
fig,ax=plt.subplots(1,2,figsize=(12,4.3))
bins=list(v["by_abs_delta_obs"].keys())
xs=np.arange(len(bins)); w=0.36
for mk,lab,col,off in [("A1_cnn_true_seed20261026","CNN seed1","#C44E52",-w/2),("A0_ridgeC","ridge C","#4C72B0",w/2)]:
    vals=[v["by_abs_delta_obs"][b].get(mk,{}).get("sign_agreement_nonzero",np.nan) if b in v["by_abs_delta_obs"] and mk in v["by_abs_delta_obs"][b] else np.nan for b in bins]
    ax[0].bar(xs+off,vals,w,label=lab,color=col)
ax[0].axhline(0.5,color="grey",ls="--",lw=0.8,label="chance")
ax[0].set_xticks(xs); ax[0].set_xticklabels(bins,fontsize=8)
ax[0].set_ylim(0.4,0.7); ax[0].set_ylabel("sign agreement (nonzero both)")
ax[0].set_title("Sign agreement by |observed delta|")
for i,b in enumerate(bins):
    ax[0].text(i,0.41,"n=%d"%v["by_abs_delta_obs"][b]["n_pairs"],ha="center",fontsize=7)
ax[0].legend(fontsize=8)
mk="A1_cnn_true_seed20261026"; cats=list(v["by_motif_length"].keys())
vals=[v["by_motif_length"][c].get(mk,{}).get("sign_agreement_nonzero",np.nan) for c in cats]
ns=[v["by_motif_length"][c]["n_pairs"] for c in cats]
ax[1].bar(np.arange(len(cats)),vals,0.6,color="#8172B3")
ax[1].axhline(0.5,color="grey",ls="--",lw=0.8)
ax[1].set_xticks(np.arange(len(cats))); ax[1].set_xticklabels(cats)
ax[1].set_xlabel("STR motif length (bp)"); ax[1].set_ylim(0.4,0.7)
ax[1].set_ylabel("sign agreement (CNN seed1)"); ax[1].set_title("Sign agreement by motif length")
for i,n in enumerate(ns): ax[1].text(i,0.41,"n=%d"%n,ha="center",fontsize=7)
fig.tight_layout(); fig.savefig(FIG+"/variant_bin_diagnostic.png",dpi=130); plt.close(fig)
print("figs 1-2 done")
