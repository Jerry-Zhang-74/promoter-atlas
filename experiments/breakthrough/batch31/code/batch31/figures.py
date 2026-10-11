#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31 three scientific figures."""
import csv,json,os,math
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
RES=os.path.join(REPO,"data/breakthrough_20261002/batch31/results")
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch31")
FIG=os.path.join(OUT,"figures"); os.makedirs(FIG,exist_ok=True)
M=json.load(open(os.path.join(OUT,"metrics.json"),encoding="utf-8"))
Q=json.load(open(os.path.join(OUT,"sure_qc_metrics.json"),encoding="utf-8"))
EXP=json.load(open(os.path.join(REPO,"data/breakthrough_20261002/batch31/results/exposure_check.json"),encoding="utf-8"))
def read_tsv(p):
    with open(p,encoding="utf-8") as f: return list(csv.DictReader(f,delimiter="\t"))

def fig1():
    parm=read_tsv(os.path.join(RES,"parm_pred_hek673_133bp.tsv"))
    pmap={r["header"]:float(r["prediction_HEK293"]) for r in parm}
    o22={r["prom_id"]:r for r in read_tsv(os.path.join(REPO,"data/breakthrough_20261002/batch22/oof_predictions.tsv"))}
    o24={r["prom_id"]:r for r in read_tsv(os.path.join(REPO,"data/breakthrough_20261002/batch24/oof_predictions_batch24.tsv"))}
    ids=sorted(i for i in pmap if i in o22 and i in o24)
    y=np.array([float(o22[i]["hek_norm"]) for i in ids])
    P=np.array([pmap[i] for i in ids]); R1=np.array([float(o22[i]["pred_R1"]) for i in ids]); H1=np.array([float(o24[i]["pred_H1"]) for i in ids])
    fig,ax=plt.subplots(1,3,figsize=(15,4.6))
    for a,name,s,c in [(ax[0],"frozen PARM HEK293",P,"#c0392b"),(ax[1],"de-novo R1 (composition)",R1,"#2471a3")]:
        a.scatter(s,y,s=9,alpha=0.45,color=c)
        rho=M["hek673"]["rho"]["PARM" if name.startswith("frozen") else "R1"]
        a.set_xlabel(name); a.set_ylabel("Haberle2019 HEK293 measured (log2)"); a.set_title(f"{name}\nSpearman rho={rho:.3f}")
    names=["PARM","R1","H1"]; rhos=[M["hek673"]["rho"][n] for n in names]
    enr=[M["hek673"]["top_decile"][n]["enrichment"] for n in names]
    ax[2].bar(names,enr,color=["#c0392b","#2471a3","#27ae60"])
    ax[2].axhline(1.0,ls="--",c="k",lw=1); ax[2].set_ylabel("top-decile enrichment (x)")
    ax[2].set_title("Top-decile enrichment\n(rank agreement)")
    bt=M["hek673"]["paired_group_bootstrap"]
    ax[2].text(0.02,0.97,f"rho PARM-R1 = +{bt['rho_PARM_minus_R1']['mean']:.3f}\n95%CI [{bt['rho_PARM_minus_R1']['ci95_lo']:.3f},{bt['rho_PARM_minus_R1']['ci95_hi']:.3f}]\n669-group paired bootstrap n=2000",
                transform=ax[2].transAxes,va="top",fontsize=8)
    fig.suptitle("Fig1 HEK673 ranking: frozen PARM HEK293 vs saved R1/H1 OOF (n=673, 669 groups)\n"
                 f"EXPOSURE: {EXP['our_check']['locus_overlap_fraction']*100:.0f}% of checked HEK673 fragments overlap official library loci (0/310 exact); exact/RC membership UNKNOWN -> retrospective cross-assay, NOT unseen-family",
                 fontsize=9.2)
    fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(os.path.join(FIG,"fig1_hek673_ranking.png"),dpi=140); plt.close(fig)

FAMC={"ACTB":"#c0392b","GAPDH":"#2471a3","CMV":"#27ae60"}
def fig2():
    L=M["local"]; recs=L["supported_constructions"]
    fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    def nrank(vals):
        order=sorted(range(len(vals)),key=lambda i:vals[i])
        r=[0]*len(vals)
        for pos,i in enumerate(order): r[i]=pos
        return np.array(r)/max(1,len(vals)-1)
    off={"ACTB":-0.06,"GAPDH":0.0,"CMV":0.06}
    for fam in ["ACTB","GAPDH","CMV"]:
        rs=sorted([r for r in recs if r["family"]==fam],key=lambda r:r["len"])
        x=np.array([r["len"] for r in rs],float); x=x/max(x)
        ro=nrank(np.array([r["obs"] for r in rs])); rp=nrank(np.array([r["pred"] for r in rs]))
        ax[0].plot(x,ro,"o-",color=FAMC[fam],label=f"{fam} measured rank")
        ax[0].plot(x,rp,"s--",color=FAMC[fam],alpha=0.6,label=f"{fam} PARM rank")
    ax[0].text(0.02,0.03,"2000bp A20/G20: unsupported (L_max=600); retained observed-only",fontsize=8,color="#333")
    ax[0].set_xlabel("construct length (normalized within family)")
    ax[0].set_ylabel("within-family rank (1=highest)")
    ax[0].set_title("Fig2a local families: measured vs PARM rank"); ax[0].legend(fontsize=7,ncol=2)
    fams=["ACTB","GAPDH","CMV"]
    acc=[L["families"][f]["direction_accuracy"] for f in fams]
    inc=[L["families"][f]["baseline_increasing_len_accuracy"] for f in fams]
    dec=[L["families"][f]["baseline_decreasing_len_accuracy"] for f in fams]
    w=0.26; xi=np.arange(len(fams))
    ax[1].bar(xi-w,acc,w,label="PARM",color="#8e44ad")
    ax[1].bar(xi,inc,w,label="baseline +length",color="#95a5a6")
    ax[1].bar(xi+w,dec,w,label="baseline -length",color="#d5d8dc")
    ax[1].axhline(0.5,ls="--",c="k",lw=1); ax[1].set_xticks(xi); ax[1].set_xticklabels(fams)
    ax[1].set_ylabel("within-family pair-direction accuracy"); ax[1].set_ylim(0,1)
    ax[1].set_title(f"Fig2b pair direction vs fixed length baselines\nmacro PARM={L['macro_family_accuracy']:.2f}  incr={L['macro_baseline_increasing']:.2f}  decr={L['macro_baseline_decreasing']:.2f}")
    ax[1].legend(fontsize=8)
    r=M["exploratory_rule"]["components"]
    ax[1].text(0.02,0.98,"rule  C1 macro>=2/3: %s | C2 each>=1/2: %s\nC3 beats both baselines: %s | C4 HEK673 rho>=R1: %s"%(
        r["C1_macro_family_direction_ge_2of3"]["pass"],r["C2_each_family_ge_1of2"]["pass"],
        r["C3_exceeds_both_length_baselines"]["pass"],r["C4_HEK673_rho_ge_saved_R1"]["pass"]),
        transform=ax[1].transAxes,va="top",fontsize=8)
    fig.suptitle("Fig2 local 293FT truncation families (12 supported; pair comparisons NOT independent n; technical not biological replicates)",fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(os.path.join(FIG,"fig2_local_families.png"),dpi=140); plt.close(fig)

def fig3():
    fig,ax=plt.subplots(1,3,figsize=(15,4.6))
    pb=list(Q["by_P_bin"].keys())
    bothp=[Q["by_P_bin"][k].get("frac_both_positive",0) for k in pb]
    bothz=[Q["by_P_bin"][k].get("frac_both_zero",0) for k in pb]
    onep=[Q["by_P_bin"][k].get("frac_one_positive",0) for k in pb]
    xi=np.arange(len(pb))
    ax[0].bar(xi,bothp,color="#27ae60",label="both positive"); ax[0].bar(xi,bothz,bottom=bothp,color="#bdc3c7",label="both zero")
    ax[0].bar(xi,onep,bottom=np.array(bothp)+np.array(bothz),color="#f39c12",label="one positive")
    ax[0].set_xticks(xi); ax[0].set_xticklabels(pb); ax[0].set_title("Fig3a repeat detection by plasmid-PCR depth P"); ax[0].legend(fontsize=7)
    tp=Q["top5"]; pr=Q["permutation_reference"]
    ax[1].bar(["observed","null mean","null p95"],[tp["intersection"],pr["null_mean_intersection"],pr["null_p95_intersection"]],
              color=["#c0392b","#95a5a6","#d5d8dc"])
    ax[1].set_ylabel("top-5% repeat1 & repeat2 intersection size")
    ax[1].set_title(f"Fig3b top-5% overlap\nenrichment={pr['enrichment_obs_over_nullmean']:.2f} vs within-stratum permutation (R={pr['R']})")
    ax[1].text(0.03,0.97,f"cutoff ties: r1 {tp['frac_tied_at_cutoff_r1']*100:.1f}% tied\nr2 {tp['frac_tied_at_cutoff_r2']*100:.1f}% tied\nconditional reference only",
               transform=ax[1].transAxes,va="top",fontsize=8)
    g=Q["resource_gate"]
    labels=["train rho>=.30","grp CI-lower>.15","P1 CI-lower>0","top5 enr>=2","top5 enr-lower>1"]
    vals=[g["comp1_train_rho_ge_0.30"],g["comp2_group_ci95_lower_gt_0.15"],g["comp3_P1_ci95_lower_gt_0"],g["comp4_top5_enrichment_ge_2"],g.get("comp5_top5_enrichment_ci95_lower_gt_1")]
    ax[2].barh(labels,[1 if v else 0 for v in vals],color=["#27ae60" if v else "#c0392b" for v in vals])
    ax[2].set_xlim(0,1.2); ax[2].set_title(f"Fig3c Astra resource gate (FUTURE priority only)\nobserved train rho={g['observed']['train_rho']:.3f}, CI-lower={g['observed']['group_ci95_lower']:.3f}")
    ax[2].text(0.02,0.02,f"train rho={g['observed']['train_rho']:.3f}\nP1 rho={g['observed']['P1_rho']:.3f}\nenr={g['observed']['top5_enrichment']:.2f}\nenr-lower={g['observed'].get('top5_enrichment_ci95_lower',0):.2f}",
               transform=ax[2].transAxes,fontsize=8)
    fig.suptitle("Fig3 SuRE TRAIN-only repeat-vs-repeat QC on declared PROXY LABELS z=log2(1+(r/N_R)/((1+P)/N_D)) "
                 "(n=7690 train inserts, 7572 groups; ZERO model inference, all zero-RNA kept; raw-count v1 rho=0.1187 preserved)",fontsize=8.6)
    fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(os.path.join(FIG,"fig3_sure_qc.png"),dpi=140); plt.close(fig)

if __name__=="__main__":
    fig1(); fig2(); fig3(); print("figures written")
