#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH31 section B: frozen PARM vs saved R1/H1 (HEK673) and local family pair directions.
ZERO fits. Ranking-only; no cross-scale MAE/R2. Exploratory diagnostic rules fixed here
BEFORE any prediction is read."""
import csv, json, math, os, hashlib, itertools
import numpy as np
from scipy import stats

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
RES=os.path.join(REPO,"data/breakthrough_20261002/batch31/results")
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch31")
SEED=20261031
def h(k): return hashlib.sha256(k.encode()).hexdigest()
def read_tsv(p):
    with open(p,encoding="utf-8") as f: return list(csv.DictReader(f,delimiter="\t"))
def spearman(x,y):
    if len(x)<3: return None
    r=stats.spearmanr(x,y)[0]
    return None if (r is None or math.isnan(r)) else float(r)
def topk(ids, vals, k):
    order=sorted(range(len(ids)), key=lambda i:(-vals[i], h(ids[i])))
    return set(ids[i] for i in order[:k])

def hek673():
    parm=read_tsv(os.path.join(RES,"parm_pred_hek673_133bp.tsv"))
    pmap={r["header"]:float(r["prediction_HEK293"]) for r in parm}
    o22={r["prom_id"]:r for r in read_tsv(os.path.join(REPO,"data/breakthrough_20261002/batch22/oof_predictions.tsv"))}
    o24={r["prom_id"]:r for r in read_tsv(os.path.join(REPO,"data/breakthrough_20261002/batch24/oof_predictions_batch24.tsv"))}
    ids=[i for i in pmap if i in o22 and i in o24]
    ids.sort()
    y=[float(o22[i]["hek_norm"]) for i in ids]
    groups=[o22[i]["group_id"] for i in ids]
    p_parm=[pmap[i] for i in ids]
    p_R1=[float(o22[i]["pred_R1"]) for i in ids]
    p_H1=[float(o24[i]["pred_H1"]) for i in ids]
    k=int(math.ceil(0.1*len(ids)))
    obs=topk(ids,y,k)
    out={"n":len(ids),"n_groups":len(set(groups)),"top_decile_k":k,
         "source_seen_rows":sum(1 for i in ids if o22[i].get("source_seen")not in(None,"0")),
         "rho":{"PARM":spearman(p_parm,y),"R1":spearman(p_R1,y),"H1":spearman(p_H1,y)}}
    for name,p in [("PARM",p_parm),("R1",p_R1),("H1",p_H1)]:
        pt=topk(ids,p,k); prec=len(pt&obs)/k
        out.setdefault("top_decile",{})[name]={"precision":prec,"enrichment":prec/0.1}
    rng=np.random.default_rng(SEED)
    g=np.array(groups); ug=np.unique(g); ng=len(ug)
    idx_by_g={gi:np.where(g==gi)[0] for gi in ug}
    dR1=[]; dH1=[]
    for it in range(2000):
        pick=rng.integers(0,ng,ng)
        sel=np.concatenate([idx_by_g[ug[j]] for j in pick])
        rp=spearman([p_parm[i] for i in sel],[y[i] for i in sel])
        r1=spearman([p_R1[i] for i in sel],[y[i] for i in sel])
        h1=spearman([p_H1[i] for i in sel],[y[i] for i in sel])
        if rp is not None and r1 is not None: dR1.append(rp-r1)
        if rp is not None and h1 is not None: dH1.append(rp-h1)
    dR1=np.array(dR1); dH1=np.array(dH1)
    out["paired_group_bootstrap"]={"draws":2000,"n_groups":ng,
        "rho_PARM_minus_R1":{"mean":float(dR1.mean()),"ci95_lo":float(np.percentile(dR1,2.5)),"ci95_hi":float(np.percentile(dR1,97.5))},
        "rho_PARM_minus_H1":{"mean":float(dH1.mean()),"ci95_lo":float(np.percentile(dH1,2.5)),"ci95_hi":float(np.percentile(dH1,97.5))},
        "scope":"paired group-level bootstrap on identical matched rows; group=669 frozen groups; ranking only"}
    out["notes"]="PARM=pretrained external HEK293 (cross-assay); R1/H1=de-novo OOF on Haberle2019; no MAE/R2 across unlike scales"
    return out

FAMS=["ACTB","GAPDH","CMV"]
def local():
    parm=read_tsv(os.path.join(RES,"parm_pred_local_supported.tsv"))
    pmap={}
    for r in parm:
        cid=r["header"].split("|")[0]; pmap[cid]=float(r["prediction_HEK293"])
    loc={r["construct_id"]:r for r in read_tsv(os.path.join(REPO,"data/breakthrough_20261002/batch23/local_constructs.tsv"))}
    recs=[]
    for cid in sorted(pmap):
        r=loc[cid]
        recs.append({"id":cid,"family":r["family"],"len":int(r["sequence_length_bp"]),
                     "obs":float(r["endpoint_target_value_log10"]),"pred":pmap[cid]})
    def sgn(a,b):
        return 0 if a==b else (1 if a>b else -1)
    fam_out={}
    pairs_all=[]
    for fam in FAMS:
        rows=[r for r in recs if r["family"]==fam]
        npairs=0; n_ok=0; n_tie_obs=0; n_tie_pred=0; n_ok_125=0; n_pairs_125=0
        plist=[]
        for a,b in itertools.combinations(rows,2):
            do=sgn(a["obs"],b["obs"]); dp=sgn(a["pred"],b["pred"])
            din=sgn(a["len"],b["len"])  # increasing-length baseline
            fold=max(a["obs"],b["obs"])-min(a["obs"],b["obs"])
            npairs+=1
            if do==0: n_tie_obs+=1
            elif dp==0: n_tie_pred+=1
            else:
                ok=(do==dp); n_ok+= 1 if ok else 0
                okin=(do==din)
                plist.append({"a":a["id"],"b":b["id"],"obs_dir":do,"pred_dir":dp,"len_dir":din,
                              "agree_parm":int(ok),"agree_len_incr":int(okin),"agree_len_decr":int(not okin),
                              "obs_fold_change_log10":round(fold,4),"obs_fc_ge_1_25":int(fold>=math.log10(1.25))})
                if fold>=math.log10(1.25):
                    n_pairs_125+=1; n_ok_125+= 1 if ok else 0
        dec_ok=sum(1 for p in plist if p["agree_len_decr"])
        inc_ok=sum(1 for p in plist if p["agree_len_incr"])
        denom=max(1,len(plist))
        fam_out[fam]={"n_supported":len(rows),"n_pairs":npairs,"n_pairs_comparable":len(plist),
            "n_pairs_obs_tie":n_tie_obs,"n_pairs_pred_tie":n_tie_pred,
            "direction_accuracy":(n_ok/denom),
            "baseline_increasing_len_accuracy":(inc_ok/denom),
            "baseline_decreasing_len_accuracy":(dec_ok/denom),
            "subgroup_obs_fc_ge_1.25":{"n_pairs":n_pairs_125,"accuracy":(n_ok_125/max(1,n_pairs_125))},
            "ids":[r["id"] for r in rows]}
        pairs_all+=plist
    macro=float(np.mean([fam_out[f]["direction_accuracy"] for f in FAMS]))
    macro_inc=float(np.mean([fam_out[f]["baseline_increasing_len_accuracy"] for f in FAMS]))
    macro_dec=float(np.mean([fam_out[f]["baseline_decreasing_len_accuracy"] for f in FAMS]))
    out={"supported_constructions":recs,"families":fam_out,"macro_family_accuracy":macro,
         "macro_baseline_increasing":macro_inc,"macro_baseline_decreasing":macro_dec,
         "pairs":pairs_all,
         "unsupported":[{"id":"A20","family":"ACTB","len":2000,"observed_log10":loc["A20"]["endpoint_target_value_log10"],"pred":None,"reason":"len>600 official L_max"},
                        {"id":"G20","family":"GAPDH","len":2000,"observed_log10":loc["G20"]["endpoint_target_value_log10"],"pred":None,"reason":"len>600 official L_max"}],
         "notes":"pair comparisons are not independent n; technical replicates not biological; no false new-family CI; fivefold spread is model disagreement only"}
    return out

def rule(hek, loc):
    rho_parm=hek["rho"]["PARM"]; rho_r1=hek["rho"]["R1"]
    macro=loc["macro_family_accuracy"]
    each=[loc["families"][f]["direction_accuracy"] for f in FAMS]
    comp={
      "C1_macro_family_direction_ge_2of3":{"value":macro,"threshold":2/3,"pass":macro>=(2/3)},
      "C2_each_family_ge_1of2":{"values":dict(zip(FAMS,each)),"pass":all(v>=0.5 for v in each)},
      "C3_exceeds_both_length_baselines":{"macro":macro,"incr":loc["macro_baseline_increasing"],
          "decr":loc["macro_baseline_decreasing"],"pass":(macro>loc["macro_baseline_increasing"] and macro>loc["macro_baseline_decreasing"])},
      "C4_HEK673_rho_ge_saved_R1":{"parm":rho_parm,"R1":rho_r1,"pass":(rho_parm is not None and rho_r1 is not None and rho_parm>=rho_r1)},
    }
    return {"components":comp,"all_pass":all(v["pass"] for v in comp.values()),
            "scope":"retrospective diagnostic rule fixed before predictions; PASS cannot certify novel-family/293FT or 2000bp reliability; failure is informative and triggers no new fit"}

def main():
    out={"hek673":hek673(),"local":local()}
    out["exploratory_rule"]=rule(out["hek673"],out["local"])
    out["chassis_note"]="Actual local chassis/target = 293FT (user correction 2026-10-03). PARM HEK293 is CROSS-CELL/CROSS-ASSAY. Public HEK293/HEK293T source identities unchanged."
    os.makedirs(OUT,exist_ok=True)
    with open(os.path.join(OUT,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({"hek673":{k:out["hek673"][k] for k in("n","n_groups","top_decile_k","rho","top_decile","paired_group_bootstrap")},
                      "local":{k:out["local"][k] for k in("macro_family_accuracy","macro_baseline_increasing","macro_baseline_decreasing")},
                      "families":{f:{"acc":out["local"]["families"][f]["direction_accuracy"],
                                     "incr":out["local"]["families"][f]["baseline_increasing_len_accuracy"],
                                     "decr":out["local"]["families"][f]["baseline_decreasing_len_accuracy"],
                                     "n_pairs":out["local"]["families"][f]["n_pairs_comparable"]} for f in FAMS},
                      "rule":out["exploratory_rule"]},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
