# -*- coding: utf-8 -*-
"""batch45 phase B secondary: reliability/effect comparisons from published prediction columns (ALL eligible unique DNA)."""
import json, math, collections, sys, pickle, statistics
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT = C.OUT
per = pickle.loads((OUT/"_per.pkl").read_bytes())
def wj(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
SYNTH = {"design","best","top_individual","MUTATED"}
def subgroup(p): return "synthetic_designed" if p["category"] in SYNTH else "natural_control"
def spearman_block(items, predkey, measkey):
    ids=[p["info"] for p in items]; pr=[p[predkey] for p in items]; ms=[p[measkey] for p in items]
    rho,n=C.spearman(pr,ms)
    return {"rho":rho,"n":n}
res={"metric_definitions":{
        "spearman":"tie-aware rank correlation between source-published PARM prediction and the fixed measured normalized-log2 activity summary",
        "gc_baseline":"same rank correlation using GC fraction of query DNA as the predictor (sequence-composition baseline)",
        "top10":"k=ceil(0.1*n); precision=overlap/k; enrichment=precision/(k/n); ties broken by ascending lexical info id; reported only when n>=30",
        "units_note":"prediction and experiment are in DIFFERENT units; NO MAE/R2/calibration computed.",
        "independence_warning":"889 unique DNA are NOT 889 independent families (overlapping genetic-optimization ancestry); 3556 barcodes and 2 RNA columns are not independent n."},
      "by_library_cell_subgroup":{}, "replicate_diagnostics":{}, "barcode_spread":{}}
for L in ["short","long"]:
    for cell,(pk,mk,r1,r2) in {"K562":("pred_K562","K562_meas","K562_r1_mean","K562_r2_mean"),
                                "HepG2":("pred_HepG2","HepG2_meas","HepG2_r1_mean","HepG2_r2_mean")}.items():
        for sg in ["synthetic_designed","natural_control","ALL"]:
            items=[p for p in per.values() if p["lib"]==L and (sg=="ALL" or subgroup(p)==sg)]
            blk={"n_unique_DNA":len(items)}
            blk["pred_vs_measured"]=spearman_block(items,pk,mk)
            gcid=items
            rho_gc,ng=C.spearman([p["GC"] for p in gcid],[p[mk] for p in gcid])
            blk["gc_vs_measured_baseline"]={"rho":rho_gc,"n":ng}
            # top10 needs finite pred & meas; build aligned lists
            pr=[p[pk] for p in items]; ms=[p[mk] for p in items]; ids=[p["info"] for p in items]
            ok=[i for i in range(len(items)) if pr[i] is not None and ms[i] is not None and math.isfinite(pr[i]) and math.isfinite(ms[i])]
            if len(ok)>=30:
                idx=list(range(len(ok)))
                order_p=sorted(idx,key=lambda i:(-pr[ok[i]],ids[ok[i]]))
                order_m=sorted(idx,key=lambda i:(-ms[ok[i]],ids[ok[i]]))
                k=math.ceil(0.1*len(ok))
                tp=set(order_p[:k]); tm=set(order_m[:k]); inter=len(tp&tm)
                blk["top10"]={"n":len(ok),"k":k,"overlap":inter,"precision":inter/k,"enrichment":(inter/k)/(k/len(ok))}
            else:
                blk["top10"]={"n":len(ok),"note":"n<30 -> descriptive only, no top-k metric"}
            res["by_library_cell_subgroup"][L+":"+cell+":"+sg]=blk
        # replicate concordance (measurement diagnostic, NOT prediction ceiling)
        items=[p for p in per.values() if p["lib"]==L and p[r1] is not None and p[r2] is not None]
        rho,ng=C.spearman([p[r1] for p in items],[p[r2] for p in items])
        res["replicate_diagnostics"][L+":"+cell]={"spearman_rep1_vs_rep2":rho,"n":len(items)}
        sds=[p["K562_barcode_spread_sd"] for p in items if p["K562_barcode_spread_sd"] is not None]
        res["barcode_spread"][L+":"+cell]={"median_barcode_sd":statistics.median(sds) if sds else None,"n":len(sds)}
wj("secondary_reliability.json",res)
for k,v in res["by_library_cell_subgroup"].items():
    t=v.get("top10",{})
    print(k,"n=",v["n_unique_DNA"],"rho=%.3f"%v["pred_vs_measured"]["rho"],"gc=%.3f"%v["gc_vs_measured_baseline"]["rho"],"top10",t.get("precision"),t.get("enrichment"))
print("--- replicates ---")
for k,v in res["replicate_diagnostics"].items(): print(k, "rho=%.3f"%v["spearman_rep1_vs_rep2"], v["n"])
