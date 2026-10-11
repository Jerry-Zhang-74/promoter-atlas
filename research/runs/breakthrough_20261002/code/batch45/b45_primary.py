# -*- coding: utf-8 -*-
"""batch45 phase B primary: label-free matched K562 design trajectories (gen0 max -> largest COMMON gen max)."""
import json, math, collections, re, sys, pickle, statistics
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT = C.OUT
per = pickle.loads((OUT/"_per.pkl").read_bytes())
by_info = {v["info"]: v for v in per.values()}
def wj(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
pat = re.compile(r"^K562_fitness_(?P<len>232bp_)?replicate(?P<rep>\d+)_generation(?P<gen>\d+)_individual(?P<ind>\d+)_(?P<sel>max|median|min|top_individual|best|best_MUTATED)$")
frag = {}
for k,p in per.items():
    m = pat.match(p["info"])
    if m: frag[p["info"]] = (p, m["rep"], int(m["gen"]), m["sel"], p["lib"])
# grid
grid = collections.defaultdict(dict)   # (lib,rep,sel) -> {gen: info}
for info,(p,rep,gen,sel,lib) in frag.items():
    grid[(lib,rep,sel)][gen] = info
libs = ["short","long"]
maturity = {}
for L in libs:
    reps = sorted({k[1] for k in grid if k[0]==L and k[2]=="max"})
    common = None
    for rep in reps:
        gs = set(grid.get((L,rep,"max"),{}).keys())
        common = gs if common is None else (common & gs)
    maturity[L] = {"n_lineages":len(reps),"reps":reps,"largest_common_generation":max(common) if common else None,
                   "common_generations":sorted(common) if common else []}
# lineage table
rows_out = []
for L in libs:
    G = maturity[L]["largest_common_generation"]
    for rep in maturity[L]["reps"]:
        i0 = grid.get((L,rep,"max"),{}).get(0)
        iG = grid.get((L,rep,"max"),{}).get(G) if G is not None else None
        rec = {"library":L,"lineage":rep,"gen0_info":i0,"genG_info":iG,"G":G,"matched":bool(i0 and iG)}
        if i0 and iG:
            p0=by_info[i0]; pG=by_info[iG]
            rec.update({
              "pred_K562_gen0":p0["pred_K562"],"pred_K562_genG":pG["pred_K562"],
              "dK562_pred":None if (p0["pred_K562"] is None or pG["pred_K562"] is None) else pG["pred_K562"]-p0["pred_K562"],
              "K562_meas_gen0":p0["K562_meas"],"K562_meas_genG":pG["K562_meas"],
              "dK562_meas":None if (p0["K562_meas"] is None or pG["K562_meas"] is None) else pG["K562_meas"]-p0["K562_meas"],
              "pred_HepG2_gen0":p0["pred_HepG2"],"pred_HepG2_genG":pG["pred_HepG2"],
              "dHepG2_pred":pG["pred_HepG2"]-p0["pred_HepG2"],
              "dHepG2_meas":pG["HepG2_meas"]-p0["HepG2_meas"]})
        rows_out.append(rec)
def summarize(vals):
    v=[x for x in vals if x is not None and math.isfinite(x)]
    if not v: return {"n":0}
    return {"n":len(v),"median":statistics.median(v),"min":min(v),"max":max(v),
            "n_positive":sum(1 for x in v if x>0),"n_negative":sum(1 for x in v if x<0),"n_zero":sum(1 for x in v if x==0)}
prim = {}
for L in libs:
    sub=[r for r in rows_out if r["library"]==L and r["matched"]]
    prim[L]={"G":maturity[L]["largest_common_generation"],"n_matched_lineages":len(sub),
             "n_lineages_total":maturity[L]["n_lineages"],
             "delta_K562_predicted_score":summarize([r["dK562_pred"] for r in sub]),
             "delta_K562_measured_summary":summarize([r["dK562_meas"] for r in sub]),
             "delta_HepG2_predicted_score_secondary":summarize([r["dHepG2_pred"] for r in sub]),
             "delta_HepG2_measured_summary_secondary":summarize([r["dHepG2_meas"] for r in sub])}
wj("primary_matched_lineages.json", {"rule":"gen0 selection=max vs largest generation common to ALL unmodified lineages; same library/length; MUTATED excluded; activity-blind freeze",
   "maturity":maturity,"per_library":prim,
   "note":"n=10 GA search lineages per library, NOT independent biological families; they share one PARM model and design pipeline."})
cols=["library","lineage","G","matched","pred_K562_gen0","pred_K562_genG","dK562_pred",
      "K562_meas_gen0","K562_meas_genG","dK562_meas","dHepG2_pred","dHepG2_meas","gen0_info","genG_info"]
with open(OUT/"per_lineage_primary.tsv","w",encoding="utf-8") as f:
    f.write("\t".join(cols)+"\n")
    for r in rows_out: f.write("\t".join(str(r.get(c)) for c in cols)+"\n")
print("primary done")
for L in libs: print(L, prim[L]["G"], prim[L]["n_matched_lineages"], "dPred", prim[L]["delta_K562_predicted_score"].get("median"), "dMeas", prim[L]["delta_K562_measured_summary"].get("median"))
