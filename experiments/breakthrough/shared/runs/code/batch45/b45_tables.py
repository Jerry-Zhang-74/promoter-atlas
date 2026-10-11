# -*- coding: utf-8 -*-
"""batch45: descriptive complete tables (controls, MUTATED vs strongest, strongest)."""
import json, re, sys, pickle, statistics, math
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT=C.OUT
per=pickle.loads((OUT/"_per.pkl").read_bytes())
def wj(n,o): (OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
# ---- controls (natural human promoters) ----
ctrl=[]
cre=re.compile(r"^control_(?P<lvl>high|low|mid)_(?P<gene>.+?)_chr(?P<chr>chr[0-9XYM]+)_(?P<s>\d+)_(?P<e>\d+)_strand(?P<str>[+-])_(?P<mer>\d+)mer_(?P<idx>\d+)$")
for p in per.values():
    if p["category"]=="control":
        m=cre.match(p["info"])
        d={"info":p["info"],"library":p["library"],"level":m["lvl"] if m else None,"gene":m["gene"] if m else None,
           "chr":m["chr"] if m else None,"start":int(m["s"]) if m else None,"end":int(m["e"]) if m else None,
           "strand":m["str"] if m else None,"mer":int(m["mer"]) if m else None,
           "q_len":p["q_len"],"GC":p["GC"],"pred_K562":p["pred_K562"],"pred_HepG2":p["pred_HepG2"],
           "K562_meas":p["K562_meas"],"HepG2_meas":p["HepG2_meas"]}
        ctrl.append(d)
ctrl.sort(key=lambda d:d["info"])
wj("table_controls.json",{"note":"natural human-promoter control fragments (descriptive, complete)","n":len(ctrl),"rows":ctrl})
# ---- strongest (best/top_individual) + MUTATED paired by lineage ----
strong={} ; mut={}
for p in per.values():
    m=re.match(r"^K562_fitness_(?:232bp_)?replicate(\d+)_generation\d+_individual\d+_(top_individual|best|best_MUTATED|top_individual_MUTATED)$",p["info"])
    if not m: continue
    rep=m.group(1); kind=m.group(2)
    key=(p["lib"],rep)
    if "MUTATED" in kind: mut[key]=p
    else: strong[key]=p
rows=[]
for k in sorted(set(strong)|set(mut)):
    s=strong.get(k); u=mut.get(k)
    rows.append({"library":k[0],"lineage":k[1],
      "strongest_info":s["info"] if s else None,"MUTATED_info":u["info"] if u else None,
      "strongest_pred_K562":s["pred_K562"] if s else None,"MUTATED_pred_K562":u["pred_K562"] if u else None,
      "delta_pred_K562":(u["pred_K562"]-s["pred_K562"]) if (s and u) else None,
      "strongest_K562_meas":s["K562_meas"] if s else None,"MUTATED_K562_meas":u["K562_meas"] if u else None,
      "delta_K562_meas":(u["K562_meas"]-s["K562_meas"]) if (s and u) else None})
dv=[r["delta_K562_meas"] for r in rows if r["delta_K562_meas"] is not None]
dp=[r["delta_pred_K562"] for r in rows if r["delta_pred_K562"] is not None]
wj("table_mutated_vs_strongest.json",{"note":"MUTATED candidates vs their LINEAGE-MATCHED strongest unmutated (separate descriptive category, NOT a pure truncation and NOT the primary comparison)",
   "n_pairs":len(dv),"median_delta_K562_meas":statistics.median(dv) if dv else None,
   "n_delta_meas_negative":sum(1 for x in dv if x<0),"n_delta_meas_positive":sum(1 for x in dv if x>0),
   "median_delta_pred_K562":statistics.median(dp) if dp else None,"rows":rows})
# ---- per-category summary ----
cat={}
for p in per.values():
    cat.setdefault((p["lib"],p["category"]),{"n":0,"K562":[], "HepG2":[]})
    cat[(p["lib"],p["category"])]["n"]+=1
    if p["K562_meas"] is not None: cat[(p["lib"],p["category"])]["K562"].append(p["K562_meas"])
    if p["HepG2_meas"] is not None: cat[(p["lib"],p["category"])]["HepG2"].append(p["HepG2_meas"])
summary={f"{k[0]}:{k[1]}":{"n_unique_DNA":v["n"],"median_K562_meas":statistics.median(v["K562"]) if v["K562"] else None,
    "median_HepG2_meas":statistics.median(v["HepG2"]) if v["HepG2"] else None} for k,v in cat.items()}
wj("table_category_summary.json",{"note":"descriptive only; no category was selected post hoc","summary":summary})
print("tables done; pairs",len(dv),"median delta meas",statistics.median(dv) if dv else None)
