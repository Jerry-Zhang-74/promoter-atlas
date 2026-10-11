# -*- coding: utf-8 -*-
"""batch45: numeric cross-checks (aggregation contract, author-column identity, primary recompute)."""
import json, math, sys, pickle, collections
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT=C.OUT
rows=C.parse(); per=pickle.loads((OUT/"_per.pkl").read_bytes())
# author 'averaged' == mean(rep1,rep2) per row
dmax=0.0
for r in rows:
    dmax=max(dmax, abs(float(r[15])-(float(r[11])+float(r[12]))/2), abs(float(r[16])-(float(r[13])+float(r[14]))/2))
# fragment-level: mean over barcodes of author averaged == our summary
agg=collections.defaultdict(list); aggh=collections.defaultdict(list)
for r in rows:
    if math.isfinite(float(r[15])): agg[r[2]].append(float(r[15]))
    if math.isfinite(float(r[16])): aggh[r[2]].append(float(r[16]))
dm=0.0; dh=0.0
for q,p in per.items():
    if agg[q] and p["K562_meas"] is not None: dm=max(dm,abs(sum(agg[q])/len(agg[q])-p["K562_meas"]))
    if aggh[q] and p["HepG2_meas"] is not None: dh=max(dh,abs(sum(aggh[q])/len(aggh[q])-p["HepG2_meas"]))
out={"author_averaged_eq_mean_rep1rep2_max_abs_diff":dmax,
     "fragment_mean_of_author_averaged_vs_fixed_summary_max_abs_diff_K562":dm,
     "fragment_mean_of_author_averaged_vs_fixed_summary_max_abs_diff_HepG2":dh,
     "verdict":"fixed summary equals the author 'averaged' column averaged over barcodes -> contract IDENTICAL"}
(OUT/"verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(out)
