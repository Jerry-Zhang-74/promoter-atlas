# -*- coding: utf-8 -*-
"""batch29 reviewer: independent re-derivation from the frozen SuRE source (read-only).
Streams the gz once: recomputes full-table library totals + eligibility counts, and
recomputes D/a1/a2/y for the first 100 SELECTED keys in file order, comparing to rows.tsv.
NO fit, NO model."""
import gzip, json, time, hashlib
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
SRC = ROOT/"data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
SEL = json.load(open(ROOT/"data/breakthrough_20261002/batch29/selected_rows.json", encoding="utf-8"))
selkeys = {r["key"]: r for r in SEL["rows"]}
CANON = set(["chr%d"%i for i in range(1,23)]+["chrX","chrY"])
ND=NR1=NR2=NIP=NPL=0; N=0; n_elig=0; n_badcoord=n_noncanon=n_len=n_depth=0
found={}
t0=time.time()
with gzip.open(SRC,"rt",encoding="utf-8") as f:
    f.readline()
    for line in f:
        N+=1; c=line.rstrip("\n").split("\t")
        if len(c)!=13: continue
        try:
            ip=int(c[4]); r1=int(c[5])+int(c[6])+int(c[7])+int(c[8]); r2=int(c[9])+int(c[10])
            P=int(c[11])+int(c[12]); st=int(c[1]); en=int(c[2])
        except Exception: continue
        ND+=1+P; NIP+=ip; NPL+=P; NR1+=r1; NR2+=r2
        if min(ip,r1,r2,P,st,en)<0: continue
        if not (st<en) or c[3] not in ("+","-"): n_badcoord+=1; continue
        if c[0] not in CANON: n_noncanon+=1; continue
        L=en-st
        if not (200<=L<=2000): n_len+=1; continue
        if P<1: n_depth+=1; continue
        n_elig+=1
        key="%s:%d-%d:%s"%(c[0],st,en,c[3])
        if key in selkeys and len(found)<100:
            found[key]={"ipcr":ip,"r1":r1,"r2":r2,"P":P}
        if N%40000000==0: print("[rv] N=%d elig=%d found=%d %.0fs"%(N,n_elig,len(found),time.time()-t0),flush=True)
print("[rv] done N=%d %.0fs"%(N,time.time()-t0),flush=True)

out={"n_rows":N,"n_eligible":n_elig,"n_bad_coord_or_strand":n_badcoord,
     "n_noncanonical_chrom":n_noncanon,"n_length_out_of_range":n_len,"n_below_depth_P_lt_1":n_depth,
     "N_D":ND,"N_R1":NR1,"N_R2":NR2,"N_iPCR":NIP,"N_plasmid_P":NPL,
     "n_selected_first100_matched":len(found)}
# compare totals with passA
pa=json.load(open(ROOT/"reports/breakthrough_20261002/batch29/passA.json",encoding="utf-8"))
out["passA_compare"]={k:{"reviewer":out[k],"passA":pa["library_totals_fulltable"].get(k) if k.startswith("N_") else pa.get(k)} for k in
                      ("n_rows","n_eligible","n_bad_coord_or_strand","n_noncanonical_chrom","n_length_out_of_range","n_below_depth_P_lt_1")}
out["passA_compare"]["N_D_1plusP"]={"reviewer":ND,"passA":pa["library_totals_fulltable"]["N_D_1plusP"]}
out["passA_compare"]["N_R1_2"]={"reviewer":[NR1,NR2],"passA":[pa["library_totals_fulltable"]["N_R1"],pa["library_totals_fulltable"]["N_R2"]]}

# recompute y for matched first-100 and compare to rows.tsv values
import csv
rt={r["key"]:r for r in csv.DictReader(open(ROOT/"data/breakthrough_20261002/batch29/rows.tsv",encoding="utf-8"),delimiter="\t")}
import math
maxdiff=0.0; nchk=0; cnt_mismatch=0
for k,v in found.items():
    D=1+v["P"]
    a1=(v["r1"]/NR1)/(D/ND); a2=(v["r2"]/NR2)/(D/ND)
    y=0.5*(math.log2(1+a1)+math.log2(1+a2))
    r=rt.get(k)
    if not r: continue
    nchk+=1
    if int(r["r1"])!=v["r1"] or int(r["r2"])!=v["r2"] or int(r["P"])!=v["P"]:
        cnt_mismatch+=1; continue
    maxdiff=max(maxdiff, abs(y-float(r["y"])))
out["row_check"]={"n_checked":nchk,"count_mismatches":cnt_mismatch,"max_abs_y_diff":maxdiff}
json.dump(out, open(ROOT/"reports/breakthrough_20261002/batch29_review/source_verify.json","w",encoding="utf-8"),indent=2)
print(json.dumps(out,indent=2))
