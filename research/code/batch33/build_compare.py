# -*- coding: utf-8 -*-
"""Batch33 A: observed family/timepoint comparison + frozen-model direction grid.
ZERO fits / ZERO inference. Reuses UNCHANGED saved scores:
  PARM = reports/breakthrough_20261002/batch31/tables/local_supported_scores.tsv
  H1   = data/breakthrough_20261002/batch24/local14_exploratory_scores.tsv
  R1   = frozen pair directions in reports/breakthrough_20261002/batch32/sectionB_metrics.json
No new calibration, no endpoint switching, no model selection by context.
"""
import csv, json, math, statistics
from pathlib import Path
ROOT = Path(r"E:/promoter_atlas")
OUT  = ROOT/"data/breakthrough_20261002/batch33"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY_CONTEXT = "S002"
ELIGIBLE = ["S004", "S001", "S002"]
CTX_LABEL = {"S004": "early_5dps_12dpt", "S001": "early_7dps_14dpt", "S002": "D28_28dps_43dpt_2ndsort"}
SUPPORTED_MAX_LEN = 600
FAMILIES = {"ACTB": ["A1","A2","A3","A4","A5","A20"],
            "GAPDH": ["G1","G2","G3","G4","G5","G20"],
            "CMV": ["CMVcore","CMVfull"]}
cells = {}
for r in csv.DictReader(open(OUT/"frozen_eligibility_matrix.tsv", encoding="utf-8"), delimiter="\t"):
    if r["clone_reported"] != "ROSA26-4": continue
    sid, cid = r["source_id"], r["construct_id"]
    cells[(sid, cid)] = {"primary": float(r["PRIMARY_log10_relative_gfp"]),
                         "secondary": float(r["SECONDARY_positive_fraction_pct"]),
                         "length": int(r["sequence_length_bp"]), "family": r["family"],
                         "parent": r["v1_gate_parent_inferred"], "dps": r["days_after_sort"]}

def spearman(pairs):
    n = len(pairs)
    if n < 3: return None
    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i]); rk = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and vals[order[j+1]] == vals[order[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    a = rank([p[0] for p in pairs]); b = rank([p[1] for p in pairs])
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x-ma)**2 for x in a)); db = math.sqrt(sum((y-mb)**2 for y in b))
    return None if da == 0 or db == 0 else num/(da*db)

def sgn(x, tol=1e-12):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)

obs_rows = []
for fam, members in FAMILIES.items():
    for sid in ELIGIBLE:
        vals = [(c, cells[(sid, c)]["primary"]) for c in members if (sid, c) in cells]
        ranks = {c: rk for rk, (c, _) in enumerate(sorted(vals, key=lambda t: -t[1]), 1)}
        for c, v in vals:
            d28 = cells[(PRIMARY_CONTEXT, c)]["primary"] if (PRIMARY_CONTEXT, c) in cells else None
            obs_rows.append({"family": fam, "construct_id": c, "source_id": sid,
                "context": CTX_LABEL[sid], "days_after_sort": cells[(sid, c)]["dps"],
                "sequence_length_bp": cells[(sid, c)]["length"],
                "PRIMARY_log10_relative_gfp": v, "rank_within_family": ranks[c],
                "D28_primary": d28, "delta_vs_D28_log10": (None if d28 is None else v-d28),
                "model_supported": int(cells[(sid, c)]["length"] <= SUPPORTED_MAX_LEN)})
with open(OUT/"observed_by_context.tsv","w",encoding="utf-8",newline="") as f:
    cols=["family","construct_id","source_id","context","days_after_sort","sequence_length_bp",
          "PRIMARY_log10_relative_gfp","rank_within_family","D28_primary","delta_vs_D28_log10","model_supported"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
    for r in obs_rows: w.writerow(r)

fam_rows = []
for fam, members in FAMILIES.items():
    mem = [c for c in members if (PRIMARY_CONTEXT, c) in cells]
    row = {"family": fam, "n_constructs": len(mem), "constructs": ";".join(mem), "note": ""}
    for sid in ["S004","S001"]:
        common = [c for c in mem if (sid, c) in cells]
        row["spearman_%s_vs_D28" % CTX_LABEL[sid]] = (spearman([(cells[(sid,c)]["primary"], cells[(PRIMARY_CONTEXT,c)]["primary"]) for c in common]) if len(common) >= 3 else None)
    if len(mem) < 3: row["note"] = "fewer_than_3_constructs_no_within_family_rank"
    fam_rows.append(row)
with open(OUT/"family_rank_vs_D28.tsv","w",encoding="utf-8",newline="") as f:
    cols=["family","n_constructs","constructs","spearman_early_5dps_12dpt_vs_D28","spearman_early_7dps_14dpt_vs_D28","note"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
    for r in fam_rows: w.writerow(r)

pair_rows = []
for fam, members in FAMILIES.items():
    mem = [c for c in members if (PRIMARY_CONTEXT, c) in cells]
    for i in range(len(mem)):
        for j in range(i+1, len(mem)):
            a, b = mem[i], mem[j]
            cur = {"family": fam, "a": a, "b": b,
                   "d28_delta": cells[(PRIMARY_CONTEXT,a)]["primary"] - cells[(PRIMARY_CONTEXT,b)]["primary"]}
            cur["dir_D28"] = sgn(cur["d28_delta"])
            for sid in ["S004","S001"]:
                if (sid,a) in cells and (sid,b) in cells:
                    d = cells[(sid,a)]["primary"] - cells[(sid,b)]["primary"]
                    cur["dir_%s" % CTX_LABEL[sid]] = sgn(d)
                    cur["delta_%s" % CTX_LABEL[sid]] = d
                    m = int(sgn(d) == sgn(cur["d28_delta"])) if sgn(d)!=0 and sgn(cur["d28_delta"])!=0 else ""
                    cur["dir_match_vs_D28_%s" % CTX_LABEL[sid]] = m
            pair_rows.append(cur)
with open(OUT/"pair_direction_by_context.tsv","w",encoding="utf-8",newline="") as f:
    cols=["family","a","b","d28_delta","dir_D28","dir_early_5dps_12dpt","delta_early_5dps_12dpt",
          "dir_match_vs_D28_early_5dps_12dpt","dir_early_7dps_14dpt","delta_early_7dps_14dpt",
          "dir_match_vs_D28_early_7dps_14dpt"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
    for r in pair_rows: w.writerow(r)

secB = json.load(open(ROOT/"reports/breakthrough_20261002/batch32/sectionB_metrics.json", encoding="utf-8"))
frozen_pred = {}
for model, fams in secB["direction_vs_frozen_models"].items():
    for fam, blk in fams.items():
        for p in blk["pairs"]:
            frozen_pred[(model, p["a"], p["b"])] = (p["pred_dir"], p["obs_dir"])
grid_rows = []
for r in pair_rows:
    a, b = r["a"], r["b"]
    if cells[(PRIMARY_CONTEXT,a)]["length"] > SUPPORTED_MAX_LEN: continue
    for model in ["PARM","H1","R1"]:
        if (model, a, b) not in frozen_pred: continue
        pred_dir = frozen_pred[(model, a, b)][0]
        grid_rows.append({"family": r["family"], "a": a, "b": b, "model": model,
            "frozen_pred_dir": pred_dir, "D28_obs_dir": r["dir_D28"],
            "agree_at_D28": int(pred_dir == r["dir_D28"]) if r["dir_D28"] != 0 else "",
            "early5_obs_dir": r.get("dir_early_5dps_12dpt"),
            "agree_at_early5": (int(pred_dir == r.get("dir_early_5dps_12dpt")) if r.get("dir_early_5dps_12dpt") not in (None,0) else ""),
            "early7_obs_dir": r.get("dir_early_7dps_14dpt"),
            "agree_at_early7": (int(pred_dir == r.get("dir_early_7dps_14dpt")) if r.get("dir_early_7dps_14dpt") not in (None,0) else "")})
with open(OUT/"model_direction_grid.tsv","w",encoding="utf-8",newline="") as f:
    cols=["family","a","b","model","frozen_pred_dir","D28_obs_dir","agree_at_D28",
          "early5_obs_dir","agree_at_early5","early7_obs_dir","agree_at_early7"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
    for r in grid_rows: w.writerow(r)

sec_rows = []
for fam, members in FAMILIES.items():
    for c in members:
        row = {"family": fam, "construct_id": c}
        for sid in ELIGIBLE:
            if (sid,c) in cells:
                row["pct_%s" % CTX_LABEL[sid]] = round(cells[(sid,c)]["secondary"], 3)
                row["parent_%s" % CTX_LABEL[sid]] = cells[(sid,c)]["parent"]
        sec_rows.append(row)
with open(OUT/"secondary_positive_fraction_by_context.tsv","w",encoding="utf-8",newline="") as f:
    cols=["family","construct_id","pct_early_5dps_12dpt","parent_early_5dps_12dpt",
          "pct_early_7dps_14dpt","parent_early_7dps_14dpt","pct_D28_28dps_43dpt_2ndsort","parent_D28_28dps_43dpt_2ndsort"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
    for r in sec_rows: w.writerow(r)

def macro(model):
    rows = [g for g in grid_rows if g["model"]==model and g["agree_at_D28"] != ""]
    e5 = [g for g in rows if g["agree_at_early5"] != ""]
    e7 = [g for g in rows if g["agree_at_early7"] != ""]
    return {"n_pairs": len(rows),
            "agree_at_D28": sum(g["agree_at_D28"] for g in rows),
            "acc_at_D28": round(sum(g["agree_at_D28"] for g in rows)/len(rows), 4) if rows else None,
            "acc_at_early5": round(sum(g["agree_at_early5"] for g in e5)/len(e5), 4) if e5 else None,
            "n_early5": len(e5),
            "acc_at_early7": round(sum(g["agree_at_early7"] for g in e7)/len(e7), 4) if e7 else None,
            "n_early7": len(e7)}
summary = {"eligible_contexts": ELIGIBLE, "primary_context": PRIMARY_CONTEXT,
           "n_observed_cells": len(obs_rows), "n_pairs": len(pair_rows),
           "macro_direction_by_model": {m: macro(m) for m in ["PARM","H1","R1"]}}
json.dump(summary, open(OUT/"comparison_summary.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
