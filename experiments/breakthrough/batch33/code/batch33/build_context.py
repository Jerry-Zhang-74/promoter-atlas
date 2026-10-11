# -*- coding: utf-8 -*-
"""Batch33 A: frozen availability/eligibility matrix + endpoint-context comparison.
ZERO fits / ZERO model inference. Parses the ORIGINAL exported CytoFLEX CSVs
(primary source) plus the frozen cellline_v2 tables, and rebuilds the documented
endpoint formula identically for every recorded measurement context.

Endpoint formula (unchanged):
  log10( mean_tubes(V1R Median B525-A) / mean_WTtubes(V1L Median B525-A) )
Numerator gate = V1R (GFP-positive gated population), statistic Median, channel B525-A.
Denominator = same-batch WT-negative reference = mean over that export's WT tubes of
the V1L (GFP-negative) Median B525-A. Tube collapse = arithmetic mean of per-tube
V1R medians over the single shared batch denominator.
CSV header names are Chinese; matched here by ASCII substrings only (no literals).
"""
import csv, json, math, hashlib, statistics
from pathlib import Path

ROOT = Path(r"E:/promoter_atlas")
OUT  = ROOT / "data/breakthrough_20261002/batch33"
OUT.mkdir(parents=True, exist_ok=True)
INV  = json.load(open(ROOT/"reports/cellline_v2_20261001/source_inventory.json", encoding="utf-8"))
PATH_BY_ID = {s["source_id"]: s["path"] for s in INV["sources"] if s.get("kind") == "csv"}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# --- exact source metadata from the frozen inventory (no hard-coding of names) ---
META = {}
for s in INV["sources"]:
    if s.get("kind") != "csv": continue
    t = s.get("time_metadata", {})
    META[s["source_id"]] = {
        "clone": t.get("clone_reported"), "date": t.get("acquisition_date_reported"),
        "dpt": t.get("days_after_transfection_reported"), "dps": t.get("days_after_sort_reported"),
        "round": t.get("sort_round_reported"), "flags": t.get("time_flags", []),
        "exp": t.get("experiment_name"),
    }
PRIMARY_SRC         = "S002"                          # D28 second-sort endpoint (unchanged)
SAME_CLONE_BASELINE = ["S004", "S001", "S002"]        # ROSA26-4: 5dps, 7dps, 28dps(2nd sort)
CROSS_CLONE         = ["S005", "S006", "S003", "S007"]# ROSA26-10: 5,7,10,14 dps

# documented local constructs (batch32 tables/local14_evidence_table.tsv)
CONSTRUCTS = [("A1","ACTB",100),("A2","ACTB",200),("A3","ACTB",300),("A4","ACTB",400),
              ("A5","ACTB",500),("A20","ACTB",2000),
              ("G1","GAPDH",100),("G2","GAPDH",200),("G3","GAPDH",300),("G4","GAPDH",400),
              ("G5","GAPDH",500),("G20","GAPDH",2000),
              ("CMV","CMV",204),("CAV","CMV",584)]
CONSTRUCT_ID = {"CMV": "CMVcore", "CAV": "CMVfull"}

def num(s):
    s = (s or "").strip()
    if s in ("", "#", "####", "#VALUE!"): return None
    if s.endswith("%"): s = s[:-1].strip()          # percentage-points columns
    try: return float(s)
    except ValueError: return None

def parse_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[2]]
    data = [r for r in rows[3:] if r and r[0].strip()]
    return hdr, data

def colidx(hdr):
    """Locate required columns by ASCII predicate on the (Chinese-suffixed) header."""
    def one(pred):
        for i, h in enumerate(hdr):
            if pred(h): return i
        return None
    cnt = lambda h: "Median" not in h and "%" not in h and "/" not in h
    last = lambda pred: [i for i, h in enumerate(hdr) if pred(h)][-1] if any(pred(h) for h in hdr) else None
    return {
        "V1R_med": one(lambda h: "V1R" in h and "Median B525-A" in h),
        "V1L_med": one(lambda h: "V1L" in h and "Median B525-A" in h),
        "V1R_pct": last(lambda h: "V1R" in h and "%" in h),   # parent-% block, not total-%
        "V1R_n":   one(lambda h: "V1R" in h and cnt(h)),
        "AE_n":    one(lambda h: h.strip() == "All Events"),
        "P2_n":    one(lambda h: h.startswith("P2") and cnt(h)),
    }

rows_out, ctx_rows, excl = [], [], []

for sid, fn in sorted(PATH_BY_ID.items()):
    path = Path(fn)
    if not path.exists():
        excl.append({"source_id": sid, "construct": "", "tube": "", "reason": "file_missing"}); continue
    sha = sha256(path)
    hdr, data = parse_csv(path)
    c = colidx(hdr)
    by = {r[0].strip(): r for r in data}
    names = list(by.keys())
    m = META[sid]

    # --- WT-negative reference tubes (documented, export-specific) ---
    if "WT" in by:                      # D28-style: WT block (WT + its 2 tubes) at end
        i = names.index("WT"); wt_like = names[i:i+3]
    else:                               # early exports: wt-3;wt-1;wt-2 block
        wt_like = [n for n in names if n.lower().startswith("wt")]
    wt_vals = [num(by[n][c["V1L_med"]]) for n in wt_like] if c["V1L_med"] is not None else []
    wt_vals = [v for v in wt_vals if v is not None]
    denom = statistics.mean(wt_vals) if wt_vals else None

    # --- V1 gate parent inferred from counts (V1L_n + V1R_n == AE or P2?) ---
    parent = "UNKNOWN"
    if c["V1R_n"] is not None and c["V1L_med"] is not None:
        v1l_n_col = None
        for i, h in enumerate(hdr):
            if "V1L" in h and ("Median" not in h and "%" not in h and "/" not in h):
                v1l_n_col = i; break
        r0 = by.get("A1", data[0])
        if v1l_n_col is not None:
            s = (num(r0[v1l_n_col]) or 0) + (num(r0[c["V1R_n"]]) or 0)
            if c["P2_n"] is not None and num(r0[c["P2_n"]]) == s: parent = "P2"
            elif c["AE_n"] is not None and num(r0[c["AE_n"]]) == s: parent = "All Events"

    ctx_rows.append({
        "source_id": sid, "file": path.name, "sha256": sha, "file_bytes": path.stat().st_size,
        "clone_reported": m["clone"], "cell_line_as_recorded": "HEK293T",
        "cell_context_current_authority": "USER 293FT (2026-10-03 correction; CELL_CONTEXT_293FT)",
        "acquisition_date": m["date"], "days_after_transfection": m["dpt"], "days_after_sort": m["dps"],
        "sort_round": m["round"], "time_flags": ";".join(m["flags"]),
        "numerator_gate": "V1R", "numerator_statistic": "Median", "numerator_channel": "B525-A",
        "denominator_gate": "V1L", "denominator_statistic": "Median", "denominator_channel": "B525-A",
        "v1_gate_parent_inferred": parent, "wt_negative_reference_tubes": ";".join(wt_like),
        "wt_negative_mean_recomputed": denom, "positive_fraction_denominator": parent,
        "assay_context": "transient transfection pool; fluorescence re-analysis export (CytoFLEX)",
        "integration_context_known": "UNKNOWN (no locus/integration site documented)",
        "n_constructs_present": sum(1 for cid,_,_ in CONSTRUCTS if cid in by),
    })

    # --- rebuild the SAME endpoint formula for every present construct ---
    for cid, fam, length in CONSTRUCTS:
        if cid not in by:
            reason = ("different_clone_ROSA26-10_unmapped_tube_sample_names" if sid in CROSS_CLONE
                      else "construct_absent_in_export")
            excl.append({"source_id": sid, "construct": cid, "tube": "", "reason": reason}); continue
        i = names.index(cid); tubes = names[i:i+3]
        v1r_med = [num(by[t][c["V1R_med"]]) for t in tubes]
        v1r_n   = [num(by[t][c["V1R_n"]]) for t in tubes] if c["V1R_n"] is not None else [None]*3
        pos_pct = [num(by[t][c["V1R_pct"]]) for t in tubes] if c["V1R_pct"] is not None else [None]*3
        for t, v in zip(tubes, v1r_n):
            if v == 0:
                excl.append({"source_id": sid, "construct": cid, "tube": t, "reason": "zero_V1R_events_excluded_from_gate_median"})
        ok = [v for v in v1r_med if v is not None]
        prim = None
        if denom and ok and all(v > 0 for v in ok):
            prim = math.log10(statistics.mean(ok) / denom)
        okp = [v for v in pos_pct if v is not None]
        sec = statistics.mean(okp) if okp else None
        rng = None
        if denom and len(ok) >= 2:
            lg = [math.log10(v/denom) for v in ok if v > 0]
            rng = max(lg) - min(lg) if len(lg) >= 2 else None
        rec = {"source_id": sid, "construct_id": CONSTRUCT_ID.get(cid, cid), "family": fam,
            "sequence_length_bp": length, "clone_reported": m["clone"],
            "days_after_transfection": m["dpt"], "days_after_sort": m["dps"], "sort_round": m["round"],
            "v1_gate_parent_inferred": parent, "technical_tube_ids": ";".join(tubes),
            "v1r_median_by_tube": ";".join("NA" if v is None else "%.1f" % v for v in v1r_med),
            "v1r_events_by_tube": ";".join("NA" if v is None else "%d" % int(v) for v in v1r_n),
            "tube_range_log10_primary": rng, "wt_negative_tube_ids": ";".join(wt_like),
            "wt_negative_mean": denom, "PRIMARY_log10_relative_gfp": prim,
            "SECONDARY_positive_fraction_pct": sec, "positive_fraction_denominator": parent,
            "gate_parent_differs_from_primary_context": (parent != "P2"), "n_tubes_used": len(ok)}
        rows_out.append(rec)

# ---------- verify recomputation against the frozen cellline_v2 reported values ----------
reported = {}
with open(ROOT/"reports/cellline_v2_20261001/construct_endpoints_D28.tsv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        reported[(r["source_id"], r["construct_id"])] = (float(r["log10_relative_gfp"]), float(r["gfp_positive_percent_parent"]))
with open(ROOT/"reports/cellline_v2_20261001/construct_time_diagnostics.tsv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        reported[(r["source_id"], r["construct_id"])] = (float(r["log10_relative_gfp"]), float(r["gfp_positive_percent_parent"]))

n_mismatch = 0; n_match = 0; n_sec_mismatch = 0
for r in rows_out:
    key = (r["source_id"], r["construct_id"])
    if key in reported and r["PRIMARY_log10_relative_gfp"] is not None:
        rp, rs = reported[key]
        r["reported_log10_relative_gfp"] = rp
        r["primary_abs_diff_vs_reported"] = abs(r["PRIMARY_log10_relative_gfp"] - rp)
        r["reported_secondary_pct"] = rs
        r["secondary_abs_diff_vs_reported"] = abs((r["SECONDARY_positive_fraction_pct"] or 0.0) - rs)
        if r["primary_abs_diff_vs_reported"] > 1e-6:
            r["primary_reproduction"] = "MISMATCH"; n_mismatch += 1
        else:
            r["primary_reproduction"] = "exact_reproduction"; n_match += 1
        if r["secondary_abs_diff_vs_reported"] > 0.02:
            n_sec_mismatch += 1
            r["primary_reproduction"] += ";SECONDARY_MISMATCH"
    else:
        r["reported_log10_relative_gfp"] = None; r["primary_abs_diff_vs_reported"] = None
        r["reported_secondary_pct"] = None; r["secondary_abs_diff_vs_reported"] = None
        r["primary_reproduction"] = "recomputed_new_context(no_frozen_reference)"

def write_tsv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows: w.writerow(row)

prim_cols = ["source_id","construct_id","family","sequence_length_bp","clone_reported",
 "days_after_transfection","days_after_sort","sort_round","v1_gate_parent_inferred",
 "technical_tube_ids","v1r_median_by_tube","v1r_events_by_tube","tube_range_log10_primary",
 "wt_negative_tube_ids","wt_negative_mean","PRIMARY_log10_relative_gfp","SECONDARY_positive_fraction_pct",
 "positive_fraction_denominator","gate_parent_differs_from_primary_context","n_tubes_used",
 "reported_log10_relative_gfp","primary_abs_diff_vs_reported","primary_reproduction",
 "reported_secondary_pct","secondary_abs_diff_vs_reported"]
write_tsv(OUT/"frozen_eligibility_matrix.tsv", rows_out, prim_cols)

ctx_cols = ["source_id","file","sha256","file_bytes","clone_reported","cell_line_as_recorded",
 "cell_context_current_authority","acquisition_date","days_after_transfection","days_after_sort",
 "sort_round","time_flags","numerator_gate","numerator_statistic","numerator_channel",
 "denominator_gate","denominator_statistic","denominator_channel","v1_gate_parent_inferred",
 "wt_negative_reference_tubes","wt_negative_mean_recomputed","positive_fraction_denominator",
 "assay_context","integration_context_known","n_constructs_present"]
write_tsv(OUT/"context_registry.tsv", ctx_rows, ctx_cols)

if excl:
    with open(OUT/"exclusions.tsv","w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["source_id","construct","tube","reason"],delimiter="\t",extrasaction="ignore")
        w.writeheader()
        for e in excl: w.writerow(e)

print("contexts=%d cells=%d exact_repro=%d mismatch=%d" % (len(ctx_rows), len(rows_out), n_match, n_mismatch))
for r in sorted(ctx_rows, key=lambda x: x["source_id"]):
    print("  %s clone=%s %sdpt/%sdps round=%s parent=%s denom=%s WT=%s" % (
        r["source_id"], r["clone_reported"], r["days_after_transfection"], r["days_after_sort"],
        r["sort_round"], r["v1_gate_parent_inferred"], r["wt_negative_mean_recomputed"], r["wt_negative_reference_tubes"]))
print("primary mismatches:")
for r in rows_out:
    if r["primary_reproduction"] == "MISMATCH":
        print("   ", r["source_id"], r["construct_id"], r["primary_abs_diff_vs_reported"])
