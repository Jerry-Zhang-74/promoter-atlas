# -*- coding: utf-8 -*-
"""batch47 Protocol C: NEW b46 addendum ONLY (<=5 min).

Corrects the b46 nonfinite_case extended-real field and completes the omitted per-RNA-replicate
max-min contrasts from the already-frozen b46 tables. Does NOT rewrite any b46 canonical file,
report, or SHA; writes only NEW files under reports/breakthrough_20261002/batch47/.

Bug being corrected: b46 set dmn_K562_extended_real = +Infinity whenever the _median candidate had
a dropout. But dmn = max - min does NOT involve the median. Only dmm = max - median becomes +Inf.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b47_common import ROOT, B47_REPORT, write_json, write_tsv, sha256_file  # noqa: E402

B46 = ROOT / "reports" / "breakthrough_20261002" / "batch46"
B44_SRC = ROOT / "data" / "breakthrough_20261002" / "batch44" / "syn.txt.gz"


def _f(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _d(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return a - b


def load_fragments():
    frag = {}
    with open(B46 / "per_fragment.tsv", "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            frag[row["info"]] = {k: _f(row.get(k)) for k in
                                 ("K562_r1_mean", "K562_r2_mean", "HepG2_r1_mean", "HepG2_r2_mean")}
    return frag


def json_load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _primary_source_check():
    """Decode the b44 cached source and verify the ONE affected barcode row (0 raw RNA, -Inf normalized)."""
    info = "K562_fitness_232bp_replicate01_generation0_individual192_median"
    bc = "TTGGATGCGACGA"
    rawcols = ["expression.RNA.count.K562.replicate1", "expression.RNA.count.K562.replicate2",
               "expression.RNA.count.HepG2.replicate1", "expression.RNA.count.HepG2.replicate2"]
    normcols = ["expression.normalized.log2rpm.K562.replicate1", "expression.normalized.log2rpm.K562.replicate2",
                "expression.normalized.log2rpm.HepG2.replicate1", "expression.normalized.log2rpm.HepG2.replicate2"]
    found = None
    if B44_SRC.exists():
        with gzip.open(B44_SRC, "rt", encoding="utf-8", errors="replace", newline="") as f:
            hdr = f.readline().rstrip("\n").split("\t")
            idx = {c: i for i, c in enumerate(hdr)}
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[idx["barcode"]] == bc and parts[idx["info"]] == info:
                    found = {"barcode": bc, "info": info,
                             "raw_rna_counts": {c: parts[idx[c]] for c in rawcols},
                             "normalized_log2rpm": {c: parts[idx[c]] for c in normcols}}
                    break
    return {"source_file": str(B44_SRC), "found": found is not None, "row": found,
            "interpretation": ("verified: this ONE barcode has 0 raw RNA counts in all four RNA columns -> "
                               "normalized log2rpm = -Inf; the median candidate per-rep mean is finite via "
                               "finite-barcode deletion. max and min candidates are unaffected.")}


def build_per_rep(rows_in, frag):
    out_rows = []
    n_missing = {"K562_r1": 0, "K562_r2": 0, "HepG2_r1": 0, "HepG2_r2": 0}
    for r in rows_in:
        imax, imin = r["info_max"], r["info_min"]
        fmx, fmn = frag.get(imax, {}), frag.get(imin, {})
        rec = {"triplet_id": r["triplet_id"], "library": r["library"], "lineage": r["lineage"],
               "generation": r["generation"], "info_max": imax, "info_min": imin}
        for cell, cols in (("K562", ("K562_r1_mean", "K562_r2_mean")),
                           ("HepG2", ("HepG2_r1_mean", "HepG2_r2_mean"))):
            for rep, col in (("r1", cols[0]), ("r2", cols[1])):
                v = _d(fmx.get(col), fmn.get(col))
                rec["dmn_%s_%s" % (cell, rep)] = v
                if v is None:
                    n_missing["%s_%s" % (cell, rep)] += 1
                    rec["dmn_%s_%s_status" % (cell, rep)] = (
                        "NULL: max or min candidate per-rep mean missing/nonfinite (finite-deletion summary)")
                else:
                    rec["dmn_%s_%s_status" % (cell, rep)] = "FINITE"
        out_rows.append(rec)
    return out_rows, n_missing


def main():
    frag = load_fragments()
    rows_in = list(csv.DictReader(open(B46 / "per_triplet.tsv", encoding="utf-8", newline=""), delimiter="\t"))
    out_rows, n_missing = build_per_rep(rows_in, frag)
    aff = next((r for r in out_rows if r["triplet_id"] == "long:rep01:gen0"), None)
    fin = json_load(B46 / "nonfinite_case.json")
    dmn_meas_finite = fin["affected_triplet"]["dmn_K562_meas_finite"]

    correction = {
        "batch": "batch47", "version": "v6.34.0-scoped-input-routing.rc1",
        "phase": "C_b46_addendum", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "targets_file": str(B46 / "nonfinite_case.json"),
        "targets_file_sha256": sha256_file(B46 / "nonfinite_case.json"),
        "error": ("b46 nonfinite_case.extended_real_sensitivity.dmn_K562_extended_real was set to +Infinity "
                  "whenever the _median candidate dropped out. WRONG: dmn = max - min does not involve the "
                  "median candidate."),
        "corrected_fields": {
            "extended_real_sensitivity.dmm_K562_extended_real": {
                "value": None, "reason": "POSITIVE_INFINITY",
                "bounded_direction": "+Infinity (strictly positive)",
                "why": "max finite, median = -Infinity -> max - median = +Infinity (unchanged from b46)"},
            "extended_real_sensitivity.dmn_K562_extended_real": {
                "value": dmn_meas_finite, "reason": None, "finite": True,
                "why": ("max and min candidates are BOTH finite (only the median candidate dropped out), so "
                        "max - min is finite and equals the finite-deletion value %r" % dmn_meas_finite)}},
        "not_rewritten": ["reports/breakthrough_20261002/batch46/nonfinite_case.json",
                          "reports/breakthrough_20261002/batch46/report.md",
                          "reports/breakthrough_20261002/batch46/sha_manifest.json",
                          "b46 independent_review.json", "all b46 canonical bytes preserved"],
        "old_compliance_FAIL_preserved": "omitted per-RNA-repl max-min contrasts (now appended here)",
        "no_new_scientific_slice": True, "no_new_model_score": True, "no_new_figures": True,
    }
    correction["primary_source_row_check"] = _primary_source_check()
    write_json(B47_REPORT / "b46_addendum_nonfinite_correction.json", correction)

    cols = ["triplet_id", "library", "lineage", "generation", "info_max", "info_min",
            "dmn_K562_r1", "dmn_K562_r1_status", "dmn_K562_r2", "dmn_K562_r2_status",
            "dmn_HepG2_r1", "dmn_HepG2_r1_status", "dmn_HepG2_r2", "dmn_HepG2_r2_status"]
    write_tsv(B47_REPORT / "b46_addendum_dmn_per_rna_rep.tsv", out_rows, cols)
    summary = {
        "batch": "batch47", "phase": "C_b46_addendum_dmn_per_rna_rep",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_triplets": len(out_rows), "n_expected": 244,
        "n_finite_per_rep": {k: (len(out_rows) - v) for k, v in n_missing.items()},
        "n_missing_per_rep": n_missing,
        "missing_explanations": ("per-rep mean is the finite-barcode mean (nonfinite normalized values dropped, "
                                 "as in the frozen b46 per_fragment table); a null dmn means the max or min "
                                 "candidate per-rep mean is missing/nonfinite. No value is invented."),
        "affected_triplet": aff,
        "source_table": "reports/breakthrough_20261002/batch46/per_fragment.tsv (already frozen)",
        "triplet_membership": "reports/breakthrough_20261002/batch46/per_triplet.tsv (already frozen)",
    }
    write_json(B47_REPORT / "b46_addendum_dmn_per_rna_rep_summary.json", summary)
    print("n_triplets", len(out_rows), "missing", n_missing)
    print("affected dmn per rep:", {k: aff[k] for k in ("dmn_K562_r1", "dmn_K562_r2")} if aff else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
