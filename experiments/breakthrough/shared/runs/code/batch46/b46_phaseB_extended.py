# -*- coding: utf-8 -*-
"""batch46 Phase B: exact -Inf / zero-RNA case + extended-real sensitivity.
Retains all raw zeros/nonfinite; invents NO pseudocount / clipping / finite replacement."""
import sys, json, math, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T

RAWCOLS = {"K562_r1": "expression.RNA.count.K562.replicate1",
           "K562_r2": "expression.RNA.count.K562.replicate2",
           "HepG2_r1": "expression.RNA.count.HepG2.replicate1",
           "HepG2_r2": "expression.RNA.count.HepG2.replicate2"}
NORMCOLS = {"K562_r1": "expression.normalized.log2rpm.K562.replicate1",
            "K562_r2": "expression.normalized.log2rpm.K562.replicate2",
            "HepG2_r1": "expression.normalized.log2rpm.HepG2.replicate1",
            "HepG2_r2": "expression.normalized.log2rpm.HepG2.replicate2"}


def main():
    rows = C.parse()
    affected_rows = []
    for r in rows:
        bad = []
        for key, col in NORMCOLS.items():
            v = C.to_float(r[C.CI[col]])
            if v is None or not math.isfinite(v):
                bad.append({"rep": key, "normalized": r[C.CI[col]],
                            "raw_rna_count": r[C.CI[RAWCOLS[key]]]})
        if bad:
            affected_rows.append({"info": r[C.CI["info"]], "barcode": r[C.CI["barcode"]],
                                  "query.seq": r[C.CI["query.seq"]],
                                  "query_length": len(r[C.CI["query.seq"]]),
                                  "library": r[C.CI["library"]], "nonfinite": bad})

    frags = T.build_fragments()
    freeze = T.load_freezer()
    triplets = T.build_triplets(frags, freeze)
    import b46_phaseB_analysis as A
    recs = A.per_triplet_records(triplets)

    affected_info = affected_rows[0]["info"] if affected_rows else None
    affected_triplet = None
    for t in triplets:
        if any(t["cands"][s]["info"] == affected_info for s in C.SELS):
            affected_triplet = t
    # exact raw records for that fragment (4 barcodes x 4 columns)
    raw_records = []
    if affected_info:
        for r in rows:
            if r[C.CI["info"]] == affected_info:
                raw_records.append({"barcode": r[C.CI["barcode"]],
                                    **{k: r[C.CI[v]] for k, v in NORMCOLS.items()},
                                    **{k + "_rawCount": r[C.CI[v]] for k, v in RAWCOLS.items()}})

    # primary (finite) contrasts for the affected triplet
    prim = None
    if affected_triplet:
        tid = affected_triplet["triplet_id"]
        row = [r for r in recs if r["triplet_id"] == tid][0]
        prim = {"triplet_id": tid, "primary_status": "LIMITED",
                "missing_reason": ("the _median candidate has one barcode with 0 raw RNA counts in ALL four "
                                   "RNA columns -> normalized log2rpm = -Inf; the finite-deletion summary uses "
                                   "the remaining 3 barcodes and is NOT the full four-barcode arithmetic mean."),
                "dmm_K562_meas_finite": row["dmm_K562_meas"], "dmn_K562_meas_finite": row["dmn_K562_meas"],
                "median_K562_meas_finite": row["K562_meas_median"]}

    # extended-real: nonfinite normalized treated as -Inf (no pseudocount), mean over 4 barcodes
    ext = None
    if affected_triplet:
        c = affected_triplet["cands"]
        def ext_meas(f, cell):
            anyneg = (f["n_nonfinite"].get(cell + "_r1", 0) > 0) or (f["n_nonfinite"].get(cell + "_r2", 0) > 0)
            return None if anyneg else f[cell + "_meas"]   # None marks -Infinity (extended real)
        # direction only (avoid literal Infinity)
        med_ext = ext_meas(c["median"], "K562")
        mx = c["max"]["K562_meas"]
        mn = c["min"]["K562_meas"]
        ext = {"definition": "extended-real: treat each nonfinite normalized value as -Infinity and take the "
                             "full four-barcode arithmetic mean (no pseudocount / no clipping).",
               "median_extended_real_is_negative_infinity": med_ext is None,
               "dmm_K562_extended_real": {"value": None, "reason": "POSITIVE_INFINITY",
                                          "bounded_direction": "+Infinity (strictly positive)"} if med_ext is None else None,
               "dmn_K562_extended_real": {"value": None, "reason": "POSITIVE_INFINITY",
                                          "bounded_direction": "+Infinity (strictly positive)"} if med_ext is None else None,
               "rank_effect": "under extended reals the _median measured summary is -Inf, so _max ranks strictly "
                              "above _median and _min; rank/direction is defined, magnitude is not finite."}

    # headline robustness: recompute the two library medians WITH and WITHOUT the affected triplet
    headline = {}
    for lib in ("short", "long"):
        with_all = [r for r in recs if r["library"] == lib]
        without = [r for r in with_all if r["triplet_id"] != (affected_triplet["triplet_id"] if affected_triplet else None)]
        headline[lib] = {
            "n_with": len(with_all), "n_without": len(without),
            "dmm_median_with": C.summarize([r["dmm_K562_meas"] for r in with_all])["median"],
            "dmm_median_without": C.summarize([r["dmm_K562_meas"] for r in without])["median"]}

    out = {"batch": C.BATCH, "version": C.VERSION, "phase": "B_nonfinite_case",
           "n_nonfinite_barcode_rows": len(affected_rows),
           "n_nonfinite_fragments": len({a["info"] for a in affected_rows}),
           "affected_records": affected_rows,
           "affected_fragment_raw_records": raw_records,
           "affected_triplet": prim, "extended_real_sensitivity": ext,
           "n_unaffected_triplets": len(recs) - (1 if affected_triplet else 0),
           "n_total_triplets": len(recs),
           "headline_dependence": headline,
           "no_rna_positive_filter": True,
           "policy": "all raw zeros/nonfinite retained; no zero-RNA filtering; no invented finite replacement."}
    C.write_json(C.OUT / "nonfinite_case.json", out)
    print("nonfinite barcode rows:", len(affected_rows))
    print("affected:", affected_info)
    print("affected triplet:", prim["triplet_id"] if prim else None)
    print("headline:", json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
