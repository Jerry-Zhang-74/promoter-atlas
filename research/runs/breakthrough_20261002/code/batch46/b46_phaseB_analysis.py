# -*- coding: utf-8 -*-
"""batch46 Phase B: primary K562 same-generation contrasts + per-triplet table.
Short and long reported separately; every eligible triplet retained (incl negative/ties)."""
import sys, json, math, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T

LINEAGES = ["%02d" % i for i in range(10)]
CELLS = ["K562", "HepG2"]


def d(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return a - b


def per_triplet_records(triplets):
    recs = []
    for t in triplets:
        c = t["cands"]
        r = {"triplet_id": t["triplet_id"], "library": t["library"], "lineage": t["lineage"],
             "generation": t["generation"], "q_len": t["actual_query_length"]}
        for s in C.SELS:
            r["info_" + s] = c[s]["info"]
            r["pred_K562_" + s] = c[s]["pred_K562"]
            r["pred_HepG2_" + s] = c[s]["pred_HepG2"]
            r["GC_" + s] = c[s]["GC"]
            for cell in CELLS:
                r["%s_r1_%s" % (cell, s)] = c[s][cell + "_r1_mean"]
                r["%s_r2_%s" % (cell, s)] = c[s][cell + "_r2_mean"]
                r["%s_meas_%s" % (cell, s)] = c[s][cell + "_meas"]
        for cell in CELLS:
            r["dmm_%s_meas" % cell] = d(c["max"][cell + "_meas"], c["median"][cell + "_meas"])
            r["dmn_%s_meas" % cell] = d(c["max"][cell + "_meas"], c["min"][cell + "_meas"])
            r["dmm_%s_r1" % cell] = d(c["max"][cell + "_r1_mean"], c["median"][cell + "_r1_mean"])
            r["dmm_%s_r2" % cell] = d(c["max"][cell + "_r2_mean"], c["median"][cell + "_r2_mean"])
        p = {s: c[s]["pred_K562"] for s in C.SELS}
        r["pred_K562_order_ok"] = bool(p["max"] is not None and p["median"] is not None
                                       and p["min"] is not None and p["max"] > p["median"] > p["min"])
        recs.append(r)
    return recs


def gen_summary(recs, lib, gen):
    rows = [r for r in recs if r["library"] == lib and r["generation"] == gen]
    return {"n": len(rows),
            "dmm_K562": C.summarize([r["dmm_K562_meas"] for r in rows]),
            "dmn_K562": C.summarize([r["dmn_K562_meas"] for r in rows])}


def lineage_summary(recs, lib, key):
    """per lineage: equal-weight mean across that lineage's eligible generations."""
    out = {}
    for lin in LINEAGES:
        vals = [r[key] for r in recs if r["library"] == lib and r["lineage"] == lin]
        fin = [v for v in vals if v is not None and math.isfinite(v)]
        out[lin] = {"n_generations": len(vals), "n_finite": len(fin),
                    "mean_equal_generation_weight": (sum(fin) / len(fin)) if fin else None,
                    "values_by_generation": {str(r["generation"]): r[key]
                                             for r in recs if r["library"] == lib and r["lineage"] == lin}}
    return out


def across_lineage(per_lin):
    means = [v["mean_equal_generation_weight"] for v in per_lin.values()
             if v["mean_equal_generation_weight"] is not None]
    return C.summarize(means)


def main():
    frags = T.build_fragments()
    freeze = T.load_freezer()
    triplets = T.build_triplets(frags, freeze)
    recs = per_triplet_records(triplets)

    # per-triplet TSV (all rows, incl negative/ties)
    hdr = (["triplet_id", "library", "lineage", "generation", "q_len",
            "info_max", "info_median", "info_min",
            "pred_K562_max", "pred_K562_median", "pred_K562_min", "pred_K562_order_ok",
            "GC_max", "GC_median", "GC_min"] +
           ["%s_meas_%s" % (cell, s) for cell in CELLS for s in C.SELS] +
           ["dmm_K562_meas", "dmn_K562_meas", "dmm_K562_r1", "dmm_K562_r2",
            "dmm_HepG2_meas", "dmn_HepG2_meas",
            "pred_HepG2_max", "pred_HepG2_median", "pred_HepG2_min"])
    with open(C.OUT / "per_triplet.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in sorted(recs, key=lambda r: r["triplet_id"]):
            fh.write("\t".join("" if r.get(h) is None else str(r.get(h)) for h in hdr) + "\n")
    C.write_json(C.OUT / "per_triplet.json", {"batch": C.BATCH, "version": C.VERSION,
                                              "n_triplets": len(recs), "rows": recs})

    # primary K562 contrasts, short and long separately -------------------------
    summary = {"batch": C.BATCH, "version": C.VERSION, "phase": "B_primary_within_generation",
               "primary_cell": "K562", "primary_cell_reason":
               "K562 is the design objective of the author GA (Methods); HepG2 is cross-cell secondary.",
               "unit": "log2 RNA/DNA (measured summary units); differences only; no cross-unit MAE/R2.",
               "note": "pair/generation/barcode counts are NOT independent n; the 10 GA search lineages "
                       "share one PARM model and pipeline.",
               "per_library": {}}
    for lib, commonG in (("short", 48), ("long", 44)):
        blk = {"n_triplets": sum(1 for r in recs if r["library"] == lib),
               "n_lineages": len({r["lineage"] for r in recs if r["library"] == lib}),
               "by_generation": {str(g): gen_summary(recs, lib, g)
                                 for g in sorted({r["generation"] for r in recs if r["library"] == lib})},
               "all_generations_per_lineage_dmm": lineage_summary(recs, lib, "dmm_K562_meas"),
               "all_generations_per_lineage_dmn": lineage_summary(recs, lib, "dmn_K562_meas"),
               "generation0": {"n_triplets": sum(1 for r in recs if r["library"] == lib and r["generation"] == 0),
                               "dmm": C.summarize([r["dmm_K562_meas"] for r in recs
                                                   if r["library"] == lib and r["generation"] == 0]),
                               "dmn": C.summarize([r["dmn_K562_meas"] for r in recs
                                                   if r["library"] == lib and r["generation"] == 0])},
               "common_generation_G%d" % commonG: {
                   "n_triplets": sum(1 for r in recs if r["library"] == lib and r["generation"] == commonG),
                   "dmm": C.summarize([r["dmm_K562_meas"] for r in recs
                                       if r["library"] == lib and r["generation"] == commonG]),
                   "dmn": C.summarize([r["dmn_K562_meas"] for r in recs
                                       if r["library"] == lib and r["generation"] == commonG])},
               "replicate_contrasts": {
                   "dmm_K562_r1": C.summarize([r["dmm_K562_r1"] for r in recs if r["library"] == lib]),
                   "dmm_K562_r2": C.summarize([r["dmm_K562_r2"] for r in recs if r["library"] == lib])},
               "all_rows_dmm": C.summarize([r["dmm_K562_meas"] for r in recs if r["library"] == lib]),
               "all_rows_dmn": C.summarize([r["dmn_K562_meas"] for r in recs if r["library"] == lib])}
        blk["all_generations_pooled_across_lineage_dmm"] = across_lineage(blk["all_generations_per_lineage_dmm"])
        blk["all_generations_pooled_across_lineage_dmn"] = across_lineage(blk["all_generations_per_lineage_dmn"])
        summary["per_library"][lib] = blk

    # score-ordering agreement / conflicts -------------------------------------
    conf = [{"triplet_id": r["triplet_id"],
             "pred_K562": {s: r["pred_K562_" + s] for s in C.SELS}}
            for r in recs if not r["pred_K562_order_ok"]]
    summary["score_ordering_check"] = {
        "rule": "does the published K562 score rank _max > _median > _min within each frozen triplet?",
        "n_triplets": len(recs), "n_agree": len(recs) - len(conf), "n_conflict": len(conf),
        "conflicts": conf,
        "policy": "labels are NEVER repaired from expression; conflicts are reported as-is."}
    C.write_json(C.OUT / "primary_contrasts.json", summary)

    print("triplets", len(recs))
    for lib in ("short", "long"):
        b = summary["per_library"][lib]
        print(lib, "n", b["n_triplets"],
              "all-rows dmm med", b["all_rows_dmm"]["median"],
              "dmn med", b["all_rows_dmn"]["median"],
              "across-lin dmm med", b["all_generations_pooled_across_lineage_dmm"]["median"])
    print("score order agree", summary["score_ordering_check"]["n_agree"], "conflict",
          summary["score_ordering_check"]["n_conflict"])


if __name__ == "__main__":
    main()
