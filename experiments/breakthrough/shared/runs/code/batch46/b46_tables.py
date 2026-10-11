# -*- coding: utf-8 -*-
"""batch46 shared builders: per-fragment fixed summaries + joined per-triplet table."""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C

CELLCOLS = {"K562": ("expression.normalized.log2rpm.K562.replicate1",
                     "expression.normalized.log2rpm.K562.replicate2"),
            "HepG2": ("expression.normalized.log2rpm.HepG2.replicate1",
                      "expression.normalized.log2rpm.HepG2.replicate2")}


def build_fragments():
    """Per unique query DNA: fixed per-replicate finite-barcode means + assay summary.
    Also counts nonfinite cells per replicate column (retained, never replaced)."""
    rows = C.parse()
    frags = {}
    for r in rows:
        seq = r[C.CI["query.seq"]]
        f = frags.get(seq)
        if f is None:
            f = frags[seq] = {"query.seq": seq, "info": r[C.CI["info"]],
                              "library": r[C.CI["library"]], "lib": C.libtag(r[C.CI["library"]]),
                              "category": C.category(r[C.CI["info"]]),
                              "q_len": len(seq), "GC": C.gc_fraction(seq),
                              "pred_K562": C.to_float(r[C.CI["K562.query.PARM.prediction"]]),
                              "pred_HepG2": C.to_float(r[C.CI["HepG2.query.PARM.prediction"]]),
                              "cells": {}, "n_nonfinite": {}, "barcodes": []}
        f["barcodes"].append(r[C.CI["barcode"]])
        for cell, (c1, c2) in CELLCOLS.items():
            for k, col in (("r1", c1), ("r2", c2)):
                v = C.to_float(r[C.CI[col]])
                key = cell + "_" + k
                f["cells"].setdefault(key, [])
                if v is None or not math.isfinite(v):
                    f["n_nonfinite"][key] = f["n_nonfinite"].get(key, 0) + 1
                else:
                    f["cells"][key].append(v)
    for seq, f in frags.items():
        for cell in CELLCOLS:
            m1 = C.mean_finite(f["cells"][cell + "_r1"])
            m2 = C.mean_finite(f["cells"][cell + "_r2"])
            f[cell + "_r1_mean"] = m1
            f[cell + "_r2_mean"] = m2
            f[cell + "_meas"] = C.mean_finite([m1, m2])
            f[cell + "_n_nonfinite"] = (f["n_nonfinite"].get(cell + "_r1", 0) +
                                        f["n_nonfinite"].get(cell + "_r2", 0))
        f["n_barcodes"] = len(f["barcodes"])
    return frags


def load_freezer():
    return json.loads((C.OUT / "freeze_triplets.json").read_text(encoding="utf-8"))


def build_triplets(frags, freeze):
    """Join frozen triplet membership (metadata) to published score + measured summaries.
    Returns list of triplet dicts with candidates keyed by selection label."""
    out = []
    for t in freeze["triplets"]:
        cands = {}
        for s in C.SELS:
            info = t["candidates"][s]["info"]
            # map info -> fragment via seq sha? we stored seq only by hash; recover by info
            f = None
            for ff in frags.values():
                if ff["info"] == info:
                    f = ff
                    break
            assert f is not None, info
            cands[s] = f
        out.append({"triplet_id": t["triplet_id"], "library": t["library"],
                    "lineage": t["lineage"], "generation": t["generation"],
                    "actual_query_length": t["actual_query_length"],
                    "cands": cands})
    return out
