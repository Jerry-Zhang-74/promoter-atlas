#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 predeclared metrics on the frozen predictions. ZERO fits (including bootstrap:
component resampling with multiplicity only, no learning).

PRIMARY: pooled 7537 finite bio Spearman(alpha, pred) for PARM, H1, GC + paired rho
differences. Fixed top-decile, fixed chr17 / non-chr17 strata, per-component rho,
paired 2000 component bootstrap (conditional uncertainty), leave-one-chromosome-out
sensitivity (no fits), controls descriptive. Controls never mixed into primary rho.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402


def load():
    with open(C.DATA / "frozen_input_table.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    h1 = json.load(open(C.DATA / "cache/h1/preds.json", encoding="utf-8"))["pred"]
    pa = json.load(open(C.DATA / "cache/parm/preds.json", encoding="utf-8"))
    parm = pa["pred"]
    recs = []
    for r in rows:
        sha = r["sequence_sha256"]
        recs.append({
            "label": r["label"], "control": int(r["control"]), "chr": r["chr"],
            "component_id": r["component_id"],
            "alpha": float(r["alpha_primary"]),
            "parm": parm[sha]["parm_mean"], "h1": h1[sha]["h1_mean"], "gc": h1[sha]["gc"],
            "parm_spread": parm[sha]["parm_fold_spread"],
            "h1_spread": h1[sha]["h1_fold_spread"],
            "parm_folds": parm[sha]["parm_folds"], "h1_folds": h1[sha]["h1_folds"],
        })
    return recs, pa["runtime"], pa["weights_sha256"]


MODELS = ("parm", "h1", "gc")


def finite_mask(recs, keys):
    m = np.ones(len(recs), dtype=bool)
    for k in keys:
        m &= np.isfinite([r[k] for r in recs])
    return m


def rho_block(recs, idx, key):
    a = [recs[i]["alpha"] for i in idx]
    p = [recs[i][key] for i in idx]
    return C.spearman(p, a)


def ties_count(vals):
    c = Counter(vals)
    return int(sum(v for v in c.values() if v > 1))


def topdecile(recs, idx, key):
    n = len(idx)
    k = int(np.ceil(C.TOP_FRAC * n))
    labels = [recs[i]["label"] for i in idx]
    alpha = [recs[i]["alpha"] for i in idx]
    pred = [recs[i][key] for i in idx]
    _, true_set = C.top_set(alpha, labels, C.TOP_FRAC)
    _, pred_set = C.top_set(pred, labels, C.TOP_FRAC)
    inter = len(true_set & pred_set)
    prec = inter / k if k else float("nan")
    enrich = (inter / k) / (k / n) if k else float("nan")
    return {"k": k, "n": n, "intersection": inter, "precision": prec, "enrichment": enrich,
            "true_ties": ties_count([round(x, 12) for x in alpha]),
            "pred_ties": ties_count([round(x, 12) for x in pred]),
            "true_top_labels_sha": C.sha256_text("|".join(sorted(labels[i] for i in true_set))),
            "pred_top_labels_sha": C.sha256_text("|".join(sorted(labels[i] for i in pred_set)))}


def main():
    t0 = time.time()
    C.set_thread_limits(4)
    recs, rt, wsha = load()
    bio = [i for i, r in enumerate(recs) if r["control"] == 0]
    ctrl = [i for i, r in enumerate(recs) if r["control"] == 1]
    bio_fin = [i for i in bio if np.isfinite(recs[i]["alpha"])]
    print("bio", len(bio), "bio_finite", len(bio_fin), "ctrl", len(ctrl))

    out = {"batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
           "record_type": "batch41_frozen_transfer_metrics", "n_bio": len(bio),
           "n_bio_finite": len(bio_fin), "n_ctrl": len(ctrl),
           "PARM_runtime": rt, "PARM_weights_sha256": wsha}

    idx = np.array(bio_fin)
    pooled = {}
    for m in MODELS:
        fr = [i for i in bio_fin if np.isfinite(recs[i][m])]
        pooled[m] = {"n": len(fr), "spearman": rho_block(recs, fr, m),
                     "top_decile": topdecile(recs, fr, m)}
    out["pooled"] = pooled
    # paired rho differences (common finite set of bio)
    common = [i for i in bio_fin if all(np.isfinite(recs[i][m]) for m in MODELS)]
    pr = {m: rho_block(recs, common, m) for m in MODELS}
    out["paired_rho_differences_common_set"] = {
        "n": len(common),
        "rho": pr,
        "PARM_minus_H1": pr["parm"] - pr["h1"],
        "PARM_minus_GC": pr["parm"] - pr["gc"],
        "H1_minus_GC": pr["h1"] - pr["gc"],
    }
    out["model_fold_spread_mean"] = {
        "parm": float(np.mean([recs[i]["parm_spread"] for i in bio_fin])),
        "h1": float(np.mean([recs[i]["h1_spread"] for i in bio_fin])),
        "note": "5-fold spread = disagreement, NOT a confidence interval",
    }

    chr17 = [i for i in bio_fin if recs[i]["chr"] == "chr17"]
    non17 = [i for i in bio_fin if recs[i]["chr"] != "chr17"]
    out["strata"] = {
        "chr17": {"n": len(chr17), "expected": 5868,
                  "rho": {m: rho_block(recs, chr17, m) for m in MODELS}},
        "non_chr17": {"n": len(non17), "expected": 1669,
                      "rho": {m: rho_block(recs, non17, m) for m in MODELS}},
        "pooled_note": "chr17 and non-chr17 BOTH reported; pool = primary; no strongest-subset selection",
    }
    C.atomic_write_json(C.DATA / "metrics_part1.json", out)
    print(json.dumps({"pooled": {m: {"n": pooled[m]["n"], "rho": round(pooled[m]["spearman"], 4),
                                    "prec": round(pooled[m]["top_decile"]["precision"], 4)} for m in MODELS},
                      "delta_parm_h1": round(out["paired_rho_differences_common_set"]["PARM_minus_H1"], 4),
                      "strata_non17_rho": {m: round(out["strata"]["non_chr17"]["rho"][m], 4) for m in MODELS}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
