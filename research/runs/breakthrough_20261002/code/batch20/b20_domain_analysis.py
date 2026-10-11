#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded domain/representation error analysis (batch20 Req4, no new download).

Uses ONLY local artifacts:
  * constructs_demo.csv  (14 HEK293T ROSA26-4 D28 constructs, observed vs the
    exploratory last-164 K562-MPRA window prediction)
  * preliminary_metrics.json (within-source grouped-CV reference)

Question: does the exploratory TSS-proximal 164 bp window carry ANY of the signal
that separates truncation lengths in the HEK293T observations? If not, the window
feature is a representation-domain failure, not a model-quality question.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b20_common as C

REPORT = C.REPORT_OUT


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def main():
    import csv
    rows = list(csv.DictReader((REPORT / "constructs_demo.csv").open(encoding="utf-8")))
    have = [r for r in rows if r["mpra_pred"] not in ("", None) and r["observed_value"] not in ("", None)]
    obs = np.array([float(r["observed_value"]) for r in have])
    prd = np.array([float(r["mpra_pred"]) for r in have])
    fam = [r["family"] for r in have]
    fam_span = {}
    for f in sorted(set(fam)):
        m = [i for i, x in enumerate(fam) if x == f]
        fam_span[f] = {
            "n": len(m),
            "observed_min": round(float(obs[m].min()), 4),
            "observed_max": round(float(obs[m].max()), 4),
            "observed_span": round(float(obs[m].max() - obs[m].min()), 4),
            "predicted_span": round(float(prd[m].max() - prd[m].min()), 4),
        }
    pm = json.load((REPORT / "preliminary_metrics.json").open(encoding="utf-8"))
    out = {
        "analysis": "exploratory_164bp_window_vs_HEK293T_construct_observations",
        "generated_at_local": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_constructs_with_both": len(have),
        "spearman_observed_vs_window_pred": round(spearman(obs, prd), 5),
        "n_distinct_window_predictions": int(len(np.unique(prd))),
        "distinct_window_values": sorted(set(round(float(x), 5) for x in prd)),
        "per_family_span": fam_span,
        "within_source_reference": {
            "metric": "batch02 grouped-by-family OOF Spearman (K562 legnet)",
            "spearman": pm["mpra_test"]["honest_grouped_cv_reference"]["spearman"],
            "mae": pm["mpra_test"]["honest_grouped_cv_reference"]["mae"],
        },
        "interpretation": (
            "The exploratory TSS-proximal 164 bp window takes only 3 distinct values across all 12 constructs "
            "and is CONSTANT within every family (per-family predicted span exactly 0.0), because the nested "
            "truncation series share an identical 3'-proximal end (A2..A20 identical; G2..G20 identical). The "
            "observed-vs-window Spearman of ~0.60 is therefore a BETWEEN-FAMILY effect only (the window happens "
            "to order ACTB > CMV > GAPDH), while the WITHIN-FAMILY length signal is exactly zero even though the "
            "observed HEK293T spans are 0.14-1.34 log10 units. Mechanism: discarding upstream context removes "
            "exactly the feature that varies with truncation length, so no ranking of truncations is possible "
            "from this window regardless of model. This is a representation/domain failure, NOT a target for "
            "further hyperparameter search on 14 constructs."
        ),
        "scope": "HEK293T ROSA26-4 D28, ZERO independent biological replicates; exploratory only.",
    }
    C.atomic_write_json(REPORT / "domain_error_analysis.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
