#!/usr/bin/env python3
"""Batch06 Phase 0 errata: n=1 vs n>=2 arm (B1/B2) differences from SAVED predictions.

Parent scientific correction (binding): same input for n=1 does NOT imply equal
outputs from separately trained B1 (repeat_bag) / B2 (diverse_bag) models.
Batch05 review limitations asserted B1==B2 / delta concentrated in the
multi-promoter subset. We reject that inference by directly computing, from the
immutable corrected saved predictions, the B2-B1 paired differences and metrics
separately for n=1 and n>=2 (and n=2-3 / n>=4). No new training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_common import (  # noqa: E402
    REPORT, atomic_write_json, metrics_dict, spearman, fast_spearman, r2, mae,
    fast_component_bootstrap_diff,
)
from b6_data import load_b6  # noqa: E402

PRED_B1 = ROOT / "reports" / "breakthrough_20261002" / "batch05" / "predictions" / \
    "pred_v6.6.0-scratchgene.rc6_s20261011.tsv"
PRED_B2 = ROOT / "reports" / "breakthrough_20261002" / "batch05" / "predictions" / \
    "pred_v6.6.0-scratchgene.rc7_s20261011.tsv"


def strata_metrics(y, p2, p1, comp, mask):
    y2 = y[mask]; p2m = p2[mask]; p1m = p1[mask]; cm = comp[mask]
    out = {"n_genes": int(mask.sum()),
           "n_components": int(len(np.unique(cm))) if mask.sum() else 0}
    if mask.sum() < 2:
        for k in ["b2", "b1", "delta"]:
            out[k] = None
        return out
    out["b2"] = metrics_dict(y2, p2m)
    out["b1"] = metrics_dict(y2, p1m)
    out["delta_spearman_point"] = fast_spearman(p2m, y2) - fast_spearman(p1m, y2)
    out["delta_r2_point"] = r2(p2m, y2) - r2(p1m, y2)
    out["delta_mae_point"] = mae(p2m, y2) - mae(p1m, y2)
    return out


def main():
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()

    b1 = pd.read_csv(PRED_B1, sep="\t")
    b2 = pd.read_csv(PRED_B2, sep="\t")
    merged = b1.merge(b2, on="gene_id", suffixes=("_b1", "_b2"))
    assert len(merged) == len(b1) == len(b2), "gene_id join must be one-to-one"
    assert (merged["Score_E_b1"] == merged["Score_E_b2"]).all(), "Score_E must match"
    assert (merged["split_b1"] == merged["split_b2"]).all(), "split must match"
    assert (merged["component_id_b1"] == merged["component_id_b2"]).all(), "component must match"
    assert (merged["n_promoters_b1"] == merged["n_promoters_b2"]).all(), "n_promoters must match"

    result = {
        "purpose": "Parent-required errata: n=1 vs n>=2 arm differences from saved predictions",
        "b1_file": str(PRED_B1),
        "b2_file": str(PRED_B2),
        "join_check": {"n_rows": int(len(merged)), "one_to_one": True,
                       "Score_E_equal": True, "split_equal": True,
                       "component_equal": True, "n_promoters_equal": True},
        "note": "B2-B1 is the paired difference of separately trained models on IDENTICAL genes; "
                "a nonzero delta at n=1 directly refutes 'B1==B2 / delta only in multi-promoter'.",
        "bootstrap_seed_note": "component-bootstrap delta_spearman seed 20266011 (same convention as batch05 paired diff)",
        "by_split": {},
    }
    for split_name in ["val", "test"]:
        sub = merged[merged["split_b1"] == split_name].reset_index(drop=True)
        y = sub["Score_E_b1"].to_numpy(dtype=float)
        p2 = sub["pred_E_4slot_b2"].to_numpy(dtype=float)
        p1 = sub["pred_E_4slot_b1"].to_numpy(dtype=float)
        comp = sub["component_id_b1"].to_numpy(dtype=np.int64)
        n = sub["n_promoters_b1"].to_numpy(dtype=np.int64)
        d = {
            "full": strata_metrics(y, p2, p1, comp, np.ones(len(y), dtype=bool)),
            "n=1": strata_metrics(y, p2, p1, comp, n == 1),
            "n>=2": strata_metrics(y, p2, p1, comp, n >= 2),
            "n=2-3": strata_metrics(y, p2, p1, comp, (n >= 2) & (n < 4)),
            "n>=4": strata_metrics(y, p2, p1, comp, n >= 4),
        }
        for k in ["full", "n=1", "n>=2"]:
            mask = {"full": np.ones(len(y), dtype=bool),
                    "n=1": n == 1, "n>=2": n >= 2}[k]
            d[k]["component_bootstrap_delta_spearman"] = fast_component_bootstrap_diff(
                y[mask], p2[mask], p1[mask], comp[mask], fast_spearman, n=2000, seed=20266011)
        result["by_split"][split_name] = d

    atomic_write_json(REPORT / "errata_n1_vs_n2_arm_differences.json", result)
    for split_name in ["val", "test"]:
        d = result["by_split"][split_name]
        print(split_name,
              "| full dSp", round(d["full"]["delta_spearman_point"], 6),
              "| n=1 dSp", round(d["n=1"]["delta_spearman_point"], 6),
              "| n>=2 dSp", round(d["n>=2"]["delta_spearman_point"], 6),
              "| n=2-3 dSp", round(d["n=2-3"]["delta_spearman_point"], 6),
              "| n>=4 dSp", round(d["n>=4"]["delta_spearman_point"], 6))
        for k in ["n=1", "n>=2"]:
            dd = d[k]
            bs = dd["component_bootstrap_delta_spearman"]
            print("   %-5s n=%d B2 sp=%.6f B1 sp=%.6f dSp=%.6f CI[%.6f,%.6f]" % (
                k, dd["n_genes"], dd["b2"]["spearman"], dd["b1"]["spearman"],
                dd["delta_spearman_point"], bs["ci_low"], bs["ci_high"]))
    print("DONE")


if __name__ == "__main__":
    main()
