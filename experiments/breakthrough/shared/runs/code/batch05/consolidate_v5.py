#!/usr/bin/env python3
"""Batch05 consolidation: B1/B2 single-seed metrics, component-bootstrap B2-B1,
matched design checks, and comparison to the best inner-selected annotation
baseline. Evidence threshold: B2 improves component-test Spearman over B1 by
>=0.03 with component-bootstrap lower bound >0, plus independent comparison to
the best inner-selected annotation baseline (baseline losses NOT hidden).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b5_common import (  # noqa: E402
    atomic_write_json, spearman, r2, mae, component_bootstrap_diff,
)

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
PRED = REPORT / "predictions"

ARMS = ["repeat_bag", "diverse_bag"]
CANDS = {"repeat_bag": "v6.6.0-scratchgene.rc6", "diverse_bag": "v6.6.0-scratchgene.rc7"}
SEED = 20261011


def load_metrics(arm):
    p = PRED / ("metrics_%s_s%d.json" % (CANDS[arm], SEED))
    return json.loads(p.read_text(encoding="utf-8"))


def load_pred(arm):
    import pandas as pd
    p = PRED / ("pred_%s_s%d.tsv" % (CANDS[arm], SEED))
    return pd.read_csv(p, sep="\t")


def main():
    summary = {"arms": {}, "paired_b2_minus_b1": {}, "matched_design_checks": {},
               "annotation_baseline_comparison": {}, "evidence": {}}

    for arm in ARMS:
        summary["arms"][arm] = {
            "candidate": CANDS[arm], "arm": arm,
            "val": load_metrics(arm)["val"],
            "test": load_metrics(arm)["test"],
        }

    init_shas = {}
    n_params = {}
    for arm in ARMS:
        res = json.loads((MODEL_DIR / ("%s_s%d" % (arm, SEED)) / "result.json").read_text(encoding="utf-8"))
        init_shas[arm] = res["initial_state_sha256"]
        n_params[arm] = res["n_params"]
    summary["matched_design_checks"] = {
        "initial_state_sha256": init_shas,
        "initial_weights_match_B1_B2": init_shas["repeat_bag"] == init_shas["diverse_bag"],
        "n_params": n_params,
        "n_params_match": n_params["repeat_bag"] == n_params["diverse_bag"],
        "note": "identical seed -> identical initial trunk+head weights; identical GeneBagModel; identical 4 slots",
    }

    pB2 = load_pred("diverse_bag")
    pB1 = load_pred("repeat_bag")
    for split_name in ["val", "test"]:
        a = pB2[pB2.split == split_name].sort_values("gene_id").reset_index(drop=True)
        b = pB1[pB1.split == split_name].sort_values("gene_id").reset_index(drop=True)
        assert list(a.gene_id) == list(b.gene_id)
        y = a.Score_E.to_numpy()
        p2 = a.pred_E_4slot.to_numpy()
        p1 = b.pred_E_4slot.to_numpy()
        comp = a.component_id.to_numpy()
        summary["paired_b2_minus_b1"][split_name] = {
            "delta_spearman": component_bootstrap_diff(y, p2, p1, comp, comp, spearman, seed=SEED + 5000),
            "delta_r2": component_bootstrap_diff(y, p2, p1, comp, comp, r2, seed=SEED + 6000),
            "delta_mae": component_bootstrap_diff(y, p2, p1, comp, comp, mae, seed=SEED + 7000),
            "n_genes": int(len(y)),
            "n_components": int(len(np.unique(comp))),
        }

    ann = json.loads((REPORT / "ann_diagnostics_v5.json").read_text(encoding="utf-8"))
    best = ann["best_inner_selected_baseline"]
    best_variant = best["variant"]
    best_test = ann["variants"][best_variant]["test"]
    b2_test_sp = summary["arms"]["diverse_bag"]["test"]["full"]["spearman"]
    b1_test_sp = summary["arms"]["repeat_bag"]["test"]["full"]["spearman"]

    summary["annotation_baseline_comparison"] = {
        "best_inner_selected_baseline": best,
        "best_baseline_test_spearman": best_test["full"]["spearman"],
        "best_baseline_test_r2": best_test["full"]["r2"],
        "b2_test_spearman": b2_test_sp,
        "b2_minus_best_baseline_spearman": b2_test_sp - best_test["full"]["spearman"],
        "b1_test_spearman": b1_test_sp,
        "note": "baseline losses NOT hidden; NN vs ridge comparison is descriptive",
    }

    d = summary["paired_b2_minus_b1"]["test"]["delta_spearman"]
    meets_improvement = (d["base"] >= 0.03) and (d["ci_low"] > 0.0)
    summary["evidence"] = {
        "b2_minus_b1_test_spearman": d["base"],
        "component_bootstrap_ci_low": d["ci_low"],
        "component_bootstrap_ci_high": d["ci_high"],
        "meets_improvement_threshold": bool(meets_improvement),
        "threshold": "B2 component-test Spearman over B1 >= 0.03 AND component-bootstrap lower bound > 0",
        "note": "single neural seed -> seed robustness unresolved; this is a batch decision threshold, never permission to halt research or claim reliable construct activity",
    }

    atomic_write_json(REPORT / "consolidated_summary_v5.json", summary)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
