#!/usr/bin/env python3
"""Batch04 consolidation: aggregate 6-run metrics, B2-B1 paired contrast, registry."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import (  # noqa: E402
    atomic_write_json, spearman, r2, mae, mse, paired_bootstrap_diff, sha256_file,
)

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"
PRED = REPORT / "predictions"

ARMS = ["repeat_bag", "diverse_bag"]
CANDS = {"repeat_bag": "v6.6.0-scratchgene.rc4", "diverse_bag": "v6.6.0-scratchgene.rc5"}
SEEDS = [20261011, 20261012, 20261013]


def load_metrics(arm, seed):
    p = PRED / ("metrics_%s_s%d.json" % (CANDS[arm], seed))
    return json.loads(p.read_text(encoding="utf-8"))


def load_pred(arm, seed):
    import pandas as pd
    p = PRED / ("pred_%s_s%d.tsv" % (CANDS[arm], seed))
    return pd.read_csv(p, sep="\t")


def main():
    per_seed = {}
    summary = {"arms": {}, "paired_b2_minus_b1": {}, "matched_design_checks": {}}
    for seed in SEEDS:
        per_seed[seed] = {}
        for arm in ARMS:
            per_seed[seed][arm] = load_metrics(arm, seed)

    # matched design checks
    init_shas = {}
    n_params = {}
    for arm in ARMS:
        res = json.loads((MODEL_DIR / ("%s_s%d" % (arm, SEEDS[0])) / "result.json").read_text(encoding="utf-8"))
        init_shas[arm] = res["initial_state_sha256"]
        n_params[arm] = res["n_params"]
    summary["matched_design_checks"] = {
        "initial_state_sha256": init_shas,
        "initial_weights_match_B1_B2": init_shas["repeat_bag"] == init_shas["diverse_bag"],
        "n_params": n_params,
        "n_params_match": n_params["repeat_bag"] == n_params["diverse_bag"],
        "note": "identical seed -> identical initial trunk+head weights; identical GeneBagModel; identical gene order; identical 4 slots",
    }

    # per-arm per-split 3-seed mean (4-slot primary + all-promoter descriptive)
    for arm in ARMS:
        cand = CANDS[arm]
        arm_sum = {"candidate": cand, "arm": arm, "seeds": {}, "mean": {}}
        for seed in SEEDS:
            m = per_seed[seed][arm]
            arm_sum["seeds"][str(seed)] = {
                "val_4slot_full_sp": m["val"]["4slot_full"]["spearman"],
                "val_4slot_unexp_sp": m["val"]["4slot_unexposed"]["spearman"],
                "test_4slot_full_sp": m["test"]["4slot_full"]["spearman"],
                "test_4slot_unexp_sp": m["test"]["4slot_unexposed"]["spearman"],
                "test_4slot_full_r2": m["test"]["4slot_full"]["r2"],
                "test_allprom_full_sp": m["test"]["allpromoter_full"]["spearman"],
                "test_allprom_unexp_sp": m["test"]["allpromoter_unexposed"]["spearman"],
            }
        for key in ["test_4slot_full_sp", "test_4slot_unexp_sp", "test_4slot_full_r2",
                    "val_4slot_full_sp", "val_4slot_unexp_sp", "test_allprom_full_sp",
                    "test_allprom_unexp_sp"]:
            vals = [arm_sum["seeds"][str(s)][key] for s in SEEDS]
            arm_sum["mean"][key] = float(np.mean(vals))
        summary["arms"][arm] = arm_sum

    # paired B2 - B1 difference (per seed) on test (primary 4-slot)
    for seed in SEEDS:
        pB2 = load_pred("diverse_bag", seed)
        pB1 = load_pred("repeat_bag", seed)
        testB2 = pB2[pB2.split == "test"].sort_values("gene_id")
        testB1 = pB1[pB1.split == "test"].sort_values("gene_id")
        assert list(testB2.gene_id) == list(testB1.gene_id)
        y = testB2.Score_E.to_numpy()
        p2 = testB2.pred_E_4slot.to_numpy()
        p1 = testB1.pred_E_4slot.to_numpy()
        summary["paired_b2_minus_b1"][str(seed)] = {
            "test_delta_spearman": paired_bootstrap_diff(y, p2, p1, spearman, seed=seed + 5000),
            "test_delta_r2": paired_bootstrap_diff(y, p2, p1, r2, seed=seed + 6000),
            "test_delta_mae": paired_bootstrap_diff(y, p2, p1, mae, seed=seed + 7000),
            "n_test_genes": int(len(y)),
        }

    deltas = [summary["paired_b2_minus_b1"][str(s)]["test_delta_spearman"]["base"] for s in SEEDS]
    summary["paired_b2_minus_b1"]["_3seed_delta_spearman"] = {
        "values": deltas,
        "mean": float(np.mean(deltas)),
        "consistent_sign": bool((np.array(deltas) > 0).all() or (np.array(deltas) < 0).all()),
        "note": "seed-consistency of the sign, NOT a significance test; genes are paired and non-independent",
    }

    atomic_write_json(REPORT / "consolidated_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
