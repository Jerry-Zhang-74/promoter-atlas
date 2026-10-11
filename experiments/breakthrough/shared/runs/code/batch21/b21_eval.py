#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 evaluation: metrics, A/B decision, top-decile utility, group bootstrap,
and the predeclared practical-improvement rule. Validation is used only to observe
early stopping; the A/B choice uses the VALIDATION split; the TEST split is reported.

Also fits ONE fixed inexpensive sequence-composition ridge baseline (GC + 3-mer),
on TRAIN rows only, clearly separated from the two neural recipes.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from b21_common import (DATA_OUT, REPORT_OUT, MODEL_OUT, BATCH02_CODE, MPRA_SUPERVISED,
                        all_metrics, spearman, mae, top_decile_precision, sha256_file,
                        atomic_write_json, now_local, BASE, sha256_text)

sys.path.insert(0, str(BATCH02_CODE))
import trackB_grouped_cv as t  # noqa: E402

SEEDS = (20261023, 20261024)
ARMS = ("A", "B")
DECISION = {"mean_spearman_gain_min": 0.02, "mae_reduction_min_frac": 0.05,
            "top_decile_precision": "not worse", "both_seeds_positive": True}


def _sha(seqs):
    return [sha256_text(s) for s in seqs]


def load_data():
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    y = df["activity"].to_numpy(float)
    sha = _sha(seqs)
    return seqs, y, sha


def ridge_baseline(seqs, y, sha, split_of):
    from sklearn.linear_model import Ridge
    Xk = t.kmer_features(seqs)          # GC + 3-mer composition, no learned arch
    tr = np.where(split_of == 0)[0]; va = np.where(split_of == 1)[0]; te = np.where(split_of == 2)[0]
    mu = Xk[tr].mean(0); sd = Xk[tr].std(0); sd[sd < 1e-12] = 1.0
    clf = Ridge(alpha=100.0).fit((Xk[tr] - mu) / sd, y[tr])   # fixed, untuned alpha
    return {"method": "sequence_composition_ridge (GC+3mer, alpha=100 fixed, train-only)",
            "val": all_metrics(clf.predict((Xk[va] - mu) / sd), y[va], [sha[i] for i in va]),
            "test": all_metrics(clf.predict((Xk[te] - mu) / sd), y[te], [sha[i] for i in te])}


def run():
    seqs, y, sha = load_data()
    g = np.load(DATA_OUT / "groups.npz", allow_pickle=True)
    split_of = g["split_of"]; comp_id = g["comp_id"]
    te = np.where(split_of == 2)[0]; va = np.where(split_of == 1)[0]
    test_sha = [sha[i] for i in te]; test_groups = [int(comp_id[i]) for i in te]
    val_sha = [sha[i] for i in va]; val_groups = [int(comp_id[i]) for i in va]

    per = {"A": {}, "B": {}}
    test_mat, val_mat = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            d = np.load(DATA_OUT / ("preds_%s_seed%d.npz" % (arm, seed)))
            assert np.array_equal(d["test_idx"], te), "split mismatch"
            test_mat[(arm, seed)] = d["test_pred"]; val_mat[(arm, seed)] = d["val_pred"]
            per[arm][seed] = {
                "test": all_metrics(d["test_pred"], y[te], test_sha, test_groups),
                "val": all_metrics(d["val_pred"], y[va], val_sha, val_groups)}
    ridge = ridge_baseline(seqs, y, sha, split_of)

    # arm means over seeds (paired: same split, same seed set)
    def arm_mean(arm, split):
        keys = ["spearman", "mae", "r2"]
        v = {k: float(np.mean([per[arm][s][split][k] for s in SEEDS])) for k in keys}
        v["top_decile_precision"] = float(np.mean([per[arm][s][split]["top_decile"]["precision"] for s in SEEDS]))
        v["top_decile_enrichment"] = float(np.mean([per[arm][s][split]["top_decile"]["enrichment"] for s in SEEDS]))
        return v
    tm = {a: arm_mean(a, "test") for a in ARMS}
    vm = {a: arm_mean(a, "val") for a in ARMS}

    sp_gain = tm["B"]["spearman"] - tm["A"]["spearman"]
    mae_red = (tm["A"]["mae"] - tm["B"]["mae"]) / tm["A"]["mae"]
    per_seed_diff = {str(s): per["B"][s]["test"]["spearman"] - per["A"][s]["test"]["spearman"] for s in SEEDS}
    top_ok = tm["B"]["top_decile_precision"] >= tm["A"]["top_decile_precision"] - 1e-12
    both_pos = all(v > 0 for v in per_seed_diff.values())
    improved = bool(sp_gain >= DECISION["mean_spearman_gain_min"] and mae_red >= DECISION["mae_reduction_min_frac"]
                    and top_ok and both_pos)

    # validation-only arm choice (as predeclared) and its test outcome
    val_choice = "B" if vm["B"]["spearman"] >= vm["A"]["spearman"] else "A"

    results = {
        "generated_at_local": now_local(),
        "primary_metric": "test family/near-homology-grouped Spearman and MAE",
        "seeds": list(SEEDS), "arms": list(ARMS),
        "decision_rule": DECISION,
        "per_arm_per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
        "arm_mean_test": tm, "arm_mean_val": vm,
        "comparison": {"mean_spearman_gain_B_minus_A": sp_gain,
                       "mae_reduction_frac_B_vs_A": mae_red,
                       "per_seed_spearman_diff": per_seed_diff,
                       "top_decile_not_worse": bool(top_ok),
                       "both_seeds_positive": bool(both_pos),
                       "predeclared_improvement_met": improved},
        "validation_only_arm_choice": val_choice,
        "ridge_baseline": ridge,
        "n_test": int(len(te)), "n_val": int(len(va)),
        "n_test_groups": int(len(set(test_groups))), "n_val_groups": int(len(set(val_groups))),
        "note": ("Task-specific decision threshold; NOT a retroactive change of the batch02 "
                 "+0.05/10 percent historical threshold and NOT compared against 0.79577. "
                 "K562 in-domain; not HEK293T."),
        "source_sha256": {"groups.npz": sha256_file(DATA_OUT / "groups.npz")},
    }
    # per-row test predictions table
    cols = {"Oligo_index": [str(i) for i in te], "sequence_sha256": test_sha,
            "component_id": test_groups, "activity": y[te]}
    df = pd.DataFrame(cols)
    for arm in ARMS:
        for s in SEEDS:
            df["pred_%s_seed%d" % (arm, s)] = test_mat[(arm, s)]
    df["ridge_pred"] = np.nan
    # ridge test preds
    from sklearn.linear_model import Ridge
    Xk = t.kmer_features(seqs)
    tr = np.where(split_of == 0)[0]
    mu = Xk[tr].mean(0); sd = Xk[tr].std(0); sd[sd < 1e-12] = 1.0
    clf = Ridge(alpha=100.0).fit((Xk[tr] - mu) / sd, y[tr])
    df["ridge_pred"] = clf.predict((Xk[te] - mu) / sd)
    df.to_csv(REPORT_OUT / "test_predictions.tsv", sep="\t", index=False)
    atomic_write_json(REPORT_OUT / "results.json", results)
    return results, df
