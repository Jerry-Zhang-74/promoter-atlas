# -*- coding: utf-8 -*-
"""batch54 ARTIFICIAL FIXTURE TESTS for the independent metric definitions.

ZERO fit / ZERO gradient / ZERO solve / ZERO GPU / ZERO network / ZERO real-data score.
Only synthetic arrays confirm the metric + bootstrap DEFINITIONS:
  duplicate bootstrap groups (multiplicity retained), constant prediction,
  NA / non-finite handling, true-zero pairs and predicted TIES, top-k stable-ID tie-break,
  and agreement with a brute-force reference.
Writes runs/breakthrough_20261002/code/batch54_review/fixture_evidence.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import indep_metrics as M

RESULTS = {}


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS[name] = "PASS"
        except Exception as e:
            RESULTS[name] = "FAIL: %r" % (e,)
            raise
        return fn
    return deco


def _close(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol, "%r != %r" % (a, b)


@check("group_equal_mae_matches_bruteforce")
def _():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50); p = rng.normal(size=50)
    g = np.array(["g%d" % (i % 7) for i in range(50)])
    got = M.group_equal_mae(y, p, g)
    ref = [np.mean(np.abs(y[g == gg] - p[g == gg])) for gg in dict.fromkeys(g.tolist())]
    _close(got, np.mean(ref))


@check("macro_sign_accuracy_truezero_and_ties")
def _():
    dt = np.array([1.0, -1.0, 0.0])
    dp = np.array([1.0, 1.0, 0.0])
    pg = np.array(["g1", "g1", "g1"])
    _, sign, aux = M.pair_group_values(dt, dp, pg)
    _close(sign["g1"], 0.5)
    assert aux["n_true_zero_pairs"] == 1
    _close(aux["true_zero_rate"], 1.0 / 3.0)
    dt2 = np.array([1.0, -1.0]); dp2 = np.array([0.0, -1.0]); pg2 = np.array(["a", "a"])
    _, sign2, aux2 = M.pair_group_values(dt2, dp2, pg2)
    _close(sign2["a"], 0.5)
    assert aux2["n_pred_ties"] == 1
    dt3 = np.array([0.0, 0.0]); dp3 = np.array([0.1, -0.1]); pg3 = np.array(["z", "z"])
    _, sign3, _ = M.pair_group_values(dt3, dp3, pg3)
    assert "z" not in sign3


@check("constant_prediction_definitions")
def _():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    c = 2.0
    p = np.full_like(y, c)
    g = np.array(["a", "a", "b", "b", "b"])
    mae, cnt = M.group_equal_mae(y, p, g, return_counts=True)
    _close(mae, np.mean([np.mean(np.abs(y[g == "a"] - c)), np.mean(np.abs(y[g == "b"] - c))]))
    assert cnt["n_groups_used"] == 2
    assert np.isnan(M.spearman_rho(y, p))
    ids = np.array(["e", "d", "c", "b", "a"])
    _close(M.topk_precision_fixed_ids(y, p, ids), 1.0)


@check("topk_fixed_id_tiebreak_deterministic")
def _():
    y = np.array([5.0, 5.0, 5.0, 1.0])
    p = np.array([9.0, 9.0, 9.0, 0.0])
    ids = np.array(["chrZ", "chrA", "chrM", "chrB"])
    prec = M.topk_precision_fixed_ids(y, p, ids)
    _close(prec, 1.0)
    perm = [3, 1, 0, 2]
    _close(M.topk_precision_fixed_ids(y[perm], p[perm], ids[perm]), prec)


@check("NA_handling_masks_nonfinite")
def _():
    y = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 2.0, np.nan, 5.0])
    g = np.array(["a", "a", "a", "b", "b"])
    mae, cnt = M.group_equal_mae(y, p, g, return_counts=True)
    _close(mae, 0.5)
    assert cnt["n_groups_used"] == 2
    _close(M.spearman_rho(y, p),
           M.spearman_rho(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 5.0])))
    y2 = np.array([np.nan, 1.0]); p2 = np.array([np.nan, 1.0]); g2 = np.array(["x", "y"])
    mae2, cnt2 = M.group_equal_mae(y2, p2, g2, return_counts=True)
    _close(mae2, 0.0); assert cnt2["n_groups_dropped_nonfinite"] == 1


@check("bootstrap_retains_multiplicity")
def _():
    G = 30
    groups = ["g%d" % i for i in range(G)]
    vals = np.array([0.0] * 15 + [1.0] * 15)
    maeA = {g: float(vals[i]) for i, g in enumerate(groups)}
    maeB = {g: 0.0 for g in groups}
    empty = {}
    seed, nb = 20261004, 2000
    boot = M.group_bootstrap_paired(groups, maeA, maeB, empty, empty, n_boot=nb, seed=seed)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(nb):
        pick = rng.integers(0, G, size=G)
        draws.append(float(vals[pick].mean()))
    _close(boot["mae_A_minus_B"]["mean"], np.mean(draws))
    uniq = []
    for k in range(200):
        pk = np.random.default_rng(seed + k).integers(0, G, size=G)
        uniq.append(float(vals[np.unique(pk)].mean()))
    assert abs(np.mean(draws) - np.mean(uniq)) > 1e-3


@check("bootstrap_paired_improvement_sign")
def _():
    G = 40
    groups = ["g%d" % i for i in range(G)]
    rng = np.random.default_rng(7)
    maeB = {g: float(rng.uniform(0.2, 0.4)) for g in groups}
    maeA = {g: v + 0.5 for g, v in maeB.items()}
    boot = M.group_bootstrap_paired(groups, maeA, maeB, {}, {}, n_boot=1000, seed=1)
    _close(boot["improvement_mae_B_over_A"]["mean"], 0.5, tol=1e-6)
    assert boot["improvement_mae_B_over_A"]["lo"] > 0.0
    small = M.group_bootstrap_paired(["a", "b", "c"], {"a": 1, "b": 1, "c": 1},
                                     {"a": 0, "b": 0, "c": 0}, {}, {}, n_boot=10)
    assert small["ok"] is False and small["n_groups"] == 3


@check("fixed_baselines_zero_delta")
def _():
    dt = np.array([1.0, -2.0, 0.0, 3.0])
    pg = np.array(["a", "a", "b", "b"])
    base = M.fixed_pair_baselines(dt, pg)
    _close(base["zero_delta"]["group_mean_mae"],
           np.mean([np.mean(np.abs([1.0, -2.0])), np.mean(np.abs([0.0, 3.0]))]))
    _close(base["zero_delta"]["macro_sign_accuracy"], 0.0)
    _close(base["always_positive"]["macro_sign_accuracy"], 0.75)


ALL = ["group_equal_mae_matches_bruteforce", "macro_sign_accuracy_truezero_and_ties",
       "constant_prediction_definitions", "topk_fixed_id_tiebreak_deterministic",
       "NA_handling_masks_nonfinite", "bootstrap_retains_multiplicity",
       "bootstrap_paired_improvement_sign", "fixed_baselines_zero_delta"]

if __name__ == "__main__":
    n_pass = sum(1 for v in RESULTS.values() if v == "PASS")
    ev = {"batch": "batch54", "kind": "artificial_no_fit_metric_fixtures",
          "fits": 0, "solves": 0, "gradients": 0, "gpu": 0, "network": 0,
          "tests": RESULTS, "n_pass": n_pass, "n_total": len(ALL),
          "all_passed": n_pass == len(ALL)}
    with open(os.path.join(HERE, "fixture_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)
    print(json.dumps(ev, ensure_ascii=False, indent=2))
    sys.exit(0 if ev["all_passed"] else 1)
