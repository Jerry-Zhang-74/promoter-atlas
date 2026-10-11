# -*- coding: utf-8 -*-
"""batch54 INDEPENDENT first-model acceptance runner (CPU only, read-only over artifacts).

Subcommands:
  fixtures   run the artificial no-fit metric fixture tests
  r0         independent metrics for the R0 composition baseline on a split
  metrics    independent metrics for one prediction artifact on a split
  checks     independent artifact/provenance/gate/coverage checks for a neural arm
  accept     assemble the acceptance JSON from checks + available frozen predictions

All metrics come from batch54 indep_metrics (own code). No producer evaluator is imported
for metric values. No fit / gradient / solve / GPU / network here.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import indep_cohort as C
import indep_metrics as M
import model_checks as K

REPORTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "reports",
                                       "breakthrough_20261002", "batch54"))


def now():
    return dt.datetime.now().astimezone().isoformat()


def _preds_for_split(co, split, preds_by_id):
    idx = C.split_idx(co, split)
    ids = co["coord_ids"][idx]
    y = co["targets"][idx]
    p = np.array([preds_by_id[c] for c in ids.tolist()], dtype=np.float64)
    groups = co["groups"][idx]
    return idx, ids, y, p, groups


def activity_metrics(co, split, preds_by_id):
    idx, ids, y, p, groups = _preds_for_split(co, split, preds_by_id)
    return M.activity_bundle(y, p, groups, ids)


def pair_metrics(co, split, preds_by_id):
    pr = co["pairs"][split]
    if pr.shape[0] == 0:
        return {"n_pairs": 0}
    ids = co["coord_ids"]
    dt = co["targets"][pr[:, 0]] - co["targets"][pr[:, 1]]
    pv = np.array([preds_by_id[c] for c in ids[pr[:, 0]].tolist()], dtype=np.float64)
    lv = np.array([preds_by_id[c] for c in ids[pr[:, 1]].tolist()], dtype=np.float64)
    pg = co["groups"][pr[:, 0]]
    return M.pair_bundle(dt, pv - lv, pg)


def pair_group_values_for(co, split, preds_by_id):
    pr = co["pairs"][split]
    ids = co["coord_ids"]
    dt = co["targets"][pr[:, 0]] - co["targets"][pr[:, 1]]
    pv = np.array([preds_by_id[c] for c in ids[pr[:, 0]].tolist()], dtype=np.float64)
    lv = np.array([preds_by_id[c] for c in ids[pr[:, 1]].tolist()], dtype=np.float64)
    pg = co["groups"][pr[:, 0]]
    return M.pair_group_values(dt, pv - lv, pg)


def load_preds_any(path):
    if path.endswith(".json"):
        return C.load_preds_json(path)
    return C.load_preds_tsv(path)


def cmd_fixtures(_):
    subprocess.check_call([sys.executable, os.path.join(HERE, "test_fixtures.py")])
    return json.load(open(os.path.join(HERE, "fixture_evidence.json"), encoding="utf-8"))


def cmd_r0(a):
    co = C.load(a.cohort_dir)
    preds = load_preds_any(a.preds)
    res = {"arm": "R0", "split": a.split, "preds": a.preds,
           "activity": activity_metrics(co, a.split, preds),
           "pair": pair_metrics(co, a.split, preds),
           "evaluator": "batch54_indep_metrics"}
    if a.split == "TEST":
        pr = co["pairs"][a.split]
        ids = co["coord_ids"]
        dt = co["targets"][pr[:, 0]] - co["targets"][pr[:, 1]]
        rest = pair_group_values_for(co, a.split, preds)
        res["fixed_baselines"] = M.fixed_pair_baselines(dt, co["groups"][pr[:, 0]])
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def cmd_metrics(a):
    co = C.load(a.cohort_dir)
    preds = load_preds_any(a.preds)
    res = {"arm": os.path.basename(a.preds), "split": a.split, "preds": a.preds,
           "activity": activity_metrics(co, a.split, preds),
           "pair": pair_metrics(co, a.split, preds),
           "evaluator": "batch54_indep_metrics"}
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def cmd_checks(a):
    events = K.read_ledger(a.ledger)
    rep = {"evaluator": "batch54", "clock": now(),
           "ledger": K.check_ledger(events),
           "arm": a.arm,
           "checks": K.check_arm(a.out_root, a.arm),
           "init_parity": K.check_init_parity_both(a.out_root)}
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rep


def cmd_compare(a):
    """Independent paired group bootstrap (multiplicity retained) for predA vs predB,
    or predA vs a constant-delta baseline when --predB is omitted."""
    co = C.load(a.cohort_dir)
    pr = co["pairs"][a.split]
    ids = co["coord_ids"]
    dt = co["targets"][pr[:, 0]] - co["targets"][pr[:, 1]]
    pg = co["groups"][pr[:, 0]]
    A = load_preds_any(a.predA)
    pvA = np.array([A[c] for c in ids[pr[:, 0]].tolist()])
    lvA = np.array([A[c] for c in ids[pr[:, 1]].tolist()])
    maeA, signA, auxA = M.pair_group_values(dt, pvA - lvA, pg)
    if a.predB:
        B = load_preds_any(a.predB)
        pvB = np.array([B[c] for c in ids[pr[:, 0]].tolist()])
        lvB = np.array([B[c] for c in ids[pr[:, 1]].tolist()])
        maeB, signB, auxB = M.pair_group_values(dt, pvB - lvB, pg)
        labelB = os.path.basename(a.predB)
    else:
        maeB, signB, auxB = M.pair_group_values(dt, np.full(dt.size, a.baseline_value), pg)
        labelB = "const_delta_%g" % a.baseline_value
    groups = list(dict.fromkeys(pg.tolist()))
    boot = M.group_bootstrap_paired(groups, maeA, maeB, signA, signB,
                                    n_boot=a.n_boot, seed=a.seed)
    res = {"evaluator": "batch54_indep", "split": a.split, "A": a.predA, "B": labelB,
           "n_pairs": int(dt.size), "n_groups": len(groups),
           "A_pair": {"group_mean_mae": M.group_mean_mae_from_values(maeA),
                      "macro_sign_accuracy": M.macro_sign_accuracy(signA), **auxA},
           "B_pair": {"group_mean_mae": M.group_mean_mae_from_values(maeB),
                      "macro_sign_accuracy": M.macro_sign_accuracy(signB), **auxB},
           "bootstrap": boot}
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def cmd_accept(a):
    ev = {}
    co = C.load(a.cohort_dir)
    events = K.read_ledger(a.ledger)
    ev["ledger"] = K.check_ledger(events)
    ev["cohort_counts"] = {s: int((co["splits"] == s).sum()) for s in C.SPLITS}
    ev["module_checks"] = K.check_arm(a.out_root, a.arm)
    ev["init_parity"] = K.check_init_parity_both(a.out_root)
    if a.a_preds:
        ev["modelA_metrics_split"] = a.a_split
        ev["modelA_activity"] = activity_metrics(co, a.a_split, load_preds_any(a.a_preds))
        ev["modelA_pair"] = pair_metrics(co, a.a_split, load_preds_any(a.a_preds))
        ev["modelA_preds"] = a.a_preds
    if a.r0_preds:
        ev["r0_dev_activity"] = activity_metrics(co, "DEV", load_preds_any(a.r0_preds))
        ev["r0_dev_pair"] = pair_metrics(co, "DEV", load_preds_any(a.r0_preds))
    ev["written_at"] = now()
    if a.out:
        json.dump(ev, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return ev


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fixtures")
    for name in ("r0", "metrics"):
        p = sub.add_parser(name)
        p.add_argument("--cohort-dir", required=True)
        p.add_argument("--preds", required=True)
        p.add_argument("--split", default="DEV")
        p.add_argument("--out", default=None)
    p = sub.add_parser("checks")
    p.add_argument("--out-root", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--arm", default="Nabs")
    p.add_argument("--out", default=None)
    p = sub.add_parser("accept")
    p.add_argument("--cohort-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--arm", default="Nabs")
    p.add_argument("--a-preds", default=None)
    p.add_argument("--a-split", default="DEV")
    p.add_argument("--r0-preds", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("compare")
    p.add_argument("--cohort-dir", required=True)
    p.add_argument("--predA", required=True)
    p.add_argument("--predB", default=None)
    p.add_argument("--baseline-value", type=float, default=0.0)
    p.add_argument("--split", default="DEV")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--out", default=None)
    a = ap.parse_args()
    fn = {"fixtures": cmd_fixtures, "r0": cmd_r0, "metrics": cmd_metrics,
          "checks": cmd_checks, "accept": cmd_accept, "compare": cmd_compare}[a.cmd]
    res = fn(a)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
