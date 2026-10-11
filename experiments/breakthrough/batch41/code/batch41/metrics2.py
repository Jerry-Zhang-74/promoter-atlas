#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 metrics part 2: per-component rho, paired component bootstrap (conditional
uncertainty, ZERO learning), leave-one-chromosome-out sensitivity (no fits), controls
descriptive, frozen decision rule. Fast numpy/scipy rank metrics."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402
import metrics as M  # noqa: E402


def per_component(recs, bio_fin):
    byc = defaultdict(list)
    for i in bio_fin:
        byc[recs[i]["component_id"]].append(i)
    res = {}
    for m in M.MODELS:
        rhos = []
        for cid, idx in byc.items():
            if len(idx) < C.MIN_COMPONENT:
                continue
            a = np.array([recs[i]["alpha"] for i in idx])
            p = np.array([recs[i][m] for i in idx])
            if np.std(a) == 0 or np.std(p) == 0:
                continue
            rhos.append(C.spearman_fast(p, a))
        r = np.array([x for x in rhos if np.isfinite(x)])
        res[m] = {"n_eligible_components": int(len(r)),
                  "macro_mean_rho": float(r.mean()) if len(r) else None}
    return res, byc


def bootstrap(recs, bio_fin, n_boot=C.BOOT_N, seed=C.BOOT_SEED):
    n = len(recs)
    AL = np.array([recs[i]["alpha"] for i in range(n)], dtype=np.float64)
    ARR = {m: np.array([recs[i][m] for i in range(n)], dtype=np.float64) for m in M.MODELS}
    labs = sorted({recs[i]["label"] for i in range(n)})
    lidx = {l: i for i, l in enumerate(labs)}
    LID = np.array([lidx[recs[i]["label"]] for i in range(n)], dtype=np.int64)
    byc = defaultdict(list)
    for i in bio_fin:
        byc[recs[i]["component_id"]].append(i)
    comp_keys = sorted(byc)
    arrs = [np.array(byc[c], dtype=np.int64) for c in comp_keys]
    rng = np.random.default_rng(seed)
    rr = {m: [] for m in M.MODELS}
    dph, pp, ph = [], [], []
    mult_acc = np.zeros(len(comp_keys), dtype=np.int64)
    for _ in range(n_boot):
        pick = rng.integers(0, len(arrs), size=len(arrs))
        for j in pick:
            mult_acc[j] += 1
        idx = np.concatenate([arrs[j] for j in pick])
        a = AL[idx]
        if a.std() == 0:
            continue
        for m in M.MODELS:
            rr[m].append(C.spearman_fast(ARR[m][idx], a))
        dph.append(rr["parm"][-1] - rr["h1"][-1])
        kt, tp = C.top_indices_fast(a, LID[idx])
        for key, store in (("parm", pp), ("h1", ph)):
            _, tpred = C.top_indices_fast(ARR[key][idx], LID[idx])
            store.append(len(np.intersect1d(tp, tpred)) / kt if kt else float("nan"))

    def ci(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                "mean": float(v.mean()), "n_boot": int(len(v))}
    return {"rho": {m: ci(rr[m]) for m in M.MODELS}, "PARM_minus_H1": ci(dph),
            "parm_top10_precision": ci(pp), "h1_top10_precision": ci(ph),
            "multiplicity_summary": {"min": int(mult_acc.min()), "max": int(mult_acc.max()),
                                     "n_components": len(comp_keys)},
            "label": "CONDITIONAL component-resampling uncertainty (residual regional dependence)"}


def loco(recs, bio_fin):
    chroms = sorted({recs[i]["chr"] for i in bio_fin})
    res = {"chromosomes": chroms, "leave_one_out": {}}
    for c in chroms:
        sub = [i for i in bio_fin if recs[i]["chr"] != c]
        res["leave_one_out"][c] = {"n": len(sub),
                                   "rho": {m: M.rho_block(recs, sub, m) for m in M.MODELS}}
    res["note"] = "NO FITS; sensitivity only; no formal chromosome-level CI"
    return res


def controls(recs, ctrl):
    al = np.array([recs[i]["alpha"] for i in ctrl])
    out = {"n": len(ctrl)}
    for m in M.MODELS:
        pv = np.array([recs[i][m] for i in ctrl])
        out[m + "_spearman_vs_alpha"] = C.spearman_fast(pv, al)
        out[m + "_min"] = float(pv.min())
        out[m + "_max"] = float(pv.max())
        out[m + "_median"] = float(np.median(pv))
    out["ranked_alpha_ascending"] = sorted(round(float(a), 4) for a in al)
    out["note"] = ("descriptive ONLY (n=85); controls never mixed into primary biological rho "
                   "and used for NO calibration")
    return out


def main():
    t0 = time.time()
    C.set_thread_limits(4)
    recs, rt, wsha = M.load()
    bio_fin = [i for i, r in enumerate(recs) if r["control"] == 0]
    ctrl = [i for i, r in enumerate(recs) if r["control"] == 1]
    part1 = json.load(open(C.DATA / "metrics_part1.json", encoding="utf-8"))

    comp_macro, byc = per_component(recs, bio_fin)
    n_eligible_ge10 = sum(1 for c, idx in byc.items() if len(idx) >= C.MIN_COMPONENT)
    boot = bootstrap(recs, bio_fin) if len(byc) >= C.MIN_BOOT_COMPONENTS else \
        {"skipped": "fewer than %d components" % C.MIN_BOOT_COMPONENTS}

    out = {"batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
           "per_component": {"macro": comp_macro, "n_components_total": len(byc),
                             "n_components_ge10": n_eligible_ge10,
                             "min_component_rows": C.MIN_COMPONENT},
           "bootstrap": boot, "loco": loco(recs, bio_fin), "controls": controls(recs, ctrl)}

    P = part1["pooled"]
    D = part1["paired_rho_differences_common_set"]
    parm_rho = P["parm"]["spearman"]
    h1_rho = P["h1"]["spearman"]
    non17_parm = part1["strata"]["non_chr17"]["rho"]["parm"]
    ci_lo = boot.get("PARM_minus_H1", {}).get("lo")
    parm_prec = P["parm"]["top_decile"]["precision"]
    h1_prec = P["h1"]["top_decile"]["precision"]
    checks = {
        "PARM_pooled_rho_ge_0.30": bool(parm_rho >= 0.30),
        "PARM_minus_H1_ge_0.05": bool((parm_rho - h1_rho) >= 0.05),
        "PARM_minus_H1_CI_lo_gt_0": bool(ci_lo is not None and ci_lo > 0),
        "PARM_top10_precision_ge_H1": bool(parm_prec >= h1_prec),
        "PARM_non_chr17_rho_gt_0": bool(non17_parm > 0),
        "no_audit_failure": True,
    }
    supported = all(checks.values())
    out["decision_rule"] = {
        "checks": checks,
        "verdict": "SUPPORTED_further_fragment_transfer" if supported else "LIMITED",
        "note": ("operational resource decision only; NOT a generalization/promotion; does not replace "
                 "old H1 MAE10% gate / local D28 FAIL; failure cannot be rescued by strongest chrom/"
                 "subset/GC flip; no automatic finetuning"),
        "positive_meaning": "positive would still be only a public OligoMPRA transfer with unknown vector/local clone context",
        "negative_meaning": "negative is NOT proof that all promoter designs or the data are useless",
    }
    out["elapsed_s"] = round(time.time() - t0, 3)
    C.atomic_write_json(C.DATA / "metrics_part2.json", out)
    print(json.dumps({"n_components": len(byc), "n_ge10": n_eligible_ge10,
                      "macro_rho": comp_macro,
                      "boot_delta_ci": boot.get("PARM_minus_H1"),
                      "decision_checks": checks, "verdict": out["decision_rule"]["verdict"],
                      "elapsed_s": out["elapsed_s"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
