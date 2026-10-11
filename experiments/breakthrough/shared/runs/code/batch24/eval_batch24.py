#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 v6.18.0-hek-kernel.rc1 evaluation. ZERO fits: reads saved OOF
predictions (cached R1 + new H1/H2) and recomputes all metrics independently."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C

C.set_thread_limits(4)


def main():
    rows = list(csv.DictReader(open(C.DATA_OUT / "oof_predictions_batch24.tsv",
                                     encoding="utf-8"), delimiter="\t"))
    pids = [r["prom_id"] for r in rows]
    y = np.array([float(r["hek_norm"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    gid = np.array([int(r["group_id"]) for r in rows])
    seen = np.array([int(r["source_seen"]) for r in rows])
    preds = {
        "R1_cached": np.array([float(r["pred_R1_cached"]) for r in rows]),
        "H1": np.array([float(r["pred_H1"]) for r in rows]),
        "H2": np.array([float(r["pred_H2"]) for r in rows]),
    }
    sha = [C.sha256_text(p)[:16] for p in pids]

    out = {
        "generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
        "units": "log2(hek exp/input)", "n": int(len(y)), "n_folds": C.N_FOLDS,
        "n_groups": int(len(np.unique(gid))),
        "tie_break": "sha256(prom_id)[:16] ascending",
        "evidence_kind": ("RETROSPECTIVE ADAPTIVE DEVELOPMENT on the reused batch22 "
                          "GSE188639 HEK293 folds/test outcomes; NOT independent "
                          "prospective validation"),
        "cached_R1_refits": 0,
        "decision_rule": {"rho_gain_min": C.GAIN_RHO, "mae_reduction_min": C.MAE_RED,
                          "top_decile_not_lower": True,
                          "min_positive_folds": "%d of %d" % (C.MIN_POS_FOLDS, C.N_FOLDS),
                          "frozen": True},
        "pooled_oof": {}, "per_fold": {}, "strata": {}, "paired_group_bootstrap": {},
        "decisions": {},
    }
    for name, p in preds.items():
        blk = C.metrics_block(p, y, sha)
        blk["spearman_ci_group_bootstrap"] = C.group_bootstrap(
            p, y, gid, metric=C.spearman, n_boot=C.N_BOOT, seed=C.SEED_BOOT)
        out["pooled_oof"][name] = blk
    for f in range(C.N_FOLDS):
        m = fold == f
        out["per_fold"][str(f)] = {"n": int(m.sum()),
                                   "n_groups": int(len(np.unique(gid[m])))}
        for name, p in preds.items():
            out["per_fold"][str(f)][name] = C.metrics_block(
                p[m], y[m], [sha[i] for i in np.where(m)[0]])
    for lab, mask in (("source_unseen", seen == 0), ("source_seen", seen == 1)):
        out["strata"][lab] = {"n": int(mask.sum()),
                              "n_groups": int(len(np.unique(gid[mask])))}
        for name, p in preds.items():
            out["strata"][lab][name] = (C.metrics_block(p[mask], y[mask],
                                                        [sha[i] for i in np.where(mask)[0]])
                                        if mask.sum() >= 3
                                        else {"n": int(mask.sum()), "note": "too few rows"})

    base = out["pooled_oof"]["R1_cached"]
    for cand in ("H1", "H2"):
        boot = C.paired_group_bootstrap(preds[cand], preds["R1_cached"], y, gid)
        out["paired_group_bootstrap"]["%s_minus_R1" % cand] = boot
        cm = out["pooled_oof"][cand]
        d_rho = cm["spearman"] - base["spearman"]
        mae_red = (base["mae"] - cm["mae"]) / base["mae"] if base["mae"] > 0 else float("nan")
        top_ok = cm["top_decile"]["precision"] >= base["top_decile"]["precision"]
        pos = sum(1 for f in range(C.N_FOLDS)
                  if out["per_fold"][str(f)][cand]["spearman"]
                  > out["per_fold"][str(f)]["R1_cached"]["spearman"])
        met = bool(d_rho >= C.GAIN_RHO and mae_red >= C.MAE_RED and top_ok
                   and pos >= C.MIN_POS_FOLDS)
        lo = boot["delta_spearman"]["ci97_5_two_comparison"]["lo"]
        lo95 = boot["delta_spearman"]["ci95"]["lo"]
        out["decisions"][cand] = {
            "delta_spearman_vs_R1": d_rho, "mae_reduction_frac_vs_R1": mae_red,
            "top_decile_not_lower": bool(top_ok), "positive_folds": int(pos),
            "thresholds_met": met,
            "delta_spearman_ci95_lo": lo95, "delta_spearman_ci97_5_lo": lo,
            "ci95_excludes_zero": bool(lo95 is not None and lo95 > 0),
            "ci97_5_excludes_zero": bool(lo is not None and lo > 0),
            "verdict": ("practical-gain thresholds met AND 97.5% two-comparison CI "
                        "excludes zero" if (met and lo is not None and lo > 0)
                        else "practical-gain thresholds met but CI includes zero "
                             "(gain uncertain)" if met
                        else "no substantive improvement over R1 (honest negative)"),
        }
    C.atomic_write_json(C.REPORT_OUT / "results.json", out)

    # independent check that the copied cached R1 reproduces batch22's published R1
    b22 = json.load(open(C.ROOT / "reports" / "breakthrough_20261002" / "batch22" / "results.json",
                         encoding="utf-8"))
    ref = b22["pooled_oof"]["R1"]
    chk = {
        "spearman_match": abs(ref["spearman"] - base["spearman"]) < 1e-12,
        "mae_match": abs(ref["mae"] - base["mae"]) < 1e-12,
        "r2_match": abs(ref["r2"] - base["r2"]) < 1e-12,
        "top_decile_precision_match": abs(ref["top_decile"]["precision"]
                                          - base["top_decile"]["precision"]) < 1e-12,
        "reference": "reports/breakthrough_20261002/batch22/results.json",
        "recomputed_R1": {"spearman": base["spearman"], "mae": base["mae"], "r2": base["r2"],
                          "top_decile_precision": base["top_decile"]["precision"]},
    }
    C.atomic_write_json(C.REPORT_OUT / "cached_r1_verification.json", chk)
    print(json.dumps(out["decisions"], indent=1))
    print("R1 verify:", json.dumps(chk))
    print("pooled:", {k: round(v["spearman"], 6) for k, v in out["pooled_oof"].items()})


if __name__ == "__main__":
    main()
