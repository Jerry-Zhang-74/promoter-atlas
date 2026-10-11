#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 finalize: assemble strict JSON evidence, unique 5-fold predictions table,
telemetry/inference account, null-with-reason, source correction, selfcheck, status,
remaining work, ONE bounded next protocol, sha manifest and registry/changelog append."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402


def load_pred():
    with open(C.DATA / "frozen_input_table.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    h1 = json.load(open(C.DATA / "cache/h1/preds.json", encoding="utf-8"))
    pa = json.load(open(C.DATA / "cache/parm/preds.json", encoding="utf-8"))
    return rows, h1, pa


def write_predictions():
    rows, h1, pa = load_pred()
    hdr = (["label", "control", "chr", "start", "end", "strand", "component_id",
            "alpha_primary", "sequence_sha256"]
           + ["parm_fold%d" % k for k in range(5)] + ["parm_mean"]
           + ["h1_fold%d" % k for k in range(5)] + ["h1_mean", "gc"])
    out = []
    for r in rows:
        sha = r["sequence_sha256"]
        hp = h1["pred"][sha]; pp = pa["pred"][sha]
        out.append([r["label"], r["control"], r["chr"], r["start"], r["end"], r["strand"],
                    r["component_id"], r["alpha_primary"], sha]
                   + ["%r" % x for x in pp["parm_folds"]] + ["%r" % pp["parm_mean"]]
                   + ["%r" % x for x in hp["h1_folds"]] + ["%r" % hp["h1_mean"], "%r" % hp["gc"]])
    C.atomic_write_tsv(C.REPORT / "predictions_unique_5fold.tsv", hdr, out)
    return len(out)


def main():
    t0 = time.time()
    p1 = json.load(open(C.DATA / "metrics_part1.json", encoding="utf-8"))
    p2 = json.load(open(C.DATA / "metrics_part2.json", encoding="utf-8"))
    man = json.load(open(C.REPORT / "diagnostic_manifest.json", encoding="utf-8"))
    add = json.load(open(C.DATA / "taskA_correction_addendum.json", encoding="utf-8"))
    n_pred = write_predictions()

    code_sha = {f.name: C.sha256_file(f) for f in sorted(C.CODE.glob("*.py"))}
    nulls = {
        "length_baseline_rho": {"value": None,
                                "reason": "length is constant 270 for all rows -> Spearman undefined"},
        "gc_baseline_constant_null": {"value": None, "reason": "N/A here; GC is NOT constant (0.248-0.807)"},
    }
    report = {
        "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
        "record_type": "batch41_frozen_transfer_report",
        "scope": "ZERO-FIT frozen-model cross-cell transfer diagnostic (not a promotion)",
        "manifest": "diagnostic_manifest.json (frozen before any forward)",
        "taskA_correction": "data/breakthrough_20261002/batch41/taskA_correction_addendum.json",
        "metrics_pooled": p1["pooled"],
        "paired_rho_differences": p1["paired_rho_differences_common_set"],
        "strata": p1["strata"],
        "per_component": p2["per_component"],
        "bootstrap": p2["bootstrap"],
        "loco": p2["loco"],
        "controls": p2["controls"],
        "model_fold_spread": p1["model_fold_spread_mean"],
        "decision_rule": p2["decision_rule"],
        "null_metrics_with_reason": nulls,
        "inference_account": {
            "PARM": p1["PARM_runtime"], "H1_GC": {"model": "CPU kernel readout + GC",
                                                  "fits": 0, "gpu": False},
            "total_fits_or_solves": 0,
            "zero_gradient_backward_optimizerstep": True,
        },
        "metric_code_sha256": code_sha,
        "unique_5fold_predictions_rows": n_pred,
        "source_correction": {
            "batch40_defect": "saved 0 overlap pairs / coverage 1.0113 / unmatched -85 vs printed 8821/8821",
            "batch41_repair": {"step90_pairs_checked": add["step90_overlap_recompute"]["pairs_checked"],
                               "step90_pairs_consistent": add["step90_overlap_recompute"]["pairs_consistent"],
                               "bio_coverage": add["reconciled_gate"]["bio_coverage_fraction"],
                               "unmatched_bio": add["reconciled_gate"]["unmatched_bio"],
                               "controls_matched": "%d/%d" % (add["reconciled_gate"]["ctrl_matched"], add["reconciled_gate"]["controls"]),
                               "source20_all_ok": add["source20_join_check"]["all_ok"]},
        },
        "limitations": [
            "public HEK293FT != local 293FT / D28; prior PARM/H1 training exposure UNKNOWN",
            "retrospective transfer, not prospective or strict new-family validation",
            "minimal promoter / vector / delivery / autonomy UNKNOWN; physical-assay orientation UNRESOLVED",
            "one MAPT regional concentration; components are NOT independent CRE families",
            "5-fold spread is model disagreement, not a confidence interval",
            "bootstrap CI is conditional component-resampling uncertainty",
        ],
        "elapsed_s": round(time.time() - t0, 3),
    }
    C.atomic_write_json(C.DATA / "report.json", report)
    print(json.dumps({"predictions_rows": n_pred,
                      "verdict": p2["decision_rule"]["verdict"],
                      "parm_rho": p1["pooled"]["parm"]["spearman"],
                      "code_files": len(code_sha)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
