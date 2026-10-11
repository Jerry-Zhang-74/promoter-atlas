#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 worker self-check.

Run by the WORKER ONLY (not a substitute for the independent reviewer). Emits:
  runs/breakthrough_20261002/batch12/worker_self_check.json
  reports/breakthrough_20261002/batch12/worker_self_check.json
  reports/breakthrough_20261002/batch12/worker_self_check.txt
  data/breakthrough_20261002/batch12/label_trace_manifest.json
Read-only on all frozen inputs; writes only the above batch12-owned files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s12_common import (  # noqa: E402
    AUDIT_TSV, BATCH, CANON_TSV, CANDIDATE_VERSION, DATA_OUT, EXPR_TSV,
    FROZEN_ARTIFACTS, MODEL_OUT, N_ELIGIBLE, N_EXCLUDED, REPORT_OUT, RUNS_OUT,
    atomic_write_json, atomic_write_text, now_local, sha256_file,
    verify_artifacts)
from s12_predict import FrozenPredictor  # noqa: E402

ACTB = "ENSG00000075624"
GAPDH = "ENSG00000111640"


def main():
    checks = {}
    ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
    checks["integrity_ok"] = bool(ok)
    checks["integrity_problems"] = problems

    p = FrozenPredictor(verify=False)
    cats = p.build_catalog()
    excl = p.build_exclusion()
    checks["catalog_count"] = len(cats)
    checks["exclusion_count"] = len(excl)
    checks["catalog_count_ok"] = (len(cats) == N_ELIGIBLE)
    checks["exclusion_count_ok"] = (len(excl) == N_EXCLUDED)
    ids = [r["gene_id"] for r in cats]
    checks["catalog_sorted_unique"] = (ids == sorted(ids) and len(set(ids)) == len(ids))
    checks["no_nan_in_catalog"] = all(
        str(r["predicted_raw_S"]) != "nan" for r in cats)

    replay = json.loads((DATA_OUT / "replay_evidence.json").read_text(encoding="utf-8"))
    checks["replay_passed"] = bool(replay["passed"])
    checks["replay_max_abs_pred_diff"] = replay["max_abs_pred_diff"]

    actb = p.predict([ACTB])[0]
    gapdh = p.predict([GAPDH])[0]
    checks["actb_predicted"] = (actb["status"] == "predicted")
    checks["gapdh_predicted"] = (gapdh["status"] == "predicted")
    checks["actb_observed"] = actb["observed_raw_S"]
    checks["gapdh_observed"] = gapdh["observed_raw_S"]

    tr = p.trace([ACTB])[0]
    checks["trace_found_actb"] = (tr["status"] == "found")
    checks["trace_formula_present"] = ("formula" in tr)

    # classification smoke
    class_checks = {}
    for gid, expect in [("ENST00000331789", "transcript_id"),
                        ("ENSMUSG00000000001", "non_human"),
                        ("ENSG99999999999", "unknown")]:
        r = p.predict([gid])[0]
        class_checks[gid] = (r["status"], r.get("reason_code"))
    checks["classification"] = class_checks

    all_ok = all([
        checks["integrity_ok"], checks["catalog_count_ok"],
        checks["exclusion_count_ok"], checks["catalog_sorted_unique"],
        checks["no_nan_in_catalog"], checks["replay_passed"],
        checks["actb_predicted"], checks["gapdh_predicted"],
        checks["trace_found_actb"], checks["trace_formula_present"],
        all(v[1] == e for v, e in zip(class_checks.values(),
            ["transcript_id", "non_human", "unknown"])),
    ])
    checks["all_ok"] = bool(all_ok)

    # Label-trace compact manifest.
    manifest = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": now_local(),
        "label_sources": [
            {"file": "labels/build/gene_expression_table.tsv",
             "sha256": FROZEN_ARTIFACTS[7]["sha256"],
             "role": "raw tissue mean_tpm -> full-precision S_raw reconstruction"},
            {"file": "labels/build/gene_expression_label_table.tsv",
             "sha256": FROZEN_ARTIFACTS[8]["sha256"],
             "role": "canonical E_raw/S_raw/B_raw + Score_E/S/B"},
            {"file": "data/breakthrough_20261002/batch08/label_audit_all_genes.tsv",
             "sha256": FROZEN_ARTIFACTS[9]["sha256"],
             "role": "58988-gene audit: full-precision S_raw + eligibility reasons + scores"},
        ],
        "trace_interface": {
            "entrypoint": "code/batch12/s12_predict.py trace",
            "schema": "returns observed E_raw/S_raw/B_raw + Score_E/S/B + "
                      "detectable_tissue_count + eligibility/missing reason + source file/hash links",
            "never_substitutes_label_lookup_for_prediction": True,
            "does_not_fabricate_absent_labels": True,
        },
        "canonical_counts": {
            "n_canonical_label_genes": 58988,
            "n_S_defined": 18947,
            "n_excluded": N_EXCLUDED,
            "n_feature_universe": 57827,
        },
    }
    atomic_write_json(DATA_OUT / "label_trace_manifest.json", manifest)

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "worker": "self_check (NOT independent review)",
        "written_at": now_local(),
        "all_ok": bool(all_ok),
        "checks": checks,
        "label_trace_manifest": str(DATA_OUT / "label_trace_manifest.json"),
    }
    atomic_write_json(RUNS_OUT / "worker_self_check.json", summary)
    atomic_write_json(REPORT_OUT / "worker_self_check.json", summary)

    lines = ["worker_self_check %s: all_ok=%s" % (CANDIDATE_VERSION, all_ok)]
    for k, v in checks.items():
        lines.append("  %-24s %s" % (k, v))
    atomic_write_text(REPORT_OUT / "worker_self_check.txt", "\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

