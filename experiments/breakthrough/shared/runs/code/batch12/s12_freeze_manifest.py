#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 frozen manifest freeze (pre-catalog).

Emits a frozen manifest/config JSON that locks, BEFORE catalog generation,
the exact artifact paths + SHA256 hashes, the target transform constants, the
model config, the population constants, and the CLI contract. This manifest is
the traceability anchor that the catalog/replay/report/tests all reference.

Read-only on frozen inputs; writes ONLY:
  data/breakthrough_20261002/batch12/freeze_manifest.json
  models/breakthrough_20261002/batch12/freeze_manifest.json
No arbitrary pickle path is accepted; no scientific fit is performed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s12_common import (  # noqa: E402
    BATCH, CANDIDATE_VERSION, DATA_OUT, FEATURE_NAMES_ALL, FROZEN_ARTIFACTS,
    HGB_CONFIG, M_S_DEFINED, MODEL_KEY, MODEL_OUT, N_CANONICAL, N_ELIGIBLE,
    N_EXCLUDED, N_FEATURE_UNIVERSE, N_TEST, N_TRAIN, N_VAL, S_TARGET_DEF,
    S_THEORETICAL_MAX, TARGET_TRAIN_MEAN, TARGET_TRAIN_STD, TRACK,
    atomic_write_json, now_local, verify_artifacts)


def main():
    ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
    manifest = {
        "candidate_version": CANDIDATE_VERSION,
        "track": TRACK,
        "batch": BATCH,
        "freeze_kind": "pre_catalog",
        "frozen_at": now_local(),
        "integrity_ok": bool(ok),
        "integrity_problems": problems,
        "scientific_fit_performed": False,
        "mode": "frozen_gene_level_S_inference",
        "population_constants": {
            "n_canonical_label_genes": N_CANONICAL,
            "n_S_defined": M_S_DEFINED,
            "n_feature_universe": N_FEATURE_UNIVERSE,
            "n_eligible_catalog": N_ELIGIBLE,
            "n_excluded_S_defined_feature_missing": N_EXCLUDED,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "relationship": "eligible = canonical S-defined AND feature-covered; "
                            "excluded = S-defined but feature-missing (178); "
                            "undefined = B_raw<0.30 or <3 detectable tissues",
        },
        "target_definition": S_TARGET_DEF,
        "target_transform": {
            "train_mean": TARGET_TRAIN_MEAN,
            "train_std": TARGET_TRAIN_STD,
            "inverse": "y_hat = z * train_std + train_mean",
            "theoretical_max": S_THEORETICAL_MAX,
            "outside_range_flag": "predicted_raw_S > 0.0 (regression unconstrained)",
        },
        "model_config": HGB_CONFIG,
        "model_key_used": MODEL_KEY,
        "feature_count": len(FEATURE_NAMES_ALL),
        "feature_names_file": "generated in s12_common.py (45 baseline + 1024 positional)",
        "frozen_artifacts": [
            {"key": a["key"], "path": a["path"], "sha256": a.get("sha256"),
             "role": a["role"]} for a in FROZEN_ARTIFACTS
        ],
        "cli_contract": {
            "entrypoint": "code/batch12/s12_predict.py",
            "subcommands": ["predict", "trace", "catalog", "version"],
            "encoding": "UTF-8 JSON/TSV, finite numeric serialization (no NaN)",
            "ordering": "deterministic (catalog sorted by gene_id; predict/trace by input order with row_id)",
            "classify_order": [
                "empty_input", "transcript_id (ENST)", "non_human (non-ENSG)",
                "unknown (canonical ENSG not in 58988 catalog / non-Ensembl)",
                "S_undefined (canonical but B_raw<0.30 or <3 detectable tissues)",
                "S_defined_feature_missing (canonical S-defined but not feature-covered)",
                "predicted (canonical S-defined AND feature-covered)",
            ],
            "no_arbitrary_pickle_loading": True,
            "read_only_on_frozen_inputs": True,
            "verifies_integrity_before_inference": True,
        },
    }
    atomic_write_json(DATA_OUT / "freeze_manifest.json", manifest)
    atomic_write_json(MODEL_OUT / "freeze_manifest.json", manifest)
    print(json.dumps({"integrity_ok": ok,
                      "data_manifest": str(DATA_OUT / "freeze_manifest.json"),
                      "model_manifest": str(MODEL_OUT / "freeze_manifest.json")},
                     ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

