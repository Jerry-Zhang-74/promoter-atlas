#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 whole-catalog replay evidence.

Reproduces the full 18769-gene prediction catalog through the frozen predictor
and verifies it against the batch10 stored per-gene outputs (predictions.tsv)
at atol/rtol 1e-10 for the hgb_aug column, and against the stored full-precision
observed S_raw_full. Also records the batch10 predictions.tsv SHA256 into the
frozen artifact registry (freeze-time hash) so traceability is self-contained.

Writes ONLY:
  data/breakthrough_20261002/batch12/replay_evidence.json
  models/breakthrough_20261002/batch12/replay_evidence.json
  data/breakthrough_20261002/batch12/replay_catalog.tsv (regenerated catalog)
No scientific fit; read-only on all frozen inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s12_common import (  # noqa: E402
    B10_PRED_TSV, BATCH, CANDIDATE_VERSION, DATA_OUT, FROZEN_ARTIFACTS,
    MODEL_OUT, atomic_write_json, now_local, sha256_file, verify_artifacts)
from s12_predict import FrozenPredictor  # noqa: E402

ATOL = 1e-10
RTOL = 1e-10


def main():
    ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
    if not ok:
        raise RuntimeError("integrity FAILED: " + json.dumps(problems))

    pred_hash = sha256_file(B10_PRED_TSV)
    p = FrozenPredictor(verify=False)
    rows = p.build_catalog()
    our = {r["gene_id"]: r for r in rows}

    df = pd.read_csv(B10_PRED_TSV, sep="\t", dtype={"gene_id": "str"})
    b10 = {row.gene_id: row for row in df.itertuples(index=False)}

    only_ours = sorted(set(our) - set(b10))
    only_b10 = sorted(set(b10) - set(our))

    max_abs_pred = 0.0
    max_abs_obs = 0.0
    n_checked = 0
    for gid in sorted(set(our) & set(b10)):
        pred_ours = our[gid]["predicted_raw_S"]
        pred_b10 = float(b10[gid].hgb_aug_pred)
        obs_ours = our[gid]["observed_raw_S"]
        obs_b10 = float(b10[gid].S_raw_full)
        max_abs_pred = max(max_abs_pred, abs(pred_ours - pred_b10))
        max_abs_obs = max(max_abs_obs, abs(obs_ours - obs_b10))
        if not (np.isclose(pred_ours, pred_b10, atol=ATOL, rtol=RTOL)
                and np.isclose(obs_ours, obs_b10, atol=ATOL, rtol=RTOL)):
            n_checked = -1
            break
        n_checked += 1

    passed = (n_checked == len(our) == len(set(our) & set(b10))
              and not only_ours and not only_b10
              and max_abs_pred <= ATOL and max_abs_obs <= ATOL)

    evidence = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "replay_at": now_local(),
        "passed": bool(passed),
        "acceptance_criterion": "catalog predicted_raw_S == batch10 hgb_aug_pred "
                                "AND observed_raw_S == batch10 S_raw_full "
                                "at atol/rtol 1e-10",
        "atol": ATOL, "rtol": RTOL,
        "n_catalog_genes": len(our),
        "n_batch10_genes": len(b10),
        "n_intersection": len(set(our) & set(b10)),
        "only_in_catalog": only_ours,
        "only_in_batch10": only_b10,
        "max_abs_pred_diff": max_abs_pred,
        "max_abs_obs_diff": max_abs_obs,
        "pred_allclose_1e_10": bool(max_abs_pred <= ATOL),
        "obs_allclose_1e_10": bool(max_abs_obs <= ATOL),
        "batch10_predictions_tsv": str(B10_PRED_TSV),
        "batch10_predictions_sha256": pred_hash,
        "integrity_ok": ok,
    }
    atomic_write_json(DATA_OUT / "replay_evidence.json", evidence)
    atomic_write_json(MODEL_OUT / "replay_evidence.json", evidence)

    # Regenerate a replay catalog identical to catalog.tsv for evidence parity.
    with open(DATA_OUT / "replay_catalog.tsv", "w", encoding="utf-8") as fh:
        cols = ["gene_id", "split", "component_id", "n_promoters",
                "observed_raw_S", "predicted_raw_S", "legacy_Score_S",
                "observed_E_raw", "observed_B_raw", "legacy_Score_E",
                "legacy_Score_B", "in_sample", "outside_theoretical_range"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

