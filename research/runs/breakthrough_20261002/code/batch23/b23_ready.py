#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 readiness: data_ready.json + next_batch_proposal.json (ZERO fits here).

No new external HEK293/HEK293T promoter-autonomy cohort was obtainable (see
public_source_inventory.json). Per protocol, output a verified negative source
inventory AND a specific alternative using already-verified data with concrete
input paths and a FIXED split. Parent must approve the exact frozen protocol
before any fitting; this batch has no fit permission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (  # noqa: E402
    ROOT, DATA_OUT, REPORT_OUT, BATCH22_DATA, BATCH22_SEQ_TABLE, BATCH22_FOLD_ASSIGN,
    BATCH22_MODEL, FROZEN_CKPT, now_local, sha256_file, atomic_write_json)

BATCH21_DATA = ROOT / "data" / "breakthrough_20261002" / "batch21"


def main():
    ov = json.load(open(REPORT_OUT / "public_source_inventory.json", encoding="utf-8"))
    inventory_shas = {
        str(BATCH22_SEQ_TABLE.relative_to(ROOT)): sha256_file(BATCH22_SEQ_TABLE),
        str(BATCH22_FOLD_ASSIGN.relative_to(ROOT)): sha256_file(BATCH22_FOLD_ASSIGN),
        str((BATCH22_DATA / "frozen_repr_A_seed20261023.npz").relative_to(ROOT)):
            sha256_file(BATCH22_DATA / "frozen_repr_A_seed20261023.npz"),
        str(FROZEN_CKPT.relative_to(ROOT)): sha256_file(FROZEN_CKPT),
        str((DATA_OUT / "local_constructs.tsv").relative_to(ROOT)):
            sha256_file(DATA_OUT / "local_constructs.tsv"),
    }
    data_ready = {
        "generated_at_local": now_local(), "batch": "batch23",
        "new_external_cohort": False,
        "new_external_cohort_reason": ov["conclusion"],
        "verified_ready_inputs": {
            "hek293_target_sequence_table": {
                "path": str(BATCH22_SEQ_TABLE.relative_to(ROOT)),
                "sha256": inventory_shas[str(BATCH22_SEQ_TABLE.relative_to(ROOT))],
                "n_rows": 673, "insert_len": 133,
                "target_units": "hek_norm = log2(hek exp / input)",
                "row_role": "reused retrospective target-domain supervision (GSE188639)",
            },
            "paired_k562_labels": {
                "path": str(BATCH22_SEQ_TABLE.relative_to(ROOT)),
                "column": "k562_norm", "note": "same 673 sequences; paired measurement; adds rows, "
                                               "NOT independent groups",
            },
            "group_folds": {
                "path": str(BATCH22_FOLD_ASSIGN.relative_to(ROOT)),
                "sha256": inventory_shas[str(BATCH22_FOLD_ASSIGN.relative_to(ROOT))],
                "n_groups": 669, "n_folds": 5, "fold_seed": 20261025,
            },
            "frozen_representation": {
                "npz": str((BATCH22_DATA / "frozen_repr_A_seed20261023.npz").relative_to(ROOT)),
                "checkpoint": str(FROZEN_CKPT.relative_to(ROOT)),
                "model_sha256": inventory_shas[str(FROZEN_CKPT.relative_to(ROOT))],
                "dim": 128,
            },
            "local14_sequences": {
                "path": str((DATA_OUT / "local_constructs.tsv").relative_to(ROOT)),
                "sha256": inventory_shas[str((DATA_OUT / "local_constructs.tsv").relative_to(ROOT))],
                "n": 14, "role": "out-of-domain external probe only (no fit, no calibration)",
            },
        },
        "claim_limits": [
            "reused retrospective HEK293 cohort; NOT an independent new study",
            "673 rows retained; earlier from-scratch gate not lowered or raised",
            "local 14 are a different cell/assay/scale (HEK293T transient D28 log10 relGFP)",
        ],
        "input_shas": inventory_shas,
    }
    atomic_write_json(REPORT_OUT / "data_ready.json", data_ready)

    proposal = {
        "generated_at_local": now_local(), "batch": "batch23",
        "status": "proposed_only; parent must approve the exact frozen protocol; ZERO fits in batch23",
        "situation": "No larger HEK293/HEK293T promoter-autonomy cohort obtainable within the bound; "
                     "673 verified HEK293 rows + 669 groups remain the target-domain supervision.",
        "primary_objective": "novel-sequence promoter-activity ranking under a mechanism that is NOT a "
                             "repeat of the K562 ranking-loss comparison",
        "cohort": {
            "name": "GSE188639 core-promoter library, HEK293 (reused retrospective)",
            "rows": 673, "groups": 669, "target": "hek_norm", "units": "log2(hek exp / input)",
            "insert_len": 133, "split": "batch22 fold_assignment.tsv, 5 group folds, seed 20261025",
        },
        "recipes": [
            {"id": "A", "name": "matched composition baseline (strong)",
             "features": "mono4+di16+tri64+GC+log1p(len) (86-d)", "model": "Ridge(alpha=100)",
             "role": "baseline; already verified in batch22"},
            {"id": "B", "name": "cell-conditioned twin-head / paired-cell multi-task",
             "features": "same supervised rows + a cell one-hot; shared composition (and optionally the "
                         "frozen 128-d) encoder with two heads (HEK293 primary, K562 auxiliary)",
             "role": "candidate; mechanism = cell-context conditioning / domain adaptation using "
                     "GENUINELY PAIRED measurements, distinct from K562 ranking loss",
             "target": "HEK293 head evaluated on held-out HEK293 groups only"},
        ],
        "budget": {"recipes": 2, "seeds": 2, "max_full_fits": 4, "max_gpu_hours": 3,
                   "note": "ridge-only variants are CPU (<=4 threads); neural variants capped at 4 fits"},
        "transforms": "all scaling / k-mer vocabulary / any feature transform fit on TRAIN folds only",
        "metrics": {"primary": ["group-heldout Spearman", "MAE", "R2"], "secondary": [],
                    "practical": "top-decile precision/enrichment (predeclared)",
                    "uncertainty": "group-level bootstrap; no technical-replicate CI",
                    "decision_rule": "predeclare thresholds BEFORE fitting; report honest negative"},
        "checkpoints": "save per-fold checkpoints + train-only stats; reuse, never retrain cached folds",
        "stop": "report negative honestly; no threshold lowering after seeing results",
        "fallback_if_rejected": "a bounded spectrum/string-kernel ridge (k=4..6) vs composition, same "
                                "fixed split, CPU-only",
        "concrete_input_paths": {
            "rows": str(BATCH22_SEQ_TABLE.relative_to(ROOT)),
            "folds": str(BATCH22_FOLD_ASSIGN.relative_to(ROOT)),
            "frozen": str((BATCH22_DATA / "frozen_repr_A_seed20261023.npz").relative_to(ROOT)),
            "local_probe": str((DATA_OUT / "local_constructs.tsv").relative_to(ROOT)),
        },
    }
    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", proposal)
    print("[b23_ready] data_ready + next_batch_proposal written")
    return data_ready, proposal


if __name__ == "__main__":
    main()
