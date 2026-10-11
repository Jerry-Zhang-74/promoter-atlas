#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 stage orchestrator. Writes runs/breakthrough_20261002/batch21/status.json
with the ACTUAL process PID/timestamp at start and refreshes it through every stage.
No planned-future completion timestamps are ever written."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b21_common import (RUN_OUT, REPORT_OUT, DATA_OUT, MODEL_OUT, atomic_write_json,
                        now_local, sha256_file, VERSION, BATCH)  # noqa: E402

STATUS = RUN_OUT / "status.json"
LOGF = RUN_OUT / "batch21_worker.log"
STARTED = now_local()
PID = os.getpid()


def log(msg):
    line = "[%s] %s" % (now_local(), msg)
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(stage, state, extra=None):
    st = {"batch": BATCH, "version": VERSION, "process": {"pid": PID,
          "launched_local": STARTED, "active_command": "b21_worker.py"},
          "stage": stage, "state": state, "updated_local": now_local(),
          "last_progress_local": now_local(),
          "checkpoints": "per-fit weights under models/breakthrough_20261002/batch21",
          "next_action": (extra or {}).get("next_action", "see stages"),
          "artifacts": (extra or {}).get("artifacts", [])}
    atomic_write_json(STATUS, st)
    return st


def stage_freeze():
    """Freeze the exact conditional design BEFORE the first fit."""
    ver = json.load(open(REPORT_OUT / "public_source_verification.json", encoding="utf-8"))
    ga = json.load(open(REPORT_OUT / "group_audit.json", encoding="utf-8"))
    frz = {
        "generated_at_local": now_local(),
        "version": VERSION, "batch": BATCH,
        "dataset_decision": ver["dataset_decision"],
        "dataset_selection_basis": ver["declared_selection_rule"],
        "primary_public_candidate": {"accession": ver["primary_candidate"]["accession"],
                                     "verdict": ver["primary_candidate"]["verdict"],
                                     "gate": ver["primary_candidate"]["gate"]},
        "selected_dataset": ver["local_selected_source"],
        "HEK293T_gap": ver["HEK293T_gap"],
        "task": "predict K562 GSE118242 per-sequence measured MPRA activity (log10) from 164 bp sequence",
        "split": {"protocol": "strict Hamming<=16 fwd+RC connected components; 80/10/10 by group",
                  "seed": ga["split"]["seed"], "components": ga["grouping"]["n_components"],
                  "rows": ga["split"]["rows_per_split"], "groups": ga["split"]["components_per_split"],
                  "cross_split_ham_le16_total": ga["cross_split_audit"]["cross_split_pairs_ham_le16_total"],
                  "strict_ok": ga["strict_ok"],
                  "evaluation_kind": "historical-development K562 (no new independent wet data)"},
        "arms": {
            "A": "MSE of train-standardized measured activity",
            "B": "MSE + 0.1*mean softplus(-sign(z_i-z_j)*(p_i-p_j)) over within-minibatch pairs "
                 "with |z_i-z_j|>=0.5 (standardized labels, TRAIN rows only; 0 if no pairs)",
            "backbone": "frozen batch02 LegNetInspired (same preprocessing for both arms)",
            "note": "objective comparison, NOT two architectures and NOT a validated length-aware solution"},
        "seeds": [20261023, 20261024], "n_full_fits": 4,
        "optimizer": "AdamW lr=3e-4 wd=1e-4", "batch": 32, "max_epochs": 20,
        "early_stop": "validation Spearman, patience 5; ties -> lower MAE then earlier epoch",
        "fit_cap_gpu_min": 45, "aggregate_cap_gpu_h": 3,
        "normalization": "train-only mean/std, identical for both arms",
        "variable_length_policy": "not applicable: measured source is uniformly 164 bp; native 164 bp kept",
        "selection": "validation-only; test reported",
        "decision_rule": {"mean_spearman_gain_min": 0.02, "mae_reduction_min_frac": 0.05,
                          "top_decile_precision": "not worse", "both_seeds_positive": True,
                          "note": "task-specific; NOT a change of the batch02 +0.05/10 percent threshold"},
        "wet_constructs_role": "retrospective external-context diagnostics only; never selection/calibration",
        "source_sha256": {"public_source_verification.json": sha256_file(REPORT_OUT / "public_source_verification.json"),
                          "group_audit.json": sha256_file(REPORT_OUT / "group_audit.json"),
                          "split_assignment.tsv": sha256_file(DATA_OUT / "split_assignment.tsv")},
    }
    atomic_write_json(REPORT_OUT / "experiment_freeze.json", frz)
    return frz


def stage_finalize():
    import b21_eval, b21_figs, pandas as pd
    from b21_common import MPRA_SUPERVISED, sha256_text
    df = pd.read_csv(MPRA_SUPERVISED, sep="\t")
    seqs = df["sequence"].astype(str).str.upper().tolist()
    y = df["activity"].to_numpy(float)
    sha = [sha256_text(s) for s in seqs]
    res, _ = b21_eval.run()
    b21_figs.run(seqs, y, sha)
    # code + artifact hashes
    man = {"generated_at_local": now_local(), "version": VERSION,
           "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("b21_*.py"))},
           "artifacts": {}}
    for p in sorted(list(REPORT_OUT.glob("*")) + list(DATA_OUT.glob("*")) + list(MODEL_OUT.glob("*"))):
        if p.is_file():
            man["artifacts"][str(p.relative_to(RUN_OUT.parents[2]))] = sha256_file(p)
    atomic_write_json(REPORT_OUT / "sha_manifest.json", man)
    return res, man


def main(only=None):
    stages = ["data", "groups", "freeze", "fit", "eval", "finalize"]
    write_status("start", "running", {"next_action": "data stage"})
    log("worker start pid=%d version=%s" % (PID, VERSION))
    for stg in stages:
        if only and stg not in only:
            continue
        write_status(stg, "running")
        log("stage %s start" % stg)
        t0 = time.time()
        try:
            if stg == "data":
                import b21_source; b21_source.run()
            elif stg == "groups":
                import b21_groups; b21_groups.run()
            elif stg == "freeze":
                stage_freeze()
            elif stg == "fit":
                import b21_train; b21_train.run()
            elif stg == "eval":
                import b21_eval; b21_eval.run()
            elif stg == "finalize":
                stage_finalize()
            log("stage %s done in %.1fs" % (stg, time.time() - t0))
            write_status(stg, "done", {"next_action": "next stage"})
        except Exception:
            log("stage %s FAILED:\n%s" % (stg, traceback.format_exc()))
            write_status(stg, "failed", {"next_action": "repair (<=3 bounded repairs)"})
            raise
    write_status("complete", "worker_complete_pending_independent_review",
                 {"next_action": "fresh independent review"})
    log("worker complete")


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    main(only)
