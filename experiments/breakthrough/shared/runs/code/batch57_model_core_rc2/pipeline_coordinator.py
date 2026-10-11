# -*- coding: utf-8 -*-
"""Coordinator: verify parent spec, then train R0 -> Nabs -> Njoint on ONE GPU.

Startup RE-READS batch52/PARENT_DATA_ACCEPTANCE.txt and batch52/PARALLEL_PIPELINE.txt and
asserts the corrected target Log2RPM_HEK293. A real fit execution requires explicit owner
opt-in flags AND accepted actual data. Every actual execution reserves a reviewed exec_gate
slot in the batch52 ledger (MAX 4, including failed/restarted runs) BEFORE its first
gradient/solve. Both neural arms are frozen (same code/config) before Nabs trains; Njoint
starts immediately after Nabs. Importing this module performs NO work.
"""
from __future__ import annotations
import datetime as dt
import json, os

from io_utils import atomic_write_json, sha256_file
from spec import TARGET_COLUMN


class GateError(Exception):
    pass


def read_run_policy(run_dir):
    """Read batch52/RUN_POLICY.json DYNAMICALLY at each execution (13:00 delivery, 12:00
    first-training target, CPU limits). Raises if missing/malformed (fail-closed)."""
    p = os.path.join(run_dir, "batch52", "RUN_POLICY.json")
    if not os.path.exists(p):
        raise GateError("RUN_POLICY.json missing: %s" % p)
    with open(p, "r", encoding="utf-8") as f:
        pol = json.load(f)
    for k in ("first_model_training_target", "first_model_delivery_deadline",
              "second_model_overall_deadline", "total_fit_execution_limit"):
        if k not in pol:
            raise GateError("RUN_POLICY.json missing key %r" % k)
    pol["_path"] = p
    pol["_sha256"] = sha256_file(p)
    return pol


def _epoch_from_iso(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).timestamp()


def assert_parent_specs(run_dir, expected_acceptance_sha=None, expected_parallel_sha=None):
    """Re-read parent acceptance + parallel pipeline and ASSERT pinned hashes.

    D5: the expected acceptance-file sha256 MUST be supplied (from the owner's frozen fit
    config) and MUST equal the actual file hash; otherwise fail closed."""
    pa = os.path.join(run_dir, "batch52", "PARENT_DATA_ACCEPTANCE.txt")
    pp = os.path.join(run_dir, "batch52", "PARALLEL_PIPELINE.txt")
    for p in (pa, pp):
        if not os.path.exists(p):
            raise GateError("required parent file missing: %s" % p)
    with open(pa, "r", encoding="utf-8") as f:
        ta = f.read()
    with open(pp, "r", encoding="utf-8") as f:
        tp = f.read()
    if TARGET_COLUMN not in ta:
        raise GateError("PARENT_DATA_ACCEPTANCE does not state target %s" % TARGET_COLUMN)
    if "Log2Norm_HEK293" not in ta or "Log2RPM_HEK293" not in ta:
        raise GateError("target-correction statement not found in PARENT_DATA_ACCEPTANCE")
    if "Nabs" not in tp or "Njoint" not in tp:
        raise GateError("PARALLEL_PIPELINE does not define Nabs/Njoint")
    actual_pa = sha256_file(pa)
    actual_pp = sha256_file(pp)
    if not expected_acceptance_sha:
        raise GateError(
            "D5: expected acceptance-file sha256 is REQUIRED (pass the value pinned in the "
            "owner's frozen fit config); refusing to proceed without it")
    if str(expected_acceptance_sha) != actual_pa:
        raise GateError(
            "D5: acceptance-file sha256 mismatch: expected %s != actual %s (acceptance file "
            "changed; re-freeze the launch config)" % (expected_acceptance_sha, actual_pa))
    if expected_parallel_sha is not None and str(expected_parallel_sha) != actual_pp:
        raise GateError("parallel-pipeline sha256 mismatch: expected %s != actual %s"
                        % (expected_parallel_sha, actual_pp))
    return {"parent_acceptance_sha256": actual_pa,
            "parent_acceptance_sha256_expected": str(expected_acceptance_sha),
            "parent_acceptance_sha256_asserted": True,
            "parallel_pipeline_sha256": actual_pp,
            "parallel_pipeline_sha256_expected":
                None if expected_parallel_sha is None else str(expected_parallel_sha),
            "target_column": TARGET_COLUMN, "target_asserted": True}


def fit_r0(cohort, out_dir, ledger, opt_in=False, accepted_data_sha=None):
    """R0 composition ridge: ONE weighted solve, gated BEFORE the solve. Saved readout +
    normalizer + IDs. R0 endpoint predictions are differenced to form the pair delta."""
    import numpy as np
    from exec_gate import open_ledger, reserve_fit, actual_status
    from r0_baseline import R0Model, features_from_codes
    if not opt_in:
        raise PermissionError("real solve refused: pass --i-understand-this-is-a-real-solve")
    if accepted_data_sha != cohort.source_sha256:
        raise PermissionError("real solve refused: accepted actual-data sha required")
    st = actual_status(ledger)
    if st["remaining"] <= 0:
        raise GateError("exec_gate budget exhausted: %r" % st)
    gate = open_ledger(ledger)
    token = reserve_fit(gate, "R0")
    os.makedirs(out_dir, exist_ok=True)
    try:
        codes = cohort.codes.numpy(); lengths = cohort.lengths.numpy()
        X = features_from_codes(codes, lengths)
        tr = cohort.split_indices("TRAIN").tolist()
        _, wtr = cohort.activity_weights("TRAIN")
        ytr = cohort.targets[cohort.split_indices("TRAIN")].numpy()
        model = R0Model().fit(X[tr], ytr, wtr.numpy())
        pred = model.predict(X)
        preds = {cid: float(pred[i]) for i, cid in enumerate(cohort.coord_ids)}
        atomic_write_json(os.path.join(out_dir, "R0_readout.json"), model.state())
        atomic_write_json(os.path.join(out_dir, "R0_predictions.json"), preds)
        atomic_write_json(os.path.join(out_dir, "R0_ids.json"), {"coord_ids": cohort.coord_ids,
                                                                 "datahash": cohort.datahash})
        gate.complete(token, identity="b57|R0|alpha=%s|data=%s" % (model.alpha, cohort.datahash[:12]))
        return {"arm": "R0", "out_dir": out_dir, "attempt_id": token["exec_id"],
                "remaining_after": actual_status(ledger)["remaining"]}
    except BaseException as e:
        gate.fail(token, e); raise


def _deadline_epoch(day, hh, mm):
    """Asia/Shanghai (+08:00) wall deadline as a POSIX epoch float."""
    import datetime as dt
    d = dt.datetime(int(day[0:4]), int(day[5:7]), int(day[8:10]), hh, mm,
                    tzinfo=dt.timezone(dt.timedelta(hours=8)))
    return d.timestamp()


def run_pipeline(cohort_dir, out_root, ledger, run_dir, opt_in_fits=False, accepted_data_sha=None,
                 device="cuda:0", day=None, first_hhmm=None, second_hhmm=None,
                 source_sha=None, first_model_only=False, expected_acceptance_sha=None,
                 expected_parallel_sha=None):
    """Train R0, then Nabs (with a one-time throughput cap), then Njoint immediately."""
    import time as _t
    from data_adapter import load_cohort
    from spec import SOURCE_SHA256, MAX_EPOCHS
    from epoch_cap import choose_epoch_cap
    from train_neural import Trainer
    from exec_gate import actual_status
    pol = read_run_policy(run_dir)                       # D6: dynamic RUN_POLICY read
    prov = assert_parent_specs(run_dir, expected_acceptance_sha=expected_acceptance_sha,
                               expected_parallel_sha=expected_parallel_sha)
    os.makedirs(out_root, exist_ok=True)
    cohort = load_cohort(cohort_dir, source_sha or SOURCE_SHA256)
    if cohort.target_column != TARGET_COLUMN:
        raise GateError("cohort target column %r != %r" % (cohort.target_column, TARGET_COLUMN))
    day = day or dt.datetime.now().strftime("%Y-%m-%d")
    # Deadlines: explicit hh:mm wins; else use RUN_POLICY (12:00 first-training / 20:00 second).
    if first_hhmm is None:
        first_dl = _epoch_from_iso(pol["first_model_training_target"])
    else:
        first_dl = _deadline_epoch(day, *first_hhmm)
    if second_hhmm is None:
        second_dl = _epoch_from_iso(pol["second_model_overall_deadline"])
    else:
        second_dl = _deadline_epoch(day, *second_hhmm)
    status = {"parent_provenance": prov, "run_policy_sha256": pol["_sha256"],
              "run_policy_budget": pol["total_fit_execution_limit"],
              "cohort_datahash": cohort.datahash, "target_column": cohort.target_column,
              "budget_before": actual_status(ledger), "arms": {}}

    r0 = fit_r0(cohort, os.path.join(out_root, "R0"), ledger, opt_in=opt_in_fits,
                accepted_data_sha=accepted_data_sha)
    status["arms"]["R0"] = r0

    def decider(epoch_seconds):
        now = _t.time()
        remaining = max(0.0, 180 * 60 - epoch_seconds)   # conservative: only epoch1 spent so far
        return choose_epoch_cap(epoch_seconds, now, first_dl, second_dl, remaining_gpu_seconds=remaining)

    tr = Trainer("Nabs", cohort, os.path.join(out_root, "Nabs"), ledger, device=device,
                 epoch_cap=MAX_EPOCHS, opt_in=opt_in_fits, accepted_data_sha=accepted_data_sha)
    nabs = tr.run(cap_decider=decider, cap_log_path=os.path.join(out_root, "epoch_cap_decision.json"))
    status["arms"]["Nabs"] = {"best": nabs["best"], "selected": nabs["selected"],
                              "epoch_cap": nabs["epoch_cap"], "init_sha": nabs["init_sha"],
                              "attempt_id": nabs["attempt_id"], "elapsed_s": nabs["elapsed_s"]}
    from spec import config_sha256
    atomic_write_json(os.path.join(out_root, "Nabs_ready.json"),
                      {"arm": "Nabs", "selected": nabs["selected"], "datahash": cohort.datahash,
                       "config_sha256": config_sha256(), "init_sha": nabs["init_sha"]})
    if first_model_only:
        atomic_write_json(os.path.join(out_root, "coordinator_status.json"), status)
        return status

    cap = nabs["epoch_cap"]
    # D7: Njoint MUST share the Nabs init bytes; assert at runtime before its first gradient.
    tj = Trainer("Njoint", cohort, os.path.join(out_root, "Njoint"), ledger, device=device,
                 epoch_cap=cap, opt_in=opt_in_fits, accepted_data_sha=accepted_data_sha,
                 expected_init_sha=nabs["init_sha"])
    njoint = tj.run()
    status["arms"]["Njoint"] = {"best": njoint["best"], "selected": njoint["selected"],
                                "epoch_cap": njoint["epoch_cap"], "init_sha": njoint["init_sha"],
                                "shared_init_ok": njoint["init_sha"] == nabs["init_sha"],
                                "attempt_id": njoint["attempt_id"], "elapsed_s": njoint["elapsed_s"]}
    status["budget_after"] = actual_status(ledger)
    atomic_write_json(os.path.join(out_root, "coordinator_status.json"), status)
    return status


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="batch53 coordinator (R0 -> Nabs -> Njoint, one GPU)")
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ledger", required=True, help="batch52 exec_gate ledger path (MAX 4)")
    ap.add_argument("--run-dir", default="runs/breakthrough_20261002")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--source-sha", default=None)
    ap.add_argument("--day", default=None)
    ap.add_argument("--first-model-only", action="store_true")
    ap.add_argument("--accepted-data-sha", default=None)
    ap.add_argument("--expected-acceptance-sha", required=True,
                    help="D5: pinned PARENT_DATA_ACCEPTANCE.txt sha256 (freeze at launch)")
    ap.add_argument("--expected-parallel-sha", default=None)
    ap.add_argument("--i-understand-this-is-a-real-fit", action="store_true", dest="opt_in")
    a = ap.parse_args(argv)
    print(json.dumps(run_pipeline(a.cohort_dir, a.out_root, a.ledger, a.run_dir, opt_in_fits=a.opt_in,
                                  accepted_data_sha=a.accepted_data_sha, device=a.device, day=a.day,
                                  first_model_only=a.first_model_only,
                                  expected_acceptance_sha=a.expected_acceptance_sha,
                                  expected_parallel_sha=a.expected_parallel_sha), indent=2, default=str))


if __name__ == "__main__":
    main()
