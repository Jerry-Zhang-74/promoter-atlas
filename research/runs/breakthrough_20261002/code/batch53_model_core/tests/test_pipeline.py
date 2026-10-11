# -*- coding: utf-8 -*-
"""ARTIFICIAL SOFTWARE TESTS for the coordinator/gating path. NO fits, NO GPU, NO network."""
from __future__ import annotations
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa
from tests.test_nofit import _make_cohort

RESULTS = {}
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # runs/breakthrough_20261002


def check(name):
    def deco(fn):
        try:
            fn(); RESULTS[name] = "PASS"
        except Exception as e:
            RESULTS[name] = "FAIL: %r" % (e,); raise
        return fn
    return deco


@check("parent_specs_reread_and_target_asserted")
def _():
    from pipeline_coordinator import assert_parent_specs
    prov = assert_parent_specs(RUN_DIR)
    assert prov["target_asserted"] and prov["target_column"] == "Log2RPM_HEK293"
    assert len(prov["parent_acceptance_sha256"]) == 64


@check("trainer_refuses_real_fit_without_optin")
def _():
    from data_adapter import load_cohort
    from train_neural import Trainer
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        coh = load_cohort(d, source_sha256=None)
        tr = Trainer("Nabs", coh, os.path.join(t, "out"), os.path.join(t, "ledger.jsonl"),
                     opt_in=False, accepted_data_sha=None)
        try:
            tr.run(); raise AssertionError("must refuse without opt-in")
        except PermissionError:
            pass


@check("coordinator_refuses_without_optin_before_any_solve")
def _():
    from pipeline_coordinator import run_pipeline
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        try:
            run_pipeline(d, os.path.join(t, "out"), os.path.join(t, "ledger.jsonl"), RUN_DIR,
                         opt_in_fits=False, accepted_data_sha=None, day="2026-10-04")
            raise AssertionError("coordinator must refuse without opt-in")
        except PermissionError:
            pass


@check("epoch_cap_logged_once_structure")
def _():
    from epoch_cap import choose_epoch_cap, log_epoch_cap
    with tempfile.TemporaryDirectory() as t:
        dec = choose_epoch_cap(60.0, 0.0, 1e18, 1e18, remaining_gpu_seconds=180 * 60)
        p = log_epoch_cap(os.path.join(t, "cap.json"), dec, 60.0)
        obj = json.load(open(p, encoding="utf-8"))
        assert obj["chosen_cap"] == 6 and "logged_at" in obj


def main():
    out = os.path.join(HERE, "pipeline_test_evidence.json")
    payload = {"kind": "artificial_no_fit_software_test", "results": RESULTS,
               "passed": sum(1 for v in RESULTS.values() if v == "PASS"),
               "failed": [k for k, v in RESULTS.items() if v != "PASS"]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
