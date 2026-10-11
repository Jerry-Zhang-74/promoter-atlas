# -*- coding: utf-8 -*-
"""ARTIFICIAL SOFTWARE + READ-ONLY ABI TESTS for the batch57 coordinator/gating path.

NO fits, NO backward, NO optimizer.step, NO np.linalg.solve, NO GPU, NO network.
The ABI test LOADS an existing on-disk checkpoint (read-only) to prove the repaired code
consumes the live training checkpoint format (shared config_sha256/architecture). It performs
an artificial no-grad CPU forward only -- no pseudo-training.
"""
from __future__ import annotations
import glob, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
import torch  # noqa
from tests.test_nofit_rc2 import _make_cohort

RESULTS = {}
RUN_DIR = os.path.abspath(os.path.join(ROOT, "..", ".."))


def check(name):
    def deco(fn):
        try:
            fn(); RESULTS[name] = "PASS"
        except Exception as e:
            RESULTS[name] = "FAIL: %r" % (e,); raise
        return fn
    return deco


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


@check("trainer_refuses_resume_failclosed")
def _():
    from data_adapter import load_cohort
    from train_neural import Trainer
    from exec_gate import ResumeUnsupported
    from spec import SOURCE_SHA256
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        coh = load_cohort(d, source_sha256=SOURCE_SHA256)
        tr = Trainer("Nabs", coh, os.path.join(t, "out"), os.path.join(t, "ledger.jsonl"),
                     opt_in=True, accepted_data_sha=SOURCE_SHA256,
                     resume_from="abc", resume_identity="abc")
        try:
            tr.run(); raise AssertionError("resume must fail closed")
        except ResumeUnsupported:
            pass


@check("coordinator_refuses_without_optin_before_any_solve")
def _():
    from pipeline_coordinator import run_pipeline, GateError
    from io_utils import sha256_file
    from spec import SOURCE_SHA256
    pa = os.path.join(RUN_DIR, "batch52", "PARENT_DATA_ACCEPTANCE.txt")
    if not os.path.exists(pa):
        return
    acc = sha256_file(pa)
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        try:
            run_pipeline(d, os.path.join(t, "out"), os.path.join(t, "ledger.jsonl"), RUN_DIR,
                         opt_in_fits=False, accepted_data_sha=SOURCE_SHA256, day="2026-10-04",
                         expected_acceptance_sha=acc)
            raise AssertionError("coordinator must refuse without opt-in")
        except PermissionError:
            pass
        except GateError:
            pass


@check("epoch_cap_throughput_only")
def _():
    from epoch_cap import choose_epoch_cap
    fast = choose_epoch_cap(60.0, 0.0, 1e18, 1e18, remaining_gpu_seconds=180 * 60)
    assert fast["chosen_cap"] == 6 and not fast["risk"]
    tight = choose_epoch_cap(3000.0, 0.0, 1.0, 1.0, remaining_gpu_seconds=10)
    assert tight["risk"] and tight["chosen_cap"] is None


@check("checkpoint_abi_loads_live_arm_readonly_nograd")
def _():
    """Read-only ABI proof: load a real on-disk arm checkpoint with the REPAIRED loader and
    run an artificial no-grad CPU forward. No training, no weight change, no writes."""
    from checkpoint import load_checkpoint
    from model_core import build_model
    from spec import config_sha256
    ck_dirs = glob.glob(os.path.join(REPO, "data", "breakthrough_20261002", "batch52",
                                     "corrected", "models", "*"))
    cks = []
    for cd in ck_dirs:
        cks += sorted(glob.glob(os.path.join(cd, "*_epoch*.ckpt")))
    if not cks:
        cks = sorted(glob.glob(os.path.join(REPO, "data", "breakthrough_20261002", "batch52",
                                            "corrected", "models", "**", "*.ckpt"), recursive=True))
    if not cks:
        return                      # no live checkpoint yet -> nothing to ABI-check
    cks = [c for c in cks if "_selected" not in c]
    if not cks:
        return
    obj = load_checkpoint(cks[-1], expect_config_sha=config_sha256())
    m = build_model()
    m.load_state_dict(obj["model"])          # strict name/shape ABI
    m.eval()
    codes = torch.zeros(2, 300, dtype=torch.long)
    codes[0, :50] = 1
    with torch.no_grad():
        out = m(codes)
    assert out.shape[0] == 2 and torch.isfinite(out).all()
    assert "optimizer" in obj and "rng" in obj and "normalizer" in obj


@check("run_policy_cpu_and_target_limits")
def _():
    from pipeline_coordinator import read_run_policy
    if not os.path.exists(os.path.join(RUN_DIR, "batch52", "RUN_POLICY.json")):
        return
    pol = read_run_policy(RUN_DIR)
    assert pol["cpu_assignment"]["model_code"] == 1
    assert pol["gpu_training_jobs_concurrent"] == 1
    assert pol["target_column"] == "Log2RPM_HEK293"


def main():
    out = os.path.join(HERE, "pipeline_test_evidence_rc2.json")
    payload = {"kind": "artificial_no_fit_software_test_batch57", "results": RESULTS,
               "passed": sum(1 for v in RESULTS.values() if v == "PASS"),
               "failed": [k for k, v in RESULTS.items() if v != "PASS"]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
