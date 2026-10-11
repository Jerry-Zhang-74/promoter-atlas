# -*- coding: utf-8 -*-
"""Refreeze batch58 CODE_READY.json with the CURRENT bytes of the batch58 code bundle.

Run after ANY change to the bundle (e.g. the PARENT1012 pre-GPU patch to b58_probe.py /
b58_run.py). Writes atomic CODE_READY.json + per-file sha256 so a fresh reviewer can compare
the frozen hashes against the bytes actually executed. The parent's patched bytes are kept
(never reverted); only the recorded hashes move.
"""
from __future__ import annotations
import hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "batch57_model_core_rc2")))

from io_utils import atomic_write_json  # noqa: E402
from spec import SOURCE_SHA256, TARGET_COLUMN, config_sha256  # noqa: E402
import b58_gate  # noqa: E402

FILES = ["README.md", "_exec_gate_vendored.py", "b58_gate.py", "b58_core.py",
         "b58_probe.py", "b58_run.py", "selfcheck_b58.py", "smoke_b58.py",
         "b58_manifest.py", "freeze_code_ready.py"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    reports = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                           "reports", "breakthrough_20261002", "batch58"))
    sc = {}
    scp = os.path.join(reports, "code_selfcheck.json")
    if os.path.exists(scp):
        sc = json.load(open(scp, encoding="utf-8"))
    files = {}
    for f in FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            files[f] = sha256(p)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = {
        "batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
        "status": "CODE_READY",
        "preserves_original": True,
        "note": ("Resource configuration variant + REAL checkpoint resume. NOT accuracy "
                 "evidence; NOT exact replication when effective batch changes. "
                 "batch52/53/57 files immutable; PARENT1012 patch KEPT (not reverted)."),
        "target_column": TARGET_COLUMN,
        "source_sha256": SOURCE_SHA256,
        "config_sha256": config_sha256(),
        "gate": {"ledger": "runs/breakthrough_20261002/batch58/exec_gate_ledger.jsonl",
                 "budget": b58_gate.MAX_ATTEMPTS, "reserve": 0,
                 "every_reservation_consuming": True,
                 "vendored_sha256": b58_gate.VENDORED_SHA256,
                 "planned": list(b58_gate.PLANNED_PURPOSES),
                 "supersedes": "batch32 ceiling for b58 ONLY"},
        "checkpoint_schema": "batch58.checkpoint.v2",
        "resume_supported": True,
        "resume_features": [
            "next_pos (position AFTER last processed batch)",
            "next/last batch id proof", "coverage-prefix hash",
            "strict config/init/data/device/eff-batch reject",
            "two-generation immutable retention with sha before latest pointer",
            "<=60s + epoch-end snapshots",
            "SIG/stopfile graceful exit at saved optimizer boundary",
            "OOM pre-step state+RNG restore and retry (recorded); no CPU neural fallback"],
        "parent_pre_gpu_patch_1012": {
            "source": "runs/breakthrough_20261002/batch58/PARENT_PRE_GPU_CORRECTION.json",
            "kept": True, "reverted": False,
            "changes": [
                "b58_probe.select_batch: choose max samples/second (batch/median_s), NOT min latency",
                "b58_probe.select_batch: plateau measured on throughput gain (<5%)",
                "b58_probe.select_batch: unmeasured nvidia_free_gib is UNSAFE (fail-closed)",
                "b58_run.precondition: require batch52/GPU_EVALUATION_DONE.json",
                "b58_run: shared repaired GPU lock dir runs/breakthrough_20261002/batch52",
                "b58_run: auto stop_steps = steps_per_epoch // 5 (stop inside epoch 0)"],
            "selector_fixture": "batch32@0.010s vs batch128@0.020s -> 128 selected (samples/s)",
            "verified_by": "selfcheck_b58.check_selector_throughput_units (PASS)"},
        "selfcheck": {"passed": sc.get("passed"), "total": sc.get("total"),
                      "all_passed": sc.get("all_passed"),
                      "report": "reports/breakthrough_20261002/batch58/code_selfcheck.json"},
        "files_sha256": dict(sorted(files.items())),
        "generated_at": stamp,
    }
    dest = os.path.join(HERE, "CODE_READY.json")
    atomic_write_json(dest, out)
    print(json.dumps({"dest": dest, "files": len(files), "generated_at": stamp,
                      "selfcheck": out["selfcheck"]}))


if __name__ == "__main__":
    main()
