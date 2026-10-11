# -*- coding: utf-8 -*-
"""batch58 artificial SELF-CHECK: serialization / resume-verification / gate / retention.

ZERO gradients, ZERO optimizer steps, ZERO real data, ZERO GPU. CUDA is hidden
(CUDA_VISIBLE_DEVICES="") before importing torch so no CUDA context is ever created and the
live trainer's GPU is never touched. A tiny in-memory fake cohort supplies only the fields
the checkpoint/verify code reads. This proves the resume mechanics, not science.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONUTF8"] = "1"

import hashlib, json, sys, tempfile, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.abspath(os.path.join(BASE, "..", "batch57_model_core_rc2")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(2)

import b58_core  # noqa: E402
import b58_gate  # noqa: E402
from b58_core import B58Trainer, ResumeRejected, _expected_epoch_coverage  # noqa: E402
from checkpoint import capture_rng, restore_rng  # noqa: E402
from model_core import build_model, param_sha256  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from spec import SEED, LR, WEIGHT_DECAY, SOURCE_SHA256, TARGET_COLUMN, config_sha256  # noqa: E402
from train_neural import epoch_order  # noqa: E402

CHECKS = []


def check(name, fn):
    try:
        fn()
        CHECKS.append({"name": name, "ok": True})
        print("PASS", name)
    except Exception as e:
        CHECKS.append({"name": name, "ok": False, "error": "%s: %s" % (type(e).__name__, e),
                       "trace": traceback.format_exc()[-800:]})
        print("FAIL", name, "->", e)


class FakeCohort:
    def __init__(self, n=300):
        self.datahash = "a" * 64
        self.source_sha256 = SOURCE_SHA256
        self.target_column = TARGET_COLUMN
        self.coord_ids = ["c%d" % i for i in range(n)]
        self.group_ids = ["g%d" % (i // 3) for i in range(n)]
        self.splits = ["TRAIN"] * n
        self.pairs = {}

    def activity_weights(self, split="TRAIN"):
        return (torch.arange(len(self.coord_ids)), torch.ones(len(self.coord_ids), dtype=torch.float64))


def _new_trainer(tmp, eff=64, n=300):
    fc = FakeCohort(n)
    tr = B58Trainer(fc, tmp, os.path.join(tmp, "ledger.jsonl"), device="cpu", effective_batch=eff,
                    epoch_cap=2, micro_batch=eff, opt_in=True, accepted_data_sha=SOURCE_SHA256,
                    run_id="SelfTest", expected_init_sha=None)
    tr.n = n
    tr.opt_steps = 0
    tr.global_samples = 0
    tr.epoch_samples = 0
    tr.pair_ptr = 0
    tr.pair_repeats = 0
    tr._perm = None
    tr._perm_epoch = None
    return tr, fc


def _mk(tmp):
    model = build_model(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    norm = TargetNormalizer(0.0, 1.0)
    return model, opt, norm, param_sha256(model)


def _cov_prefix(perm, upto):
    h = hashlib.sha256(); h.update(b"b58-epoch-coverage|")
    h.update(np.asarray(perm[0:upto], dtype=np.int64).tobytes())
    return h.hexdigest()


def check_gate_cap3():
    tmp = tempfile.mkdtemp(prefix="b58gate_")
    led = os.path.join(tmp, "ledger.jsonl")
    st = b58_gate.actual_status(led)
    assert st["budget"] == 3 and st["consumed"] == 0, st
    g = b58_gate.open_ledger(led)
    toks = [b58_gate.reserve(g, p) for p in ("b58-profile", "b58-Nabs-throughput-variant",
                                             "b58-resume-continuation")]
    assert all(t["consumes"] for t in toks), "every b58 reservation MUST consume (no free resume)"
    st = b58_gate.actual_status(led)
    assert st["consumed"] == 3 and st["remaining"] == 0, st
    try:
        b58_gate.reserve(g, "b58-overflow")
        raise AssertionError("4th reservation must raise BudgetExceeded")
    except b58_gate.BudgetExceeded:
        pass


def check_save_verify_roundtrip():
    tmp = tempfile.mkdtemp(prefix="b58rt_")
    tr, fc = _new_trainer(tmp)
    model, opt, norm, init_sha = _mk(tmp)
    perm = tr._perm_of(0)
    nxt = 128
    tr.opt_steps = 2; tr.global_samples = 128; tr.epoch_samples = 128
    tr._save_generation(model, opt, norm, init_sha, 0, nxt, [], {"unset": True},
                        _cov_prefix(perm, nxt), perm[tr._prev_batch_start(nxt):nxt],
                        perm[nxt:nxt + tr.eff], attempt_id="tok1")
    ptr, obj = tr._load_latest(os.path.join(tmp, "latest.json"))
    assert obj["format"] == b58_core.FORMAT and obj["next_pos"] == nxt
    tr2, _ = _new_trainer(tmp + "_b")
    m2, o2, n2, is2 = _mk(tmp + "_b")
    e, p = tr2._verify_and_restore(ptr, obj, m2, o2, n2, is2)
    assert (e, p) == (0, nxt), (e, p)
    assert tr2._perm_slice(e, p, tr2.eff) == list(map(int, obj["next_batch_ids"]))
    assert obj["opt_steps"] == 2 and obj["global_samples"] == 128


def check_tamper_nextbatch_reject():
    tmp = tempfile.mkdtemp(prefix="b58tm_")
    tr, fc = _new_trainer(tmp)
    model, opt, norm, init_sha = _mk(tmp)
    perm = tr._perm_of(0)
    nxt = 128
    tr.opt_steps = 2
    tr._save_generation(model, opt, norm, init_sha, 0, nxt, [], {}, _cov_prefix(perm, nxt),
                        perm[tr._prev_batch_start(nxt):nxt], perm[nxt:nxt + tr.eff], attempt_id="t")
    tr2, _ = _new_trainer(tmp + "_b")
    m2, o2, n2, is2 = _mk(tmp + "_b")
    ptr, obj = tr._load_latest(os.path.join(tmp, "latest.json"))
    obj["next_batch_ids"] = [999, 998]
    try:
        tr2._verify_and_restore(ptr, obj, m2, o2, n2, is2)
        raise AssertionError("tampered next_batch_ids must be rejected")
    except ResumeRejected:
        pass
    ptr2, obj2 = tr._load_latest(os.path.join(tmp, "latest.json"))
    obj2["coverage_hash"] = "0" * 64
    try:
        tr2._verify_and_restore(ptr2, obj2, m2, o2, n2, is2)
        raise AssertionError("tampered coverage hash must be rejected")
    except ResumeRejected:
        pass


def check_old_schema_reject():
    tmp = tempfile.mkdtemp(prefix="b58old_")
    tr, fc = _new_trainer(tmp)
    model, opt, norm, init_sha = _mk(tmp)
    old = {"format": "batch53.checkpoint.v1", "config_sha256": config_sha256(),
           "datahash": tr.cohort.datahash, "source_sha256": SOURCE_SHA256,
           "target_column": TARGET_COLUMN, "effective_batch": tr.eff, "seed": SEED,
           "device_is_cuda": False, "init_sha": init_sha, "epoch": 0,
           "sampler_state": {"epoch": 0, "pos": 128}, "extra": {"mid": True},
           "model": model.state_dict(), "optimizer": opt.state_dict(),
           "normalizer": norm.state(), "rng": capture_rng()}
    p = os.path.join(tmp, "old.ckpt")
    torch.save(old, p)
    tr2, _ = _new_trainer(tmp + "_b")
    m2, o2, n2, is2 = _mk(tmp + "_b")
    try:
        tr2._verify_and_restore({"path": p}, old, m2, o2, n2, is2)
        raise AssertionError("old b57-schema mid checkpoint must be rejected")
    except ResumeRejected:
        pass


def check_two_generation_retention():
    tmp = tempfile.mkdtemp(prefix="b58ret_")
    tr, fc = _new_trainer(tmp)
    model, opt, norm, init_sha = _mk(tmp)
    perm = tr._perm_of(0)
    for k, nxt in enumerate([64, 128, 192]):
        ls = tr._prev_batch_start(nxt)
        tr.opt_steps = k + 1
        tr._save_generation(model, opt, norm, init_sha, 0, nxt, [], {}, _cov_prefix(perm, nxt),
                            perm[ls:nxt], perm[nxt:nxt + tr.eff], attempt_id="t")
    cks = [f for f in os.listdir(tmp) if f.startswith("SelfTest_gen") and f.endswith(".ckpt")]
    assert len(cks) == 2, "must keep exactly TWO generations, got %r" % cks
    ptr = json.load(open(os.path.join(tmp, "latest.json"), encoding="utf-8"))
    assert os.path.basename(ptr["path"]) in cks


def check_coverage_hash():
    tr, _ = _new_trainer(tempfile.mkdtemp(prefix="b58cov_"), eff=64, n=300)
    perm = tr._perm_of(0)
    exp = _expected_epoch_coverage(perm)
    h = hashlib.sha256(); h.update(b"b58-epoch-coverage|")
    for bi in range(0, tr.n, tr.eff):
        h.update(np.asarray(perm[bi:bi + tr.eff], dtype=np.int64).tobytes())
    assert h.hexdigest() == exp, "full-pass coverage hash must match"
    h2 = hashlib.sha256(); h2.update(b"b58-epoch-coverage|")
    for bi in range(0, tr.n, tr.eff):
        if bi == 64:
            continue
        h2.update(np.asarray(perm[bi:bi + tr.eff], dtype=np.int64).tobytes())
    assert h2.hexdigest() != exp, "a skipped batch MUST change the coverage hash"


def check_rng_determinism():
    import random
    st = capture_rng()
    restore_rng(st)
    a = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())
    restore_rng(st)
    b = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())
    assert a == b, "RNG restore must be exact"


def check_snapshot_restore():
    tr, _ = _new_trainer(tempfile.mkdtemp(prefix="b58snap_"))
    model, opt, norm, init_sha = _mk(tr.out_dir)
    before = param_sha256(model)
    snap = tr._snapshot_pre_step(model, opt)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert param_sha256(model) != before
    tr._restore_pre_step(model, opt, snap)
    assert param_sha256(model) == before, "pre-step restore must return identical model bytes"


def check_selector_throughput_units():
    """PARENT1012 selector fixture: batch32 @0.010s (latency 0.010) vs batch128 @0.020s
    (latency 0.020). Highest THROUGHPUT (samples/s) must win -> 128, even though its per-step
    latency is WORSE. Also, a candidate with UNMEASURED memory must be treated as UNSAFE
    (fail-closed), never selected silently."""
    import b58_probe
    results = [
        {"batch": 32, "ok": True, "median_s": 0.010, "nvidia_free_gib": 3.0, "nvidia_used_gib": 2.0},
        {"batch": 128, "ok": True, "median_s": 0.020, "nvidia_free_gib": 1.2, "nvidia_used_gib": 6.9},
    ]
    sel = b58_probe.select_batch(results)
    assert sel["selected_batch"] == 128, "samples/s selection must pick 128, got %r" % sel
    assert sel["selected"]["batch"] / sel["selected"]["median_s"] == 6400.0
    # latency-only rule would have picked 32; prove we are NOT doing that
    assert min(results, key=lambda r: r["median_s"])["batch"] == 32
    # fail-closed memory: the only measured candidate has no memory reading -> UNSAFE
    unsafe = [{"batch": 64, "ok": True, "median_s": 0.005, "nvidia_free_gib": None,
               "nvidia_used_gib": None}]
    assert b58_probe.select_batch(unsafe)["selected_batch"] is None, "unmeasured memory must fail closed"
    # OOM candidate never selected
    withoom = [{"batch": 256, "ok": False, "oom": True, "nvidia_free_gib": 2.0},
               {"batch": 32, "ok": True, "median_s": 0.010, "nvidia_free_gib": 3.0, "nvidia_used_gib": 2.0}]
    assert b58_probe.select_batch(withoom)["selected_batch"] == 32


def main():
    for name, fn in [("gate_cap3_consuming", check_gate_cap3),
                     ("selector_throughput_units", check_selector_throughput_units),
                     ("save_verify_roundtrip_next_pos", check_save_verify_roundtrip),
                     ("tamper_reject", check_tamper_nextbatch_reject),
                     ("old_schema_reject", check_old_schema_reject),
                     ("two_generation_retention", check_two_generation_retention),
                     ("coverage_full_pass", check_coverage_hash),
                     ("rng_determinism", check_rng_determinism),
                     ("snapshot_restore_noop", check_snapshot_restore)]:
        check(name, fn)
    npass = sum(1 for c in CHECKS if c["ok"])
    out = {"batch": "batch58", "suite": "selfcheck_b58", "passed": npass, "total": len(CHECKS),
           "all_passed": npass == len(CHECKS), "checks": CHECKS,
           "artificial_fixture": True, "real_data": 0, "gpu": 0, "backward": 0,
           "optimizer_step": 0, "accepted_data": 0,
           "note": "serialization/resume/gate checks only; CUDA hidden; no fit"}
    dest = os.path.abspath(os.path.join(BASE, "..", "..", "..", "..", "reports",
                                        "breakthrough_20261002", "batch58", "code_selfcheck.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": npass, "total": len(CHECKS), "dest": dest}))
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
