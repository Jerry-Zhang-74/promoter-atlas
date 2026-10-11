# -*- coding: utf-8 -*-
"""ARTIFICIAL CPU NO-FIT SOFTWARE TESTS for the batch57 repaired model core.

SOFTWARE checks on artificial tensors only: ZERO training, ZERO backward, ZERO optimizer.step,
ZERO real-data model score, ZERO GPU/network. This file deliberately does NOT call
np.linalg.solve (the batch53 tests/test_nofit.py did, via R0Model.fit).
Run:  python tests/test_nofit_rc2.py   (writes tests/no_fit_evidence_rc2.json)
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import torch

from spec import SEED, MAX_LEN
from model_core import build_model, param_sha256, encode_bases
from losses import weighted_smooth_l1, pair_difference_loss
from normalizer import TargetNormalizer
from metrics import top10_precision
from gpu_lock import GpuLock, _proc_creation
import train_neural

RESULTS = {}


def check(name):
    def deco(fn):
        try:
            fn(); RESULTS[name] = "PASS"
        except Exception as e:
            RESULTS[name] = "FAIL: %r" % (e,)
            raise
        return fn
    return deco


def _codes(lengths, seed=0):
    rng = np.random.default_rng(seed)
    c = np.full((len(lengths), MAX_LEN), 4, dtype=np.int64)
    for i, L in enumerate(lengths):
        c[i, :L] = rng.integers(0, 4, size=L)
    return torch.from_numpy(c)


def _make_cohort(d, lengths=(120, 200, 130, 210), splits=("TRAIN", "TRAIN", "DEV", "DEV")):
    os.makedirs(d, exist_ok=True)
    gs = ["g1", "g1", "g2", "g2"]
    with open(os.path.join(d, "endpoints.tsv"), "w", encoding="utf-8") as f:
        f.write("coord_id\tunion_id\tsplit\tlength\tLog2RPM_HEK293\n")
        for i, (L, s, g) in enumerate(zip(lengths, splits, gs)):
            f.write("c%d\t%s\t%s\t%d\t%.4f\n" % (i, g, s, L, 0.1 * i))
    with open(os.path.join(d, "groups.tsv"), "w", encoding="utf-8") as f:
        f.write("union_id\tsplit\n")
        for g, s in (("g1", "TRAIN"), ("g2", "DEV")):
            f.write("%s\t%s\n" % (g, s))
    with open(os.path.join(d, "pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("union_id\tsplit\tshort_coord_id\tlong_coord_id\tshort_len\tlong_len\n")
        f.write("g1\tTRAIN\tc0\tc1\t120\t200\n")
    rng = np.random.default_rng(1)
    with open(os.path.join(d, "sequences.fa"), "w", encoding="utf-8") as f:
        for i, L in enumerate(lengths):
            f.write(">c%d\n%s\n" % (i, "".join(rng.choice(list("ACGT"), size=L))))
    with open(os.path.join(d, "reference_provenance.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"assembly": "hg19", "proxy_not_donor_clone": True}))
    return d


@check("init_bytes_identical_both_arms")
def _():
    assert param_sha256(build_model(SEED)) == param_sha256(build_model(SEED))


@check("onehot_and_mask")
def _():
    c = torch.tensor([[0, 1, 2, 3, 4, 0]])
    oh, mask = encode_bases(c)
    assert mask.tolist() == [[True, True, True, True, False, True]]
    assert oh[:, :, 4].abs().sum().item() == 0.0


@check("masked_max_ignores_padding")
def _():
    m = build_model(SEED).eval()
    c = _codes([120, 200], seed=3)
    base = m(c)
    cp = c.clone(); cp[:, 200:] = 4
    m2 = build_model(SEED).eval(); m2.load_state_dict(m.state_dict())
    assert torch.allclose(base, m2(cp), atol=1e-5)


@check("weighted_smooth_l1_scale_invariant")
def _():
    p = torch.zeros(4); t = torch.ones(4); w = torch.tensor([1.0, 2.0, 1.0, 2.0])
    assert abs(float(weighted_smooth_l1(p, t, w)) - float(weighted_smooth_l1(p, t, w * 5.0))) < 1e-9


@check("pair_difference_zero_intercept")
def _():
    ps = torch.tensor([0.5, -0.2]); pl = torch.tensor([0.1, 0.3])
    ys = torch.tensor([1.0, 0.0]); yl = torch.tensor([0.4, 0.1]); pw = torch.ones(2)
    assert abs(float(pair_difference_loss(ps, pl, ys, yl, pw))
               - float(pair_difference_loss(ps + 3.0, pl + 3.0, ys, yl, pw))) < 1e-6


@check("normalizer_roundtrip")
def _():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n = TargetNormalizer.fit(y, torch.ones(4))
    assert torch.allclose(n.inverse(n.forward(y)), y, atol=1e-5)


@check("D1_proc_creation_is_stable")
def _():
    c1 = _proc_creation(os.getpid())
    x = 0
    for i in range(200000):
        x += i * i
    c2 = _proc_creation(os.getpid())
    assert c1 is not None and c1 == c2, "process creation time changed within one process"


@check("D1_live_holder_is_not_reclaimed_twoprocess")
def _():
    child = (
        "import sys, time, json\n"
        "from gpu_lock import GpuLock\n"
        "L = GpuLock(sys.argv[1]); ident = L.acquire('child-hold')\n"
        "print(json.dumps(ident), flush=True)\n"
        "time.sleep(float(sys.argv[2]))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory() as t:
        p = subprocess.Popen([sys.executable, "-X", "utf8", "-c", child, t, "2.5"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            line = p.stdout.readline().strip()
            assert line, "child failed to acquire: %s" % p.stderr.read()
            child_ident = json.loads(line)
            cpid = child_ident["pid"]
            assert isinstance(cpid, int) and cpid > 0
            assert child_ident.get("creation") is not None
            L = GpuLock(t)
            try:
                L.acquire("parent", timeout_s=1.0)
                raise AssertionError("parent reclaimed a lock held by a LIVE child")
            except TimeoutError:
                pass
            h = L.holder()
            assert h and h["pid"] == cpid and h["creation"] == child_ident["creation"]
            p.wait(timeout=20)
            for _ in range(120):
                if _proc_creation(cpid) is None:
                    break
                time.sleep(0.1)
        finally:
            try:
                p.kill()
            except Exception:
                pass
        ident = L.acquire("parent", timeout_s=10)
        assert ident["pid"] == os.getpid()
        L.release()
        assert not os.path.exists(L.path)


@check("D2_resume_unsupported_always_consumes")
def _():
    from exec_gate import open_ledger, reserve_fit, ResumeUnsupported, actual_status, MAX_ATTEMPTS
    with tempfile.TemporaryDirectory() as t:
        led = os.path.join(t, "ledger.jsonl")
        g = open_ledger(led)
        tok = reserve_fit(g, "R0")
        g.complete(tok, identity="id0")
        try:
            reserve_fit(g, "R0", resume_from=tok["exec_id"], resume_identity="id0")
            raise AssertionError("resume must be refused (fail-closed)")
        except ResumeUnsupported:
            pass
        st = actual_status(led)
        assert st["consumed"] == 1 and st["remaining"] == MAX_ATTEMPTS - 1 and st["reserve"] == 0
        assert st["resume_supported"] is False


@check("D6_actual_status_real_ledger")
def _():
    from exec_gate import actual_status
    real = os.path.abspath(os.path.join(ROOT, "..", "..", "batch52", "exec_gate_ledger.jsonl"))
    if os.path.exists(real):
        st = actual_status(real)
        assert st["reserve"] == 0 and st["remaining"] == max(0, st["budget"] - st["consumed"])


@check("D3_best_baseline_is_max")
def _():
    import eval_replay, inspect
    src = inspect.getsource(eval_replay.final_compare)
    assert "best_base = max(" in src, "D3: best baseline must use max()"
    assert "best_base = min(" not in src, "D3: min() baseline must be gone"


@check("D8_top10_stable_ties")
def _():
    y = np.array([1.0, 1.0, 0.0, 0.0]); p = np.array([1.0, 1.0, 0.0, 0.0])
    ids = ["b", "a", "d", "c"]
    assert top10_precision(y, p, ids) == 1.0


@check("D7_shared_init_assertion")
def _():
    from model_core import assert_same_init
    a, b = build_model(SEED), build_model(SEED)
    assert assert_same_init(a, b) == param_sha256(a)
    m = build_model(SEED); m.head2.bias.data += 1.0
    try:
        assert_same_init(a, m); raise AssertionError("perturbed init must differ")
    except AssertionError as e:
        assert "differ" in str(e)


@check("D10_pair_order_deterministic_and_shuffled")
def _():
    o1 = train_neural.pair_order(50, 0)
    o2 = train_neural.pair_order(50, 0)
    assert torch.equal(o1, o2), "pair order must be deterministic for a fixed epoch"
    assert not torch.equal(o1, train_neural.pair_order(50, 1)), "epoch must change the order"
    assert sorted(o1.tolist()) == list(range(50)), "must be a permutation"
    assert o1.tolist() != list(range(50)), "must actually shuffle"


@check("D9_trainer_records_opt_steps")
def _():
    import inspect
    src = inspect.getsource(train_neural.Trainer.run)
    assert "opt_steps += 1" in src and "global_step=opt_steps" in src


@check("D11_selected_checkpoint_sidecar")
def _():
    from checkpoint import save_checkpoint, select_checkpoint, load_checkpoint
    from spec import config_sha256
    with tempfile.TemporaryDirectory() as t:
        m = build_model(SEED); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        src = os.path.join(t, "e0.ckpt")
        save_checkpoint(src, m, opt, epoch=0, global_step=7, datahash="dh", attempt_id="a0",
                        normalizer_state={})
        dst = os.path.join(t, "sel.ckpt")
        side = select_checkpoint(src, dst, attempt_id="a0", datahash="dh")
        assert os.path.exists(dst + ".sha256.json")
        obj = load_checkpoint(dst, expect_config_sha=config_sha256(), expect_datahash="dh")
        assert obj["global_step"] == 7 and side["selected"] is True


@check("D12_rejects_ragged_row")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        with open(os.path.join(d, "endpoints.tsv"), "a", encoding="utf-8") as f:
            f.write("cX\tg1\tTRAIN\t120\n")
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("ragged row must reject")
        except AdapterError:
            pass


@check("D12_rejects_ambiguous_by_default")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        fa = os.path.join(d, "sequences.fa")
        lines = open(fa, encoding="utf-8").read().split("\n")
        idx = lines.index(">c0")
        lines[idx + 1] = "N" + lines[idx + 1][1:]
        open(fa, "w", encoding="utf-8").write("\n".join(lines))
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("ambiguous must reject by default")
        except AdapterError:
            pass
        coh = load_cohort(d, source_sha256=None, allow_ambiguous=True)
        assert coh.lengths[0].item() == 119.0


@check("D12_npz_byte_ids_decode")
def _():
    from data_adapter import load_cohort
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        os.remove(os.path.join(d, "sequences.fa"))
        rng = np.random.default_rng(2)
        ids = np.array([b"c0", b"c1", b"c2", b"c3"])
        codes = np.full((4, 210), 4, dtype=np.uint8)
        for i, L in enumerate((120, 200, 130, 210)):
            codes[i, :L] = rng.integers(0, 4, size=L)
        np.savez(os.path.join(d, "sequences.npz"), coord_ids=ids, codes=codes)
        info = load_cohort(d, source_sha256=None, validate_only=True)
        assert info["ok"] and info["n_endpoints"] == 4


@check("pipeline_parent_sha_required_and_asserted")
def _():
    from pipeline_coordinator import assert_parent_specs, GateError
    from io_utils import sha256_file
    run_dir = os.path.abspath(os.path.join(ROOT, "..", ".."))
    pa = os.path.join(run_dir, "batch52", "PARENT_DATA_ACCEPTANCE.txt")
    if not os.path.exists(pa):
        return
    real = sha256_file(pa)
    prov = assert_parent_specs(run_dir, expected_acceptance_sha=real)
    assert prov["parent_acceptance_sha256_asserted"] and prov["parent_acceptance_sha256"] == real
    for bad in ("0" * 64, None):
        try:
            assert_parent_specs(run_dir, expected_acceptance_sha=bad)
            raise AssertionError("bad acceptance sha must fail")
        except GateError:
            pass


@check("pipeline_read_run_policy")
def _():
    from pipeline_coordinator import read_run_policy
    run_dir = os.path.abspath(os.path.join(ROOT, "..", ".."))
    if not os.path.exists(os.path.join(run_dir, "batch52", "RUN_POLICY.json")):
        return
    pol = read_run_policy(run_dir)
    assert pol["total_fit_execution_limit"] == 4
    assert pol["first_model_training_target"].endswith("12:00:00+08:00")


def main():
    out = os.path.join(HERE, "no_fit_evidence_rc2.json")
    payload = {"kind": "artificial_cpu_no_fit_software_test_batch57",
               "guarantees": ["no training", "no backward", "no optimizer.step",
                              "no np.linalg.solve", "no gpu", "no network"],
               "results": RESULTS,
               "passed": sum(1 for v in RESULTS.values() if v == "PASS"),
               "failed": [k for k, v in RESULTS.items() if v != "PASS"]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
