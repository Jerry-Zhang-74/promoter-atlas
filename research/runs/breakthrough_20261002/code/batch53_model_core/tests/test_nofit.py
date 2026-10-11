# -*- coding: utf-8 -*-
"""ARTIFICIAL CPU NO-FIT SOFTWARE TESTS for batch53 model core.

These are SOFTWARE correctness checks on artificial tensors only. There is ZERO training,
ZERO backward, ZERO optimizer.step, ZERO real-data model score and ZERO GPU/network use.
Run:  python tests/test_nofit.py   (writes tests/no_fit_evidence.json)
"""
from __future__ import annotations
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from spec import SEED, MAX_LEN
from model_core import build_model, param_sha256, encode_bases, SeqCNN
from losses import weighted_smooth_l1, pair_difference_loss, total_loss
from normalizer import TargetNormalizer
from metrics import group_mean_pair_mae, macro_sign_accuracy, spearman, top10_precision

RESULTS = {}


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS[name] = "PASS"
        except Exception as e:
            RESULTS[name] = "FAIL: %r" % (e,)
            raise
        return fn
    return deco


def _codes(lengths, seed=0):
    rng = np.random.default_rng(seed)
    N = len(lengths)
    c = np.full((N, MAX_LEN), 4, dtype=np.int64)
    for i, L in enumerate(lengths):
        c[i, :L] = rng.integers(0, 4, size=L)
    return torch.from_numpy(c)


@check("init_bytes_identical_both_arms")
def _():
    a, b = build_model(SEED), build_model(SEED)
    assert param_sha256(a) == param_sha256(b), "same seed must give same init bytes"


@check("different_seed_differs")
def _():
    a, b = build_model(SEED), build_model(SEED + 1)
    assert param_sha256(a) != param_sha256(b)


@check("onehot_matches_manual")
def _():
    c = torch.tensor([[0, 1, 2, 3, 4, 0]])
    oh, mask = encode_bases(c)
    assert mask.tolist() == [[True, True, True, True, False, True]]
    assert oh[0, 0, :].tolist() == [1, 0, 0, 0, 0, 1]
    assert oh[0, 1, :].tolist() == [0, 1, 0, 0, 0, 0]
    assert oh[:, :, 4].abs().sum().item() == 0.0     # ambiguous -> zero channels


@check("reject_gt600_and_null")
def _():
    for bad in (torch.zeros(1, 601, dtype=torch.long), torch.tensor([[-1]])):
        try:
            encode_bases(bad); raise AssertionError("should have rejected")
        except ValueError:
            pass


@check("masked_max_ignores_padding")
def _():
    m = build_model(SEED).eval()
    c = _codes([120, 200], seed=3)
    base = m(c)
    cp = c.clone(); cp[:, 200:] = 4                  # extend padding of longest to full width
    m2 = build_model(SEED).eval(); m2.load_state_dict(m.state_dict())
    out = m2(cp)
    assert torch.allclose(base, out, atol=1e-5), "padding changed the pooled output"


@check("constant_sequence_invariance_to_trailing_pad")
def _():
    m = build_model(SEED).eval()
    seq = torch.full((1, 100), 0, dtype=torch.long)  # all A
    a = m(seq)
    seq2 = torch.cat([seq, torch.full((1, 300), 4, dtype=torch.long)], dim=1)
    b = m(seq2)
    assert torch.allclose(a, b, atol=1e-5)


@check("weighted_smooth_l1_normalization")
def _():
    p = torch.zeros(4); t = torch.ones(4); w = torch.tensor([1.0, 2.0, 1.0, 2.0])
    v = weighted_smooth_l1(p, t, w)
    w2 = w * 5.0
    v2 = weighted_smooth_l1(p, t, w2)
    assert abs(float(v) - float(v2)) < 1e-9, "weights must be scale invariant (mean 1)"


@check("pair_difference_zero_intercept")
def _():
    ps = torch.tensor([0.5, -0.2]); pl = torch.tensor([0.1, 0.3])
    ys = torch.tensor([1.0, 0.0]); yl = torch.tensor([0.4, 0.1])
    pw = torch.ones(2)
    l1 = pair_difference_loss(ps, pl, ys, yl, pw)
    l2 = pair_difference_loss(ps + 3.0, pl + 3.0, ys, yl, pw)   # common shift cancels
    assert abs(float(l1) - float(l2)) < 1e-6, "pair loss must have no intercept"


@check("normalizer_roundtrip")
def _():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0]); w = torch.ones(4)
    n = TargetNormalizer.fit(y, w)
    z = n.forward(y)
    assert torch.allclose(n.inverse(z), y, atol=1e-5)
    assert abs(float(z.mean())) < 1e-6


@check("metrics_sanity")
def _():
    y = np.array([1.0, 2, 3, 4, 5]); g = ["a", "a", "b", "b", "b"]
    assert abs(group_mean_pair_mae(y, y, g)) < 1e-9
    assert abs(spearman(y, y) - 1.0) < 1e-9
    assert top10_precision(y, y) == 1.0
    dt = np.array([0.1, -0.2, 0.3]); dp = np.array([0.05, -0.1, 0.2]); pg = ["x", "x", "y"]
    assert abs(macro_sign_accuracy(dt, dp, pg) - 1.0) < 1e-9


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


@check("adapter_valid_cohort_loads")
def _():
    from data_adapter import load_cohort
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        info = load_cohort(d, source_sha256=None, validate_only=True)
        assert info["ok"] and info["counts"]["TRAIN"] == 2 and info["pairs"]["TRAIN"] == 1
        coh = load_cohort(d, source_sha256=None)
        assert coh.codes.shape[1] == MAX_LEN and coh.datahash


@check("adapter_rejects_wrong_target_column")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        p = os.path.join(d, "endpoints.tsv")
        s = open(p, encoding="utf-8").read().replace("Log2RPM_HEK293", "Log2Norm_HEK293")
        open(p, "w", encoding="utf-8").write(s)
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("must reject Log2Norm target")
        except AdapterError:
            pass


@check("adapter_rejects_length_mismatch")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        p = os.path.join(d, "sequences.fa")
        s = open(p, encoding="utf-8").read().replace(">c0", ">c0", 1)
        s = s.replace("\n", "\n", 1)
        open(p, "w", encoding="utf-8").write(s.replace("c0\n", "c0\n") + "")
        # truncate c0 by rewriting it shorter
        lines = open(p, encoding="utf-8").read().split("\n")
        idx = lines.index(">c0")
        lines[idx + 1] = lines[idx + 1][:50]
        open(p, "w", encoding="utf-8").write("\n".join(lines))
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("length mismatch must reject")
        except AdapterError:
            pass


@check("adapter_rejects_group_split_leak")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t, splits=("TRAIN", "DEV", "DEV", "DEV"))
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("group spanning splits must reject")
        except AdapterError:
            pass


@check("adapter_rejects_missing_provenance")
def _():
    from data_adapter import load_cohort, AdapterError
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        os.remove(os.path.join(d, "reference_provenance.json"))
        try:
            load_cohort(d, source_sha256=None); raise AssertionError("missing provenance must reject")
        except AdapterError:
            pass


@check("checkpoint_atomic_strict_load_and_mismatch")
def _():
    from checkpoint import save_checkpoint, load_checkpoint, CheckpointMismatch
    from spec import config_sha256
    with tempfile.TemporaryDirectory() as t:
        m = build_model(SEED)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        p = os.path.join(t, "x.ckpt")
        save_checkpoint(p, m, opt, epoch=0, global_step=0, datahash="deadbeef",
                        attempt_id="aid", normalizer_state={})
        obj = load_checkpoint(p, expect_config_sha=config_sha256(), expect_datahash="deadbeef")
        assert obj["gate_attempt_id"] == "aid" and "rng" in obj and "sampler_state" in obj
        try:
            load_checkpoint(p, expect_config_sha="0" * 64); raise AssertionError("config mismatch must raise")
        except CheckpointMismatch:
            pass
        try:
            load_checkpoint(p, expect_datahash="nope"); raise AssertionError("datahash mismatch must raise")
        except CheckpointMismatch:
            pass


@check("exec_gate_budget_max4_and_resume")
def _():
    from exec_gate import open_ledger, BudgetExceeded, MAX_ATTEMPTS
    assert MAX_ATTEMPTS == 4
    with tempfile.TemporaryDirectory() as t:
        g = open_ledger(os.path.join(t, "ledger.jsonl"))
        toks = [g.reserve("R0"), g.reserve("Nabs"), g.reserve("Njoint"), g.reserve("reserve")]
        for tk in toks[:3]:
            g.complete(tk, identity="id-" + tk["exec_id"])
        try:
            g.reserve("extra"); raise AssertionError("must exceed budget at 4")
        except BudgetExceeded:
            pass
        # genuine resume of a completed exec does NOT consume
        base = toks[0]
        rt = g.reserve("R0-resume", resume_of=base["exec_id"], resume_identity="id-" + base["exec_id"])
        assert rt["consumes"] is False


@check("epoch_cap_throughput_only")
def _():
    from epoch_cap import choose_epoch_cap
    fast = choose_epoch_cap(60.0, 0.0, 1e18, 1e18, remaining_gpu_seconds=180 * 60)
    assert fast["chosen_cap"] == 6 and not fast["risk"], fast
    slow = choose_epoch_cap(900.0, 0.0, 1e18, 1e18, remaining_gpu_seconds=180 * 60)
    assert slow["chosen_cap"] == 4, slow
    tight = choose_epoch_cap(3000.0, 0.0, 1.0, 1.0, remaining_gpu_seconds=10)
    assert tight["risk"] and tight["chosen_cap"] is None, tight


@check("gpu_lock_acquire_release_identity")
def _():
    from gpu_lock import GpuLock
    with tempfile.TemporaryDirectory() as t:
        L = GpuLock(t)
        ident = L.acquire("test")
        assert ident["pid"] == os.getpid() and "creation" in ident
        assert L.holder()["pid"] == os.getpid()
        L.release()
        assert not os.path.exists(L.path)


@check("r0_ridge_one_solve_and_delta")
def _():
    from r0_baseline import R0Model, features_from_codes
    c = _codes([120, 200, 130, 210], seed=7).numpy()
    L = np.array([120, 200, 130, 210])
    X = features_from_codes(c, L)
    assert X.shape[1] == 342
    y = np.array([0.1, 0.2, 0.3, 0.4]); w = np.ones(4) / 1.0
    m = R0Model().fit(X, y, w)
    p = m.predict(X)
    assert np.isfinite(p).all() and abs(m.intercept - 0.25) < 1e-6


@check("eval_metrics_and_bootstrap_guard")
def _():
    from data_adapter import load_cohort
    from eval_replay import evaluate_split, fixed_baselines, paired_group_bootstrap
    with tempfile.TemporaryDirectory() as t:
        d = _make_cohort(t)
        coh = load_cohort(d, source_sha256=None)
        preds = {cid: float(i) for i, cid in enumerate(coh.coord_ids)}
        ev = evaluate_split(coh, preds, "DEV")
        assert "activity" in ev
        fb = fixed_baselines(coh, "TRAIN")
        assert "zero_delta" in fb and "always_positive" in fb
        boot = paired_group_bootstrap(coh, "TRAIN", preds, preds)
        assert boot["ok"] is False and boot["n_groups"] < 30


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_fit_evidence.json")
    payload = {"kind": "artificial_cpu_no_fit_software_test",
               "guarantees": ["no training", "no backward", "no optimizer.step", "no gpu", "no network"],
               "results": RESULTS,
               "passed": sum(1 for v in RESULTS.values() if v == "PASS"),
               "failed": [k for k, v in RESULTS.items() if v != "PASS"]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
