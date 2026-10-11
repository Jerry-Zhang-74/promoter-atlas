# -*- coding: utf-8 -*-
"""batch30 ZERO-GRADIENT preflight.  NO estimator fit, NO optimizer step, NO gradient.
Structural readiness evidence ONLY (NOT empirical equivalence of training continuation).

T1 native-length forward-hook widths (cuda:0, eval + no_grad): exact width == L for
   train/val/infer paths, plus- and minus-strand cached inputs; no code-4 at network input.
T2 storage-padding invariance: same row with ALTERED DISCARDED padding -> identical native
   prediction (native slice).  (We do NOT require raw padded output to differ at init.)
T3 checkpoint identity: init model + EMPTY optimizer roundtrip; SYNTHETIC standalone populated
   fixture (clearly dummy) + RNG/cursor/earlystop roundtrip; mutation after capture cannot
   alter the snapshot.
T4 execution gate: failure+retry consumes 2; max-4 rejects the 5th BEFORE its callback runs;
   cross-process lock is exclusive.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C
import b30_data as D
import b30_ckpt as K
import b30_model as MOD
from b30_exec_gate import ExecGate, BudgetExceeded

DEV = "cuda:0"
TESTS = []
T0 = time.time()


def _rec(tid, ok, **kw):
    r = {"id": tid, "ok": bool(ok)}
    r.update(kw)
    TESTS.append(r)
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", tid, json.dumps(kw)[:300]), flush=True)
    return ok


def probe(model, codes, lens, idx, device, torch):
    cap = {}
    h1 = model.stem[0].register_forward_pre_hook(lambda m, inp: cap.update({"stem_in": tuple(inp[0].shape)}))
    h2 = model.register_forward_pre_hook(lambda m, inp: cap.update({"model_in": tuple(inp[0].shape)}))
    try:
        with torch.no_grad():
            t, L = D.native_batch(codes, lens, idx, device, torch)
            y = model(t)
    finally:
        h1.remove(); h2.remove()
    return cap, L, t, y


def main():
    import torch
    torch.backends.cudnn.deterministic = True
    C.set_thread_limits(4); torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA required (no CPU fallback permitted)"
    device = DEV
    DATA = C.DATA30 if (C.DATA30 / "encoded.npz").exists() else C.DATA29
    SPLIT = (C.DATA30 / "split.tsv") if (C.DATA30 / "split.tsv").exists() else (C.REP29 / "split_manifest.tsv")
    codes, lens = (lambda z: (z["codes"], z["lens"]))(np.load(DATA / "encoded.npz"))
    man = {}
    with open(SPLIT, encoding="utf-8") as f:
        next(f)
        for line in f:
            k, ch, st, en, sd, g, b = line.rstrip("\n").split("\t")
            man[k] = (ch, int(st), int(en), sd, g, b)
    rows = []
    for i, k in enumerate(man.keys()):
        rows.append(k)
    # need lens aligned to encoded.npz row order = rows.tsv order
    rrows = D.load_rows(DATA / "rows.tsv")
    key2i = {r["key"]: i for i, r in enumerate(rrows)}
    by_len = {}
    for r in rrows:
        by_len.setdefault(int(r["len"]), []).append(r)
    lens_sorted = sorted(by_len)
    Lmin, Lmed, Lmax = lens_sorted[0], lens_sorted[len(lens_sorted) // 2], lens_sorted[-1]

    model = MOD.build_cnn("legnet").to(device)
    nparam = MOD.count_params(model)
    model.eval()

    # ---- T1 native width at forward hook ----
    checks = []
    for tag, L in (("min", Lmin), ("median", Lmed), ("max", Lmax)):
        cands = by_len[L]
        for sd in ("+", "-"):
            sub = [r for r in cands if r["strand"] == sd]
            if not sub:
                continue
            i = key2i[sub[0]["key"]]
            cap, LL, t, y = probe(model, codes, lens, np.array([i]), device, torch)
            okw = (LL == L) and (cap.get("model_in", (None, None))[-1] == L) and \
                  (cap.get("stem_in", (None, None))[-1] == L) and (int(t.min()) >= 0) and (int(t.max()) <= 3)
            checks.append({"bucket": tag, "L": L, "strand": sd, "key": sub[0]["key"],
                           "model_in_width": cap.get("model_in", [None])[-1],
                           "tensor_width": int(t.shape[-1]), "ok": bool(okw)})
    _rec("T1_native_width_hooks", all(c["ok"] for c in checks) and len(checks) >= 4,
         device=str(device), n_params=nparam, checks=checks, torch=torch.__version__)

    # ---- T2 storage-padding invariance ----
    nrows, maxlen = codes.shape
    padrow = None
    for r in rrows:
        i = key2i[r["key"]]
        if int(r["len"]) < maxlen:
            padrow = (i, int(r["len"])); break
    ca = codes.copy()
    cb = codes.copy()
    i, Lr = padrow
    cb[i, Lr:] = 3                       # all-T garbage in DISCARDED storage
    with torch.no_grad():
        ta, _ = D.native_batch(ca, lens, np.array([i]), device, torch)
        tb, _ = D.native_batch(cb, lens, np.array([i]), device, torch)
        ya = model(ta)[:, 0].item(); yb = model(tb)[:, 0].item()
    _rec("T2_padding_invariance", (ya == yb) and (int(ca[i, Lr:].max()) == 4) and (int(cb[i, Lr:].min()) == 3),
         key=rrows[i]["key"], L=Lr, maxlen=maxlen, pad_orig_code=int(ca[i, Lr:].max()) if Lr < maxlen else None,
         pred_orig=ya, pred_altpad=yb, abs_diff=abs(ya - yb))

    # ---- T3 checkpoint identity / mutation-proof snapshot ----
    m2 = MOD.build_cnn("legnet").to("cpu")
    opt = torch.optim.AdamW(m2.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)
    hs = {"config": "cfg", "data_rows": "dr", "data_codes": "dc", "split": "sp"}
    early = {"best_gbmae": 1.0, "best_epoch": 0, "epochs_no_improve": 0, "best_val": None}
    snap = K.capture_state(m2, opt, sched, epoch=0, next_batch_cursor=0, global_step=0,
                           early=early, cfg={"dummy": True}, hashes=hs, exec_id="EXEC")
    digest_before = K.state_digest(snap)
    # empty optimizer must stay empty (no fake moments)
    empty_ok = all(len(v) == 0 for v in snap["optimizer"]["state"].values())
    # mutate live objects AFTER capture; snapshot must not change
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(1.0)
    snap2 = K.capture_state(m2, opt, sched, epoch=0, next_batch_cursor=0, global_step=0,
                            early=early, cfg={"dummy": True}, hashes=hs, exec_id="EXEC")
    frozen_ok = (K.state_digest(snap) == digest_before) and (K.state_digest(snap2) != digest_before)
    # roundtrip: fresh model restores original params bitwise
    m3 = MOD.build_cnn("legnet").to("cpu")
    o3 = torch.optim.AdamW(m3.parameters(), lr=0.001)
    s3 = torch.optim.lr_scheduler.LambdaLR(o3, lr_lambda=lambda e: 1.0)
    K.restore_state(snap, m3, o3, s3)
    same = all(torch.equal(snap["model"][k], m3.state_dict()[k]) for k in snap["model"])
    _rec("T3a_init_state_roundtrip", bool(empty_ok and frozen_ok and same),
         empty_optimizer=bool(empty_ok), snapshot_frozen=bool(frozen_ok), restored_bitwise=bool(same))

    # SYNTHETIC populated fixture (clearly dummy data, NOT derived from any fit)
    m4 = MOD.build_cnn("legnet").to("cpu")
    o4 = torch.optim.AdamW(m4.parameters(), lr=0.001)
    for p in m4.parameters():
        p.data = p.data * 0.0 + 0.123
    for pg in o4.param_groups:
        for p in pg["params"]:
            st = o4.state[p]
            st["step"] = torch.tensor(7.0)
            st["exp_avg"] = torch.full_like(p, 0.5)      # DUMMY moments, not produced by training
            st["exp_avg_sq"] = torch.full_like(p, 0.25)
    import random as _r
    _r.seed(7)
    syn = K.capture_state(m4, o4, s3, epoch=3, next_batch_cursor=5, global_step=42,
                          early={"best_gbmae": 0.9, "best_epoch": 3, "epochs_no_improve": 1, "best_val": {"v": 1}},
                          cfg={"dummy": "synthetic"}, hashes=hs, exec_id="EXEC")
    sd_before = K.state_digest(syn)
    with torch.no_grad():
        for p in m4.parameters():
            p.mul_(0.0)
        for st in o4.state.values():
            st["exp_avg"].zero_()
    froze = K.state_digest(syn) == sd_before
    m5 = MOD.build_cnn("legnet").to("cpu"); o5 = torch.optim.AdamW(m5.parameters(), lr=0.001); sp5 = torch.optim.lr_scheduler.LambdaLR(o5, lr_lambda=lambda e: 1.0)
    ep, cur, step, e2 = K.restore_state(syn, m5, o5, sp5)
    syn_digest_after = K.state_digest(K.capture_state(m5, o5, sp5, epoch=ep, next_batch_cursor=cur,
                                                      global_step=step, early=e2, cfg={"dummy": "synthetic"},
                                                      hashes=hs, exec_id="EXEC"))
    _rec("T3b_synthetic_fixture_roundtrip", bool(froze and (ep, cur, step) == (3, 5, 42) and syn_digest_after == sd_before),
         clearly_dummy=True, epoch=ep, cursor=cur, step=step, snapshot_frozen=bool(froze),
         restored_digest_matches=bool(syn_digest_after == sd_before))

    # ---- T4 execution gate (mock) ----
    led = C.RUN30 / "exec_gate_mock" / "ledger.jsonl"
    if led.exists():
        led.unlink()
    gate = ExecGate(led, 4)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("mock failure")

    def okrun():
        calls["n"] += 1
        return "ok"

    g1 = None
    try:
        gate.run("mock_fail", boom)
    except RuntimeError:
        g1 = "raised"
    g2 = gate.run("mock_retry", okrun)          # same identity? no -> new execution
    st_after = gate.status()
    retry_ok = (g1 == "raised") and (g2 == "ok") and (st_after["consumed"] == 2)
    # fill to budget, then the 5th must be rejected BEFORE its callback runs
    gate.run("m3", okrun); gate.run("m4", okrun)
    pre_calls = calls["n"]
    rejected_before_callback = False
    try:
        gate.run("m5", okrun)
    except BudgetExceeded:
        rejected_before_callback = (calls["n"] == pre_calls)
    st_final = gate.status()
    # lock exclusivity
    lockp = str(led) + ".lock"
    gate._acquire()
    locked = os.path.exists(lockp)
    relock_failed = False
    try:
        os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        relock_failed = True
    gate._release()
    released = not os.path.exists(lockp)
    _rec("T4_exec_gate_mock", bool(retry_ok and rejected_before_callback and st_final["remaining"] == 0
                                   and locked and relock_failed and released),
         failure_then_retry_consumed=st_after["consumed"], callback_before_callback_guard=bool(rejected_before_callback),
         remaining_after_5th=st_final["remaining"], lock_exclusive=bool(locked and relock_failed and released))

    out = {"batch": "batch30", "kind": "zero_gradient_preflight", "device": DEV,
           "gpu_used": True, "gradients_taken": False, "optimizer_steps": 0, "estimator_fits": 0,
           "n_tests": len(TESTS), "n_pass": sum(1 for t in TESTS if t["ok"]),
           "n_fail": sum(1 for t in TESTS if not t["ok"]),
           "tests": TESTS, "elapsed_s": round(time.time() - T0, 1), "written_at": C.now_local(),
           "note": "Structural readiness only; NOT empirical equivalence of training continuation."}
    C.atomic_write_json(C.REP30 / "tests.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "tests"}, indent=2))


if __name__ == "__main__":
    main()
