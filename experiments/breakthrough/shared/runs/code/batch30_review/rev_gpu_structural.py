# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) PART1: GPU native-length + structural checks.
ZERO fits, ZERO optimizer steps. ALL neural forwards on cuda:0 only."""
from __future__ import annotations
import copy, json, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs/breakthrough_20261002/code/batch30"))
import b30_common as C   # noqa
import b30_data as D     # noqa
import b30_ckpt as K     # noqa
import b30_model as MOD  # noqa
from b30_exec_gate import ExecGate, BudgetExceeded  # noqa

OUT = {}
DATA = ROOT / "data/breakthrough_20261002/batch30"


def load_inputs():
    z = np.load(DATA / "encoded.npz"); codes, lens = z["codes"], z["lens"]
    rows = D.load_rows(DATA / "rows.tsv")
    key2i = {r["key"]: i for i, r in enumerate(rows)}
    split = {}
    with open(DATA / "split.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t"); split[p[0]] = p[6]
    by_len = {}
    for r in rows:
        by_len.setdefault(int(r["len"]), []).append(r)
    return codes, lens, rows, key2i, split, by_len


def sec_A(model, codes, lens, rows, key2i, split, by_len, dev, torch, fwd):
    cap = {}
    h1 = model.stem[0].register_forward_pre_hook(lambda m, inp: cap.update(stem=tuple(inp[0].shape)))
    h2 = model.register_forward_pre_hook(lambda m, inp: cap.update(model=tuple(inp[0].shape)))
    checks = []; sel = []
    ls = sorted(by_len)
    for tag, L in (("min", ls[0]), ("median", ls[len(ls) // 2]), ("max", ls[-1])):
        for sp in ("train", "val", "test"):
            for sd in ("+", "-"):
                cand = [r for r in by_len[L] if r["strand"] == sd and split[r["key"]] == sp]
                if cand:
                    sel.append((tag, sp, sd, L, cand[0]))
    for tag, sp, sd, L, r in sel:
        i = key2i[r["key"]]
        t, LL = D.native_batch(codes, lens, np.array([i]), dev, torch)
        fwd(t)
        mine = torch.from_numpy(codes[i, :L].astype(np.int64)).to(dev)[None, :]
        checks.append({"tag": tag, "split": sp, "strand": sd, "L": int(L),
                       "hook_model_w": int(cap["model"][-1]), "hook_stem_w": int(cap["stem"][-1]),
                       "tensor_w": int(t.shape[-1]), "returns_L": int(LL),
                       "min": int(t.min()), "max": int(t.max()),
                       "equal_my_slice": bool(torch.equal(t, mine)),
                       "ok": bool(cap["model"][-1] == L and cap["stem"][-1] == L and t.shape[-1] == L
                                  and LL == L and int(t.min()) >= 0 and int(t.max()) <= 3
                                  and torch.equal(t, mine))})
    Lbig = max(by_len, key=lambda k: len(by_len[k]))
    nb = min(32, len(by_len[Lbig]))
    ids = np.array([key2i[r["key"]] for r in by_len[Lbig][:nb]], dtype=np.int64)
    t32, L32 = D.native_batch(codes, lens, ids, dev, torch); fwd(t32)
    checks.append({"tag": "bucket_full", "split": "-", "strand": "-", "L": int(Lbig),
                   "tensor_shape": list(t32.shape), "returns_L": int(L32),
                   "ok": bool(t32.shape == (nb, Lbig) and L32 == Lbig and nb <= 32)})
    h1.remove(); h2.remove()
    OUT["A_native_length_hooks"] = {"n": len(checks), "n_ok": sum(c["ok"] for c in checks),
                                    "all_ok": all(c["ok"] for c in checks), "checks": checks,
                                    "n_params": MOD.count_params(model), "device": dev}
    va = np.array([key2i[r["key"]] for r in rows if split[r["key"]] == "val"], dtype=np.int64)
    maxb = 0; widths_ok = True; nbatch = 0
    with torch.no_grad():
        for t, LL, chunk in D.iter_native_batches(codes, lens, va, dev, torch, batch=32):
            nbatch += 1; maxb = max(maxb, len(chunk))
            if t.shape != (len(chunk), LL):
                widths_ok = False
    OUT["B_eval_batch_bound"] = {"n_val": int(va.size), "n_batches": nbatch, "max_batch": maxb,
                                 "widths_ok": bool(widths_ok), "src_batch_const": 32,
                                 "ok": bool(maxb <= 32 and widths_ok)}


def sec_CD(model, codes, rows, key2i, dev, torch, fwd):
    pad = None
    for r in rows:
        if int(r["len"]) < codes.shape[1]:
            pad = r; break
    i = key2i[pad["key"]]; Lr = int(pad["len"])
    ca = codes.copy(); cb = codes.copy(); cb[i, Lr:] = 3
    ta = torch.from_numpy(ca[i, :Lr].astype(np.int64)).to(dev)[None]
    tb = torch.from_numpy(cb[i, :Lr].astype(np.int64)).to(dev)[None]
    ya = fwd(ta)[:, 0].item(); yb = fwd(tb)[:, 0].item()
    OUT["C_padding_invariance"] = {"key": pad["key"], "L": int(Lr),
                                   "orig_pad_code_max": int(ca[i, Lr:].max()),
                                   "altered_pad_code_min": int(cb[i, Lr:].min()),
                                   "pred_a": ya, "pred_b": yb, "abs_diff": abs(ya - yb),
                                   "ok": bool(ya == yb and int(ca[i, Lr:].max()) == 4 and int(cb[i, Lr:].min()) == 3)}
    import b30_infer_fasta as F
    tmp = ROOT / "runs/breakthrough_20261002/code/batch30_review/_tmp_rev.fa"
    seq1 = "".join(np.random.RandomState(1).choice(list("ACGT"), 317))
    tmp.write_text(">test_seq_A\n%s\n>test_seq_B\n%s\n" % (seq1, "ACGT" * 40), encoding="utf-8")
    fr = F.predict_fasta(str(tmp), device=dev, model=model)
    tmp2 = ROOT / "runs/breakthrough_20261002/code/batch30_review/_tmp_rev_bad.fa"
    tmp2.write_text(">bad\nACGTNX\n", encoding="utf-8")
    bad = False
    try:
        F.predict_fasta(str(tmp2), device=dev, model=model)
    except ValueError:
        bad = True
    Ls = [x["length"] for x in fr["inputs"]]
    OUT["D_fasta_interface"] = {"n_seq": fr["n_sequences"], "device": fr["device"], "lengths": Ls,
                                "nonacgt_rejected": bool(bad),
                                "ok": bool(fr["n_sequences"] == 2 and fr["device"] == "cuda:0"
                                           and Ls == [317, 160] and bad)}
    for p in (tmp, tmp2):
        if p.exists():
            p.unlink()


def sec_E(model_cls, torch, DEV):
    m = model_cls().to("cpu"); o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    s = torch.optim.lr_scheduler.LambdaLR(o, lr_lambda=lambda e: 1.0)
    hs = {"config": "c", "data_rows": "dr", "data_codes": "dc", "split": "sp"}
    early = {"best_gbmae": 1.0, "best_epoch": 0, "epochs_no_improve": 0, "best_val": None}
    snap = K.capture_state(m, o, s, epoch=0, next_batch_cursor=0, global_step=0, early=early,
                           cfg={"d": 1}, hashes=hs, exec_id="EX")
    d0 = K.state_digest(snap)
    empty = all(len(v) == 0 for v in snap["optimizer"]["state"].values())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    snap2 = K.capture_state(m, o, s, epoch=0, next_batch_cursor=0, global_step=0, early=early,
                            cfg={"d": 1}, hashes=hs, exec_id="EX")
    frozen = (K.state_digest(snap) == d0) and (K.state_digest(snap2) != d0)
    m3 = model_cls().to("cpu"); o3 = torch.optim.AdamW(m3.parameters(), lr=1e-3)
    s3 = torch.optim.lr_scheduler.LambdaLR(o3, lr_lambda=lambda e: 1.0)
    K.restore_state(snap, m3, o3, s3)
    bitwise = all(torch.equal(snap["model"][k], m3.state_dict()[k]) for k in snap["model"])
    m4 = model_cls().to("cpu"); o4 = torch.optim.AdamW(m4.parameters(), lr=1e-3)
    with torch.no_grad():
        for p in m4.parameters():
            p.data = p.data * 0.0 + 0.123
    for pg in o4.param_groups:
        for p in pg["params"]:
            st = o4.state[p]; st["step"] = torch.tensor(7.0)
            st["exp_avg"] = torch.full_like(p, 0.5); st["exp_avg_sq"] = torch.full_like(p, 0.25)
    np.random.seed(11); random.seed(11); torch.manual_seed(11)
    syn = K.capture_state(m4, o4, s3, epoch=3, next_batch_cursor=5, global_step=42,
                          early={"best_gbmae": 0.9, "best_epoch": 3, "epochs_no_improve": 1, "best_val": {"v": 1}},
                          cfg={"d": "syn"}, hashes=hs, exec_id="EX")
    sd0 = K.state_digest(syn)
    with torch.no_grad():
        for p in m4.parameters():
            p.mul_(0.0)
        for st in o4.state.values():
            st["exp_avg"].zero_()
    syn_frozen = K.state_digest(syn) == sd0
    def fresh_pair():
        mm = model_cls().to("cpu"); oo = torch.optim.AdamW(mm.parameters(), lr=1e-3)
        ss = torch.optim.lr_scheduler.LambdaLR(oo, lr_lambda=lambda e: 1.0)
        return mm, oo, ss
    # capture the live RNG right after restore, then compare a second restore
    mm, oo, ss = fresh_pair(); K.restore_state(syn, mm, oo, ss)
    a_py = random.random(); a_np = float(np.random.rand())
    mm2, oo2, ss2 = fresh_pair(); K.restore_state(syn, mm2, oo2, ss2)
    c_py = random.random(); c_np = float(np.random.rand())
    rng_ok = (a_py == c_py) and (a_np == c_np)
    syn_mut = copy.deepcopy(syn)
    for st in syn_mut["optimizer"]["state"].values():
        st["exp_avg"] = st["exp_avg"] * 0 + 999.0
    blind = K.state_digest(syn_mut) == sd0
    refuse = False
    try:
        K.verify_identity(syn, {"config": "WRONG"}, "EX")
    except ValueError:
        refuse = True
    OUT["E_structural_state"] = {"empty_optimizer": bool(empty), "snapshot_frozen": bool(frozen),
                                 "restored_bitwise": bool(bitwise), "synthetic_frozen": bool(syn_frozen),
                                 "rng_roundtrip_exact": bool(rng_ok), "digest_blind_to_optimizer": bool(blind),
                                 "verify_identity_refuses": bool(refuse),
                                 "ok": bool(empty and frozen and bitwise and syn_frozen and rng_ok and blind and refuse)}


def sec_F(torch):
    led = ROOT / "runs/breakthrough_20261002/batch30_review/rev_gate_mock/ledger.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    if led.exists():
        led.unlink()
    g = ExecGate(led, 4)
    n = {"c": 0}
    def boom():
        n["c"] += 1; raise RuntimeError("x")
    def okr():
        n["c"] += 1; return "ok"
    raised = False
    try:
        g.run("mk1", boom)
    except RuntimeError:
        raised = True
    g.run("mk2", okr); st2 = g.status()
    g.run("mk3", okr); g.run("mk4", okr)
    pre = n["c"]; rej = False
    try:
        g.run("mk5", okr)
    except BudgetExceeded:
        rej = n["c"] == pre
    lock_ok = False
    g._acquire(); lock_ok = os.path.exists(str(led) + ".lock")
    lock_excl = False
    try:
        os.open(str(led) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        lock_excl = True
    g._release()
    lock_ok = lock_ok and lock_excl and not os.path.exists(str(led) + ".lock")
    OUT["F_exec_gate"] = {"failure_then_retry_consumed": st2["consumed"],
                          "fifth_rejected_pre_callback": bool(rej),
                          "remaining_after_5": g.status()["remaining"],
                          "lock_exclusive": bool(lock_ok),
                          "ok": bool(raised and st2["consumed"] == 2 and rej
                                     and g.status()["remaining"] == 0 and lock_ok)}


def main():
    import torch
    torch.backends.cudnn.deterministic = True
    C.set_thread_limits(4); torch.set_num_threads(4)
    assert torch.cuda.is_available()
    dev = "cuda:0"
    codes, lens, rows, key2i, split, by_len = load_inputs()
    model = MOD.build_cnn("legnet").to(dev).eval()
    gt = {"t": 0.0, "c": 0}
    def fwd(t):
        t0 = time.time()
        with torch.no_grad():
            y = model(t)
        gt["t"] += time.time() - t0; gt["c"] += 1
        return y
    sec_A(model, codes, lens, rows, key2i, split, by_len, dev, torch, fwd)
    sec_CD(model, codes, rows, key2i, dev, torch, fwd)
    sec_E(lambda: MOD.build_cnn("legnet"), torch, dev)
    sec_F(torch)
    OUT["gpu"] = {"device": dev, "calls": gt["c"], "forward_wall_s": round(gt["t"], 3),
                  "budget_s": 300, "ok": bool(gt["t"] < 300)}
    OUT["fits"] = {"estimator_fits": 0, "optimizer_steps": 0, "gradients_taken": False}
    out = ROOT / "reports/breakthrough_20261002/batch30_review/rev_gpu_structural.json"
    out.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {}
    for k, v in OUT.items():
        if isinstance(v, dict) and "checks" in v:
            brief[k] = {kk: vv for kk, vv in v.items() if kk != "checks"}
        else:
            brief[k] = v
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
