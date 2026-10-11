# -*- coding: utf-8 -*-
"""BATCH29 step: the FOUR planned fits (1 ridge + 2 true CNN + 1 shuffled CNN) behind the
reviewed batch27 exec_gate.  HARD CAP 4 estimator EXECUTIONS.  Test labels NOT read here."""
from __future__ import annotations
import json, math, os, sys, time, csv, subprocess
from pathlib import Path
import numpy as np

ROOT = Path("E:/promoter_atlas")
CB = ROOT / "runs/breakthrough_20261002/code"
sys.path.insert(0, str(CB / "batch26"))
sys.path.insert(0, str(CB / "batch02"))
import b26_common as C
import b26_metrics as M
import trackB_grouped_cv as T
sys.path.insert(0, str(CB / "batch27"))
from exec_gate import ExecGate, BudgetExceeded

DB = ROOT / "data/breakthrough_20261002/batch29"
REP = ROOT / "reports/breakthrough_20261002/batch29"
RUN = ROOT / "runs/breakthrough_20261002/batch29"
ATT = RUN / "attempts"
LED = RUN / "exec_gate" / "ledger.jsonl"
TELEM = RUN / "gpu_telemetry.jsonl"

LR, WD, BATCH = 0.001, 0.0001, 32
MAX_EPOCHS, PATIENCE, GRAD_CLIP = 20, 5, 1.0
RIDGE_ALPHA = 100
TRUE_SEEDS = (20261101, 20261102)
SHUFFLE_MODEL_SEED = 20261101
PERM_SEED = 20261103
BUDGET = 4

gate = ExecGate(LED, BUDGET)


def load_rows():
    rows = []
    with open(DB / "rows.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def load_encoded():
    z = np.load(DB / "encoded.npz")
    return z["codes"], z["lens"]


def features342(seqs):
    km = {k: {} for k in (1, 2, 3, 4)}
    for k in (1, 2, 3, 4):
        for i in range(4 ** k):
            s = ""; x = i
            for _ in range(k): s = "ACGT"[x % 4] + s; x //= 4
            km[k][s] = i
    off = {1: 0, 2: 4, 3: 20, 4: 84}
    X = np.zeros((len(seqs), 342), dtype=np.float64)
    for r, s in enumerate(seqs):
        n = len(s)
        for k in (1, 2, 3, 4):
            cnt = np.zeros(4 ** k, dtype=np.float64)
            for i in range(n - k + 1):
                j = km[k].get(s[i:i + k])
                if j is not None: cnt[j] += 1.0
            d = n - k + 1
            if d > 0: cnt = cnt / d
            o = off[k]; X[r, o:o + 4 ** k] = cnt
        X[r, 340] = (s.count("G") + s.count("C")) / n if n else 0.0
        X[r, 341] = math.log1p(n)
    return X


def load_assay_seqs():
    d = {}; k = None
    for line in open(DB / "sure29_assay_oriented.fa", encoding="utf-8"):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            p = line[1:].split("_"); k = "%s:%s:%s" % (p[0], p[1], p[2])
        else: d[k] = line
    return d


def nvidia():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        out = "ERR " + repr(e)[:120]
    return out


def proc_start(pid):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-Process -Id %d).StartTime.ToString('o')" % pid],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception:
        return "unknown"


def telem(ev, extra=None):
    rec = {"ts": C.now_local(), "pid": os.getpid(), "proc_start": proc_start(os.getpid()),
           "event": ev, "nvidia": nvidia()}
    if extra: rec.update(extra)
    C.append_jsonl(TELEM, rec)


def preflight():
    import torch
    torch.backends.cudnn.deterministic = True
    C.set_thread_limits(4); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = "cuda:0"
    model = T.build_cnn("legnet").to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, 4, (2, 400), device=dev)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)
    ck = {"model": model.state_dict(), "config": {"dummy": True}}
    tmp = RUN / "_preflight_ckpt.pt.tmp"
    torch.save(ck, tmp); ck2 = torch.load(tmp, map_location="cpu", weights_only=False)
    os.remove(tmp)
    out = {"n_params": nparam, "forward_ok": True, "save_load_ok": True,
           "input_device": str(x.device), "param_device": str(next(model.parameters()).device),
           "torch": torch.__version__, "cuda": torch.version.cuda, "written_at": C.now_local()}
    C.atomic_write_json(REP / "preflight.json", out)
    telem("preflight", {"n_params": nparam})
    print("[preflight]", json.dumps(out), flush=True)
    return out


def run_ridge():
    d = ATT / "E1_ridge"; d.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); seq = load_assay_seqs()
    y = np.array([float(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows], dtype=object)
    loc = np.array([r["group"] for r in rows], dtype=object)
    tr = np.where(sp == "train")[0]; va = np.where(sp == "val")[0]
    X = features342([seq[r["key"]] for r in rows])
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0); sd_x = np.where(sd_x > 0, sd_x, 1.0)
    Z = (X - mu_x) / sd_x
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    zt = (y - mu_y) / sd_y
    A = Z[tr].T @ Z[tr] + RIDGE_ALPHA * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T @ zt[tr])
    pv = Z[va] @ w * sd_y + mu_y
    np.save(d / "ridge_weights.npy", np.concatenate([w, mu_x, sd_x, np.array([mu_y, sd_y])]))
    np.savez(d / "val_pred.npz", idx=va, pred=pv, true=y[va])
    gbm, ng = M.group_balanced_mae(y[va], pv, loc[va])
    res = {"attempt_id": "E1_ridge", "method": "ridge", "n_features": int(X.shape[1]),
           "alpha": RIDGE_ALPHA, "component_fits": 1, "n_train": int(len(tr)), "n_val": int(len(va)),
           "val_spearman": M.spearman(pv, y[va]), "val_mae": M.mae(pv, y[va]),
           "val_group_balanced_mae": gbm, "val_groups": ng, "mu_y": mu_y, "sd_y": sd_y,
           "zero_variance_features": int(np.sum(X[tr].std(0) == 0)),
           "weights_sha": C.sha256_file(d / "ridge_weights.npy"), "finished_at": C.now_local()}
    C.atomic_write_json(d / "result.json", res)
    print("[E1] val_sp=%.4f val_gbMAE=%.4f" % (res["val_spearman"], gbm), flush=True)
    return res


def length_buckets(lens, idx):
    b = {}
    for i in idx: b.setdefault(int(lens[i]), []).append(i)
    return {k: np.array(sorted(v)) for k, v in b.items()}


def predict_all(model, codes, lens, idx, dev, torch):
    model.eval(); out = np.zeros(len(idx)); pos = {v: i for i, v in enumerate(idx)}
    for L, ids in length_buckets(lens, idx).items():
        for i in range(0, len(ids), 64):
            ch = ids[i:i + 64]
            xb = torch.from_numpy(codes[ch].astype(np.int64)).to(dev)
            p = model(xb)[:, 0].detach().cpu().numpy()
            for k, g in enumerate(ch): out[pos[g]] = p[k]
    return out


def train_cnn(labels, seed):
    import torch, random as _r
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    C.set_thread_limits(4); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise SystemExit("CUDA not available")
    dev = "cuda:0"
    rows = load_rows(); codes, lens = load_encoded()
    y = np.array([float(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows], dtype=object)
    loc = np.array([r["group"] for r in rows], dtype=object)
    tr = np.where(sp == "train")[0]; va = np.where(sp == "val")[0]
    cnt = {}
    for i in tr: cnt[loc[i]] = cnt.get(loc[i], 0) + 1
    w = np.array([1.0 / cnt[loc[i]] for i in tr]); w = w / w.mean()
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    z_tr = (y[tr] - mu_y) / sd_y
    pos_tr = {v: k for k, v in enumerate(tr)}
    perm_sha = None
    if labels == "shuffled":
        rs = np.random.RandomState(PERM_SEED); perm = rs.permutation(len(tr))
        perm_sha = C.sha256_bytes(perm.astype(np.int64).tobytes()); z_tr = z_tr[perm]
    C.set_seed(seed)
    model = T.build_cnn("legnet").to(dev)
    assert next(model.parameters()).is_cuda
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)
    nparam = sum(p.numel() for p in model.parameters())
    best = {"gbmae": float("inf"), "epoch": -1, "state": None, "val": None, "opt": None, "sched": None}
    log = []; t0 = time.time(); step = 0
    telem("train_start", {"labels": labels, "seed": seed, "n_train": int(len(tr)), "n_val": int(len(va))})
    for epoch in range(MAX_EPOCHS):
        model.train()
        buckets = length_buckets(lens, tr)
        rs = np.random.RandomState((seed * 1000003 + epoch) % (2 ** 31 - 1))
        order = []
        for L in sorted(buckets):
            ids = buckets[L].copy(); rs.shuffle(ids)
            for i in range(0, len(ids), BATCH): order.append(ids[i:i + BATCH])
        tot = 0.0
        for b in order:
            if len(b) == 1:
                j = b[0]; b = np.array([j, j])
                zb = np.array([z_tr[pos_tr[j]], z_tr[pos_tr[j]]])
                wb = np.array([w[pos_tr[j]] * 0.5, w[pos_tr[j]] * 0.5])
            else:
                zb = z_tr[np.array([pos_tr[j] for j in b])]
                wb = np.array([w[pos_tr[j]] for j in b])
            xb = torch.from_numpy(codes[b].astype(np.int64)).to(dev)
            assert xb.is_cuda
            ztb = torch.from_numpy(zb.astype(np.float32)).to(dev)
            wbt = torch.from_numpy(wb.astype(np.float32)).to(dev)
            pred = model(xb)[:, 0]
            loss = (((pred - ztb) ** 2) * wbt).sum() / wbt.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step(); step += 1; tot += float(loss.detach())
        pv = predict_all(model, codes, lens, va, dev, torch) * sd_y + mu_y
        gbmae, ng = M.group_balanced_mae(y[va], pv, loc[va])
        log.append({"epoch": epoch, "train_loss": tot / max(len(order), 1),
                    "val_spearman": M.spearman(pv, y[va]), "val_mae": M.mae(pv, y[va]),
                    "val_group_balanced_mae": gbmae, "elapsed_s": round(time.time() - t0, 1)})
        print("[%s seed=%d] ep=%d loss=%.4f val_sp=%.4f val_gbMAE=%.4f" %
              (labels, seed, epoch, log[-1]["train_loss"], log[-1]["val_spearman"], gbmae), flush=True)
        if epoch == 0: telem("train_mid", {"labels": labels, "seed": seed, "epoch": 0})
        if gbmae < best["gbmae"] - 1e-12:
            best = {"gbmae": gbmae, "epoch": epoch, "val": log[-1],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict()}
        if epoch - best["epoch"] >= PATIENCE: break
    return {"rows": rows, "codes": codes, "lens": lens, "y": y, "sp": sp, "loc": loc, "tr": tr, "va": va,
            "mu_y": mu_y, "sd_y": sd_y, "best": best, "log": log,
            "nparam": nparam, "step": step, "perm_sha": perm_sha, "elapsed_s": round(time.time() - t0, 1)}


def run_cnn(labels, seed, aid):
    import torch, random as _r
    d = ATT / aid; d.mkdir(parents=True, exist_ok=True)
    out = train_cnn(labels, seed)
    dev = "cuda:0"
    model = T.build_cnn("legnet").to(dev)
    model.load_state_dict(out["best"]["state"]); model.eval()
    va = out["va"]
    pv = predict_all(model, out["codes"], out["lens"], va, dev, torch) * out["sd_y"] + out["mu_y"]
    np.savez(d / "val_pred.npz", idx=va, pred=pv, true=out["y"][va])
    ckpt = {"model": out["best"]["state"], "optimizer": out["best"]["opt"], "scheduler": out["best"]["sched"],
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": _r.getstate(),
            "epoch": out["best"]["epoch"], "global_step": out["step"],
            "best_val_group_balanced_mae": out["best"]["gbmae"], "best_val": out["best"]["val"],
            "seed": seed, "labels": labels,
            "config": {"lr": LR, "wd": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                       "grad_clip": GRAD_CLIP, "arch": "LegNetInspired(1,ch=64)", "batch_policy": "equal-exact-length",
                       "mu_y": out["mu_y"], "sd_y": out["sd_y"], "perm_seed": PERM_SEED, "perm_sha": out["perm_sha"]}}
    tmp = str(d / "ckpt.pt.tmp"); torch.save(ckpt, tmp); os.replace(tmp, d / "ckpt.pt")
    res = {"attempt_id": aid, "labels": labels, "seed": seed, "component_fits": 1, "n_params": out["nparam"],
           "epochs_run": len(out["log"]), "best_epoch": out["best"]["epoch"],
           "best_val_group_balanced_mae": out["best"]["gbmae"], "best_val_spearman": out["best"]["val"]["val_spearman"],
           "best_val_mae": out["best"]["val"]["val_mae"], "mu_y": out["mu_y"], "sd_y": out["sd_y"],
           "n_train": int(len(out["tr"])), "n_val": int(len(out["va"])), "perm_sha": out["perm_sha"],
           "elapsed_s": out["elapsed_s"], "train_pid": os.getpid(), "device": dev,
           "log": out["log"], "ckpt_sha": C.sha256_file(d / "ckpt.pt"), "finished_at": C.now_local()}
    C.atomic_write_json(d / "result.json", res)
    telem("train_end", {"attempt_id": aid, "best_epoch": out["best"]["epoch"],
                        "best_val_gbmae": out["best"]["gbmae"], "elapsed_s": out["elapsed_s"]})
    print("[%s] best_ep=%d val_gbMAE=%.4f epochs=%d t=%.1fs" %
          (aid, out["best"]["epoch"], out["best"]["gbmae"], len(out["log"]), out["elapsed_s"]), flush=True)
    return res


def run_select():
    res = {}
    for aid in ("E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102", "E4_cnn_shuffle_seed20261101"):
        p = ATT / aid / "result.json"
        if p.is_file(): res[aid] = C.read_json(p)
    cands = [(a, r) for a, r in res.items() if a.startswith("E2_") or a.startswith("E3_")]
    if len(cands) < 2:
        raise SystemExit("both true-seed fits required before selection")
    cands.sort(key=lambda x: (x[1]["best_val_group_balanced_mae"], int(x[1]["seed"])))
    primary = cands[0]
    man = {"batch": "batch29", "selected_primary": primary[0], "selected_seed": primary[1]["seed"],
           "criterion": "lowest validation group-balanced MAE (proxy y units); ties -> lower seed",
           "ranking": [{"attempt_id": a, "seed": r["seed"], "val_group_balanced_mae": r["best_val_group_balanced_mae"],
                        "val_spearman": r["best_val_spearman"]} for a, r in cands],
           "both_true_seeds_reported": True,
           "shuffle_val_group_balanced_mae": res.get("E4_cnn_shuffle_seed20261101", {}).get("best_val_group_balanced_mae"),
           "baseline_E1_val_group_balanced_mae": res.get("E1_ridge", {}).get("val_group_balanced_mae"),
           "test_targets_opened": False, "written_at": C.now_local()}
    C.atomic_write_json(REP / "selection_manifest.json", man)
    print("[select] primary=%s" % primary[0], flush=True)
    return man


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "preflight":
        preflight()
    elif mode == "e1":
        gate.run("E1_ridge", run_ridge, identity="ridge342_alpha100")
    elif mode == "e2":
        gate.run("E2_cnn_true_seed20261101", lambda: run_cnn("true", 20261101, "E2_cnn_true_seed20261101"), identity="legnet64_seed20261101_true")
    elif mode == "e3":
        gate.run("E3_cnn_true_seed20261102", lambda: run_cnn("true", 20261102, "E3_cnn_true_seed20261102"), identity="legnet64_seed20261102_true")
    elif mode == "e4":
        gate.run("E4_cnn_shuffle_seed20261101", lambda: run_cnn("shuffled", 20261101, "E4_cnn_shuffle_seed20261101"), identity="legnet64_seed20261101_shuffle")
    elif mode == "select":
        run_select()
    elif mode == "status":
        print(json.dumps(gate.status(), indent=2))
