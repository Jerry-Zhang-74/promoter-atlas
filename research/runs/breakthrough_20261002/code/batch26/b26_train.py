# -*- coding: utf-8 -*-
"""batch26 step 2: the FOUR planned full fits (1 ridge + 2 true CNN + 1 shuffled CNN).

Hard bounds: total estimator fits <= 4.  Every attempt is pre-registered in
runs/batch26/attempts/_ledger.jsonl before compute; each attempt has an
immutable per-attempt directory.  Test targets are NOT read here.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C
import b26_metrics as M

BATCH02_CODE = C.BATCH02_CODE
sys.path.insert(0, str(BATCH02_CODE))

LR, WD, BATCH = 0.001, 0.0001, 32
MAX_EPOCHS, PATIENCE, GRAD_CLIP = 25, 5, 1.0
RIDGE_ALPHA = 100
TRUE_SEEDS = (20261026, 20261027)
SHUFFLE_SEED = 20261028
ATTEMPTS = C.RUN26 / "attempts"
LEDGER = ATTEMPTS / "_ledger.jsonl"
PREREG = ATTEMPTS / "_preregistered.json"


def load_rows():
    import csv
    rows = []
    with open(C.DATA26 / "str_variant_rows.tsv", "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            rows.append(r)
    return rows


def ledger_counts():
    n = 0
    if LEDGER.is_file():
        for line in open(LEDGER, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if d.get("kind") == "full_fit":
                    n += 1
    return n


def preregister():
    if PREREG.is_file():
        return C.read_json(PREREG)
    exp = {
        "batch": "batch26", "version": "v6.19.0-strvariant.rc1",
        "total_estimator_fits_max": 4,
        "attempts": [
            {"attempt_id": "A0_ridgeC", "kind": "classical", "method": "ridge", "alpha": RIDGE_ALPHA},
            {"attempt_id": "A1_cnn_true_seed20261026", "kind": "neural", "labels": "true", "seed": 20261026},
            {"attempt_id": "A2_cnn_true_seed20261027", "kind": "neural", "labels": "true", "seed": 20261027},
            {"attempt_id": "A3_cnn_shuffle_seed20261026", "kind": "neural", "labels": "shuffled", "seed": 20261026,
             "permutation_seed": SHUFFLE_SEED},
        ],
        "architecture": "batch02 trackB_grouped_cv LegNetInspired(1, ch=64) read-only import",
        "lr": LR, "weight_decay": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "grad_clip": GRAD_CLIP, "optimizer": "AdamW",
        "scheduler": "constant LambdaLR(1.0)",
        "registered_at": C.now_local(),
        "inputs": {"str_variant_rows.tsv": C.sha256_file(C.DATA26 / "str_variant_rows.tsv"),
                   "encoded.npz": C.sha256_file(C.DATA26 / "encoded.npz"),
                   "split_manifest.tsv": C.sha256_file(C.DATA26 / "split_manifest.tsv")},
        "code_sha256": {"b26_train.py": C.sha256_file(Path(__file__)),
                        "b26_data.py": C.sha256_file(C.CODE26 / "b26_data.py"),
                        "b26_metrics.py": C.sha256_file(C.CODE26 / "b26_metrics.py")},
        "architecture_sha256": C.sha256_file(BATCH02_CODE / "trackB_grouped_cv.py"),
    }
    C.atomic_write_json(PREREG, exp)
    return exp


def attempt_dir(aid):
    d = ATTEMPTS / aid
    d.mkdir(parents=True, exist_ok=True)
    return d


def record(kind, aid, payload):
    rec = {"kind": kind, "attempt_id": aid, "ts": C.now_local(), "pid": os.getpid()}
    rec.update(payload)
    C.append_jsonl(LEDGER, rec)


# ------------------------------------------------------------------ features
def features342(seqs):
    """Fixed complete vocab: mono/di/tri/tetra frequencies (4+16+64+256) + GC + log1p(len)."""
    km = {1: {}, 2: {}, 3: {}, 4: {}}
    for k in (1, 2, 3, 4):
        for i in range(4 ** k):
            s = ""
            x = i
            for _ in range(k):
                s = "ACGT"[x % 4] + s
                x //= 4
            km[k][s] = i
    off = {1: 0, 2: 4, 3: 20, 4: 84}
    X = np.zeros((len(seqs), 342), dtype=np.float64)
    for r, s in enumerate(seqs):
        n = len(s)
        for k in (1, 2, 3, 4):
            cnt = np.zeros(4 ** k, dtype=np.float64)
            for i in range(n - k + 1):
                j = km[k].get(s[i:i + k])
                if j is not None:
                    cnt[j] += 1.0
            denom = n - k + 1
            if denom > 0:
                cnt = cnt / denom
            o = off[k]
            X[r, o:o + 4 ** k] = cnt
        X[r, 340] = (s.count("G") + s.count("C")) / n if n else 0.0
        X[r, 341] = math.log1p(n)
    return X


# ------------------------------------------------------------------ baseline C
def run_baseline():
    aid = "A0_ridgeC"
    d = attempt_dir(aid)
    rows = load_rows()
    seqs = [r["vreg_seq"] for r in rows]
    y = np.array([float(r["mean_log2_ratio"]) for r in rows], dtype=np.float64)
    sp = np.array([int(r["split"]) for r in rows], dtype=np.int64)
    loc = np.array([r["locus_id"] for r in rows], dtype=object)
    gid = np.array([r["group_id"] for r in rows], dtype=object)
    tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]
    X = features342(seqs)
    mu_x, sd_x = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)
    Z = (X - mu_x) / sd_x
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    zt = (y - mu_y) / sd_y
    A = Z[tr].T @ Z[tr] + RIDGE_ALPHA * np.eye(Z.shape[1])
    b = Z[tr].T @ zt[tr]
    w = np.linalg.solve(A, b)
    pv = Z[va] @ w * sd_y + mu_y
    pred_w = np.concatenate([w, mu_x, sd_x, np.array([mu_y, sd_y], dtype=np.float64)])
    np.save(d / "ridge_weights.npy", pred_w)
    np.savez(d / "val_pred.npz", idx=va, pred=pv, true=y[va])
    gbm, ng = M.group_balanced_mae(y[va], pv, loc[va])
    res = {"attempt_id": aid, "n_features": int(X.shape[1]), "component_fits": 1,
           "mu_y": mu_y, "sd_y": sd_y, "mu_x_sha": C.sha256_bytes(mu_x.tobytes()),
           "sd_x_sha": C.sha256_bytes(sd_x.tobytes()),
           "alpha": RIDGE_ALPHA, "n_train": int(len(tr)), "n_val": int(len(va)),
           "zero_variance_features": int(np.sum(X[tr].std(axis=0) == 0)),
           "val_spearman": M.spearman(pv, y[va]), "val_mae": M.mae(pv, y[va]),
           "val_group_balanced_mae": gbm, "val_groups": ng,
           "train_mean_val_spearman": M.spearman(y[va], np.full(len(va), mu_y)),
           "train_mean_val_mae": M.mae(y[va], np.full(len(va), mu_y)),
           "weights_file": "ridge_weights.npy",
           "weights_sha": C.sha256_file(d / "ridge_weights.npy"),
           "finished_at": C.now_local()}
    C.atomic_write_json(d / "result.json", res)
    record("full_fit", aid, {"phase": "classical", "result": res})
    print("[A0] val_sp=%.4f val_gbMAE=%.4f" % (res["val_spearman"], gbm), flush=True)
    return res


def load_encoded():
    z = np.load(C.DATA26 / "encoded.npz")
    return z["codes"], z["lens"]


def length_buckets(codes, lens, idx):
    b = {}
    for i in idx:
        b.setdefault(int(lens[i]), []).append(i)
    return {k: np.array(sorted(v)) for k, v in b.items()}


def predict_all(model, codes, lens, idx, dev, torch):
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    pos = {v: i for i, v in enumerate(idx)}
    buckets = length_buckets(codes, lens, idx)
    with torch.no_grad():
        for L, ids in buckets.items():
            for i in range(0, len(ids), 64):
                chunk = ids[i:i + 64]
                xb = torch.from_numpy(codes[chunk].astype(np.int64)).to(dev)
                p = model(xb)[:, 0].detach().cpu().numpy()
                for k, g in enumerate(chunk):
                    out[pos[g]] = p[k]
    return out


def epoch_batches(codes, lens, idx, weights, seed, epoch, torch, dev):
    buckets = length_buckets(codes, lens, idx)
    rs = np.random.RandomState((seed * 1000003 + epoch) % (2 ** 31 - 1))
    order = []
    for L in sorted(buckets):
        ids = buckets[L].copy()
        rs.shuffle(ids)
        for i in range(0, len(ids), BATCH):
            order.append(ids[i:i + BATCH])
    return order


def train_cnn(labels, seed, tag):
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    C.set_thread_limits(4)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing CPU fallback")
    dev = "cuda:0"
    import trackB_grouped_cv as T
    rows = load_rows()
    codes, lens = load_encoded()
    y = np.array([float(r["mean_log2_ratio"]) for r in rows], dtype=np.float64)
    sp = np.array([int(r["split"]) for r in rows], dtype=np.int64)
    loc = np.array([r["locus_id"] for r in rows], dtype=object)
    var = np.array([r["variant"] for r in rows], dtype=object)
    tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]
    # inverse locus-frequency weights (train rows), normalized mean 1 over train
    cnt = {}
    for i in tr:
        cnt[loc[i]] = cnt.get(loc[i], 0) + 1
    w = np.array([1.0 / cnt[loc[i]] for i in tr], dtype=np.float64)
    w = w / w.mean()
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    z_tr = (y[tr] - mu_y) / sd_y
    perm_sha = None
    if labels == "shuffled":
        rs = np.random.RandomState(SHUFFLE_SEED)
        perm = rs.permutation(len(tr))
        perm_sha = C.sha256_bytes(perm.astype(np.int64).tobytes())
        z_tr = z_tr[perm]
    C.set_seed(seed)
    model = T.build_cnn("legnet").to(dev)
    assert next(model.parameters()).is_cuda
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)
    nparam = sum(p.numel() for p in model.parameters())
    nstate = sum(v.numel() for v in model.state_dict().values())
    best = {"gbmae": float("inf"), "epoch": -1, "state": None, "val": None}
    log = []
    t0 = time.time()
    step = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        batches = epoch_batches(codes, lens, tr, w, seed, epoch, torch, dev)
        tot = 0.0
        for b in batches:
            xb = torch.from_numpy(codes[b].astype(np.int64)).to(dev)
            zb = torch.from_numpy(z_tr[np.searchsorted(tr, b)].astype(np.float32)).to(dev)
            wb = torch.from_numpy(w[np.searchsorted(tr, b)].astype(np.float32)).to(dev)
            pred = model(xb)[:, 0]
            loss = (((pred - zb) ** 2) * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step(); step += 1
            tot += float(loss.detach())
        pv = predict_all(model, codes, lens, va, dev, torch) * sd_y + mu_y
        gbmae, ng = M.group_balanced_mae(y[va], pv, loc[va])
        log.append({"epoch": epoch, "train_loss": tot / max(len(batches), 1),
                    "val_spearman": M.spearman(pv, y[va]), "val_mae": M.mae(pv, y[va]),
                    "val_group_balanced_mae": gbmae, "elapsed_s": round(time.time() - t0, 1)})
        print("[%s seed=%d] ep=%d loss=%.4f val_sp=%.4f val_gbMAE=%.4f" %
              (tag, seed, epoch, log[-1]["train_loss"], log[-1]["val_spearman"], gbmae), flush=True)
        if gbmae < best["gbmae"] - 1e-12:
            best = {"gbmae": gbmae, "epoch": epoch, "val": log[-1],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict()}
        if epoch - best["epoch"] >= PATIENCE:
            break
    return {"rows": rows, "codes": codes, "lens": lens, "y": y, "sp": sp, "loc": loc, "var": var,
            "tr": tr, "va": va, "w": w, "mu_y": mu_y, "sd_y": sd_y, "z_tr": z_tr,
            "model": model, "opt": opt, "sched": sched, "best": best, "log": log,
            "nparam": nparam, "nstate": nstate, "step": step, "perm_sha": perm_sha,
            "elapsed_s": round(time.time() - t0, 1), "T": T}


def run_cnn(labels, seed):
    import random as _r
    import torch
    aid = ("A1_cnn_true_seed%d" % seed) if labels == "true" else ("A3_cnn_shuffle_seed%d" % seed)
    if labels == "true" and seed == 20261027:
        aid = "A2_cnn_true_seed%d" % seed
    d = attempt_dir(aid)
    out = train_cnn(labels, seed, "true" if labels == "true" else "shuffle")
    T = out["T"]
    dev = "cuda:0"
    codes, lens, va = out["codes"], out["lens"], out["va"]
    model = T.build_cnn("legnet").to(dev)
    model.load_state_dict(out["best"]["state"]); model.eval()
    pv = predict_all(model, codes, lens, va, dev, torch) * out["sd_y"] + out["mu_y"]
    np.savez(d / "val_pred.npz", idx=va, pred=pv, true=out["y"][va])
    ckpt = {"model": out["best"]["state"], "optimizer": out["best"]["opt"],
            "scheduler": out["best"]["sched"],
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": _r.getstate(),
            "epoch": out["best"]["epoch"], "global_step": out["step"],
            "best_val_group_balanced_mae": out["best"]["gbmae"],
            "best_val": out["best"]["val"], "seed": seed, "labels": labels,
            "config": {"lr": LR, "wd": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS,
                       "patience": PATIENCE, "grad_clip": GRAD_CLIP,
                       "scheduler": "constant", "arch": "LegNetInspired(1,ch=64)",
                       "mu_y": out["mu_y"], "sd_y": out["sd_y"], "shuffle_seed": SHUFFLE_SEED,
                       "perm_sha": out["perm_sha"]}}
    tmp = str(d / "ckpt.pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, d / "ckpt.pt")
    res = {"attempt_id": aid, "labels": labels, "seed": seed, "component_fits": 1,
           "n_params": out["nparam"], "n_state_entries": out["nstate"],
           "epochs_run": len(out["log"]), "best_epoch": out["best"]["epoch"],
           "best_val_group_balanced_mae": out["best"]["gbmae"],
           "best_val_spearman": out["best"]["val"]["val_spearman"],
           "best_val_mae": out["best"]["val"]["val_mae"],
           "mu_y": out["mu_y"], "sd_y": out["sd_y"], "n_train": int(len(out["tr"])),
           "n_val": int(len(out["va"])), "perm_sha": out["perm_sha"],
           "elapsed_s": out["elapsed_s"], "train_pid": os.getpid(), "device": dev,
           "log": out["log"], "log_sha": C.sha256_bytes(json.dumps(out["log"]).encode()),
           "ckpt_sha": C.sha256_file(d / "ckpt.pt"),
           "val_pred_sha": C.sha256_file(d / "val_pred.npz"),
           "finished_at": C.now_local()}
    C.atomic_write_json(d / "result.json", res)
    record("full_fit", aid, {"phase": "neural", "labels": labels, "seed": seed,
                             "n_params": out["nparam"], "epochs_run": len(out["log"]),
                             "best_val_gbmae": out["best"]["gbmae"],
                             "ckpt_sha": res["ckpt_sha"]})
    print("[%s] best_ep=%d val_gbMAE=%.4f epochs=%d t=%.1fs" %
          (aid, out["best"]["epoch"], out["best"]["gbmae"], len(out["log"]), out["elapsed_s"]), flush=True)
    return res


def run_select():
    res = {}
    for aid in ("A1_cnn_true_seed20261026", "A2_cnn_true_seed20261027",
                "A3_cnn_shuffle_seed20261026", "A0_ridgeC"):
        p = ATTEMPTS / aid / "result.json"
        if p.is_file():
            res[aid] = C.read_json(p)
    cands = [(aid, r) for aid, r in res.items() if aid.startswith("A1_") or aid.startswith("A2_")]
    if len(cands) < 2:
        raise SystemExit("both true-seed fits required before selection")
    cands.sort(key=lambda x: (x[1]["best_val_group_balanced_mae"], int(x[1]["seed"])))
    primary = cands[0]
    man = {"batch": "batch26", "selected_primary": primary[0],
           "selected_seed": primary[1]["seed"],
           "criterion": "lowest validation group-balanced MAE (published log2 units); ties -> lower seed",
           "ranking": [{"attempt_id": a, "seed": r["seed"],
                        "val_group_balanced_mae": r["best_val_group_balanced_mae"],
                        "val_spearman": r["best_val_spearman"]} for a, r in cands],
           "shuffle_control": res.get("A3_cnn_shuffle_seed20261026", {}).get("best_val_group_balanced_mae"),
           "baseline_C": res.get("A0_ridgeC", {}).get("val_group_balanced_mae"),
           "test_targets_opened": False, "written_at": C.now_local()}
    C.atomic_write_json(C.REPORT26 / "selection_manifest.json", man)
    print("[select] primary=%s gbMAE=%.4f" % (primary[0], primary[1]["best_val_group_balanced_mae"]), flush=True)
    return man


def run_resume_check(aid="A1_cnn_true_seed20261026"):
    import torch
    d = ATTEMPTS / aid
    ck = torch.load(d / "ckpt.pt", map_location="cpu", weights_only=False)
    have = {"model": "model" in ck, "optimizer": "optimizer" in ck, "scheduler": "scheduler" in ck,
            "torch_rng": "torch_rng" in ck, "numpy_rng": "numpy_rng" in ck,
            "python_rng": "python_rng" in ck, "epoch": "epoch" in ck,
            "global_step": "global_step" in ck, "best_val_group_balanced_mae" in ck: True}
    import trackB_grouped_cv as T
    model = T.build_cnn("legnet")
    model.load_state_dict(ck["model"]); model.eval()
    rows = load_rows()
    codes, lens = load_encoded()
    z = np.load(d / "val_pred.npz")
    idx, pred_saved = z["idx"], z["pred"]
    mu_y, sd_y = ck["config"]["mu_y"], ck["config"]["sd_y"]
    dev = "cuda:0"
    torch.backends.cudnn.deterministic = True
    model = model.to(dev)
    p = predict_all(model, codes, lens, idx, dev, torch) * sd_y + mu_y
    maxdiff = float(np.max(np.abs(p - pred_saved)))
    out = {"attempt_id": aid, "checkpoint_keys_present": have,
           "n_val_rows": int(len(idx)), "resume_inference_max_abs_diff": maxdiff,
           "reset_weights": False, "extra_fit_performed": False,
           "note": "resume STATE/INFERENCE check only (loaded existing checkpoint, no weight reset, no extra fit)",
           "checked_at": C.now_local()}
    C.atomic_write_json(d / "resume_check.json", out)
    print("[resume] maxdiff=%.3e keys=%s" % (maxdiff, all(have.values())), flush=True)
    return out


def run_null_trainmean():
    rows = load_rows()
    y = np.array([float(r["mean_log2_ratio"]) for r in rows])
    sp = np.array([int(r["split"]) for r in rows])
    tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]
    mu = float(y[tr].mean())
    out = {"train_mean": mu, "val_spearman": M.spearman(y[va], np.full(len(va), mu)),
           "val_mae": M.mae(y[va], np.full(len(va), mu)),
           "n_fits": 0, "note": "bookkeeping only; no estimator fit"}
    C.atomic_write_json(C.REPORT26 / "null_train_mean.json", out)
    print("[null] val_sp=%.4f" % out["val_spearman"], flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prereg":
        preregister()
    elif mode == "baseline":
        preregister(); run_baseline()
    elif mode == "cnn":
        preregister()
        run_cnn(sys.argv[2], int(sys.argv[3]))
    elif mode == "select":
        run_select()
    elif mode == "resume-check":
        run_resume_check()
    elif mode == "null":
        run_null_trainmean()
    else:
        raise SystemExit("unknown mode %s" % mode)
