# -*- coding: utf-8 -*-
"""batch30 trainer (REPAIRED).  NOT executed in batch30 (ZERO estimator fits / ZERO gradient
steps this batch).  Structural readiness is exercised by b30_preflight.py under no_grad only.

Repairs vs batch29:
  * native-length batches via b30_data.native_batch / iter_native_batches (batch <= 32, eval too;
    batch29 predict_all used 64).
  * loss: per-batch weighted mean  loss = sum_i w_i (pred-z)_i^2 / sum_i w_i, with group weights
    w_i = (1/group_count_i) / mean_over_train(1/group_count) so the objective is invariant to
    batch composition across batches.  Singletons are duplicated ONLY for BatchNorm (BatchNorm1d
    forbids a batch of 1): the duplicated copy carries EXACTLY HALF the original per-example mass
    (0.5*w each), so total group mass is unchanged -- but the same biological example is seen twice
    by BatchNorm running statistics (disclosed; NOT a new biological example).
  * durable latest.pt after every completed epoch + exact best.pt at improvement; --resume restores
    ALL states and continues at the next unconsumed batch/epoch.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C
import b30_data as D
import b30_metrics as M
import b30_ckpt as K
import b30_model as MOD

LR, WD, BATCH = 0.001, 0.0001, 32
MAX_EPOCHS, PATIENCE, GRAD_CLIP = 20, 5, 1.0
TRUE_SEEDS = (20261101, 20261102)
SHUFFLE_MODEL_SEED = 20261101
PERM_SEED = 20261103


def config_dict():
    return {"lr": LR, "wd": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "grad_clip": GRAD_CLIP, "arch": "LegNetInspired(1,ch=64)", "batch_policy": "equal-exact-length-native",
            "n_params": 110145, "perm_seed": PERM_SEED}


def hashes_dict(rows_tsv, codes_npz, split_tsv):
    return {"config": C.sha256_text(json.dumps(config_dict(), sort_keys=True)),
            "data_rows": C.sha256_file(rows_tsv),
            "data_codes": C.sha256_file(codes_npz),
            "split": C.sha256_file(split_tsv)}


def epoch_batches(lens, tr, seed, epoch):
    """Deterministic epoch order: length buckets ascending, ids shuffled by (seed,epoch)."""
    rs = np.random.RandomState((seed * 1000003 + epoch) % (2 ** 31 - 1))
    order = []
    for L in sorted(D.length_buckets(lens, tr)):
        ids = D.length_buckets(lens, tr)[L].copy()
        rs.shuffle(ids)
        for i in range(0, len(ids), BATCH):
            order.append(ids[i:i + BATCH])
    return order


def predict_native(model, codes, lens, idx, device, torch):
    """Eval-mode native prediction; batch <= 32 (eval obeys the same bound).  Returns raw y."""
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    pos = {int(v): i for i, v in enumerate(idx)}
    with torch.no_grad():
        for t, L, chunk in D.iter_native_batches(codes, lens, idx, device, torch, batch=BATCH):
            p = model(t)[:, 0].detach().cpu().numpy()
            for k, g in enumerate(chunk):
                out[pos[int(g)]] = p[k]
    return out


def fit(rows, codes, lens, split_tsv, seed, labels="true", exec_id=None, resume_path=None,
        out_dir=None, device="cuda:0", max_epochs=MAX_EPOCHS, stop_after_epoch=None, torch=None):
    """Train (or resume) ONE CNN fit.  Returns a summary dict.  NOTE: not run in batch30."""
    import torch as _t
    torch = torch or _t
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    C.set_thread_limits(4); torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA required"
    assert BATCH <= 32
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"; bestf = out_dir / "best.pt"
    y = np.array([float(r["y"]) for r in rows]); sp = np.array([r["split"] for r in rows], dtype=object)
    loc = np.array([r["group"] for r in rows], dtype=object)
    tr = np.where(sp == "train")[0]; va = np.where(sp == "val")[0]
    cnt = {}; [cnt.__setitem__(loc[i], cnt.get(loc[i], 0) + 1) for i in tr]
    w = np.array([1.0 / cnt[loc[i]] for i in tr]); w = w / w.mean()
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    z_tr = (y[tr] - mu_y) / sd_y
    perm_sha = None
    if labels == "shuffled":
        rs = np.random.RandomState(PERM_SEED); perm = rs.permutation(len(tr))
        perm_sha = C.sha256_bytes(perm.astype(np.int64).tobytes()); z_tr = z_tr[perm]
    C.set_seed(seed)
    model = MOD.build_cnn("legnet").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)
    cfg = config_dict(); cfg.update({"mu_y": mu_y, "sd_y": sd_y, "perm_sha": perm_sha, "seed": seed, "labels": labels})
    hs = hashes_dict(C.DATA30 / "rows.tsv", C.DATA30 / "encoded.npz", split_tsv)
    early = {"best_gbmae": float("inf"), "best_epoch": -1, "epochs_no_improve": 0, "best_val": None}
    start_epoch = 0; cursor0 = 0; step = 0; bestp = None
    t0 = time.time(); log = []
    if resume_path:
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        K.verify_identity(st, hs, exec_id)
        ep, cursor0, step, early = K.restore_state(st, model, opt, sched)
        start_epoch = ep + 1
        bestp = st.get("best_snapshot")
        C.append_jsonl(C.RUN30 / "telemetry.jsonl",
                       {"ts": C.now_local(), "event": "resume", "epoch": ep, "cursor": cursor0, "step": step})
    for epoch in range(start_epoch, max_epochs):
        model.train(); order = epoch_batches(lens, tr, seed, epoch)
        pos_tr = {int(v): i for i, v in enumerate(tr)}
        start_b = cursor0 if epoch == start_epoch else 0
        for bi in range(start_b, len(order)):
            b = order[bi]
            idxs = np.array([pos_tr[int(j)] for j in b])
            xb, L = D.native_batch(codes, lens, b, device, torch)
            zb = z_tr[idxs]; wb = w[idxs]
            if len(b) == 1:  # BatchNorm singleton duplication: HALF mass each, disclosed
                xb = torch.cat([xb, xb], dim=0); zb = np.array([zb[0], zb[0]]); wb = np.array([wb[0] * 0.5, wb[0] * 0.5])
            ztb = torch.from_numpy(zb.astype(np.float32)).to(device)
            wbt = torch.from_numpy(wb.astype(np.float32)).to(device)
            pred = model(xb)[:, 0]
            loss = (((pred - ztb) ** 2) * wbt).sum() / wbt.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step(); step += 1
        pv = predict_native(model, codes, lens, va, device, torch) * sd_y + mu_y
        gbmae, ng = M.group_balanced_mae(y[va], pv, loc[va])
        log.append({"epoch": epoch, "val_spearman": M.spearman(pv, y[va]), "val_mae": M.mae(pv, y[va]),
                    "val_group_balanced_mae": gbmae, "elapsed_s": round(time.time() - t0, 1)})
        if gbmae < early["best_gbmae"] - 1e-12:
            early = {"best_gbmae": gbmae, "best_epoch": epoch, "epochs_no_improve": 0, "best_val": log[-1]}
            bestp = K.capture_state(model, opt, sched, epoch=epoch, next_batch_cursor=0, global_step=step,
                                    early=early, cfg=cfg, hashes=hs, exec_id=exec_id)
            K.atomic_torch_save(bestp, bestf)   # exact snapshot, never mutated again
        else:
            early["epochs_no_improve"] += 1
        snap = K.capture_state(model, opt, sched, epoch=epoch, next_batch_cursor=0, global_step=step,
                               early=early, cfg=cfg, hashes=hs, exec_id=exec_id)
        snap["best_snapshot"] = bestp
        K.atomic_torch_save(snap, latest)       # atomic replace each completed epoch
        C.append_jsonl(C.RUN30 / "telemetry.jsonl",
                       {"ts": C.now_local(), "event": "epoch_done", "epoch": epoch, "step": step,
                        "val_gbmae": gbmae, "digest": K.state_digest(snap)})
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if early["epochs_no_improve"] >= PATIENCE:
            break
    return {"epochs_run": len(log), "best_epoch": early["best_epoch"], "best_val": early["best_val"],
            "log": log, "global_step": step, "elapsed_s": round(time.time() - t0, 1)}


def _main():
    ap = argparse.ArgumentParser(description="batch30 trainer (next-batch entry point; not run in batch30)")
    ap.add_argument("mode", choices=["fit", "resume"])
    ap.add_argument("--seed", type=int, default=TRUE_SEEDS[0])
    ap.add_argument("--labels", choices=["true", "shuffled"], default="true")
    ap.add_argument("--exec-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume-path", default=None)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    a = ap.parse_args()
    rows = D.load_rows(C.DATA30 / "rows.tsv")
    z = np.load(C.DATA30 / "encoded.npz"); codes, lens = z["codes"], z["lens"]
    rp = a.resume_path if a.mode == "resume" else None
    res = fit(rows, codes, lens, C.DATA30 / "split.tsv", a.seed, labels=a.labels, exec_id=a.exec_id,
              resume_path=rp, out_dir=a.out_dir, stop_after_epoch=a.stop_after_epoch)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _main()
