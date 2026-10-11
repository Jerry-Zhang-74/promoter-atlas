# -*- coding: utf-8 -*-
"""batch54 INDEPENDENT GPU replay of SAVED weights -> frozen predictions (<=60 s, exclusive).

Optional short exclusive GPU step. Acquires the shared GPU-exclusive lock (PID+creation) so
it never competes with a training/replay job; refuses if the lock cannot be taken within a
short timeout. Loads the checkpoint with a STRICT own loader (sidecar SHA + config SHA +
datahash), builds the frozen architecture (batch53 model_core), runs a no_grad forward in
small chunks from a memory-mapped codes array (low RAM), inverse-normalises, writes a frozen
prediction TSV, then RELEASES the GPU lock before any CPU metric work. No fit, no gradient,
no CPU neural fallback.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "batch53_model_core"))
sys.path.insert(0, HERE)
sys.path.insert(0, HELPER)
import indep_cohort as C


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_load(ckpt, cohort_datahash=None, expect_config_sha=None):
    import torch
    side_path = ckpt + ".sha256.json"
    actual = sha256_file(ckpt)
    if os.path.exists(side_path):
        side = json.load(open(side_path, "r", encoding="utf-8"))
        if side.get("checkpoint_sha256") and side["checkpoint_sha256"] != actual:
            raise RuntimeError("checkpoint sidecar SHA mismatch")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if obj.get("format") != "batch53.checkpoint.v1":
        raise RuntimeError("unexpected checkpoint format %r" % obj.get("format"))
    if expect_config_sha and obj.get("config_sha256") != expect_config_sha:
        raise RuntimeError("config sha mismatch")
    if cohort_datahash and obj.get("datahash") != cohort_datahash:
        raise RuntimeError("datahash mismatch")
    return obj, actual


def replay(ckpt, cohort_dir, split, out_path, device="cuda:0", budget_s=60.0, lock_dir=None,
           expect_config_sha=None):
    import torch
    import importlib.util as _lock_import
    _lock_spec = _lock_import.spec_from_file_location("b57_verified_lock", r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch57_model_core_rc2/gpu_lock.py")
    _lock_module = _lock_import.module_from_spec(_lock_spec)
    _lock_spec.loader.exec_module(_lock_module)
    GpuLock = _lock_module.GpuLock
    from model_core import build_model
    co = C.load(cohort_dir)
    idx = C.split_idx(co, split)
    coord_ids = co["coord_ids"]
    codes = C.codes_memmap(cohort_dir)
    norm = None
    lock = GpuLock(r"E:/promoter_atlas/runs/breakthrough_20261002/batch52")
    t0 = time.time()
    lock.acquire("batch54_replay:%s" % os.path.basename(ckpt), timeout_s=min(budget_s, 30))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: no CPU neural fallback permitted")
        obj, art = strict_load(ckpt, expect_config_sha=expect_config_sha)
        if obj["config"]["target_column"] != "Log2RPM_HEK293":
            raise RuntimeError("checkpoint target != Log2RPM_HEK293")
        nz = obj.get("normalizer", {})
        mean, std = float(nz.get("mean", 0.0)), float(nz.get("std", 1.0))
        model = build_model().to(device)
        model.load_state_dict(obj["model"])
        model.eval()
        preds = np.empty(len(idx), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s + 128]
                c = torch.from_numpy(np.ascontiguousarray(codes[b])).to(torch.long).to(device)
                z = model(c)
                preds[s:s + len(b)] = (z * std + mean).detach().cpu().numpy()
                if time.time() - t0 > budget_s:
                    raise RuntimeError("replay exceeded %.0fs budget" % budget_s)
        elapsed = time.time() - t0
    finally:
        lock.release()          # GPU free before CPU work
    lines = ["coord_id\tsplit\tactivity_pred"]
    for k in range(len(idx)):
        lines.append("%s\t%s\t%.8f" % (coord_ids[idx[k]], split, preds[k]))
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)
    meta = {"phase": "batch54_indep_replay", "checkpoint": ckpt, "checkpoint_sha256": art,
            "split": split, "n_rows": int(len(idx)), "pred_sha256": sha256_file(out_path),
            "elapsed_s": round(elapsed, 3), "device": device, "budget_s": budget_s,
            "evaluator": "batch54"}
    with open(out_path + ".ready.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget-seconds", type=float, default=60.0)
    ap.add_argument("--lock-dir", default=None)
    ap.add_argument("--expect-config-sha", default=None)
    a = ap.parse_args()
    print(json.dumps(replay(a.checkpoint, a.cohort_dir, a.split, a.out, a.device,
                            a.budget_seconds, a.lock_dir, a.expect_config_sha),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
