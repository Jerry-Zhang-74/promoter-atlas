# -*- coding: utf-8 -*-
"""batch52 evaluation: GPU replay of SAVED weights (<=3 min, under GPU lock) + CPU metrics.

Reuses reviewed batch53 model_core/checkpoint/metrics/gpu_lock. Uses the batch52 lean cohort
loader (same datahash as the reviewer's data_adapter). Immutable outputs: refuses to overwrite.
TEST is opened only in the final 'final' subcommand with the frozen DEV-selected checkpoints.
"""
from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "breakthrough_20261002", "code", "batch53_model_core"))
sys.path.insert(0, HERE); sys.path.insert(0, HELPER)

import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(4)

from checkpoint import load_checkpoint  # noqa: E402
import importlib.util as _lock_import
_lock_spec = _lock_import.spec_from_file_location("b57_verified_lock", r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch57_model_core_rc2/gpu_lock.py")
_lock_module = _lock_import.module_from_spec(_lock_spec)
_lock_spec.loader.exec_module(_lock_module)
GpuLock = _lock_module.GpuLock  # noqa: E402
from io_utils import atomic_write_json, atomic_write_text, sha256_file, write_ready  # noqa: E402
from metrics import (spearman, rmse, group_equal_mae, top10_precision, group_mean_pair_mae,  # noqa: E402
                     macro_sign_accuracy, true_zero_rate)
from model_core import build_model  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from spec import SOURCE_SHA256, config_sha256, TARGET_COLUMN  # noqa: E402
import cohort_v2  # noqa: E402


def _refuse(path):
    if os.path.exists(path):
        raise RuntimeError("immutable output exists: %s" % path)


def replay(ckpt, cohort_dir, split, out_path, ready_path, device="cuda:0", lock_dir=None, budget_seconds=180):
    _refuse(out_path)
    cohort = cohort_v2.load_cohort_v2(cohort_dir, source_sha256=SOURCE_SHA256)
    obj = load_checkpoint(ckpt, expect_config_sha=config_sha256(), expect_datahash=cohort.datahash)
    if obj["config"]["target_column"] != TARGET_COLUMN:
        raise RuntimeError("checkpoint target != %s" % TARGET_COLUMN)
    lock = GpuLock(r"E:/promoter_atlas/runs/breakthrough_20261002/batch52")
    t0 = time.time()
    try:
        lock.acquire("replay:%s" % os.path.basename(ckpt), timeout_s=budget_seconds)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: no CPU neural fallback")
        model = build_model().to(device)
        model.load_state_dict(obj["model"]); model.eval()
        norm = TargetNormalizer(**{k: v for k, v in obj["normalizer"].items() if k in ("mean", "std")})
        idx = cohort.split_indices(split)
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s + 128]
                preds.append(norm.inverse(model(cohort.codes[b].to(device))).cpu())
                if time.time() - t0 > budget_seconds:
                    raise RuntimeError("replay budget exceeded")
        pred = torch.cat(preds) if preds else torch.zeros(0)
    finally:
        lock.release()
    lines = ["coord_id\tsplit\tactivity_pred"]
    for k, g in enumerate(idx.tolist()):
        lines.append("%s\t%s\t%.8f" % (cohort.coord_ids[g], split, float(pred[k])))
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    art = sha256_file(out_path)
    meta = {"phase": "replay", "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
            "config_sha256": config_sha256(), "datahash": cohort.datahash, "split": split,
            "n_rows": int(len(idx)), "activity_pred_sha256": art, "elapsed_s": round(time.time() - t0, 3)}
    if ready_path:
        write_ready(ready_path, {"artifact": out_path, "artifact_sha256": art, **meta})
    return meta


def load_preds(path):
    if path.endswith(".json"):
        return {k: float(v) for k, v in json.load(open(path, encoding="utf-8")).items()}
    d = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for ln in f:
            if ln.strip():
                a = ln.rstrip("\n").split("\t")
                d[a[0]] = float(a[2])
    return d


def metrics_for(cohort, preds, split):
    idx = cohort.split_indices(split)
    y = cohort.targets[idx].numpy()
    p = np.array([preds[cohort.coord_ids[g]] for g in idx.tolist()], dtype=np.float64)
    groups = [cohort.group_ids[g] for g in idx.tolist()]
    out = {"activity": {"rho": spearman(y, p), "group_mae": group_equal_mae(y, p, groups),
                        "rmse": rmse(y, p),
                        # stable-ID tie-break (canonical review D8): never array order
                        "top10_precision": top10_precision(y, p, stable_ids=[cohort.coord_ids[g] for g in idx.tolist()])},
           "n_rows": int(len(idx))}
    pr = cohort.pairs.get(split)
    if pr is not None and pr.numel() > 0:
        dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
        pv = np.array([preds[cohort.coord_ids[int(a)]] for a in pr[:, 0]])
        lv = np.array([preds[cohort.coord_ids[int(b)]] for b in pr[:, 1]])
        dp = pv - lv
        pg = [cohort.group_ids[int(a)] for a in pr[:, 0]]
        out["pair"] = {"n_pairs": int(pr.shape[0]), "n_groups": len(set(pg)),
                       "group_mean_mae": group_mean_pair_mae(dt, dp, pg),
                       "macro_sign_accuracy": macro_sign_accuracy(dt, dp, pg),
                       "true_zero_rate": true_zero_rate(dt), "rho_pooled": spearman(dt, dp)}
    return out


def fixed_baselines(cohort, split):
    pr = cohort.pairs.get(split)
    res = {}
    if pr is None or pr.numel() == 0:
        return res
    dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
    pg = [cohort.group_ids[int(a)] for a in pr[:, 0]]
    n = len(dt)
    for name, dp in (("zero_delta", np.zeros(n)), ("always_positive", np.ones(n)),
                     ("always_negative", -np.ones(n)), ("length_only_shorter_lower", -np.ones(n))):
        res[name] = {"group_mean_mae": group_mean_pair_mae(dt, dp, pg),
                     "macro_sign_accuracy": macro_sign_accuracy(dt, dp, pg)}
    return res


def paired_bootstrap(cohort, split, predA, predB, n_boot=2000, seed=20261004):
    pr = cohort.pairs.get(split)
    dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
    pvA = np.array([predA[cohort.coord_ids[int(a)]] for a in pr[:, 0]])
    lvA = np.array([predA[cohort.coord_ids[int(b)]] for b in pr[:, 1]])
    pvB = np.array([predB[cohort.coord_ids[int(a)]] for a in pr[:, 0]])
    lvB = np.array([predB[cohort.coord_ids[int(b)]] for b in pr[:, 1]])
    dpA, dpB = pvA - lvA, pvB - lvB
    pg = np.array([cohort.group_ids[int(a)] for a in pr[:, 0]])
    groups = list(dict.fromkeys(pg.tolist()))
    if len(groups) < 30:
        return {"ok": False, "reason": "fewer than 30 groups", "n_groups": len(groups)}
    gidx = {g: np.where(pg == g)[0] for g in groups}
    rng = np.random.default_rng(seed)
    mae_imp, sign_diff = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([gidx[groups[k]] for k in pick])
        gg = pg[rows]
        mae_imp.append(group_mean_pair_mae(dt[rows], dpA[rows], gg) - group_mean_pair_mae(dt[rows], dpB[rows], gg))
        sign_diff.append(macro_sign_accuracy(dt[rows], dpB[rows], gg) - macro_sign_accuracy(dt[rows], dpA[rows], gg))

    def ci(a):
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}
    return {"ok": True, "n_groups": len(groups), "n_boot": n_boot,
            "mae_A_minus_B": ci(mae_imp), "sign_B_minus_A": ci(sign_diff)}


def final(cohort_dir, r0_json, nabs_preds, njoint_preds, split, out_path, ready_path=None, n_boot=2000):
    _refuse(out_path)
    cohort = cohort_v2.load_cohort_v2(cohort_dir, source_sha256=SOURCE_SHA256)
    preds = {"R0": load_preds(r0_json), "Nabs": load_preds(nabs_preds), "Njoint": load_preds(njoint_preds)}
    res = {"split": split, "config_sha256": config_sha256(), "datahash": cohort.datahash,
           "arms": {k: metrics_for(cohort, v, split) for k, v in preds.items()},
           "fixed_baselines": fixed_baselines(cohort, split),
           "bootstrap_Nabs_vs_Njoint": paired_bootstrap(cohort, split, preds["Nabs"], preds["Njoint"], n_boot=n_boot)}
    a = res["arms"]["Nabs"].get("pair"); b = res["arms"]["Njoint"].get("pair")
    base = res["fixed_baselines"]
    checks = {}
    imp = None
    if a and b:
        imp = (a["group_mean_mae"] - b["group_mean_mae"]) / max(a["group_mean_mae"], 1e-12)
        # canonical review D3: the bar is the BEST fixed direction (max), never the worst (min)
        best_base = max((v["macro_sign_accuracy"] for v in base.values()
                         if v["macro_sign_accuracy"] == v["macro_sign_accuracy"]), default=float("nan"))
        res["best_fixed_baseline_macro_sign"] = best_base
        bt = res["bootstrap_Nabs_vs_Njoint"]
        rho_ok = (b["rho_pooled"] == b["rho_pooled"] and a["rho_pooled"] == a["rho_pooled"]
                  and b["rho_pooled"] >= a["rho_pooled"] - 0.05)
        checks = {"pair_mae_improve_ge_10pct": bool(imp >= 0.10),
                  "bootstrap_lowerCI_gt_0": bool(bt.get("ok") and bt["mae_A_minus_B"]["lo"] > 0),
                  "macro_dir_ge_5pp_above_best_baseline": bool(
                      b["macro_sign_accuracy"] == b["macro_sign_accuracy"] and best_base == best_base
                      and b["macro_sign_accuracy"] >= best_base + 0.05),
                  "activity_rho_not_worse_than_0.05": bool(rho_ok),
                  "zero_delta_baseline_reported": "zero_delta" in base}
    res["investigation_target"] = {"met": (all(checks.values()) if checks else None), "checks": checks,
                                   "improvement_fraction": imp}
    res["verdict"] = "TARGET_MET" if res["investigation_target"]["met"] else "LIMITED"
    atomic_write_json(out_path, res)
    if ready_path:
        write_ready(ready_path, {"artifact": out_path, "artifact_sha256": sha256_file(out_path),
                                 "verdict": res["verdict"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--checkpoint", required=True); r.add_argument("--cohort-dir", required=True)
    r.add_argument("--split", required=True); r.add_argument("--out", required=True)
    r.add_argument("--ready", default=None); r.add_argument("--device", default="cuda:0")
    r.add_argument("--lock-dir", default=None); r.add_argument("--budget-seconds", type=float, default=180)
    f = sub.add_parser("final")
    f.add_argument("--cohort-dir", required=True); f.add_argument("--r0-json", required=True)
    f.add_argument("--nabs-preds", required=True); f.add_argument("--njoint-preds", required=True)
    f.add_argument("--split", default="TEST"); f.add_argument("--out", required=True)
    f.add_argument("--ready", default=None); f.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "replay":
        print(json.dumps(replay(a.checkpoint, a.cohort_dir, a.split, a.out, a.ready, a.device,
                                a.lock_dir, a.budget_seconds), indent=2))
    else:
        print(json.dumps(final(a.cohort_dir, a.r0_json, a.nabs_preds, a.njoint_preds, a.split,
                               a.out, a.ready, a.n_boot), indent=2, default=str))


if __name__ == "__main__":
    main()
