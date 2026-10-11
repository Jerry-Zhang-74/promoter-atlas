# -*- coding: utf-8 -*-
"""Reusable INDEPENDENT evaluation / replay program (separate process, immutable contract).

Two strictly separated phases:
  replay  : GPU, <=3 min total. Acquires the GPU-exclusive lock (PID+creation), loads a saved
            checkpoint with STRICT hash checks, regenerates predictions with NO gradients and
            NO CPU neural fallback, writes them atomically, then RELEASES the GPU lock before
            any slow CPU work.
  metrics : CPU only, may run while Njoint trains. Never touches the GPU or a neural net.

Immutability: an output path that already exists is refused (frozen contract). A ready
marker is written only AFTER the prediction artifact is complete, so no partial read races.

First-model TEST results are written to an evaluator-owned QUARANTINE directory. The training
process must never read them; only the final parent comparison may.
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch

from checkpoint import load_checkpoint
from data_adapter import load_cohort
from gpu_lock import GpuLock
from io_utils import atomic_write_json, atomic_write_text, sha256_file, write_ready
from metrics import (spearman, rmse, group_equal_mae, top10_precision, group_mean_pair_mae,
                     macro_sign_accuracy, true_zero_rate)
from model_core import build_model
from spec import SOURCE_SHA256, MAX_LEN, config_sha256
from normalizer import TargetNormalizer


class ContractError(Exception):
    pass


def _refuse_existing(path):
    if os.path.exists(path):
        raise ContractError("immutable output already exists: %s" % path)


def replay_predictions(ckpt_path, cohort_dir, split, out_path, ready_path=None,
                       device="cuda:0", lock_dir=None, budget_seconds=180,
                       source_sha=SOURCE_SHA256, expect_datahash=None, expect_config_sha=None):
    """Regenerate predictions from SAVED weights (zero fit). Holds the GPU lock only for the
    forward pass and releases it before returning (no CPU work under the lock)."""
    _refuse_existing(out_path)
    cohort = load_cohort(cohort_dir, source_sha)
    expect_config_sha = expect_config_sha or config_sha256()
    obj = load_checkpoint(ckpt_path, expect_config_sha=expect_config_sha,
                          expect_datahash=expect_datahash or cohort.datahash)
    if obj["config"]["target_column"] != "Log2RPM_HEK293":
        raise ContractError("checkpoint target != Log2RPM_HEK293")
    lock = GpuLock(lock_dir or os.path.dirname(out_path) or ".")
    t0 = time.time()
    try:
        lock.acquire("replay:%s" % os.path.basename(ckpt_path), timeout_s=budget_seconds)
        if not torch.cuda.is_available():
            raise ContractError("CUDA unavailable: no CPU neural fallback is permitted")
        model = build_model().to(device)
        model.load_state_dict(obj["model"])
        model.eval()
        norm = TargetNormalizer(**{k: v for k, v in obj["normalizer"].items() if k in ("mean", "std")})
        idx = cohort.split_indices(split)
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s + 128]
                preds.append(norm.inverse(model(cohort.codes[b].to(device))).cpu())
                if time.time() - t0 > budget_seconds:
                    raise ContractError("replay exceeded %ss budget" % budget_seconds)
        pred = torch.cat(preds) if preds else torch.zeros(0)
    finally:
        lock.release()          # GPU free before any metrics/CPU work
    lines = ["coord_id\tsplit\tactivity_pred"]
    for k, g in enumerate(idx.tolist()):
        lines.append("%s\t%s\t%.8f" % (cohort.coord_ids[g], split, float(pred[k])))
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    art_sha = sha256_file(out_path)
    meta = {"phase": "replay", "checkpoint": ckpt_path, "checkpoint_sha256": sha256_file(ckpt_path),
            "config_sha256": expect_config_sha, "datahash": cohort.datahash, "split": split,
            "n_rows": int(len(idx)), "activity_pred_sha256": art_sha,
            "elapsed_s": round(time.time() - t0, 3), "gpu_held_released": True}
    if ready_path:
        write_ready(ready_path, {"artifact": out_path, "artifact_sha256": art_sha, **meta})
    return meta


# ------------------------------------------------------------- CPU metrics ---
def load_preds(pred_path):
    d = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        head = f.readline()
        for ln in f:
            if not ln.strip():
                continue
            cid, _split, val = ln.rstrip("\n").split("\t")[:3]
            d[cid] = float(val)
    return d


def evaluate_split(cohort, preds_by_id, split):
    idx = cohort.split_indices(split)
    y = cohort.targets[idx].numpy()
    p = np.array([preds_by_id[cohort.coord_ids[g]] for g in idx.tolist()], dtype=np.float64)
    groups = [cohort.group_ids[g] for g in idx.tolist()]
    out = {"activity": {"rho": spearman(y, p), "group_mae": group_equal_mae(y, p, groups),
                        "rmse": rmse(y, p), "top10_precision": top10_precision(y, p)},
           "n_rows": int(len(idx))}
    pr = cohort.pairs.get(split)
    if pr is not None and pr.numel() > 0:
        dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
        pv = np.array([preds_by_id[cohort.coord_ids[int(a)]] for a in pr[:, 0]], dtype=np.float64)
        lv = np.array([preds_by_id[cohort.coord_ids[int(b)]] for b in pr[:, 1]], dtype=np.float64)
        dp = pv - lv
        pg = [cohort.group_ids[int(a)] for a in pr[:, 0]]
        out["pair"] = {"n_pairs": int(pr.shape[0]), "n_groups": len(set(pg)),
                       "group_mean_mae": group_mean_pair_mae(dt, dp, pg),
                       "macro_sign_accuracy": macro_sign_accuracy(dt, dp, pg),
                       "true_zero_rate": true_zero_rate(dt),
                       "rho_pooled": spearman(dt, dp)}
    return out


def fixed_baselines(cohort, split):
    """Prespecified fixed direction baselines on the SAME rows (never tuned on TEST)."""
    pr = cohort.pairs.get(split)
    res = {}
    if pr is None or pr.numel() == 0:
        return res
    dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
    pg = [cohort.group_ids[int(a)] for a in pr[:, 0]]
    n = len(dt)
    for name, dp in (("zero_delta", np.zeros(n)), ("always_positive", np.ones(n)),
                     ("always_negative", -np.ones(n)),
                     ("length_only_shorter_lower", -np.ones(n))):
        res[name] = {"group_mean_mae": group_mean_pair_mae(dt, dp, pg),
                     "macro_sign_accuracy": macro_sign_accuracy(dt, dp, pg)}
    return res


def paired_group_bootstrap(cohort, split, predA, predB, n_boot=2000, seed=20261004):
    """Paired group-resamples retaining sampling multiplicity (>=30 groups). Returns the
    distribution of (MAE_A - MAE_B) improvement and (sign_B - sign_A) difference."""
    pr = cohort.pairs.get(split)
    if pr is None or pr.numel() == 0:
        return {"ok": False, "reason": "no pairs"}
    dt = (cohort.targets[pr[:, 0]] - cohort.targets[pr[:, 1]]).numpy()
    pvA = np.array([predA[cohort.coord_ids[int(a)]] for a in pr[:, 0]])
    lvA = np.array([predA[cohort.coord_ids[int(b)]] for b in pr[:, 1]])
    pvB = np.array([predB[cohort.coord_ids[int(a)]] for a in pr[:, 0]])
    lvB = np.array([predB[cohort.coord_ids[int(b)]] for b in pr[:, 1]])
    dpA, dpB = pvA - lvA, pvB - lvB
    pg = np.array([cohort.group_ids[int(a)] for a in pr[:, 0]])
    groups = list(dict.fromkeys(pg.tolist()))
    if len(groups) < 30:
        return {"ok": False, "reason": "fewer than 30 groups: %d" % len(groups), "n_groups": len(groups)}
    gidx = {g: np.where(pg == g)[0] for g in groups}
    rng = np.random.default_rng(seed)
    mae_imp, sign_diff = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([gidx[groups[k]] for k in pick])
        dtt, dA, dB = dt[rows], dpA[rows], dpB[rows]
        gg = pg[rows]
        mae_imp.append(group_mean_pair_mae(dtt, dA, gg) - group_mean_pair_mae(dtt, dB, gg))
        sign_diff.append(macro_sign_accuracy(dtt, dB, gg) - macro_sign_accuracy(dtt, dA, gg))
    def ci(a):
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}
    return {"ok": True, "n_groups": len(groups), "n_boot": n_boot,
            "mae_A_minus_B": ci(mae_imp), "sign_B_minus_A": ci(sign_diff)}


def final_compare(cohort_dir, predA_path, predB_path, split, out_path, ready_path=None,
                  source_sha=SOURCE_SHA256, n_boot=2000):
    """Prespecified frozen comparison of Nabs (A) vs Njoint (B). No refits, no tuning."""
    _refuse_existing(out_path)
    cohort = load_cohort(cohort_dir, source_sha)
    A, B = load_preds(predA_path), load_preds(predB_path)
    mA = evaluate_split(cohort, A, split)
    mB = evaluate_split(cohort, B, split)
    base = fixed_baselines(cohort, split)
    boot = paired_group_bootstrap(cohort, split, A, B, n_boot=n_boot)
    res = {"split": split, "Nabs": mA, "Njoint": mB, "fixed_baselines": base, "bootstrap": boot}
    target = {"met": None, "checks": {}}
    pa, pb = mA.get("pair"), mB.get("pair")
    if pa and pb:
        imp = (pa["group_mean_mae"] - pb["group_mean_mae"]) / max(pa["group_mean_mae"], 1e-12)
        best_base = min((b["macro_sign_accuracy"] for b in base.values()
                         if b["macro_sign_accuracy"] == b["macro_sign_accuracy"]), default=float("nan"))
        rho_ok = (pb["rho_pooled"] == pb["rho_pooled"] and pa["rho_pooled"] == pa["rho_pooled"]
                  and pb["rho_pooled"] >= pa["rho_pooled"] - 0.05)
        checks = {"pair_mae_improve_ge_10pct": bool(imp >= 0.10),
                  "bootstrap_lowerCI_gt_0": bool(boot.get("ok") and boot["mae_A_minus_B"]["lo"] > 0),
                  "macro_dir_ge_5pp_above_best_baseline": bool(
                      pb["macro_sign_accuracy"] == pb["macro_sign_accuracy"]
                      and best_base == best_base and pb["macro_sign_accuracy"] >= best_base + 0.05),
                  "activity_rho_not_worse_than_0.05": bool(rho_ok),
                  "zero_delta_baseline_reported": "zero_delta" in base}
        target = {"met": all(checks.values()), "improvement_fraction": imp, "checks": checks}
    res["investigation_target"] = target
    res["verdict"] = "TARGET_MET" if target.get("met") else "LIMITED"
    atomic_write_json(out_path, res)
    if ready_path:
        write_ready(ready_path, {"artifact": out_path, "artifact_sha256": sha256_file(out_path),
                                 "verdict": res["verdict"]})
    return res


def write_sealed_quarantine(out_dir, payload: dict, name: str):
    """Evaluator-owned sealed/quarantine output for the first-model TEST evaluation. Training
    must not read this. Ready marker created AFTER the atomic artifact."""
    os.makedirs(out_dir, exist_ok=True)
    art = os.path.join(out_dir, name)
    _refuse_existing(art + ".json")
    atomic_write_json(art + ".json", payload)
    write_ready(art + ".ready.json", {"artifact": art + ".json",
                                      "artifact_sha256": sha256_file(art + ".json"),
                                      "quarantine": True})
    return art + ".json"


def main(argv=None):
    ap = argparse.ArgumentParser(description="independent eval/replay (replay=GPU<=3min, metrics/compare=CPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--checkpoint", required=True); r.add_argument("--cohort-dir", required=True)
    r.add_argument("--split", required=True); r.add_argument("--out", required=True)
    r.add_argument("--ready"); r.add_argument("--device", default="cuda:0")
    r.add_argument("--lock-dir"); r.add_argument("--budget-seconds", type=float, default=180)
    m = sub.add_parser("metrics")
    m.add_argument("--cohort-dir", required=True); m.add_argument("--preds", required=True)
    m.add_argument("--split", required=True); m.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--cohort-dir", required=True); c.add_argument("--predA", required=True)
    c.add_argument("--predB", required=True); c.add_argument("--split", required=True)
    c.add_argument("--out", required=True); c.add_argument("--ready")
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        print(json.dumps(replay_predictions(a.checkpoint, a.cohort_dir, a.split, a.out, a.ready,
                                            a.device, a.lock_dir, a.budget_seconds), indent=2))
    elif a.cmd == "metrics":
        cohort = load_cohort(a.cohort_dir)
        res = {"metrics": evaluate_split(cohort, load_preds(a.preds), a.split),
               "fixed_baselines": fixed_baselines(cohort, a.split)}
        atomic_write_json(a.out, res); print(json.dumps(res, indent=2))
    else:
        print(json.dumps(final_compare(a.cohort_dir, a.predA, a.predB, a.split, a.out, a.ready), indent=2))


if __name__ == "__main__":
    main()
