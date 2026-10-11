# -*- coding: utf-8 -*-
"""batch54 INDEPENDENT model-artifact / provenance / gate / coverage checks (CPU, read-only).

Reads a saved first-model (Nabs) checkpoint + ready marker + batch52 ledger and verifies:
  * ready marker present and parseable;
  * selected checkpoint SHA (strict, own loader, sidecar match);
  * frozen config_sha256, cohort datahash, target column Log2RPM_HEK293;
  * normalizer present/finite;
  * ALL eligible TRAIN rows covered in every reported full epoch (global_step == n_TRAIN);
  * DEV checkpoint-selection rule reproduced (best group-mean pair MAE, earliest tie);
  * shared init bytes (param_sha256 of a fresh build == recorded init_sha) and, when the
    second-arm checkpoint exists, that both arms share init;
  * exec_gate ledger accounting (reserve/complete/fail, MAX 4);
  * runnable load check: model constructs + loads state_dict + one SYNTHETIC forward.

No fit / no solve / no gradient / no optimizer step / no network. The single synthetic
forward is a serialization/runnability check on an artificial zero tensor only; it is NOT an
evaluation of any real sequence and produces no metric.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np

N_TRAIN_EXPECTED = 531860
FROZEN_CONFIG_SHA = "12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157"
COHORT_DATAHASH = "a17fcb741bc115c886f23e895d93e290b766b872644d5cd964d7d81a570f302d"
SOURCE_SHA256 = "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96"
TARGET_COLUMN = "Log2RPM_HEK293"
MAX_FITS = 4
HELPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "batch53_model_core"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_load_ckpt(path):
    import torch
    side = None
    sp = path + ".sha256.json"
    if os.path.exists(sp):
        side = json.load(open(sp, "r", encoding="utf-8"))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    actual = sha256_file(path)
    return obj, side, actual


def read_ledger(path):
    events = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    events.append(json.loads(ln))
    return events


def check_ledger(events):
    """Count EXECUTIONS by distinct reserve exec_id (reserve+complete carry consumes=True;
    they must not be double-counted). MAX 4 executions total."""
    reserved = [e for e in events if e.get("kind") == "reserved"]
    completed = [e for e in events if e.get("kind") == "completed"]
    failed = [e for e in events if e.get("kind") == "failed"]
    exec_ids = []
    for e in reserved:
        if e.get("exec_id") not in exec_ids:
            exec_ids.append(e.get("exec_id"))
    return {"n_events": len(events), "n_executions_reserved": len(exec_ids),
            "n_completed": len(completed), "n_failed": len(failed),
            "within_max": len(exec_ids) <= MAX_FITS,
            "purposes": [e.get("purpose") for e in reserved],
            "exec_ids": exec_ids}


def check_arm(out_root, arm, cohort_datahash=COHORT_DATAHASH, n_train=N_TRAIN_EXPECTED):
    """Return a structured independent report for one arm's saved artifacts."""
    import torch
    rep = {"arm": arm, "out_root": out_root, "issues": [], "ok": True}
    out_dir = os.path.join(out_root, arm)
    ready = os.path.join(out_root, "%s_ready.json" % arm)
    rep["ready_marker"] = ready
    rep["ready_exists"] = os.path.exists(ready)
    if not rep["ready_exists"]:
        rep["ok"] = False
        rep["issues"].append("ready marker missing")
        return rep
    marker = json.load(open(ready, "r", encoding="utf-8"))
    rep["marker"] = marker

    sel = marker.get("selected")
    rep["selected"] = sel
    rep["selected_exists"] = bool(sel and os.path.exists(sel))
    if not rep["selected_exists"]:
        rep["ok"] = False
        rep["issues"].append("selected checkpoint missing")
        return rep

    epoch_files = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("%s_epoch" % arm) and fn.endswith(".ckpt"):
            epoch_files.append(os.path.join(out_dir, fn))
    rep["n_epoch_checkpoints"] = len(epoch_files)
    cov = []
    for ef in epoch_files:
        obj = torch.load(ef, map_location="cpu", weights_only=False)
        ex = obj.get("extra", {}) or {}
        cov.append({"file": os.path.basename(ef), "epoch": obj.get("epoch"),
                    "global_step": obj.get("global_step"),
                    "full_epoch": obj.get("global_step") == n_train,
                    "train_loss": ex.get("train_loss"), "dev_pair_mae": ex.get("dev_pair_mae"),
                    "init_sha": ex.get("init_sha"), "epoch_cap": ex.get("epoch_cap"),
                    "datahash": obj.get("datahash")})
    rep["epoch_coverage"] = cov
    epochs_present = sorted(int(c["epoch"]) for c in cov if c["epoch"] is not None)
    rep["epochs_present"] = epochs_present
    all_full = all(c["full_epoch"] for c in cov) and len(cov) > 0
    contiguous = epochs_present == list(range(len(epochs_present)))
    rep["all_epochs_full_train"] = all_full
    rep["epochs_contiguous"] = contiguous
    rep["n_epochs_saved"] = len(cov)
    if not all_full:
        rep["issues"].append("an epoch did not cover all %d TRAIN rows" % n_train)
    if not contiguous:
        rep["issues"].append("epoch indices not contiguous")

    dev = [(c["epoch"], c["dev_pair_mae"]) for c in cov if c["dev_pair_mae"] is not None
           and c["dev_pair_mae"] == c["dev_pair_mae"]]
    if dev:
        best_ep = min(dev, key=lambda t: (t[1], t[0]))[0]
        rep["selection_recomputed_epoch"] = int(best_ep)
        rep["selection_matches_marker"] = bool(marker.get("best", {}).get("epoch") == best_ep)
        if not rep["selection_matches_marker"]:
            rep["issues"].append("recomputed DEV selection epoch != marker best epoch")
        src = os.path.join(out_dir, "%s_epoch%02d.ckpt" % (arm, int(best_ep)))
        if os.path.exists(src):
            rep["selected_equals_best_epoch_bytes"] = sha256_file(src) == sha256_file(sel)
            # The selected ckpt is a re-serialized copy of the best epoch file (torch.save of
            # the re-loaded object), so bytes may legitimately differ. The meaningful check is
            # TENSOR equality of model params to the sidecar-hashed epoch file.
            import torch as _t
            o_sel = _t.load(sel, map_location="cpu", weights_only=False)
            o_src = _t.load(src, map_location="cpu", weights_only=False)
            same = True
            for k, v in o_src["model"].items():
                if k not in o_sel["model"] or o_sel["model"][k].shape != v.shape or \
                        not _t.equal(o_sel["model"][k], v):
                    same = False
                    break
            rep["selected_tensors_equal_best_epoch"] = bool(same)
            rep["selected_same_epoch_index"] = bool(o_sel.get("epoch") == o_src.get("epoch"))
            rep["selected_same_datahash"] = bool(o_sel.get("datahash") == o_src.get("datahash"))
            if not (same and o_sel.get("epoch") == o_src.get("epoch")):
                rep["issues"].append("selected checkpoint model tensors != best DEV epoch file")

    obj, side, actual = strict_load_ckpt(sel)
    rep["selected_sha256"] = actual
    rep["selected_has_sidecar"] = bool(side)
    rep["selected_sidecar_sha_match"] = bool(side and side.get("checkpoint_sha256") == actual)
    if side and side.get("checkpoint_sha256") != actual:
        rep["issues"].append("selected sidecar sha mismatch")
    cfg = obj.get("config", {}) or {}
    rep["config_sha256"] = obj.get("config_sha256")
    rep["config_sha_ok"] = obj.get("config_sha256") == FROZEN_CONFIG_SHA
    rep["target_column"] = cfg.get("target_column")
    rep["target_ok"] = cfg.get("target_column") == TARGET_COLUMN
    rep["datahash"] = obj.get("datahash")
    rep["datahash_ok"] = obj.get("datahash") == cohort_datahash
    norm = obj.get("normalizer", {}) or {}
    rep["normalizer"] = {k: norm.get(k) for k in ("mean", "std", "frozen")}
    rep["normalizer_finite"] = bool(np.isfinite(norm.get("mean", float("nan"))) and
                                    np.isfinite(norm.get("std", float("nan"))))
    rep["gate_attempt_id"] = obj.get("gate_attempt_id")
    for k in ("config_sha_ok", "target_ok", "datahash_ok", "normalizer_finite"):
        if not rep.get(k):
            rep["issues"].append("check failed: %s" % k)

    try:
        import sys
        if HELPER not in sys.path:
            sys.path.insert(0, HELPER)
        from model_core import build_model, param_sha256
        fresh = param_sha256(build_model())
        rep["fresh_init_param_sha256"] = fresh
        rec = marker.get("init_sha") or (cov[0]["init_sha"] if cov else None)
        rep["recorded_init_sha"] = rec
        rep["shared_init_ok"] = bool(rec and fresh == rec)
        if not rep["shared_init_ok"]:
            rep["issues"].append("recorded init_sha != fresh build init bytes")
        m = build_model()
        m.load_state_dict(obj["model"], strict=True)
        m.eval()
        with torch.no_grad():
            syn = torch.zeros(1, 137, dtype=torch.long)
            out = m(syn)
        rep["runnable_load_check"] = {"constructed": True, "state_dict_loaded": True,
                                      "synthetic_forward_out": float(out[0]),
                                      "finite": bool(np.isfinite(float(out[0])))}
    except Exception as e:
        rep["runnable_load_check"] = {"error": repr(e)}
        rep["issues"].append("runnable load check failed: %r" % (e,))

    rep["ok"] = len(rep["issues"]) == 0
    return rep


def check_init_parity_both(out_root):
    a = os.path.join(out_root, "Nabs_ready.json")
    b = os.path.join(out_root, "Njoint_ready.json")
    if not (os.path.exists(a) and os.path.exists(b)):
        return {"both_present": False}
    ma = json.load(open(a, "r", encoding="utf-8")).get("init_sha")
    mb = json.load(open(b, "r", encoding="utf-8")).get("init_sha")
    return {"both_present": True, "Nabs_init_sha": ma, "Njoint_init_sha": mb,
            "shared_init_ok": ma == mb}
