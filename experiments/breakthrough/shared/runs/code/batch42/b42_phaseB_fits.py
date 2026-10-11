# -*- coding: utf-8 -*-
"""batch42 Phase B: EXACTLY THREE fixed estimator solves (R0, R1, Rnull).

Freeze gate PASSED (train 7219 rows/92 groups, test 318 rows/21 groups).  Each
solve is reserved through the reviewed batch27 ExecGate BEFORE the estimator is
instantiated; a failed solve consumes its slot and is never retried.  No CV, no
hyperparameter search, no multi-seed, no full-data final fit.  Test labels are
NOT touched here (sealed for Phase C).
"""
from __future__ import annotations
import importlib.util
import json
import time

import numpy as np

import b42_common as C

C.set_thread_limits(4)
T0 = time.time()

ledger = C.B42_RUN / "exec_gate_ledger.jsonl"
spec = importlib.util.spec_from_file_location("exec_gate", C.EXEC_GATE)
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)
GATE = eg.ExecGate(ledger, C.MAX_FITS)

frozen = json.load(open(C.B42_DATA / "frozen_split.json", encoding="utf-8"))
gate = frozen["gate"]
if not gate["PASS"]:
    C.atomic_write_json(C.B42_RUN / "phaseB_fits.json",
                        {"phase": "B_fits", "decision": "ZERO_FIT_GATE_FAILED", "gate": gate})
    raise SystemExit("gate not PASS; refusing to instantiate any estimator")

# ---- data (TRAIN labels only) ----------------------------------------------
rows = C.load_frozen_rows(with_labels=True)
gmap = {}
split_of = {}
with open(C.B42_DATA / "groups.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        lab, grp, canon, sp, *_ = line.rstrip("\n").split("\t")
        gmap[lab] = grp
        split_of[lab] = sp

bio = [r for r in rows if r["control"] == 0]
TRAIN = [r for r in bio if split_of[r["label"]] == "TRAIN"]
TEST = [r for r in bio if split_of[r["label"]] == "TEST"]
assert len(TRAIN) == gate["train_rows"] and len(TEST) == gate["test_rows"]

train_groups = {}
for r in TRAIN:
    train_groups.setdefault(gmap[r["label"]], 0)
    train_groups[gmap[r["label"]]] += 1
w = np.array([1.0 / train_groups[gmap[r["label"]]] for r in TRAIN], dtype=np.float64)
w = w / w.mean()
y = np.array([r["alpha_primary"] for r in TRAIN], dtype=np.float64)
seq_tr = [r["sequence"] for r in TRAIN]
ids_tr = [r["sequence_sha256"] for r in TRAIN]
print("TRAIN", len(TRAIN), "groups", len(train_groups), "weight mean", w.mean())

def fit_weighted_ridge(Z, yv, wv, alpha):
    """TRAIN-only weighted standardization + weighted ridge (no ORIGINAL intercept).
    Returns mu, scale, ymean, coeff so that pred = ymean + ((Z-mu)/scale) @ coeff."""
    wsum = float(wv.sum())
    mu = (wv[:, None] * Z).sum(0) / wsum
    var = (wv[:, None] * (Z - mu) ** 2).sum(0) / wsum
    scale = np.where(var > 0.0, np.sqrt(var), 1.0)
    Zs = (Z - mu) / scale
    ymean = float((wv * yv).sum() / wsum)
    yc = yv - ymean
    ZW = Zs * wv[:, None]
    A = Zs.T @ ZW + alpha * np.eye(Z.shape[1])
    coeff = np.linalg.solve(A, ZW.T @ yc)
    return mu, scale, ymean, coeff


CODE_SHA = C.sha256_file(C.B42_CODE / "b42_common.py")
MODELS = C.B42_DATA / "models"
MODELS.mkdir(parents=True, exist_ok=True)
attempts = []


def save_model(rid, mu, scale, ymean, coeff, alpha, feat, extra=None):
    obj = {"exec_id": rid, "version": C.VERSION, "generated_at": C.now_local(),
           "alpha": float(alpha), "intercept_weighted_train_label_mean": float(ymean),
           "feature_mean": mu.tolist(), "feature_scale": scale.tolist(),
           "coeff": coeff.tolist(), "dim": int(coeff.shape[0]),
           "train_n": int(len(TRAIN)), "train_groups": int(len(train_groups)),
           "train_ids_sha256": C.sha256_text("|".join(ids_tr)),
           "weight_rule": "1/train_group_size normalized to mean 1",
           "feature_code_sha256": CODE_SHA, "feature": feat}
    if extra:
        obj.update(extra)
    C.atomic_write_json(MODELS / (rid + ".json"), obj)
    return obj

R1_IDENTITY = "b42|H1-1280|alpha=1|w=1/gsize|seed=none"
R0_IDENTITY = "b42|comp342|alpha=100|w=1/gsize|seed=none"
RN_IDENTITY = "b42|H1-1280|alpha=1|w=1/gsize|shuffle=20261043"

Z1 = None


def solve_R0():
    Z = np.vstack([C.composition_features(s) for s in seq_tr])
    mu, scale, ymean, coeff = fit_weighted_ridge(Z, y, w, C.R0_ALPHA)
    o = save_model("R0", mu, scale, ymean, coeff, C.R0_ALPHA, "comp342",
                   {"feature_dim": 342, "feature_order": "mono4|di2|tri3|tetra4|GC|loglength",
                    "identity": R0_IDENTITY})
    return {"dim": o["dim"], "coeff_sha": C.sha256_bytes(np.asarray(o["coeff"], float).tobytes())}


def solve_R1():
    global Z1
    Z1 = C.h1_features(seq_tr)
    mu, scale, ymean, coeff = fit_weighted_ridge(Z1, y, w, C.R1_ALPHA)
    o = save_model("R1", mu, scale, ymean, coeff, C.R1_ALPHA, "H1_1280_primal",
                   {"feature_dim": 1280, "feature_order": "sqrt(.5)*L2(4mer256)|sqrt(.5)*L2(5mer1024)",
                    "gram_contract_sha256": C.H1_GRAM_SHA, "identity": R1_IDENTITY,
                    "kernel_equiv": "K=0.5*cos(k4_full)+0.5*cos(k5_full) (batch24 H1)"})
    return {"dim": o["dim"], "coeff_sha": C.sha256_bytes(np.asarray(o["coeff"], float).tobytes())}


def solve_Rnull():
    global Z1
    if Z1 is None:
        Z1 = C.h1_features(seq_tr)
    rng = np.random.default_rng(C.NULL_SEED)
    perm = rng.permutation(len(y))
    ys = y[perm]
    mu, scale, ymean, coeff = fit_weighted_ridge(Z1, ys, w, C.R1_ALPHA)
    o = save_model("Rnull", mu, scale, ymean, coeff, C.R1_ALPHA, "H1_1280_primal_shuffled",
                   {"feature_dim": 1280, "feature_order": "sqrt(.5)*L2(4mer256)|sqrt(.5)*L2(5mer1024)",
                    "gram_contract_sha256": C.H1_GRAM_SHA, "identity": RN_IDENTITY,
                    "shuffle_seed": C.NULL_SEED, "shuffle_rule": "numpy default_rng(20261043).permutation(train_labels) once",
                    "null_note": "single-shuffle sanity reference, NOT a permutation p-value"})
    return {"dim": o["dim"], "coeff_sha": C.sha256_bytes(np.asarray(o["coeff"], float).tobytes())}


for rid, fn, ident in (("R0", solve_R0, R0_IDENTITY), ("R1", solve_R1, R1_IDENTITY),
                       ("Rnull", solve_Rnull, RN_IDENTITY)):
    t = time.time()
    try:
        res = GATE.run(rid, fn, identity=ident)
        attempts.append({"exec_id": rid, "status": "completed", "seconds": round(time.time() - t, 2), **res})
        print("FIT", rid, "done", res["dim"], "%.1fs" % (time.time() - t), flush=True)
    except BaseException as e:  # noqa: BLE001 - any solve failure consumes the slot
        attempts.append({"exec_id": rid, "status": "failed", "error": repr(e),
                         "seconds": round(time.time() - t, 2)})
        print("FIT", rid, "FAILED", repr(e), flush=True)
        raise

status_after = GATE.status()
C.atomic_write_json(C.B42_RUN / "phaseB_fits.json",
                    {"batch": C.BATCH, "version": C.VERSION, "phase": "B_fits",
                     "generated_at": C.now_local(), "elapsed_s": round(time.time() - T0, 2),
                     "gate": gate, "attempts": attempts, "status_after": status_after,
                     "max_fits": C.MAX_FITS, "no_cv_no_tuning_no_fullfit": True,
                     "script_sha256": C.sha256_file(__file__)})
print("PHASE B DONE in %.1fs" % (time.time() - T0), flush=True)
print("STATUS", json.dumps(status_after))
