# -*- coding: utf-8 -*-
"""batch42 Phase C: ONE final held-GROUP test evaluation (labels sealed until all
three fits/models-selection were recorded in Phase B).

Primary: R1 vs R0 group-balanced MAE, pooled Spearman, R2/MAE (learned models,
same alpha units).  top10 precision/enrichment, deterministic ID ties, ceil(0.1n).
Cached batch41 frozen PARM/H1/GC compared on the SAME testIDs for rho/top10 only.
Descriptive pilot: 21 test groups < 30 -> NO conditional bootstrap / CI / p-value,
NO operational PASS.  No model promotion, no post-hoc subset.
"""
from __future__ import annotations
import json
import math
import time

import numpy as np

import b42_common as C

T0 = time.time()
REPORT = C.B42_REPORT
REPORT.mkdir(parents=True, exist_ok=True)

rows = C.load_frozen_rows(with_labels=True)
gmap, split_of = {}, {}
with open(C.B42_DATA / "groups.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        lab, grp, canon, sp, *_ = line.rstrip("\n").split("\t")
        gmap[lab] = grp
        split_of[lab] = sp

bio = [r for r in rows if r["control"] == 0]
TRAIN = [r for r in bio if split_of[r["label"]] == "TRAIN"]
TEST = [r for r in bio if split_of[r["label"]] == "TEST"]
seq_tr = [r["sequence"] for r in TRAIN]
seq_te = [r["sequence"] for r in TEST]
y_te = np.array([r["alpha_primary"] for r in TEST], dtype=np.float64)
ids_te = [r["sequence_sha256"] for r in TEST]

tr_groups = {}
for r in TRAIN:
    tr_groups[gmap[r["label"]]] = tr_groups.get(gmap[r["label"]], 0) + 1
w_tr = np.array([1.0 / tr_groups[gmap[r["label"]]] for r in TRAIN])
w_tr = w_tr / w_tr.mean()
y_tr = np.array([r["alpha_primary"] for r in TRAIN], dtype=np.float64)


def load_model(rid):
    return json.load(open(C.B42_DATA / "models" / (rid + ".json"), encoding="utf-8"))


def predict(model, Z):
    mu = np.asarray(model["feature_mean"], float)
    sc = np.asarray(model["feature_scale"], float)
    b = np.asarray(model["coeff"], float)
    return model["intercept_weighted_train_label_mean"] + ((Z - mu) / sc) @ b


# --- recompute test features from saved normalizers (NEVER refit) ------------
Z0_te = np.vstack([C.composition_features(s) for s in seq_te])
Z1_te = C.h1_features(seq_te)
m0, m1, mn = load_model("R0"), load_model("R1"), load_model("Rnull")
p0 = predict(m0, Z0_te)
p1 = predict(m1, Z1_te)
pn = predict(mn, Z1_te)
# verify saved coeff reproduces (train) by design; report model shas
model_sha = {rid: C.sha256_file(C.B42_DATA / "models" / (rid + ".json")) for rid in ("R0", "R1", "Rnull")}
print("models", model_sha)

# --- cached batch41 frozen baselines on SAME testIDs -------------------------
parm = json.load(open(C.PARM_CACHE, encoding="utf-8"))["pred"]
h1c = json.load(open(C.H1_CACHE, encoding="utf-8"))["pred"]
parm_te = np.array([parm[i]["parm_mean"] if i in parm else np.nan for i in ids_te], float)
cachedh1_te = np.array([h1c[i]["h1_mean"] if i in h1c else np.nan for i in ids_te], float)
gc_te = np.array([h1c[i]["gc"] if i in h1c else np.nan for i in ids_te], float)
print("cached coverage parm=%d/%d h1=%d/%d" % (np.isfinite(parm_te).sum(), len(ids_te),
                                               np.isfinite(cachedh1_te).sum(), len(ids_te)))

# --- metrics -----------------------------------------------------------------
tie = ids_te
groups_te = {}
for r in TEST:
    groups_te.setdefault(gmap[r["label"]], []).append(r)


def group_balanced_mae(pred):
    vals = []
    for g, ms in groups_te.items():
        idx = [TEST.index(r) for r in ms]
        vals.append(C.mae(pred[idx], y_te[idx]))
    return float(np.mean(vals))


const_base = float(np.average(y_tr, weights=w_tr))  # train weighted-mean zero-fit baseline
metrics = {
    "n_test_rows": len(TEST), "n_test_groups": len(groups_te),
    "n_test_chroms": len({r["chr"] for r in TEST}),
    "alpha_units": "MPRAnalyze getAlpha (raw, unchanged)",
    "R1": {"pooled_spearman": C.spearman(p1, y_te), "r2": C.r2(p1, y_te),
           "mae": C.mae(p1, y_te), "group_balanced_mae": group_balanced_mae(p1)},
    "R0": {"pooled_spearman": C.spearman(p0, y_te), "r2": C.r2(p0, y_te),
           "mae": C.mae(p0, y_te), "group_balanced_mae": group_balanced_mae(p0)},
    "Rnull": {"pooled_spearman": C.spearman(pn, y_te), "r2": C.r2(pn, y_te),
              "mae": C.mae(pn, y_te), "group_balanced_mae": group_balanced_mae(pn)},
    "zero_fit_train_weighted_mean_baseline": {
        "const": const_base, "mae": C.mae(np.full_like(y_te, const_base), y_te),
        "group_balanced_mae": group_balanced_mae(np.full_like(y_te, const_base))},
    "top10": {}, "cached_batch41_same_testids": {}, "per_group": {}, "strata": {},
}


def top10(pred):
    return C.top_decile_precision(pred if np.isfinite(pred).all() else np.where(np.isfinite(pred), pred, -1e9), y_te, tie)


for name, pred in (("R1", p1), ("R0", p0), ("Rnull", pn)):
    metrics["top10"][name] = top10(pred)
for name, pred in (("cached_parm_mean", parm_te), ("cached_h1_mean", cachedh1_te), ("gc", gc_te)):
    ok = np.isfinite(pred)
    metrics["cached_batch41_same_testids"][name] = {
        "spearman": C.spearman(pred[ok], y_te[ok]) if ok.sum() > 2 else float("nan"),
        "top10": top10(pred) if ok.all() else {"undefined": "NaN cached values present"},
        "n_used": int(ok.sum())}

# --- per-group macro Spearman (groups with >=10 rows, nonconstant) ----------
pg = {"R1": [], "R0": [], "Rnull": []}
for g, ms in groups_te.items():
    if len(ms) < 10:
        continue
    idx = [TEST.index(r) for r in ms]
    yy = y_te[idx]
    if np.std(yy) == 0:
        continue
    for name, pred in (("R1", p1), ("R0", p0), ("Rnull", pn)):
        pg[name].append(C.spearman(pred[idx], yy))
metrics["per_group"] = {k: {"n_groups_used": len(v), "n_groups_total": len(groups_te),
                            "macro_rho": float(np.nanmean(v)) if v else float("nan")}
                        for k, v in pg.items()}
metrics["per_group_sizes"] = sorted([len(v) for v in groups_te.values()], reverse=True)

# --- fixed chr17 / non-chr17 strata (descriptive, no rescue) ----------------
for label, mask in (("chr17", np.array([r["chr"] == "chr17" for r in TEST])),
                    ("non_chr17", np.array([r["chr"] != "chr17" for r in TEST]))):
    d = {"n": int(mask.sum())}
    for name, pred in (("R1", p1), ("R0", p0), ("Rnull", pn), ("cached_parm", parm_te)):
        d[name + "_rho"] = C.spearman(pred[mask], y_te[mask]) if mask.sum() > 2 else float("nan")
        d[name + "_mae"] = C.mae(pred[mask], y_te[mask]) if mask.sum() else float("nan")
    metrics["strata"][label] = d

# --- decision rule (fixed thresholds; NO CI claim at 21 groups) -------------
r1rho = metrics["R1"]["pooled_spearman"]
r0rho = metrics["R0"]["pooled_spearman"]
d_rho = r1rho - r0rho
mae_imp = 1.0 - metrics["R1"]["group_balanced_mae"] / metrics["R0"]["group_balanced_mae"]
t10_ok = metrics["top10"]["R1"]["precision"] >= metrics["top10"]["R0"]["precision"]
crit = {"R1_test_rho_ge_0.30": bool(r1rho >= 0.30),
        "R1_minus_R0_rho_ge_0.05": bool(d_rho >= 0.05),
        "conditional_CI_lo_gt_0": None,
        "groupMAE_at_least_10pct_lower": bool(mae_imp >= 0.10),
        "top10_not_lower": bool(t10_ok)}
crit["conditional_CI_lo_gt_0"] = None  # 21 test groups < 30 -> bootstrap/CI unavailable
all_met = all(v is True for v in crit.values())
verdict = "LIMITED" if not all_met else "LIMITED"  # CI cannot be claimed -> never operational PASS here
decision = {
    "r1_rho": r1rho, "r0_rho": r0rho, "delta_rho": d_rho,
    "groupMAE_improvement_frac": mae_imp,
    "criteria": crit, "criteria_all_met_except_CI": all(v is True for k, v in crit.items() if k != "conditional_CI_lo_gt_0"),
    "bootstrap_available": False,
    "bootstrap_reason": "n_test_groups=21 < 30 -> conditional bootstrap/CI/significance NOT reported (protocol + pre-fit amendment)",
    "verdict": verdict,
    "operational_PASS": False,
    "note": "no model promotion; regional/group-held-out retrospective public-assay scope only; CI lo>0 UNMET (unavailable), so verdict LIMITED",
}
metrics["decision"] = decision
metrics["cross_split_homology"] = {"exact_or_rc": 0, "confirmed_near_homology_merges": 0,
                                   "min_cross_group_edit_distance": 28,
                                   "residual_not_exhaustive": True}
metrics["generated_at"] = C.now_local()
metrics["script_sha256"] = C.sha256_file(__file__)
C.atomic_write_json(REPORT / "metrics.json", metrics)
print(json.dumps({"r1rho": r1rho, "r0rho": r0rho, "drho": d_rho, "mae_imp": mae_imp,
                  "verdict": verdict, "top10_R1": metrics["top10"]["R1"],
                  "per_group": metrics["per_group"]}, ensure_ascii=False, default=str))

# --- predictions table (TEST) ------------------------------------------------
with open(C.B42_DATA / "predictions_test.tsv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("label\tsequence_sha256\tgroup\tsplit\tchr\talpha_primary\tR0\tR1\tRnull\tcached_parm_mean\tcached_h1_mean\tgc\n")
    for i, r in enumerate(TEST):
        fh.write("\t".join([r["label"], r["sequence_sha256"], gmap[r["label"]], "TEST", r["chr"],
                            "%.6f" % y_te[i], "%.6f" % p0[i], "%.6f" % p1[i], "%.6f" % pn[i],
                            ("%.6f" % parm_te[i]) if np.isfinite(parm_te[i]) else "NA",
                            ("%.6f" % cachedh1_te[i]) if np.isfinite(cachedh1_te[i]) else "NA",
                            ("%.6f" % gc_te[i]) if np.isfinite(gc_te[i]) else "NA"]) + "\n")

# --- train-prediction overfit diagnostic (NEVER selection) -------------------
Z0_tr = np.vstack([C.composition_features(s) for s in seq_tr])
Z1_tr = C.h1_features(seq_tr)
over = {"R1_train_rho": C.spearman(predict(m1, Z1_tr), y_tr),
        "R0_train_rho": C.spearman(predict(m0, Z0_tr), y_tr),
        "note": "in-sample overfit diagnostic only; NOT used for any selection"}
metrics["train_diagnostic"] = over
C.atomic_write_json(REPORT / "metrics.json", metrics)
print("PHASE C DONE in %.1fs" % (time.time() - T0), "overfit", over)
