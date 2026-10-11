# -*- coding: utf-8 -*-
"""batch43 Phase B: fixed group-error MSE decomposition + rank diagnostics.

Uses STORED batch42 TEST predictions for R0/R1/Rnull/train-mean-const and cached
b41 PARM/H1/GC.  TRAIN predictions were NOT persisted, so this replay uses ONLY
the saved R0/R1/Rnull coefficients with their frozen feature/normalizer
contracts (ONE pass, CPU<=4, no fitting, no neural scoring).  Replay of a fixed
20-TEST-row selection is verified against the stored predictions first.
"""
from __future__ import annotations
import csv
import json
import math
import time

import numpy as np

import b43_common as C

T0 = time.time()
C.set_thread_limits(4)
R = C.B43_REPORT
D = C.B43_DATA

# ---- load frozen cohort -----------------------------------------------------
rows = C.B42.load_frozen_rows(with_labels=True)
gmap = C.load_groups()
bio = [r for r in rows if r["control"] == 0]
ctl = [r for r in rows if r["control"] == 1]
assert len(bio) == 7537 and len(ctl) == 85, (len(bio), len(ctl))
TRAIN = [r for r in bio if gmap[r["label"]]["split"] == "TRAIN"]
TEST = [r for r in bio if gmap[r["label"]]["split"] == "TEST"]
assert len(TRAIN) == 7219 and len(TEST) == 318
gtr = {gmap[r["label"]]["group"] for r in TRAIN}
gte = {gmap[r["label"]]["group"] for r in TEST}
assert len(gtr) == 92 and len(gte) == 21, (len(gtr), len(gte))

models = C.load_models()
metrics42 = C.jload(C.B42_METRICS)
CONST = float(metrics42["zero_fit_train_weighted_mean_baseline"]["const"])

# ---- stored TEST predictions ------------------------------------------------
stored = {}
with open(C.PRED_TEST, encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        stored[r["label"]] = r
assert len(stored) == 318
p_test_stored = {k: np.array([float(stored[r["label"]][k]) for r in TEST]) for k in ("R0", "R1", "Rnull")}
y_test = np.array([r["alpha_primary"] for r in TEST])
for i, r in enumerate(TEST):
    assert abs(y_test[i] - float(stored[r["label"]]["alpha_primary"])) < 1e-6, r["label"]

# ---- TEST replay verification (fixed 20-row selection first) ----------------
Z0_te = np.vstack([C.composition_features(r["sequence"]) for r in TEST])
Z1_te = C.h1_features([r["sequence"] for r in TEST])
rep_te = {"R0": C.predict(models["R0"], Z0_te), "R1": C.predict(models["R1"], Z1_te),
          "Rnull": C.predict(models["Rnull"], Z1_te)}
verify20 = {}
for k in ("R0", "R1", "Rnull"):
    d20 = float(np.max(np.abs(rep_te[k][:20] - p_test_stored[k][:20])))
    dall = float(np.max(np.abs(rep_te[k] - p_test_stored[k])))
    verify20[k] = {"max_abs_diff_first20": d20, "max_abs_diff_all318": dall}
VERIFY_TOL = 1e-6  # predictions published at 6 decimals -> rounding <= 5e-7
assert all(v["max_abs_diff_all318"] <= VERIFY_TOL for v in verify20.values()), verify20

# use stored values as canonical TEST predictions (identical to replay within rounding)
p_test = {"R0": p_test_stored["R0"], "R1": p_test_stored["R1"], "Rnull": p_test_stored["Rnull"]}
p_test["const"] = np.full_like(y_test, CONST)

# cached cross-unit raw scores on the SAME TEST ids (rank-only)
parm = C.jload(C.PARM_CACHE)["pred"]
h1c = C.jload(C.H1_CACHE)["pred"]
parm_te = np.array([parm[r["sequence_sha256"]]["parm_mean"] if r["sequence_sha256"] in parm else np.nan for r in TEST])
h1_te = np.array([h1c[r["sequence_sha256"]]["h1_mean"] if r["sequence_sha256"] in h1c else np.nan for r in TEST])
gc_te = np.array([h1c[r["sequence_sha256"]]["gc"] if r["sequence_sha256"] in h1c else np.nan for r in TEST])

# ---- TRAIN replay (saved coefficients only; ONE pass) -----------------------
t_r = time.time()
Z0_tr = np.vstack([C.composition_features(r["sequence"]) for r in TRAIN])
Z1_tr = C.h1_features([r["sequence"] for r in TRAIN])
rep_tr = {"R0": C.predict(models["R0"], Z0_tr), "R1": C.predict(models["R1"], Z1_tr),
          "Rnull": C.predict(models["Rnull"], Z1_tr)}
train_replay_seconds = round(time.time() - t_r, 2)
y_train = np.array([r["alpha_primary"] for r in TRAIN])
p_train = dict(rep_tr)
p_train["const"] = np.full_like(y_train, CONST)

# sanity: replay reproduces the batch42 in-sample train diagnostic rho
tr_diag = metrics42.get("train_diagnostic", {})
train_diag_recheck = {"replayed_R1_train_rho": C.spearman(rep_tr["R1"], y_train),
                      "replayed_R0_train_rho": C.spearman(rep_tr["R0"], y_train),
                      "batch42_saved_R1_train_rho": tr_diag.get("R1_train_rho"),
                      "batch42_saved_R0_train_rho": tr_diag.get("R0_train_rho")}

parm_tr = np.array([parm[r["sequence_sha256"]]["parm_mean"] if r["sequence_sha256"] in parm else np.nan for r in TRAIN])
h1_tr = np.array([h1c[r["sequence_sha256"]]["h1_mean"] if r["sequence_sha256"] in h1c else np.nan for r in TRAIN])
gc_tr = np.array([h1c[r["sequence_sha256"]]["gc"] if r["sequence_sha256"] in h1c else np.nan for r in TRAIN])

with open(D / "predictions_train.tsv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("label\tsequence_sha256\tgroup\tsplit\tchr\talpha_primary\tR0\tR1\tRnull\tcached_parm_mean\tcached_h1_mean\tgc\n")
    for i, r in enumerate(TRAIN):
        def fmt(v):
            return "NA" if not np.isfinite(v) else "%.6f" % v
        fh.write("\t".join([r["label"], r["sequence_sha256"], gmap[r["label"]]["group"], "TRAIN", r["chr"],
                            "%.6f" % y_train[i], "%.6f" % p_train["R0"][i], "%.6f" % p_train["R1"][i],
                            "%.6f" % p_train["Rnull"][i], fmt(parm_tr[i]), fmt(h1_tr[i]), fmt(gc_tr[i])]) + "\n")

# ---- group index maps -------------------------------------------------------
def group_index(part_rows):
    gi = {}
    for i, r in enumerate(part_rows):
        gi.setdefault(gmap[r["label"]]["group"], []).append(i)
    return {g: np.array(v) for g, v in gi.items()}

gi_test, gi_train = group_index(TEST), group_index(TRAIN)


def decompose(pred, y, gi):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    N = len(pred); G = len(gi)
    e = pred - y
    mse_g, b_g, within_g, ns = [], [], [], []
    per = []
    for g, idx in gi.items():
        eg = e[idx]
        b = float(np.mean(eg))
        mseg = float(np.mean(eg ** 2))
        within = float(np.mean((eg - b) ** 2))
        per.append({"group": g, "n": int(len(idx)), "b_g": b, "MSE_g": mseg, "within_g": within,
                    "identity_resid": abs(mseg - (b ** 2 + within))})
        mse_g.append(mseg); b_g.append(b); within_g.append(within); ns.append(len(idx))
    mse_g, b_g, within_g, ns = map(np.array, (mse_g, b_g, within_g, ns))
    # row-weighted
    rw_mse = float(np.mean(e ** 2))
    rw_between = float(np.sum(ns * b_g ** 2) / N)
    rw_within = float(np.sum(ns * within_g) / N)
    rw_meanerr = float(np.mean(e))
    # group-equal
    ge_mse = float(np.mean(mse_g))
    ge_between = float(np.mean(b_g ** 2))
    ge_within = float(np.mean(within_g))
    ge_meanerr = float(np.mean(b_g))
    idres = max(p["identity_resid"] for p in per)
    rwid = abs(rw_mse - (rw_between + rw_within))
    geid = abs(ge_mse - (ge_between + ge_within))
    return {
        "N": N, "G": G,
        "row_weighted": {"MSE": rw_mse, "between_b2": rw_between, "within": rw_within,
                         "between_frac": rw_between / rw_mse if rw_mse else None,
                         "within_frac": rw_within / rw_mse if rw_mse else None,
                         "mean_error": rw_meanerr},
        "group_equal": {"MSE": ge_mse, "between_b2": ge_between, "within": ge_within,
                        "between_frac": ge_between / ge_mse if ge_mse else None,
                        "within_frac": ge_within / ge_mse if ge_mse else None,
                        "mean_error": ge_meanerr},
        "identity_max_abs_resid_per_group": idres,
        "identity_rowweighted_resid": rwid,
        "identity_groupequal_resid": geid,
        "mae": C.mae(pred, y),
        "group_balanced_mae": float(np.mean([C.mae(pred[idx], y[idx]) for idx in gi.values()])),
        "r2": C.r2(pred, y),
        "per_group": per,
    }

# ---- MSE decomposition for learned same-unit models + const -----------------
decomp = {"TRAIN": {}, "TEST": {}}
for part, preds, y, gi in (("TRAIN", p_train, y_train, gi_train), ("TEST", p_test, y_test, gi_test)):
    for m in ("R0", "R1", "Rnull", "const"):
        decomp[part][m] = decompose(preds[m], y, gi)

# ---- rank diagnostics (all models; cross-unit = rank only) ------------------
def _isconst(x):
    """Tolerance-based constancy: catches the constant baseline whose fill_like()
    carries ~2e-16 float jitter (a stricter std==0 test would rank that jitter)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return True
    return float(np.ptp(x)) <= 1e-9 * max(1.0, abs(float(np.mean(x))))


def rank_metrics(pred, y, gi, min_rows=10):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    gp, gy, per = [], [], []
    n_ge10 = n_defined = 0
    reasons = {"lt_min_rows": 0, "constant_alpha": 0, "constant_pred": 0}
    rhos = []
    for g, idx in gi.items():
        pg, yg = pred[idx], y[idx]
        ok = np.isfinite(pg)
        gp.append(float(np.mean(pg[ok])) if ok.sum() else np.nan)
        gy.append(float(np.mean(yg[ok])) if ok.sum() else np.nan)
        if len(idx) < min_rows:
            reasons["lt_min_rows"] += 1
            per.append({"group": g, "n": int(len(idx)), "within_rho": None, "reason": "lt10rows"})
            continue
        n_ge10 += 1
        if _isconst(yg):
            reasons["constant_alpha"] += 1
            per.append({"group": g, "n": int(len(idx)), "within_rho": None, "reason": "constant_alpha"})
            continue
        if ok.sum() < min_rows or _isconst(pg[ok]):
            reasons["constant_pred"] += 1
            per.append({"group": g, "n": int(len(idx)), "within_rho": None, "reason": "constant_or_incomplete_pred"})
            continue
        rho = C.spearman(pg[ok], yg[ok])
        rhos.append(rho); n_defined += 1
        per.append({"group": g, "n": int(len(idx)), "within_rho": rho, "reason": "defined"})
    gp = np.array(gp); gy = np.array(gy)
    m = np.isfinite(gp) & np.isfinite(gy)
    if m.sum() < 3:
        gm_rho, gm_reason = None, "fewer_than_3_finite_groups"
    elif _isconst(gp[m]):
        gm_rho, gm_reason = None, "group_mean_prediction_constant"
    elif _isconst(gy[m]):
        gm_rho, gm_reason = None, "group_mean_alpha_constant"
    else:
        gm_rho, gm_reason = C.spearman(gp[m], gy[m]), "defined"
    return {"n_groups_total": len(gi), "n_groups_ge10": n_ge10,
            "n_groups_rho_defined": n_defined,
            "coverage_frac_of_ge10": (n_defined / n_ge10) if n_ge10 else None,
            "macro_within_rho": (float(np.mean(rhos)) if rhos else None),
            "group_mean_spearman": gm_rho, "group_mean_spearman_reason": gm_reason,
            "undefined_reasons": reasons, "per_group": per}


rank = {"TRAIN": {}, "TEST": {}}
cohort = {
    "TRAIN": {"R0": p_train["R0"], "R1": p_train["R1"], "Rnull": p_train["Rnull"], "const": p_train["const"],
              "parm": parm_tr, "h1": h1_tr, "gc": gc_tr},
    "TEST": {"R0": p_test["R0"], "R1": p_test["R1"], "Rnull": p_test["Rnull"], "const": p_test["const"],
             "parm": parm_te, "h1": h1_te, "gc": gc_te},
}
for part in ("TRAIN", "TEST"):
    y = y_train if part == "TRAIN" else y_test
    gi = gi_train if part == "TRAIN" else gi_test
    for m in ("R0", "R1", "Rnull", "const", "parm", "h1", "gc"):
        rank[part][m] = rank_metrics(cohort[part][m], y, gi)

# ---- fixed distribution table -------------------------------------------------
all_labels = [r["label"] for r in bio]
fam_of = C.author_parent_families(all_labels)
ctl_labels = [r["label"] for r in ctl]


def stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return None
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(med), "q25": float(q25), "q75": float(q75),
            "iqr": float(q75 - q25), "min": float(x.min()), "max": float(x.max())}


def dist_block(part_rows, part_name):
    out = {}
    for name, sel in (("overall", lambda r: True),
                      ("chr17", lambda r: r["chr"] == "chr17"),
                      ("non_chr17", lambda r: r["chr"] != "chr17")):
        sub = [r for r in part_rows if sel(r)]
        gset = {gmap[r["label"]]["group"] for r in sub}
        gi = {}
        for r in sub:
            gi.setdefault(gmap[r["label"]]["group"], 0)
            gi[gmap[r["label"]]["group"]] += 1
        sizes = sorted(gi.values(), reverse=True)
        fams = {fam_of[r["label"]] for r in sub}
        comps = {gmap[r["label"]]["component_id"] for r in sub}
        yrs = [r["alpha_primary"] for r in sub]
        gc = [h1c[r["sequence_sha256"]]["gc"] if r["sequence_sha256"] in h1c else np.nan for r in sub]
        blk = {"rows": len(sub), "groups": len(gset), "source_components_batch41": len(comps),
               "source_parent_families": len(fams),
               "alpha": stats(yrs), "gc": stats(gc),
               "group_sizes_min_med_max": [min(sizes), float(np.median(sizes)), max(sizes)] if sizes else None}
        if name == "overall":
            blk["group_sizes_sorted_desc"] = sizes
            blk["gc_coverage_rows_with_gc"] = int(np.isfinite(gc).sum())
        out[name] = blk
    # frozen weight distribution (TRAIN-only by design)
    if part_name == "TRAIN":
        tr_groups = {}
        for r in part_rows:
            tr_groups[gmap[r["label"]]["group"]] = tr_groups.get(gmap[r["label"]]["group"], 0) + 1
        w = np.array([1.0 / tr_groups[gmap[r["label"]]["group"]] for r in part_rows])
        w = w / w.mean()
        out["frozen_weight_distribution"] = {"rule": "1/train_group_size normalized to mean 1",
                                             "n_rows": int(len(w)),
                                             **{k: v for k, v in stats(w).items() if k != "n"}}
    else:
        out["frozen_weight_distribution"] = None
        out["frozen_weight_reason"] = "evaluation metrics are UNWEIGHTED; the frozen 1/group-size weights are TRAIN-only"
    return out


distribution = {
    "TRAIN": dist_block(TRAIN, "TRAIN"),
    "TEST": dist_block(TEST, "TEST"),
    "controls_scrambled_context": {
        "note": ("source scrambled controls are descriptive context only; real bio tiles are not universal "
                 "positives; no invented binary truth/AUC and no replicate noise ceiling"),
        "n_controls": len(ctl),
        "alpha": stats([r["alpha_primary"] for r in ctl]),
        "gc": stats([h1c[r["sequence_sha256"]]["gc"] if r["sequence_sha256"] in h1c else np.nan for r in ctl]),
    },
}

# ---- group-level error table (complete; no outcome-based exclusions) --------
with open(D / "group_error_table.tsv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("partition\tmodel\tgroup\tn_rows\tb_g\tMSE_g\twithin_g\tidentity_resid\t"
             "group_mean_pred\tgroup_mean_alpha\twithin_rho\twithin_rho_reason\n")
    for part in ("TRAIN", "TEST"):
        y = y_train if part == "TRAIN" else y_test
        gi = gi_train if part == "TRAIN" else gi_test
        for m in ("R0", "R1", "Rnull", "const"):
            clist = decomp[part][m]["per_group"]
            rmap = {p["group"]: p for p in rank[part][m]["per_group"]}
            for p in clist:
                g = p["group"]; idx = gi[g]
                r = rmap[g]
                fh.write("\t".join([part, m, g, str(p["n"]), "%.8f" % p["b_g"], "%.8f" % p["MSE_g"],
                                    "%.8f" % p["within_g"], "%.2e" % p["identity_resid"],
                                    "%.8f" % float(np.mean(cohort[part][m][idx])),
                                    "%.8f" % float(np.mean(y[idx])),
                                    ("%.8f" % r["within_rho"]) if r["within_rho"] is not None else "NA",
                                    r["reason"]]) + "\n")

# ---- outputs -----------------------------------------------------------------
orig_unchanged = {
    "R1_group_balanced_mae": metrics42["R1"]["group_balanced_mae"],
    "R0_group_balanced_mae": metrics42["R0"]["group_balanced_mae"],
    "Rnull_group_balanced_mae": metrics42["Rnull"]["group_balanced_mae"],
    "const_group_balanced_mae": metrics42["zero_fit_train_weighted_mean_baseline"]["group_balanced_mae"],
    "R1_mae": metrics42["R1"]["mae"], "R0_mae": metrics42["R0"]["mae"], "Rnull_mae": metrics42["Rnull"]["mae"],
    "verdict": metrics42["decision"]["verdict"], "operational_PASS": metrics42["decision"]["operational_PASS"],
}

error_decomp = {
    "batch": C.BATCH, "version": C.VERSION,
    "generated_at": C.now_local(),
    "scope": "posthoc diagnostic reuse of the EXPOSED batch42 group-heldout test; NOT new validation",
    "oracle_warning": "group means b_g use TEST labels; they are diagnostic ORACLE quantities, NOT a deployable correction or improved predictions",
    "definition_note": "MSE decomposition, NOT an additive MAE decomposition; identity MSE_g=b_g^2+within_g verified per group",
    "denominators": {"row_weighted": "N rows", "group_equal": "G groups"},
    "original_metrics_unchanged": orig_unchanged,
    "TRAIN": {m: {k: v for k, v in decomp["TRAIN"][m].items() if k != "per_group"}
              | {"per_group": decomp["TRAIN"][m]["per_group"]} for m in decomp["TRAIN"]},
    "TEST": {m: {k: v for k, v in decomp["TEST"][m].items() if k != "per_group"}
             | {"per_group": decomp["TEST"][m]["per_group"]} for m in decomp["TEST"]},
}
rank_diag = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
    "note": ("group-mean-prediction vs group-mean-alpha Spearman (all groups, no CI/significance); "
             "within-group rho for every group with >=10 rows and nonconstant vectors; constant "
             "baseline rho is null plus reason, NOT zero; PARM/H1/GC are RANK comparisons only"),
    "min_rows_for_within_rho": 10,
    "TRAIN": rank["TRAIN"], "TEST": rank["TEST"],
}
replay_ver = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
    "traink_predictions_persisted": False,
    "method": ("TRAIN predictions were NOT saved in batch42; replayed ONCE from the SAVED R0/R1/Rnull "
               "coefficients + frozen feature/normalizer contracts (additive projection), no fitting, no neural scoring"),
    "test_replay_verification": verify20,
    "test_replay_tolerance": VERIFY_TOL,
    "test_replay_tolerance_reason": "predictions_test.tsv published at 6 decimals -> rounding <= 5e-7",
    "train_replay_seconds": train_replay_seconds,
    "train_diagnostic_recheck": train_diag_recheck,
    "cpu_threads": 4,
}
C.atomic_write_json(R / "error_decomposition.json", error_decomp)
C.atomic_write_json(R / "rank_diagnostics.json", rank_diag)
C.atomic_write_json(R / "distribution_table.json", distribution)
C.atomic_write_json(R / "replay_verification.json", replay_ver)

# identity assertions
maxid = max(decomp[p][m]["identity_max_abs_resid_per_group"] for p in decomp for m in decomp[p])
maxrw = max(decomp[p][m]["identity_rowweighted_resid"] for p in decomp for m in decomp[p])
maxge = max(decomp[p][m]["identity_groupequal_resid"] for p in decomp for m in decomp[p])
assert maxid < 1e-10 and maxrw < 1e-10 and maxge < 1e-10, (maxid, maxrw, maxge)

summary = {
    "TEST_decomposition": {m: {"MSE_row_weighted": decomp["TEST"][m]["row_weighted"]["MSE"],
                               "between_frac_rw": decomp["TEST"][m]["row_weighted"]["between_frac"],
                               "within_frac_rw": decomp["TEST"][m]["row_weighted"]["within_frac"],
                               "MSE_group_equal": decomp["TEST"][m]["group_equal"]["MSE"],
                               "between_frac_ge": decomp["TEST"][m]["group_equal"]["between_frac"],
                               "within_frac_ge": decomp["TEST"][m]["group_equal"]["within_frac"]}
                           for m in ("R0", "R1", "Rnull", "const")},
    "TRAIN_decomposition": {m: {"MSE_row_weighted": decomp["TRAIN"][m]["row_weighted"]["MSE"],
                                "between_frac_rw": decomp["TRAIN"][m]["row_weighted"]["between_frac"],
                                "within_frac_rw": decomp["TRAIN"][m]["row_weighted"]["within_frac"]}
                            for m in ("R0", "R1", "Rnull", "const")},
    "identity_max_resid": {"per_group": maxid, "row_weighted": maxrw, "group_equal": maxge},
    "replay_max_abs_diff_all318": {k: v["max_abs_diff_all318"] for k, v in verify20.items()},
    "elapsed_s": round(time.time() - T0, 2),
}
C.atomic_write_json(R / "phaseB_summary.json", summary)
C.atomic_write_json(C.B43_RUN / "phaseB_decomposition.json",
                    {"batch": C.BATCH, "phase": "B_decomposition", "generated_at": C.now_local(),
                     "elapsed_s": summary["elapsed_s"], "identity_max_resid": summary["identity_max_resid"],
                     "replay_max_abs_diff_all318": summary["replay_max_abs_diff_all318"],
                     "script_sha256": C.sha256_file(__file__)})
print(json.dumps(summary, ensure_ascii=False, indent=1))
