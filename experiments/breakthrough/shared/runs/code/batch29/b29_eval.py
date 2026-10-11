# -*- coding: utf-8 -*-
"""BATCH29 evaluation: TEST scoring for the four attempts AFTER selection manifest is frozen.
Primary metrics on ALL test rows (including zero-RNA), secondary detection, strata, target
sensitivity (D=2+P, D=0.5+P on the SAME saved predictions).  NO new fits."""
from __future__ import annotations
import json, os, sys, math, csv
from pathlib import Path
import numpy as np

ROOT = Path("E:/promoter_atlas")
CB = ROOT / "runs/breakthrough_20261002/code"
sys.path.insert(0, str(CB / "batch26")); sys.path.insert(0, str(CB / "batch02"))
import b26_common as C
import b26_metrics as M
import trackB_grouped_cv as T
sys.path.insert(0, str(CB / "batch29"))
import b29_train as TR

DB = ROOT / "data/breakthrough_20261002/batch29"
REP = ROOT / "reports/breakthrough_20261002/batch29"
ATT = ROOT / "runs/breakthrough_20261002/batch29/attempts"

man = C.read_json(REP / "selection_manifest.json")
rows = TR.load_rows()
keys = [r["key"] for r in rows]
sp = np.array([r["split"] for r in rows], dtype=object)
loc = np.array([r["group"] for r in rows], dtype=object)
y = np.array([float(r["y"]) for r in rows])
r1 = np.array([int(r["r1"]) for r in rows]); r2 = np.array([int(r["r2"]) for r in rows])
P = np.array([int(r["P"]) for r in rows]); L = np.array([int(r["len"]) for r in rows])
both = np.array([int(r["both"]) for r in rows])
te = np.where(sp == "test")[0]
tr = np.where(sp == "train")[0]
sha12 = np.array([k for k in keys])

# alternate-target labels (sensitivity only)
N = C.read_json(REP / "passA.json")["library_totals_fulltable"]
N_D1, N_R1, N_R2 = N["N_D_1plusP"], N["N_R1"], N["N_R2"]
def y_alt(delta):
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        D = delta + int(r["P"])
        a1 = (int(r["r1"]) / N_R1) / (D / N_D1); a2 = (int(r["r2"]) / N_R2) / (D / N_D1)
        out[i] = 0.5 * (math.log2(1 + a1) + math.log2(1 + a2))
    return out
y_delta2 = y_alt(2.0); y_delta05 = y_alt(0.5)


def topdecile(yt, yp):
    n = len(yt); k = int(math.ceil(0.1 * n))
    tie = np.array([int(h[:16], 16) for h in [C.sha256_bytes(x.encode()) for x in sha12]])
    order_p = np.lexsort((tie, -np.asarray(yp)))
    order_t = np.lexsort((tie, -np.asarray(yt)))
    pt = set(order_p[:k].tolist()); tt = set(order_t[:k].tolist())
    inter = len(pt & tt); base = k / n
    return {"k": k, "n": n, "intersection": inter, "precision": inter / k,
            "baseline_precision": base, "enrichment": (inter / k) / base if base > 0 else float("nan")}


def auroc(score, label):
    s = np.asarray(score); lb = np.asarray(label)
    pos = s[lb == 1]; neg = s[lb == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ssum = np.zeros(len(cnt)); np.add.at(ssum, inv, ranks)
    ranks = (ssum / cnt)[inv]
    return float((ranks[lb == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ap(score, label):
    s = np.asarray(score); lb = np.asarray(label)
    order = np.argsort(-s); lb = lb[order]
    tp = np.cumsum(lb); prec = tp / np.arange(1, len(lb) + 1)
    npos = lb.sum()
    return float((prec * lb).sum() / npos) if npos > 0 else float("nan")


def bundle(pred, mask):
    yt = y[mask]; yp = np.asarray(pred)[mask]; g = loc[mask]
    gbm, ng = M.group_balanced_mae(yt, yp, g)
    return {"n": int(mask.sum()), "spearman": M.spearman(yp, yt), "mae": M.mae(yp, yt),
            "r2": M.r2(yt, yp), "group_balanced_mae": gbm, "n_groups": ng,
            "top_decile": topdecile(yt, yp), "y_zero_frac": float(np.mean(yt == 0))}


def detect_metrics(pred, mask):
    lab = both[mask]
    if lab.sum() == 0 or lab.sum() == len(lab):
        return {"auroc": None, "ap": None, "n_pos": int(lab.sum()), "n": int(len(lab))}
    return {"auroc": auroc(np.asarray(pred)[mask], lab), "ap": ap(np.asarray(pred)[mask], lab),
            "n_pos": int(lab.sum()), "n": int(len(lab))}


def ridge_test_pred():
    d = ATT / "E1_ridge"
    w = np.load(d / "ridge_weights.npy")
    ww = w[:342]; mu_x = w[342:684]; sd_x = w[684:1026]; mu_y = float(w[1026]); sd_y = float(w[1027])
    seq = TR.load_assay_seqs()
    X = TR.features342([seq[keys[i]] for i in te])
    return ((X - mu_x) / sd_x) @ ww * sd_y + mu_y


def cnn_test_pred(aid):
    import torch
    d = ATT / aid
    ck = torch.load(d / "ckpt.pt", map_location="cpu", weights_only=False)
    codes, lens = TR.load_encoded()
    dev = "cuda:0"
    model = T.build_cnn("legnet").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    p = TR.predict_all(model, codes, lens, te, dev, torch) * ck["config"]["sd_y"] + ck["config"]["mu_y"]
    return p


preds = {}
preds["E1_ridge"] = ridge_test_pred()
for aid in ("E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102", "E4_cnn_shuffle_seed20261101"):
    if (ATT / aid / "ckpt.pt").is_file():
        preds[aid] = cnn_test_pred(aid)
base = np.full(len(te), float(y[tr].mean()))
mask_all = np.ones(len(te), dtype=bool)

results = {"batch": "batch29", "primary_cohort": "ALL test rows incl. zero-RNA",
           "n_test": int(len(te)), "n_train": int(len(tr)),
           "baseline_train_mean": bundle(base, mask_all),
           "models": {}}
det = {}
strata = {}

depth_bins = [("P=1", P == 1), ("P=2-3", (P >= 2) & (P <= 3)), ("P=4-8", (P >= 4) & (P <= 8)), ("P>=9", P >= 9)]
len_bins = [("200-499", (L >= 200) & (L <= 499)), ("500-999", (L >= 500) & (L <= 999)),
            ("1000-1499", (L >= 1000) & (L <= 1499)), ("1500-2000", (L >= 1500) & (L <= 2000))]

for name, p in list(preds.items()) + [("baseline_train_mean", base)]:
    results["models"][name] = {"all_test": bundle(p, mask_all)}
    det[name] = detect_metrics(p, mask_all)
    st = {"depth": {}, "length": {}}
    for lab, m in depth_bins:
        st["depth"][lab] = bundle(p, m)
    for lab, m in len_bins:
        st["length"][lab] = bundle(p, m)
    st["both_RNA_positive_conditional"] = bundle(p, both[te] == 1)
    strata[name] = st
results["detection_secondary"] = det

# target-sensitivity: recompute rank agreement of SAME predictions to alternate fixed D
sens = {}
for name, p in preds.items():
    sens[name] = {"rho_vs_primary_y": M.spearman(p, y[te]),
                  "rho_vs_D2plus_y": M.spearman(p, y_delta2[te]),
                  "rho_vs_D05plus_y": M.spearman(p, y_delta05[te]),
                  "rank_agreement_primary_vs_D2plus": M.spearman(y[te], y_delta2[te]),
                  "rank_agreement_primary_vs_D05plus": M.spearman(y[te], y_delta05[te])}
results["target_sensitivity"] = sens

# criteria (candidate = selected primary vs baseline C = E1 ridge)
sel = man["selected_primary"]
def g(name):
    return results["models"][name]["all_test"]
Cb = g("E1_ridge")
Pc = g(sel)
boot = M.group_bootstrap(preds[sel], preds["E1_ridge"], y[te], loc[te], n=2000)
second = "E3_cnn_true_seed20261102" if sel.startswith("E2_") else "E2_cnn_true_seed20261101"
P2 = g(second)
crit = {
    "selected_rho>=0.30": Pc["spearman"] >= 0.30,
    "rho_minus_C>=0.05": (Pc["spearman"] - Cb["spearman"]) >= 0.05,
    "groupMAE_reduction>=10pct": (Cb["group_balanced_mae"] - Pc["group_balanced_mae"]) / Cb["group_balanced_mae"] >= 0.10,
    "topdecile>=C": Pc["top_decile"]["precision"] >= Cb["top_decile"]["precision"],
    "second_true_seed_rho>C": P2["spearman"] > Cb["spearman"],
    "paired95CI_delta_rho_lower>0": (boot["delta_rho_ci95"] is not None and boot["delta_rho_ci95"][0] > 0),
}
results["selected_primary"] = sel
results["criteria"] = crit
results["criteria_all_met"] = all(crit.values())
results["paired_group_bootstrap_2000"] = boot
results["criteria_definition"] = ("predeclared: selected rho>=.30 AND rho-C>=.05 AND groupMAE reduction>=10% "
                                  "AND topdecile>=C AND second true seed rho>C AND paired95%CI delta-rho lower>0; "
                                  "C = E1 ridge; null collapse (constant prediction) rho is UNDEFINED not zero")
C.atomic_write_json(REP / "test_metrics.json", results)
C.atomic_write_json(REP / "strata.json", strata)
np.savez(REP / "test_predictions.npz", test_idx=te, y=y[te], y_delta2=y_delta2[te], y_delta05=y_delta05[te],
         **{n: p for n, p in preds.items()}, baseline_train_mean=base)
print(json.dumps({"selected": sel, "criteria": crit, "all_met": results["criteria_all_met"],
                  "E2_rho": g("E2_cnn_true_seed20261101")["spearman"],
                  "E3_rho": g("E3_cnn_true_seed20261102")["spearman"],
                  "E1_rho": Cb["spearman"], "boot": boot}, indent=2))
