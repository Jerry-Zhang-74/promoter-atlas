#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch37 step 3 - ZERO-fit metrics from cached scores (CPU only, no rescoring).

Spearman(d, logFC), strict sign accuracy, zero-fit baselines (delta GC fraction,
delta length), fixed direction baselines, group-balanced sign accuracy, paired
group bootstrap (2000), label-free groups by interval-union overlap. NaN labels are
excluded from numeric metrics only after row-count disclosure; no undefined rho->0.
"""
import csv, json, os, sys, math, hashlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "data/breakthrough_20261002/batch37")
RPT = os.path.join(REPO, "reports/breakthrough_20261002/batch37")
THRESH = {"KV_rho_warrants_separate_path": 0.20, "min_effective_groups_for_bootstrap": 30,
          "per_region_min_pairs": 10}
RNG = np.random.default_rng(20261004)

def _rank(a):
    order = np.argsort(a, kind="mergesort"); r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    out = r.copy(); i = 0; s = np.sort(a)
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]: j += 1
        if j > i:
            avg = (i + j) / 2.0
            for k in range(i, j + 1): out[order[k]] = avg
        i = j + 1
    return out

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]): return float("nan")
    rx = _rank(x); ry = _rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")

def gc_frac(s): return (s.count("G") + s.count("C")) / len(s)

def strict_sign(y, pred):
    m = np.isfinite(y) & (y != 0)
    if m.sum() == 0: return float("nan"), 0
    return float(np.mean(np.sign(y[m]) == np.sign(pred[m]))), int(m.sum())

def fixed_dir_acc(y, sign):
    m = np.isfinite(y) & (y != 0)
    if m.sum() == 0: return float("nan"), 0
    return float(np.mean(np.sign(y[m]) == sign)), int(m.sum())

def load_pairs(orders):
    pairs = []
    with open(os.path.join(OUTDIR, "paired_inputs.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = r["label_logFC"]
            lab = float(lab) if lab not in ("", "None") else float("nan")
            a_id = hashlib.sha256(r["alt_insert"].encode()).hexdigest()[:20]
            rf_id = hashlib.sha256(r["ref_insert"].encode()).hexdigest()[:20]
            try:
                adjp = float(r["label_adjP"])
            except Exception:
                adjp = float("nan")
            pairs.append({"arm": r["arm"], "orig": r["original_id"], "pos": int(r["pos"]),
                          "vclass": r["variant_class"], "status": r["pair_status"],
                          "group": int(r["group_id"]), "y": lab, "d": orders[a_id] - orders[rf_id],
                          "alt": r["alt_insert"], "ref": r["ref_insert"],
                          "dgcf": gc_frac(r["alt_insert"]) - gc_frac(r["ref_insert"]),
                          "dlen": len(r["alt_insert"]) - len(r["ref_insert"]), "adjP": adjp})
    return pairs

def summarize(sub):
    y = np.array([p["y"] for p in sub]); d = np.array([p["d"] for p in sub])
    dg = np.array([p["dgcf"] for p in sub]); dl = np.array([p["dlen"] for p in sub])
    fin = np.isfinite(y)
    out = {"n_rows": len(sub), "n_label_finite": int(fin.sum()), "n_label_missing": int((~fin).sum()),
           "n_pred_ties_zero_d": int((d == 0).sum()),
           "spearman_d_logFC": spearman(d[fin], y[fin]),
           "spearman_deltaGC_logFC": spearman(dg[fin], y[fin]),
           "spearman_deltaLen_logFC": spearman(dl[fin], y[fin]),
           "mean_d": float(np.mean(d)), "mean_logFC": float(np.mean(y[fin])) if fin.sum() else float("nan")}
    acc, n = strict_sign(y, d); out["model_strict_sign_accuracy"] = acc; out["sign_denominator"] = n
    ap, _ = fixed_dir_acc(y, 1.0); an, _ = fixed_dir_acc(y, -1.0)
    out["fixed_always_positive_accuracy"] = ap; out["fixed_always_negative_accuracy"] = an
    return out

def main():
    orders = {}
    foldvals = []
    foldcols = ["HEK293_fold%d.parm" % i for i in range(5)]
    with open(os.path.join(OUTDIR, "unique_scores.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            orders[r["seq_id"]] = float(r["mean_score"])
            foldvals.append([float(r[c]) for c in foldcols])
    fold_sd = float(np.array(foldvals).std(axis=1).mean())

    pairs = load_pairs(orders)
    sat = [p for p in pairs if p["arm"] == "SAT"]
    kv = [p for p in pairs if p["arm"] == "KV"]

    res = {"kv_all": summarize(kv),
           "kv_valid_only": summarize([p for p in kv if p["status"] == "VALID_SNV"]),
           "kv_degenerate_only": summarize([p for p in kv if p["status"] == "DEGENERATE_ALT_EQ_REF"]),
           "sat_all": summarize(sat),
           "sat_snv": summarize([p for p in sat if p["vclass"] == "SNV"]),
           "sat_del": summarize([p for p in sat if p["vclass"] == "deletion"])}

    by_group = {}
    for p in kv: by_group.setdefault(p["group"], []).append(p)
    gscores = []; gsign = []
    for g, ps in sorted(by_group.items()):
        y = np.array([p["y"] for p in ps]); d = np.array([p["d"] for p in ps])
        fin = np.isfinite(y)
        if fin.sum() >= THRESH["per_region_min_pairs"]:
            gscores.append({"group": g, "n": int(fin.sum()), "rho": spearman(d[fin], y[fin]),
                            "start": min(p["pos"] for p in ps), "end": max(p["pos"] for p in ps)})
        a, n = strict_sign(y, d)
        if n > 0: gsign.append(a)
    eff_groups = len([g for g, ps in by_group.items() if np.isfinite([p["y"] for p in ps]).sum() > 0])
    res["kv_group_stats"] = {"n_groups": len(by_group), "n_effective_groups": eff_groups,
                             "n_groups_ge10pairs": len(gscores),
                             "n_groups_constant_or_small": len(by_group) - len(gscores),
                             "group_balanced_mean_strict_sign_accuracy": float(np.mean(gsign)) if gsign else float("nan")}

    boot = {"applicable": eff_groups >= THRESH["min_effective_groups_for_bootstrap"],
            "n_effective_groups": eff_groups, "n_resamples": 2000, "multiplicity_retained": True}
    if boot["applicable"]:
        gids = list(by_group.keys()); rhos = []; drhos = []
        for _ in range(2000):
            pick = RNG.choice(len(gids), size=len(gids), replace=True)
            ys = []; ds = []; dgs = []
            for gi in pick:
                for p in by_group[gids[gi]]:
                    if np.isfinite(p["y"]):
                        ys.append(p["y"]); ds.append(p["d"]); dgs.append(p["dgcf"])
            ys = np.array(ys); ds = np.array(ds); dgs = np.array(dgs)
            r = spearman(ds, ys); rg = spearman(dgs, ys)
            if np.isfinite(r): rhos.append(r)
            if np.isfinite(r) and np.isfinite(rg): drhos.append(r - rg)
        rhos = np.array(rhos); drhos = np.array(drhos)
        boot["rho_point"] = res["kv_all"]["spearman_d_logFC"]
        boot["rho_ci95"] = [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))] if len(rhos) else None
        boot["rho_lower_gt_0"] = bool(len(rhos) and np.percentile(rhos, 2.5) > 0)
        boot["delta_rho_vs_deltaGC_point"] = res["kv_all"]["spearman_d_logFC"] - res["kv_all"]["spearman_deltaGC_logFC"]
        boot["delta_rho_ci95"] = [float(np.percentile(drhos, 2.5)), float(np.percentile(drhos, 97.5))] if len(drhos) else None
        boot["n_finite_boot_rho"] = int(len(rhos)); boot["n_finite_boot_drho"] = int(len(drhos))
    res["kv_bootstrap"] = boot

    sat_fin = [p for p in sat if np.isfinite(p["y"])]
    res["sat_secondary_descriptive"] = {
        "abs_logFC_lt_0.1": summarize([p for p in sat_fin if abs(p["y"]) < 0.1]),
        "abs_logFC_ge_0.1": summarize([p for p in sat_fin if abs(p["y"]) >= 0.1]),
        "adjP_le_0.05": summarize([p for p in sat_fin if np.isfinite(p["adjP"]) and p["adjP"] <= 0.05]),
        "note": "descriptive interpretation only; primary is never replaced by a significant subset",
    }

    sat_iv = [(p["pos"] - 109, p["pos"] + 109) for p in sat]
    sat_min, sat_max = min(i[0] for i in sat_iv), max(i[1] for i in sat_iv)
    kv_overlap = sum(1 for p in kv if (p["pos"] - 109) <= sat_max and (p["pos"] + 109) >= sat_min)
    res["sat_kv_overlap"] = {"sat_region": [sat_min, sat_max], "kv_variants_overlapping_sat_region": kv_overlap,
                             "note": "SAT and KV arms are never pooled"}
    res["fold_score_mean_abs_sd"] = fold_sd
    res["thresholds"] = THRESH

    rr = res["kv_all"]["spearman_d_logFC"]
    macro = res["kv_group_stats"]["group_balanced_mean_strict_sign_accuracy"]
    bp = res["kv_all"]["fixed_always_positive_accuracy"]; bn = res["kv_all"]["fixed_always_negative_accuracy"]
    audit_caveat = (res["kv_all"]["n_label_finite"] > 0 and
                    (res["kv_all"]["n_pred_ties_zero_d"] / res["kv_all"]["n_rows"] > 0.01))
    cond = [bool(np.isfinite(rr) and rr >= 0.20), bool(boot.get("rho_lower_gt_0", False)),
            bool(np.isfinite(macro) and np.isfinite(bp) and np.isfinite(bn) and macro > bp and macro > bn),
            not audit_caveat]
    res["decision_rule"] = {"conditions": {"rho_ge_0.20": cond[0], "bootstrap_lower_gt_0": cond[1],
                            "macro_sign_gt_both_fixed": cond[2], "no_audit_failure": cond[3]},
                            "all_met": all(cond), "audit_caveat_present": audit_caveat,
                            "verdict_note": "LIMITED unless all four met; groups<30 or CI undefined -> LIMITED not PASS"}

    with open(os.path.join(RPT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RPT, "table_kv_groups.tsv"), "w", encoding="utf-8", newline="") as f:
        f.write("group\tstart\tend\tn\trho\n")
        for g in gscores:
            f.write("%d\t%d\t%d\t%d\t%s\n" % (g["group"], g["start"], g["end"], g["n"],
                                              "" if not np.isfinite(g["rho"]) else "%.6f" % g["rho"]))
    print(json.dumps({"sat_all_rho": res["sat_all"]["spearman_d_logFC"],
                      "sat_snv_rho": res["sat_snv"]["spearman_d_logFC"],
                      "sat_del_rho": res["sat_del"]["spearman_d_logFC"],
                      "kv_all_rho": res["kv_all"]["spearman_d_logFC"],
                      "kv_valid_only_rho": res["kv_valid_only"]["spearman_d_logFC"],
                      "kv_bootstrap": boot, "decide": res["decision_rule"]}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
