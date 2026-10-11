#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 deliverable 1: corrected release metrics recomputed from SAVED targets/predictions.

Independent recomputation (no retraining) of E/S/B metrics for every batch16 arm/seed
and the shipped reference seed, plus matched HGB baselines on identical rows.
Read-only: consumes only existing batch16 artifacts; writes only under
reports/breakthrough_20261002/batch17/.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
B16 = ROOT / "models" / "breakthrough_20261002" / "batch16"
DATA16 = ROOT / "data" / "breakthrough_20261002" / "batch16"
REPORT17 = ROOT / "reports" / "breakthrough_20261002" / "batch17"
TASKS = ("E", "S", "B")
SEEDS = (20261016, 20261017)
REFERENCE_SEED = 20261016
ARMS = ("A", "B", "C")
HGB_KEY = {"E": "E45", "S": "S1069", "B": "B45"}
HGB_LABEL = {"E": "HGB_E45", "S": "HGB_S1069", "B": "HGB_B45"}
BOOT_SEED = 20261012
BOOT_N = 2000
CANDIDATE = "v6.15.1-release.rc1"


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i, n = 0, len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    rx = rankdata_average(x) - rankdata_average(x).mean()
    ry = rankdata_average(y) - rankdata_average(y).mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


def r2_score(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    ss_res = float(((yt - yp) ** 2).sum()); ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae(yt, yp):
    return float(np.mean(np.abs(np.asarray(yt, float) - np.asarray(yp, float))))


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "spearman": float("nan"), "r2": float("nan"), "mae": float("nan")}
    return {"n": int(ok.sum()), "spearman": spearman(p[ok], y[ok]),
            "r2": r2_score(y[ok], p[ok]), "mae": mae(y[ok], p[ok])}


def main():
    t0 = time.time()
    REPORT17.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA16 / "data.npz", allow_pickle=True)
    gene_split = data["gene_split"].astype(str)
    mask = {t: data["%s_mask" % t].astype(bool) for t in TASKS}
    Y = {t: data[t].astype(np.float64) for t in TASKS}
    comp = data["new_comp_dense"].astype(np.int64)
    gene_gid = data["gene_gid"].astype(str)
    sc = np.load(DATA16 / "scalers.npz", allow_pickle=True)

    def inv(task, p):
        return p * float(sc["%s_std" % task]) + float(sc["%s_mean" % task])

    predmap = {}
    load_notes = []
    for arm in ARMS:
        for seed in SEEDS:
            for task in TASKS:
                for split in ("val", "test"):
                    f = B16 / ("%s_s%d" % (arm, seed)) / ("pred_%s_%s.npy" % (task, split))
                    if not f.exists():
                        load_notes.append("missing %s" % f.name)
                        continue
                    p = np.load(f)
                    pred = inv(task, p[0].astype(np.float64))
                    y = p[1].astype(np.float64)
                    if arm == "A":
                        idx = np.where((gene_split == split) & mask[task])[0]
                    else:
                        idx = np.where(gene_split == split)[0]
                    if len(pred) != len(idx):
                        load_notes.append("LEN MISMATCH %s %s %s %s: %d vs %d" % (arm, seed, task, split, len(pred), len(idx)))
                        continue
                    if not np.allclose(y, Y[task][idx], equal_nan=True):
                        load_notes.append("Y MISMATCH %s %s %s %s" % (arm, seed, task, split))
                    predmap[(arm, seed, task, split)] = (idx, pred)

    per_seed = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = "%s_s%d" % (arm, seed)
            per_seed[key] = {}
            for task in TASKS:
                for split in ("val", "test"):
                    if (arm, seed, task, split) in predmap:
                        idx, pred = predmap[(arm, seed, task, split)]
                        per_seed[key]["%s_%s" % (task, split)] = metrics(Y[task][idx], pred)

    def split_index(arm, task, split):
        if arm == "A":
            return np.where((gene_split == split) & mask[task])[0]
        return np.where(gene_split == split)[0]

    def mean_of_seed(arm, task, split):
        rows = []
        for s in SEEDS:
            d = per_seed.get("%s_s%d" % (arm, s), {}).get("%s_%s" % (task, split))
            if d:
                rows.append(d)
        if not rows:
            return None
        return {"n_per_seed": sorted({r["n"] for r in rows}), "n": rows[0]["n"],
                "spearman": float(np.mean([r["spearman"] for r in rows])),
                "r2": float(np.mean([r["r2"] for r in rows])),
                "mae": float(np.mean([r["mae"] for r in rows])),
                "estimand": "unweighted mean of per-seed metrics"}

    def ensemble(arm, task, split):
        seeds_avail = [s for s in SEEDS if (arm, s, task, split) in predmap]
        if not seeds_avail:
            return None, None, None
        base_idx = np.where(gene_split == split)[0]
        pos = {int(g): i for i, g in enumerate(base_idx)}
        acc = np.zeros(len(base_idx)); cnt = np.zeros(len(base_idx))
        for s in seeds_avail:
            idx, pred = predmap[(arm, s, task, split)]
            for i, g in enumerate(idx):
                j = pos[int(g)]; acc[j] += pred[i]; cnt[j] += 1
        ens = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        valid = mask[task][base_idx]
        m = metrics(Y[task][base_idx][valid], ens[valid])
        m["estimand"] = "Spearman/R2/MAE of mean-over-seed predicted value (ensemble), valid-mask rows only"
        m["n_seeds"] = len(seeds_avail)
        return m, base_idx, ens

    ats = {}
    for arm in ARMS:
        ats[arm] = {}
        for task in TASKS:
            ats[arm][task] = {}
            for split in ("val", "test"):
                ats[arm][task][split] = {"mean_of_seed": mean_of_seed(arm, task, split),
                                         "ensemble": ensemble(arm, task, split)[0]}

    # matched HGB baseline + same-row comparison
    hgb_rows = {}
    with open(B16 / "hgb_predictions.tsv", encoding="utf-8") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        cidx = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(hdr):
                continue
            key = (parts[cidx["baseline"]], parts[cidx["task"]], parts[cidx["split"]])
            hgb_rows.setdefault(key, {})[parts[cidx["gene_id"]]] = float(parts[cidx["yhat"]])
    gid_to_pos = {g: i for i, g in enumerate(gene_gid)}

    hgb_metrics = {}
    same_row = {}
    for task in TASKS:
        hgb_metrics[task] = {}
        same_row[task] = {}
        for split in ("val", "test"):
            rows = hgb_rows.get((HGB_KEY[task], task, split), {})
            gidx = np.array([gid_to_pos[g] for g in rows if g in gid_to_pos], dtype=np.int64)
            pv = np.array([rows[gene_gid[g]] for g in gidx], dtype=np.float64)
            d = metrics(Y[task][gidx], pv)
            d["estimand"] = "single matched HGB baseline (%s) on the corrected split" % HGB_LABEL[task]
            hgb_metrics[task][split] = d
            ens_m, baseA, ensA = ensemble("A", task, split)
            posA = {int(g): i for i, g in enumerate(baseA)}
            common = [g for g in gidx if int(g) in posA]
            ci = np.array([posA[int(g)] for g in common], dtype=np.int64)
            yc = Y[task][baseA[ci]]
            same_row[task][split] = {
                "n_common_rows": int(len(common)),
                "neural_A_ensemble": metrics(yc, ensA[ci]),
                "hgb": metrics(yc, np.array([rows[gene_gid[baseA[j]]] for j in ci], dtype=np.float64)),
                "note": "both metrics computed on the identical gene rows (matched comparison)"}

    def rule_for(arm):
        per_target = {}
        gains = []
        for task in TASKS:
            a = ats["A"][task]["val"]["mean_of_seed"]["spearman"]
            x = ats[arm][task]["val"]["mean_of_seed"]["spearman"]
            g = x - a
            per_target[task] = {"A_val_rho_meanseed": a,
                                "%s_val_rho_meanseed" % arm: x, "gain": g}
            gains.append(g)
        mg = float(np.mean(gains))
        no_drop = all(g > -0.02 for g in gains)
        return {"per_target": per_target, "mean_gain": mg, "no_drop_gt_0.02": bool(no_drop),
                "rule_pass": bool(mg >= 0.02 and no_drop),
                "rule": "mean per-target val rho gain >= 0.02 AND no target drop > 0.02",
                "estimand": "mean-of-seed VALIDATION Spearman (frozen rule uses validation only)"}

    validation_rule = {arm: rule_for(arm) for arm in ("B", "C")}

    def component_bootstrap(task, split, arm_x, n=BOOT_N, seed=BOOT_SEED):
        _, baseA, ensA = ensemble("A", task, split)
        ensX = ensemble(arm_x, task, split)[2]
        valid = mask[task][baseA]
        y = Y[task][baseA][valid]
        pa = ensA[valid]; px = ensX[valid]; c = comp[baseA][valid]
        ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(px)
        y, pa, px, c = y[ok], pa[ok], px[ok], c[ok]
        base = spearman(px, y) - spearman(pa, y)
        comps = np.unique(c)
        order = np.argsort(c, kind="mergesort"); cs = c[order]
        bounds = np.flatnonzero(np.diff(cs)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(cs)]])
        rng = np.random.default_rng(seed)
        vals = np.empty(n)
        for i in range(n):
            chosen = rng.integers(0, len(comps), size=len(comps))
            idx = np.concatenate([order[int(starts[k]):int(ends[k])] for k in chosen])
            vals[i] = spearman(px[idx], y[idx]) - spearman(pa[idx], y[idx])
        return {"estimand": "ENSEMBLE delta = Spearman(mean-over-seed pred_%s, y) - Spearman(mean-over-seed pred_A, y)" % arm_x,
                "contrast": "%s minus A" % arm_x,
                "source_seeds": list(SEEDS),
                "valid_rows": int(len(y)),
                "component_grouping": "new_comp_dense (component-union); whole components resampled with replacement",
                "n_components": int(len(comps)),
                "resamples": int(n), "bootstrap_seed": int(seed),
                "base_point_estimate": float(base),
                "boot_mean": float(np.mean(vals)),
                "ci95_low": float(np.percentile(vals, 2.5)),
                "ci95_high": float(np.percentile(vals, 97.5)),
                "note": "ensemble estimand does NOT equal the mean-of-seed rho difference; both reported"}

    bootstrap = {arm_x: {t: component_bootstrap(t, "test", arm_x) for t in TASKS}
                 for arm_x in ("B", "C")}

    tsv = ["group\tarm\tseed\ttask\tsplit\tspearman\tr2\tmae\tn"]
    for arm in ARMS:
        for seed in SEEDS:
            key = "%s_s%d" % (arm, seed)
            for task in TASKS:
                for split in ("val", "test"):
                    d = per_seed[key].get("%s_%s" % (task, split))
                    if d:
                        tsv.append("per_seed\t%s\t%d\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (arm, seed, task, split, d["spearman"], d["r2"], d["mae"], d["n"]))
    for arm in ARMS:
        for task in TASKS:
            for split in ("val", "test"):
                m = ats[arm][task][split]["mean_of_seed"]
                if m:
                    tsv.append("mean_of_seed\t%s\t.\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (arm, task, split, m["spearman"], m["r2"], m["mae"], m["n"]))
                e = ats[arm][task][split]["ensemble"]
                if e:
                    tsv.append("ensemble\t%s\t.\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (arm, task, split, e["spearman"], e["r2"], e["mae"], e["n"]))
    for task in TASKS:
        for split in ("val", "test"):
            d = hgb_metrics[task][split]
            tsv.append("baseline_hgb\t%s\t.\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (HGB_LABEL[task], task, split, d["spearman"], d["r2"], d["mae"], d["n"]))
            sr = same_row[task][split]
            tsv.append("same_row_neuralA_ensemble\tA\t.\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (task, split, sr["neural_A_ensemble"]["spearman"], sr["neural_A_ensemble"]["r2"], sr["neural_A_ensemble"]["mae"], sr["neural_A_ensemble"]["n"]))
            tsv.append("same_row_hgb\t%s\t.\t%s\t%s\t%.8f\t%.8f\t%.8f\t%d" % (HGB_LABEL[task], task, split, sr["hgb"]["spearman"], sr["hgb"]["r2"], sr["hgb"]["mae"], sr["hgb"]["n"]))
    (REPORT17 / "corrected_release_metrics.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")

    out = {
        "candidate_version": CANDIDATE,
        "source_candidate": "v6.15.0-multitask.rc1 (batch16 weights; packaging repair only)",
        "batch": "batch17",
        "kind": "corrected release metrics recomputed from saved targets/predictions (no retraining)",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "provenance": {"source_dir": str(B16), "data_dir": str(DATA16),
                       "reference_seed": REFERENCE_SEED, "seeds": list(SEEDS),
                       "note": "saved per-fit prediction arrays; no new fits; test retrospective only"},
        "load_notes": load_notes,
        "known_confounds_prominent": [
            "TRAINER/INIT/DATA-ORDER CONFOUND (batch16 F1): A_s20261016(all) and A_s20261017 E/S were fit by b16_train.py (global torch-RNG shuffle); B/C arms and A_s20261017 B were fit by b16_train_resume.py after process loss (deterministic per-epoch sampler seed gen_seed+epoch*1000003). Same algorithm/optimizer/loss/epochs, but DIFFERENT initialization RNG stream and data-order schedule. This is a real confound in EVERY B-vs-A and C-vs-A contrast. It is not biased toward A, so it does not manufacture the negative shared-network result, but it blocks a strictly controlled causal task-sharing claim.",
            "AUXILIARY MISSINGNESS UNRESOLVED (batch16 F6): promoter_omics_targets_v1.tsv encodes uncovered assays as 0.0 with no NaN and no per-cell missingness flag. Zero-valued auxiliary measurements are treated as UNRESOLVED absence-vs-missing. No established benefit OR detriment of epigenetic auxiliary supervision is claimed.",
            "RECOVERY-ERA TEST IS TOY (batch16 F2): the batch16 interrupted-vs-continuous checkpoint test used Linear(1,1) on CPU. See resume_diagnostic.json for the batch17 production-path (CNN/AdamW/CUDA/memmap/masks/sampler) diagnostic.",
        ],
        "estimand_definitions": {
            "per_seed": "one-fit saved predictions, raw target units",
            "mean_of_seed": "unweighted mean of per-seed rho/r2/mae (n per seed listed)",
            "ensemble": "Spearman/R2/MAE of the mean-over-seed predicted value per gene",
            "baseline_hgb": "matched HistGradientBoosting from scratch on the corrected train split",
            "same_row": "neural-A ensemble vs matched HGB on identical gene rows",
            "validation_rule": "frozen: mean-of-seed per-target validation rho gain >= 0.02 AND no target drop > 0.02",
        },
        "per_seed": per_seed,
        "reference_shipped_seed": {"seed": REFERENCE_SEED,
                                   "note": "actual deployed checkpoints (A_s20261016); split-key primary seed; not test-selected",
                                   "per_seed_metrics": per_seed["A_s%d" % REFERENCE_SEED]},
        "mean_of_seed_and_ensemble": ats,
        "baseline_hgb": hgb_metrics,
        "same_row_neural_vs_hgb": same_row,
        "validation_rule_applied": validation_rule,
        "paired_component_bootstrap": bootstrap,
        "test_use_statement": "Test predictions existed from batch16 but were NOT used to pick the seed, arm or architecture; selection used validation only. Reported test numbers are retrospective heldout, not prospective validation.",
    }
    with open(REPORT17 / "corrected_release_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
        fh.write("\n")

    print(json.dumps({
        "reference_seed_per_seed": per_seed["A_s%d" % REFERENCE_SEED],
        "validation_rule": {k: {"mean_gain": v["mean_gain"], "rule_pass": v["rule_pass"]} for k, v in validation_rule.items()},
        "load_notes": load_notes,
        "wrote": [str(REPORT17 / "corrected_release_metrics.json"), str(REPORT17 / "corrected_release_metrics.tsv")],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
