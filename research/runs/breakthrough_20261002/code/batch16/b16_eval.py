#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 evaluation + task-sharing comparison + figures.

Primary contrast: shared (B) vs independent (A) per-target validation Spearman
gain (mean over seeds). Frozen acceptance rule: mean per-target rho gain >= 0.02
AND no target drops by more than 0.02. Secondary: R2/MAE, C vs A/B, heldout
paired component-bootstrap delta. No test-based selection.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT, CANDIDATE_VERSION,
                        TASKS, SEEDS, set_thread_limits, spearman, atomic_write_json)

set_thread_limits(4)


def load_neural():
    p = MODEL_OUT / "neural_results.json"
    if not p.exists():
        return {}
    return json.load(open(p, encoding="utf-8"))


def load_hgb():
    p = MODEL_OUT / "hgb_results.json"
    if not p.exists():
        return {}
    return json.load(open(p, encoding="utf-8"))["results"]


def mean_over_seeds(metrics_by_key, arm, task, split):
    vals = []
    for key, v in metrics_by_key.items():
        if key.startswith(arm + "_s"):
            d = v["metrics"].get("%s_%s" % (task, split))
            if d and d.get("n", 0) > 0:
                vals.append(d["spearman"])
    return float(np.mean(vals)) if vals else float("nan")


def component_bootstrap_diff(pred_b, pred_a, y, comp, seed=20261012, n=2000):
    """Paired component-level bootstrap of spearman(B) - spearman(A)."""
    y = np.asarray(y, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    comp = np.asarray(comp)
    ok = np.isfinite(y) & np.isfinite(pb) & np.isfinite(pa)
    y, pb, pa, comp = y[ok], pb[ok], pa[ok], comp[ok]
    if len(y) < 2:
        return None
    base = spearman(pb, y) - spearman(pa, y)
    comps = np.unique(comp)
    order = np.argsort(comp, kind="mergesort")
    cs = comp[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(cs)]])
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen = rng.integers(0, len(comps), size=len(comps))
        idx = np.concatenate([order[int(starts[c]):int(ends[c])] for c in chosen])
        vals[i] = spearman(pb[idx], y[idx]) - spearman(pa[idx], y[idx])
    return {"base": float(base), "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_components": int(len(comps)), "n_boot": int(n)}


def main():
    t0 = time.time()
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    neural = load_neural()
    hgb = load_hgb()

    arms_seeds = sorted(neural.get("results", {}).keys())
    comparison = {"hgb": {}, "neural": {}}
    for task in TASKS:
        comparison["hgb"][task] = {}
        for split in ("val", "test"):
            base_key = "S1069" if task == "S" else ("E45" if task == "E" else "B45")
            d = hgb.get(base_key, {}).get(split, {})
            comparison["hgb"][task][split] = {"spearman": d.get("spearman"),
                                              "r2": d.get("r2"), "mae": d.get("mae"),
                                              "n": d.get("n")}

    for arm in ("A", "B", "C"):
        comparison["neural"][arm] = {}
        for task in TASKS:
            comparison["neural"][arm][task] = {}
            for split in ("val", "test"):
                d = mean_over_seeds(neural.get("results", {}), arm, task, split)
                comparison["neural"][arm][task][split] = {"spearman": d}

    # ---- primary contrast B vs A (validation rho) ----
    primary = {}
    for task in TASKS:
        a = comparison["neural"]["A"][task]["val"]["spearman"]
        b = comparison["neural"]["B"][task]["val"]["spearman"]
        primary[task] = {"A_val_rho": a, "B_val_rho": b, "gain": (b - a) if not (np.isnan(a) or np.isnan(b)) else float("nan")}
    gains = [primary[t]["gain"] for t in TASKS if not np.isnan(primary[t]["gain"])]
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    no_drop = all(g > -0.02 for g in gains)
    pass_rule = (mean_gain >= 0.02) and no_drop
    primary["summary"] = {"mean_gain": mean_gain, "no_drop_gt_0.02": no_drop,
                          "rule_pass": pass_rule,
                          "rule": "mean per-target val rho gain >= 0.02 AND no target drop > 0.02"}

    # ---- C vs A and C vs B ----
    for task in TASKS:
        c = comparison["neural"]["C"][task]["val"]["spearman"]
        primary[task]["C_val_rho"] = c
        primary[task]["C_vs_A_gain"] = (c - primary[task]["A_val_rho"]) if not np.isnan(c) else float("nan")
        primary[task]["C_vs_B_gain"] = (c - primary[task]["B_val_rho"]) if not np.isnan(c) else float("nan")

    # ---- heldout (test) metrics ----
    heldout = {}
    for arm in ("A", "B", "C"):
        heldout[arm] = {}
        for task in TASKS:
            heldout[arm][task] = mean_over_seeds(neural.get("results", {}), arm, task, "test")

    # ---- paired component bootstrap (B vs A) on heldout, per task ----
    bootstrap = {}
    data = np.load(DATA_OUT / "data.npz", allow_pickle=True)
    comp = data["new_comp_dense"].astype(np.int64)
    gene_split = data["gene_split"].astype(str)
    te = gene_split == "test"
    for task in TASKS:
        # aggregate predictions over seeds (mean) from saved npy
        preds_b, preds_a = [], []
        for seed in SEEDS:
            for arm, arr in (("B", preds_b), ("A", preds_a)):
                f = MODEL_OUT / ("%s_s%d" % (arm, seed)) / ("pred_%s_test.npy" % task)
                if f.exists():
                    p = np.load(f)
                    arr.append(p[0])
        if preds_b and preds_a:
            y = data[task][te]
            m = data["%s_mask" % task][te]
            # B/C pred files span the full test set (5851); A pred files are already
            # restricted to masked-valid genes (e.g. 1880 for S). Normalize both to
            # the valid-gene vector before pooling across seeds.
            def _mask_valid(pred):
                return pred[m] if len(pred) == len(m) else pred

            pb = np.mean([_mask_valid(p) for p in preds_b], axis=0)
            pa = np.mean([_mask_valid(p) for p in preds_a], axis=0)
            assert len(pb) == len(pa) == int(m.sum()), (task, len(pb), len(pa), int(m.sum()))
            comp_te = comp[te][m]
            y_m = y[m]
            bootstrap[task] = component_bootstrap_diff(pb, pa, y_m, comp_te)

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": "batch16",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "arms_seeds_done": arms_seeds,
        "comparison": comparison,
        "primary_contrast_B_vs_A": primary,
        "heldout_test_rho": heldout,
        "paired_component_bootstrap_delta": bootstrap,
    }
    atomic_write_json(REPORT_OUT / "comparison.json", summary)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _make_figs(comparison, primary, heldout, REPORT_OUT)
    except Exception as e:
        summary["figure_error"] = str(e)

    print(json.dumps({
        "primary": primary,
        "heldout_test_rho": heldout,
        "bootstrap": bootstrap,
        "comparison_json": str(REPORT_OUT / "comparison.json"),
    }, ensure_ascii=False, indent=2))


def _make_figs(comparison, primary, heldout, out):
    import matplotlib.pyplot as plt

    # fig1: per-target validation rho across arms (A/B/C) + HGB
    fig, ax = plt.subplots(figsize=(8, 5))
    targets = list(TASKS)
    x = np.arange(len(targets))
    width = 0.2
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for i, arm in enumerate(("A", "B", "C")):
        vals = [comparison["neural"][arm][t]["val"]["spearman"] for t in targets]
        ax.bar(x + (i - 1) * width, vals, width, label=arm, color=colors[arm])
    hgb_vals = [comparison["hgb"][t]["val"]["spearman"] for t in targets]
    ax.scatter(x, hgb_vals, marker="x", color="black", s=80, label="HGB", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("validation Spearman rho")
    ax.set_title("Batch16 validation per-target rho (A/B/C vs HGB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_val_rho.png", dpi=150)
    fig.savefig(out / "fig_val_rho.svg")
    plt.close(fig)

    # fig2: B - A gain per target
    fig, ax = plt.subplots(figsize=(6, 5))
    gains = [primary[t]["gain"] for t in targets]
    ax.bar(targets, gains, color=["green" if g >= 0 else "red" for g in gains])
    ax.axhline(0.02, color="green", linestyle="--", label="+0.02 threshold")
    ax.axhline(0, color="gray", linestyle="-")
    ax.set_ylabel("B - A validation rho gain")
    ax.set_title("Shared vs independent task-sharing gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_gain.png", dpi=150)
    fig.savefig(out / "fig_gain.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
