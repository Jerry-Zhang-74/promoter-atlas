# -*- coding: utf-8 -*-
"""batch26 step 3: held-out TEST evaluation of the four frozen fits.

Runs ONLY after the selection manifest exists (test targets opened once).
No fits here: inference from saved checkpoints / saved ridge weights only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C
import b26_metrics as M
import b26_train as TR

sys.path.insert(0, str(C.BATCH02_CODE))
import trackB_grouped_cv as T  # noqa: E402


def _vhex(v):
    return C.sha256_bytes(v.encode())[:16]


def load_test():
    rows = TR.load_rows()
    codes, lens = TR.load_encoded()
    te = np.array(sorted(i for i, r in enumerate(rows) if int(r["split"]) == 2))
    return rows, codes, lens, te


def infer_all(rows, codes, lens, te):
    import torch
    dev = "cuda:0"
    import trackB_grouped_cv as TT
    preds = {}
    # baseline C from saved weights (no fit)
    W = np.load(TR.ATTEMPTS / "A0_ridgeC" / "ridge_weights.npy")
    w = W[:342]; mu_x = W[342:684]; sd_x = W[684:1026]; mu_y = float(W[1026]); sd_y = float(W[1027])
    seqs = [rows[i]["vreg_seq"] for i in te]
    X = TR.features342(seqs)
    preds["A0_ridgeC"] = ((X - mu_x) / sd_x) @ w * sd_y + mu_y
    for aid in ("A1_cnn_true_seed20261026", "A2_cnn_true_seed20261027", "A3_cnn_shuffle_seed20261026"):
        p = TR.ATTEMPTS / aid / "ckpt.pt"
        if not p.is_file():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = TT.build_cnn("legnet")
        model.load_state_dict(ck["model"]); model.eval().to(dev)
        pt = TR.predict_all(model, codes, lens, te, dev, torch)
        preds[aid] = pt * ck["config"]["sd_y"] + ck["config"]["mu_y"]
    return preds


def _boot(pred_a, pred_b, y, groups, n=2000):
    return M.group_bootstrap(np.asarray(pred_a), np.asarray(pred_b), y, groups, n=n)


def within_locus(rows, te, preds):
    """predicted/observed mutant-vs-ref deltas per test locus."""
    ref_pred = {}
    ref_obs = {}
    for k, i in enumerate(te):
        r = rows[i]
        if r["variant"].rsplit("_", 1)[1] == "ref":
            ref_pred[r["locus_id"]] = {m: np.asarray(p[k]) for m, p in preds.items()}
            ref_obs[r["locus_id"]] = float(r["mean_log2_ratio"])
    out = {}
    for m in preds:
        dp, do, gg, ll = [], [], [], []
        for k, i in enumerate(te):
            r = rows[i]
            if r["variant"].rsplit("_", 1)[1] == "ref":
                continue
            L = r["locus_id"]
            if L not in ref_pred:
                continue
            dp.append(float(preds[m][k]) - float(ref_pred[L][m]))
            do.append(float(r["mean_log2_ratio"]) - ref_obs[L])
            gg.append(r["group_id"]); ll.append(r["locus_id"])
        dp = np.asarray(dp); do = np.asarray(do); gg = np.asarray(gg, dtype=object)
        if len(dp) < 10:
            out[m] = {"n_pairs": int(len(dp))}
            continue
        nz = (dp != 0) & (do != 0)
        sign = float(np.mean(np.sign(dp[nz]) == np.sign(do[nz]))) if nz.sum() else float("nan")
        rho_d = M.spearman(dp, do)
        # single-rho group bootstrap of the delta-Spearman (NOT a difference vs a
        # zero vector, which is degenerate and yields 0 resamples).
        dboot = M.bootstrap_rho(dp, do, gg, n=2000, seed=20261030)
        out[m] = {"n_pairs": int(len(dp)), "n_loci": int(len(set(ll))),
                  "n_groups_bootstrapped": int(len(set(gg))),
                  "n_nonzero_both": int(nz.sum()),
                  "ties_in_delta": int(np.sum((dp == 0) | (do == 0))),
                  "delta_spearman": rho_d,
                  "delta_spearman_defined": bool(np.isfinite(rho_d)),
                  "delta_mae": M.mae(dp, do),
                  "sign_agreement_nonzero": sign,
                  "delta_rho_bootstrap": dboot,
                  "delta_rho_ci95": dboot["rho_ci95"]}
    # within-locus centered scores (controls between-locus differences)
    cen = {}
    for m in preds:
        pp = np.asarray(preds[m])
        ptrue = np.asarray([float(rows[i]["mean_log2_ratio"]) for i in te])
        g = np.asarray([rows[i]["locus_id"] for i in te], dtype=object)
        pc = pp.copy(); tc = ptrue.copy()
        for L in np.unique(g):
            msk = g == L
            if msk.sum() >= 2:
                pc[msk] -= pc[msk].mean(); tc[msk] -= tc[msk].mean()
        cen[m] = M.spearman(pc, tc)
    return out, cen


def main():
    man = C.read_json(C.REPORT26 / "selection_manifest.json")
    if not man:
        raise SystemExit("selection manifest missing: refusing to open test targets")
    primary = man["selected_primary"]
    rows, codes, lens, te = load_test()
    y = np.array([float(rows[i]["mean_log2_ratio"]) for i in te])
    loc = np.array([rows[i]["locus_id"] for i in te], dtype=object)
    grp = np.array([rows[i]["group_id"] for i in te], dtype=object)
    vh = np.array([_vhex(rows[i]["variant"]) for i in te])
    preds = infer_all(rows, codes, lens, te)
    train_mean = float(np.mean([float(r["mean_log2_ratio"]) for r in rows if int(r["split"]) == 0]))
    tm = {}
    for m, p in preds.items():
        gb, ng = M.group_balanced_mae(y, p, loc)
        pv = np.asarray(p, dtype=np.float64)
        tm[m] = {"spearman": M.spearman(pv, y), "mae": M.mae(pv, y), "r2": M.r2(y, pv),
                 "group_balanced_mae": gb, "n_groups": ng,
                 "pred_std": float(pv.std()), "pred_nunique": int(len(np.unique(np.round(pv, 6)))),
                 "pred_rho_defined": bool(np.isfinite(M.spearman(pv, y))),
                 "top_decile": M.top_decile(y, pv, vh)}
    tm["train_mean"] = {"spearman": M.spearman(np.full(len(y), train_mean), y),
                        "mae": M.mae(np.full(len(y), train_mean), y),
                        "r2": M.r2(y, np.full(len(y), train_mean)), "value": train_mean,
                        "pred_nunique": 1}
    boot = {}
    for m in ("A1_cnn_true_seed20261026", "A2_cnn_true_seed20261027",
              "A3_cnn_shuffle_seed20261026"):
        if m in preds:
            boot["%s_vs_A0_ridgeC" % m] = _boot(preds[m], preds["A0_ridgeC"], y, grp)
    wl, cen = within_locus(rows, te, preds)
    # predeclared task criteria
    c = tm.get("A0_ridgeC", {})
    sel = tm.get(primary, {})
    other = "A2_cnn_true_seed20261027" if primary.startswith("A1") else "A1_cnn_true_seed20261026"
    oth = tm.get(other, {})
    sh = tm.get("A3_cnn_shuffle_seed20261026", {})
    dboot = boot.get("%s_vs_A0_ridgeC" % primary, {})
    ci = dboot.get("delta_rho_ci95") or [None, None]
    sel_rho = sel.get("spearman")
    sh_rho = sh.get("spearman") if sh else None
    sh_defined = bool(sh) and np.isfinite(sh_rho) if sh_rho is not None else False
    crit = {
        "rho_increase_ge_0.05": (sel_rho - c.get("spearman", 9)) >= 0.05,
        "group_balanced_mae_reduction_ge_5pct": (
            (c.get("group_balanced_mae", 9) - sel.get("group_balanced_mae", 9))
            / c.get("group_balanced_mae", 9) >= 0.05) if c.get("group_balanced_mae") else False,
        "top_decile_precision_not_lower": (
            sel.get("top_decile", {}).get("precision", -1) >= c.get("top_decile", {}).get("precision", 9)),
        "other_true_seed_rho_gt_C": oth.get("spearman", -9) > c.get("spearman", 9),
        "paired_ci95_delta_rho_lower_gt_0": (ci[0] is not None and ci[0] > 0),
    }
    # shuffled-control sanity margin: predeclared criterion, but the control's test
    # prediction is a single constant (Spearman undefined).  Undefined is reported as
    # undefined (None) -- NOT substituted by zero, which would manufacture a pass.
    if sh is None:
        crit["shuffle_rho_at_least_0.05_below_selected"] = None
        shuffle_note = "null control absent (not applicable)"
    elif not sh_defined:
        crit["shuffle_rho_at_least_0.05_below_selected"] = None
        sh_const = float(np.mean(preds["A3_cnn_shuffle_seed20261026"]))
        shuffle_note = ("shuffled-label control rho UNDEFINED: test predictions are the single "
                        "constant %.6f (pred_nunique=%s, pred_std=%.4g); Spearman has zero rank "
                        "variance.  The numeric 0.05 margin is NOT evaluated and NOT substituted "
                        "with 0.  Qualitatively this is a collapsed negative control with no "
                        "ranking signal: top-decile precision %.4f vs chance %.4f."
                        % (sh_const, sh.get("pred_nunique", "?"), sh.get("pred_std", float("nan")),
                           sh.get("top_decile", {}).get("precision", float("nan")),
                           sh.get("top_decile", {}).get("baseline_precision", float("nan"))))
    else:
        crit["shuffle_rho_at_least_0.05_below_selected"] = (sel_rho - sh_rho) >= 0.05
        shuffle_note = "evaluated"
    w = wl.get(primary, {})
    dsp = w.get("delta_spearman")
    crit["variant_delta_spearman_gt_0.10"] = (dsp is not None and np.isfinite(dsp) and dsp > 0.10)
    dci = w.get("delta_rho_ci95") or [None, None]
    crit["variant_delta_ci95_lower_gt_0"] = (dci[0] is not None and np.isfinite(dci[0]) and dci[0] > 0)
    sa = w.get("sign_agreement_nonzero")
    crit["variant_sign_agreement_gt_0.55"] = (sa is not None and np.isfinite(sa) and sa > 0.55)
    all_met = all(v is True for v in crit.values())
    n_true = sum(1 for v in crit.values() if v is True)
    n_false = sum(1 for v in crit.values() if v is False)
    n_undef = sum(1 for v in crit.values() if v is None)
    out = {"batch": "batch26", "version": "v6.19.0-strvariant.rc1", "split": "test",
           "n_test_rows": int(len(te)), "n_test_loci": int(len(set(loc))),
           "n_test_groups": int(len(set(grp))), "selected_primary": primary,
           "test_metrics": tm, "bootstrap_vs_baseline": boot,
           "within_locus": wl, "within_locus_centered_spearman": cen,
           "predeclared_criteria": crit,
           "criteria_all_met": all_met,
           "criteria_n_true": n_true, "criteria_n_false": n_false, "criteria_n_undefined": n_undef,
           "criteria_undefined_names": [k for k, v in crit.items() if v is None],
           "shuffled_control_note": shuffle_note,
           "variant_direction_support_met": bool(
               crit["variant_delta_spearman_gt_0.10"] and crit["variant_delta_ci95_lower_gt_0"]
               and crit["variant_sign_agreement_gt_0.55"]),
           "units": "log2(cDNA/gDNA) (same-study); predictions in the same units",
           "test_opened_at": C.now_local(),
           "note": ("internal retrospective locus-held-out test; NOT external or prospective "
                    "validation.  STR+flank fragment in a fixed minP-GFP context, not autonomous "
                    "promoter activity.")}
    C.atomic_write_json(C.REPORT26 / "test_metrics.json", out)
    lines = ["variant\tlocus_id\tgroup_id\ttrue\t" + "\t".join(sorted(preds))]
    for k, i in enumerate(te):
        lines.append("\t".join([rows[i]["variant"], rows[i]["locus_id"], rows[i]["group_id"],
                                "%.6f" % y[k]] + ["%.6f" % preds[m][k] for m in sorted(preds)]))
    C.atomic_write_text(C.DATA26 / "test_predictions.tsv", "\n".join(lines) + "\n")
    print("[eval] primary=%s rho=%.4f (C=%.4f) gbMAE=%.4f (C=%.4f) all_met=%s" %
          (primary, sel.get("spearman", float("nan")), c.get("spearman", float("nan")),
           sel.get("group_balanced_mae", float("nan")), c.get("group_balanced_mae", float("nan")),
           out["criteria_all_met"]), flush=True)
    return out


if __name__ == "__main__":
    main()
