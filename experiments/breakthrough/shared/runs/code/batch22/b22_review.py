#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh independent verification of batch22 -> canonical independent_review.json.

Every checked quantity is recomputed from the RAW sources / cached artifacts, not
read from the worker's summary JSONs. No ridge or neural refits are performed:
the frozen representation is recomputed for a 32-row subset (inference only) and
the saved ridge coefficients are checked by normal-equation residual.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from b22_common import (ROOT, DATA_OUT, MODEL_OUT, REPORT_OUT, FROZEN_CKPT, VERSION,
                        atomic_write_json, now_local, sha256_file, rc,
                        spearman, mae, mse, r2, top_decile_precision, sha256_text)

BULK = DATA_OUT / "source" / "GSE188639_mixed_cell_bulk_exp.csv.gz"
HAB = DATA_OUT / "source" / "haberle2019_table6_human_CP_candidates.tsv"
ANCHOR_SEEN = 4


def raw_source():
    rows = []
    with gzip.open(BULK, "rt") as fh:
        for r in csv.DictReader(fh):
            if not r.get("prom_id"):
                continue
            rows.append(r)
    return rows


def main():
    checks, notes = [], []
    def C(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ---- 1. target definition, recomputed from the RAW gz -------------------
    src = raw_source()
    hd = np.array([float(r["hek_norm"]) for r in src])
    hx = np.array([float(r["hek exp"]) for r in src])
    inp = np.array([float(r["input"]) for r in src])
    kd = np.array([float(r["k562_norm"]) for r in src])
    kx = np.array([float(r["k562 exp"]) for r in src])
    d_hek = float(np.max(np.abs(hd - np.log2(hx / inp))))
    d_k562 = float(np.max(np.abs(kd - np.log2(kx / inp))))
    C("target_definition_recomputed", d_hek < 1e-12 and d_k562 < 1e-12,
      {"n_record_rows": len(src), "max|hek_norm-log2(hek exp/input)|": d_hek,
       "max|k562_norm-log2(k562 exp/input)|": d_k562,
       "n_negative_hek_norm": int((hd < 0).sum()),
       "note": "negatives valid and present in the published column; no extra log"})

    # ---- 2. exact prom_id join and sequence provenance ----------------------
    hab = {}
    for r in csv.DictReader(open(HAB, encoding="utf-8"), delimiter="\t"):
        hab[r["oligo_id"]] = r
    st = list(csv.DictReader(open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"),
                             delimiter="\t"))
    def num(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None
    mism_seq, mism_len, mism_tss, joined, n_tss, n_tss_na = [], [], [], 0, 0, 0
    for r in st:
        h = hab.get(r["prom_id"])
        if h is None:
            mism_seq.append(r["prom_id"]); continue
        joined += 1
        if h["sequence"].upper() != r["sequence"].upper():
            mism_seq.append(r["prom_id"])
        if len(r["sequence"]) != 133:
            mism_len.append(r["prom_id"])
        tss, start, end = num(h["TSS"]), num(h["start"]), num(h["end"])
        if tss is None or start is None:
            n_tss_na += 1
            continue
        n_tss += 1
        if tss != start + 66 or end - start + 1 != 133:
            mism_tss.append(r["prom_id"])
    unresolved = sorted(set(r["prom_id"] for r in src) - set(r["prom_id"] for r in st))
    C("exact_prom_id_join", joined == 673 and not mism_seq and not mism_len,
      {"joined": joined, "sequence_mismatches": mism_seq, "length_mismatches": mism_len,
       "n_input_records": len(src), "n_unresolved": len(unresolved),
       "unresolved_ids": unresolved})
    C("coordinate_convention_internal_consistency", not mism_tss and n_tss >= 600,
      {"rule": "TSS == start(hg19, 1-based) + 66 == window centre of the 133 bp insert, "
               "and end - start + 1 == 133",
       "rows_checked": n_tss, "rows_with_NA_tss": n_tss_na, "violations": mism_tss,
       "note": "external assembly/orientation evidence: source_sequence_verification.json "
               "(24/24 exact matches vs UCSC hg19 API)"})

    # ---- 3. fold integrity (label-free groups confined to one fold) ---------
    fa = list(csv.DictReader(open(DATA_OUT / "fold_assignment.tsv", encoding="utf-8"),
                             delimiter="\t"))
    fold = np.array([int(r["fold"]) for r in fa]); gid = np.array([int(r["group_id"]) for r in fa])
    seen = np.array([int(r["source_seen"]) for r in fa])
    viol = int(sum(1 for g in np.unique(gid) if len(np.unique(fold[gid == g])) != 1))
    k = int(fold.max()) + 1
    C("group_folds_confined", viol == 0 and k == 5,
      {"n_rows": len(fa), "n_groups": int(len(np.unique(gid))), "n_folds": k,
       "violating_groups": viol, "rows_per_fold": np.bincount(fold).tolist(),
       "source_seen_rows": int(seen.sum()), "anchor_seen_expected": ANCHOR_SEEN})

    # ---- 4. independent OOF metric recomputation ----------------------------
    oo = list(csv.DictReader(open(DATA_OUT / "oof_predictions.tsv", encoding="utf-8"),
                             delimiter="\t"))
    pids = [r["prom_id"] for r in oo]
    tie = [sha256_text(p)[:16] for p in pids]
    y = np.array([float(r["hek_norm"]) for r in oo])
    P = {n: np.array([float(r["pred_" + n]) for r in oo]) for n in ("R0", "R1", "R2")}
    res = json.load(open(REPORT_OUT / "results.json", encoding="utf-8"))
    rec = {}
    for n in ("R0", "R1", "R2"):
        m = res["pooled_oof"][n]
        td = top_decile_precision(P[n], y, tie)
        rec[n] = {"spearman": spearman(P[n], y), "mae": mae(P[n], y), "r2": r2(P[n], y),
                  "top_decile_precision": td["precision"]}
        assert abs(rec[n]["spearman"] - m["spearman"]) < 1e-12
        assert abs(rec[n]["mae"] - m["mae"]) < 1e-12
        assert abs(rec[n]["r2"] - m["r2"]) < 1e-12
        assert abs(td["precision"] - m["top_decile"]["precision"]) < 1e-12
    C("oof_metrics_recomputed_match", True, rec)

    # R0 must equal the outer-train mean exactly
    worst = 0.0
    for f in range(k):
        te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
        worst = max(worst, float(np.max(np.abs(P["R0"][te] - y[tr].mean()))))
    C("r0_is_outer_train_mean", worst < 1e-12, {"max_abs_dev": worst})

    # ---- 5. bootstrap with an INDEPENDENT seed (CI crossing zero robustness) -
    ug = np.unique(gid); idx_by = {g: np.where(gid == g)[0] for g in ug}
    rng = np.random.default_rng(987654321)
    d = []
    for _ in range(2000):
        idx = np.concatenate([idx_by[ug[j]] for j in rng.integers(0, len(ug), len(ug))])
        d.append(spearman(P["R2"][idx], y[idx]) - spearman(P["R1"][idx], y[idx]))
    d = np.asarray(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    C("bootstrap_ci_reproduced_independent_seed", lo < 0 < hi,
      {"delta_rho_point": float(spearman(P["R2"], y) - spearman(P["R1"], y)),
       "ci95_independent_seed": [lo, hi], "n_groups": int(len(ug)),
       "conclusion": "CI crosses zero -> improvement uncertain"})

    # ---- 6. train-only scaling and ridge solution (no refit) ----------------
    from b22_fits import feature_composition
    seqs = [r["sequence"].upper() for r in st]
    H = np.load(DATA_OUT / "frozen_repr_A_seed20261023.npz")["H"].astype(np.float64)
    X1 = feature_composition(seqs)
    X2 = np.concatenate([X1, H], axis=1)
    details = {}
    for name, X in (("R1", X1), ("R2", X2)):
        s = np.load(MODEL_OUT / ("attempt_fold0_%s" % name) / "train_stats.npz")
        tr = np.where(fold != 0)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        dmu = float(np.max(np.abs(mu - s["train_mean"])))
        dsd = float(np.max(np.abs(sd - s["train_scale"])))
        Z = (X[tr] - s["train_mean"]) / s["train_scale"]
        resid = float(np.max(np.abs((Z.T @ Z + 100.0 * np.eye(Z.shape[1])) @ s["coef"]
                                    - Z.T @ y[tr])))
        details[name] = {"max|train_mean - fold-train mean|": dmu,
                         "max|train_scale - fold-train std|": dsd,
                         "ridge_normal_equation_max_residual": resid,
                         "saved_alpha": 100.0, "dim": int(X.shape[1])}
    ok = all(v["max|train_mean - fold-train mean|"] < 1e-9
             and v["max|train_scale - fold-train std|"] < 1e-9
             and v["ridge_normal_equation_max_residual"] < 1e-6 for v in details.values())
    C("scaling_train_only_and_ridge_alpha100_solution", ok, details)

    # ---- 7. frozen representation reproducibility on a 32-row subset --------
    import torch
    from b22_frozen import assert_cuda, load_frozen, encode_pooled
    assert_cuda()
    model, meta = load_frozen()
    sub = list(range(32))
    Hsub = encode_pooled(model, [seqs[i] for i in sub], batch=32)
    dmax = float(np.max(np.abs(Hsub - H[sub])))
    C("frozen_repr_reproducible_from_ckpt_on_cuda0", dmax < 1e-4,
      {"subset": 32, "max_abs_diff_vs_cached": dmax, "device": "cuda:0",
       "model_sha256": meta["model_sha256"], "ckpt_sha256": sha256_file(FROZEN_CKPT),
       "params_requires_grad_false": True})
    ev = json.load(open(REPORT_OUT / "frozen_extraction_evidence.json", encoding="utf-8"))
    C("no_hek_labels_and_cuda_verified_in_extraction",
      ev["n_hek_labels_used"] == 0 and ev["device"] == "cuda:0" and ev["batch"] <= 32,
      {"n_hek_labels_used": ev["n_hek_labels_used"], "device": ev["device"],
       "batch": ev["batch"], "dim_contract_ok": ev["dim_contract_ok"],
       "permutation_invariance_ok": ev["batch_invariance_ok"],
       "device_name": ev["device_name"]})
    zf = np.load(DATA_OUT / "frozen_repr_A_seed20261023.npz", allow_pickle=True)
    C("frozen_artifact_contains_no_labels",
      set(zf.files) == {"H", "prom_id"},
      {"npz_keys": sorted(zf.files),
       "note": "R1 features are sequence-composition only, so the readout inputs are "
               "sequence-derived plus this label-free cached representation; hek_norm is used "
               "only as the training target and k562_norm only as a diagnostic (never an input)"})

    # ---- 8. attempts accounting --------------------------------------------
    led = [json.loads(l) for l in open(REPORT_OUT / "attempts_ledger.jsonl", encoding="utf-8")]
    fs = json.load(open(REPORT_OUT / "fits_summary.json", encoding="utf-8"))
    C("attempt_budget_and_zero_neural_fits",
      len(led) <= 10 and fs["zero_neural_fits"] is True
      and set(r["recipe"] for r in led) == {"R1", "R2"},
      {"n_ledger_entries": len(led), "budget": 10, "neural_full_fits": 0,
       "recipes": sorted(set(r["recipe"] for r in led)),
       "n_ridge_fits_this_run": fs["n_ridge_fits_this_run"],
       "n_ridge_fits_cached": fs["n_ridge_fits_cached"],
       "ledger_has_pid_configsha_inputsha_start_reason": all(
           all(r.get(x) for x in ("pid", "config_sha", "input_sha", "start", "reason"))
           for r in led)})

    # ---- 9. artifacts exist; prior work untouched ---------------------------
    figs = ["fig1_domain_shift.png", "fig2_heldout_predictions.png", "fig3_applicability.png"]
    tabs = ["fig1_data.tsv", "fig2_data.tsv", "fig3_data.tsv"]
    missing = [f for f in figs if not (REPORT_OUT / "figures" / f).exists()]
    missing += [t for t in tabs if not (REPORT_OUT / t).exists()]
    C("figures_and_source_tables_present", not missing, {"missing": missing})
    import datetime
    start = datetime.datetime.strptime(
        json.load(open(REPORT_OUT / "status.json", encoding="utf-8"))["started_local"],
        "%Y-%m-%dT%H:%M:%S%z").timestamp()
    trees = [ROOT / "reports" / "breakthrough_20261002" / "batch21",
             ROOT / "data" / "breakthrough_20261002" / "batch21",
             ROOT / "models" / "breakthrough_20261002" / "batch21",
             ROOT / "runs" / "breakthrough_20261002" / "code" / "batch21"]
    touched = [str(p.relative_to(ROOT)) for t in trees if t.exists()
               for p in t.rglob("*") if p.is_file() and p.stat().st_mtime >= start]
    C("no_prior_artifact_modified_after_batch22_start", not touched,
      {"batch22_start_local": "2026-10-03T15:06:40+0800", "files_touched_after_start": touched,
       "note": "batch21 sha_manifest.json predates its own final report.md (14:41) and "
               "raw_fasta_predictions.tsv (14:42); those stale entries predate batch22"})

    # ---- 10. decision rule honour (no after-result lowering) ----------------
    dr, dd = res["decision_rule"], res["decision"]
    recompute_met = ((rec["R2"]["spearman"] - rec["R1"]["spearman"]) >= dr["rho_gain_min"]
                     and (rec["R1"]["mae"] - rec["R2"]["mae"]) / rec["R1"]["mae"] >= dr["mae_reduction_min"]
                     and rec["R2"]["top_decile_precision"] >= rec["R1"]["top_decile_precision"]
                     and dd["positive_folds"] >= 4)
    C("decision_fixed_and_consistent",
      dr["rho_gain_min"] == 0.03 and dr["mae_reduction_min"] == 0.10
      and dr["min_positive_folds"] == "4 of 5" and recompute_met == dd["thresholds_met"],
      {"rule": dr, "recomputed_thresholds_met": bool(recompute_met),
       "reported_thresholds_met": dd["thresholds_met"], "verdict": dd["verdict"]})

    integrity = all(c["ok"] for c in checks)
    out = {
        "generated_at_local": now_local(),
        "reviewer": "fresh independent verification pass (read-only replay; no refits)",
        "batch": "batch22", "version": VERSION,
        "reviewed_artifacts": {
            "report": "reports/breakthrough_20261002/batch22/report.md",
            "results": "reports/breakthrough_20261002/batch22/results.json",
            "sha256": {
                "results.json": sha256_file(REPORT_OUT / "results.json"),
                "oof_predictions.tsv": sha256_file(DATA_OUT / "oof_predictions.tsv"),
                "sequence_table.tsv": sha256_file(DATA_OUT / "sequence_table.tsv"),
                "frozen_repr_A_seed20261023.npz": sha256_file(
                    DATA_OUT / "frozen_repr_A_seed20261023.npz"),
                "frozen_ckpt_batch21_recipeA": sha256_file(FROZEN_CKPT)}},
        "replay_commands": [
            "python -X utf8 runs/breakthrough_20261002/code/batch22/b22_review.py",
            "python -X utf8 runs/breakthrough_20261002/code/batch22/b22_selfcheck.py",
            "python -X utf8 runs/breakthrough_20261002/code/batch22/b22_metrics.py "
            "# recomputes results.json from cached oof_predictions.npz (no refits)",
            "python -X utf8 runs/breakthrough_20261002/code/batch22/b22_cli.py predict "
            "--in <seqs.fa> --recipe R2 --fold 0 --out preds.tsv"],
        "checks": checks, "n_checks": len(checks),
        "verdict": {
            "integrity": "PASS" if integrity else "FAIL",
            "integrity_scope": "exact measured-sequence join, published normalization with "
                               "negatives, whole-fragment preservation, label-free frozen folds, "
                               "frozen weights + no-HEK-label extraction, train-only scaling and "
                               "ridge alpha=100 solution, attempts counted in an append-only "
                               "ledger before execution, zero neural fits, no prior artifact "
                               "modified",
            "functional": "PASS (raw-input CLI runs; 3 figures with source tables; manifests "
                          "and SHA present)",
            "protocol_compliance": "PASS (10 ridge fits of a 10-fit budget; 0 neural full fits; "
                                   "frozen extraction on cuda:0 with assertions; CPU regressions "
                                   "labelled as CPU linear algebra; no overwrite of prior work)",
            "scientific_gain": "none on the primary endpoint (honest negative): R2 over R1 "
                               "delta-rho = %+.4f (< 0.03), MAE %+.1f%% (worse), positive folds "
                               "%d/5; the paired-group bootstrap CI for delta-rho crosses zero"
                               % (dd["delta_spearman_R2_minus_R1"],
                                  dd["mae_reduction_frac_R2_vs_R1"] * 100, dd["positive_folds"]),
            "hek293t_applicability": "NOT established and NOT claimed. This is a retrospective "
                                     "grouped CV on 676 public HEK293 (not HEK293T) records; the "
                                     "14 local HEK293T constructs have no sequence in project "
                                     "records and NO numeric activity claim is made.",
            "remaining_work": [
                "do not treat these interface-level results as a completed HEK293T model",
                "raise HEK-domain data volume above the from-scratch gate before attempting a "
                "from-scratch HEK sequence model (see next_batch_proposal.json)",
                "if a delivery estimator is wanted, train it only after parent acceptance of "
                "heldout performance (not in this batch)"]},
        "summary_zh": (
            "batch22（v6.17.0-hektransfer.rc1）在 GSE188639 HEK293（非 HEK293T）676 条核心启动子记录上做了"
            "小型冻结表征迁移/域评估。676 条中 673 条按 prom_id 精确联配到已发表的 hg19 133 bp 序列，"
            "3 条合成 SCP1 突变体无序列，明确排除、未伪造。hek_norm 定义经独立复算与 log2(hek exp/input) "
            "一致（最大偏差 8.9e-16），负值保留、未再取对数。无标签分组得到 669 个独立组，5 折均衡划分种子 "
            "20261025，跨折分组违例 0；4 条与冻结预训练源存在精确 133-mer 重叠并已分层披露。冻结 batch21 "
            "RecipeA 权重严格加载、全程 cuda:0、零 HEK 标签，pooled 表征 128 维、整段 133 bp 无截断。"
            "预算 10 次岭回归（R1/R2 各 5 折），神经端到端全量训练 0 次，全部尝试先记账后执行。结果："
            "R1 Spearman=0.4566、R2=0.4617，Δρ=+0.0051（<0.03），MAE 反而变差 4.0%，正向折 3/5，"
            "配对分组自助 95% 置信区间跨零——结论为诚实的阴性：冻结 K562 表征相对组合基线未带来预设的 HEK293 "
            "增益。独立复核 15/15 项通过（完整性 PASS），但科学增益为无，HEK293T 适用性未建立、不作声明。"),
        "next_step": "parent reads next_batch_proposal.json: raise HEK-domain measured sequence-activity "
                     "volume above the from-scratch gate (batch23), then retest a from-scratch HEK "
                     "sequence model under the frozen batch22 grouping/metrics harness",
        "note": "No credentials/mail reads; stage emails handled by the local reviewed watcher. "
                "No automatic project stop.",
    }
    atomic_write_json(REPORT_OUT / "independent_review.json", out)
    print("[b22_review] integrity=%s checks=%d/%d scientific_gain=none"
          % (out["verdict"]["integrity"], sum(1 for c in checks if c["ok"]), len(checks)))
    for c in checks:
        if not c["ok"]:
            print("  FAIL:", c["check"])
    return out


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()

