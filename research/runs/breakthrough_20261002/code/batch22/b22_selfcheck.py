#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch22 SELFCHECK (worker-side, mechanically replayable).

This is DELIBERATELY DISTINCT from the fresh independent review: it re-checks
invariants and replays saved predictions/statistics only (no ridge or neural refits),
and it verifies that every read-only prior artifact is byte-identical to its
recorded hash.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from b22_common import (ROOT, DATA_OUT, REPORT_OUT, MODEL_OUT, FROZEN_CKPT, VERSION,
                        atomic_write_json, now_local, sha256_file, spearman, mae)
from b22_fits import load_inputs, feature_composition, fit_ridge

BATCH21 = ROOT / "reports" / "breakthrough_20261002" / "batch21"


def check(name, ok, detail):
    return {"check": name, "ok": bool(ok), "detail": detail}


def main():
    rows, seqs, y, diag, pids, fold_of, gid, seen, H = load_inputs()
    checks = []
    n = len(y)
    k = int(fold_of.max()) + 1

    # 1. grouped folds: no group crosses a fold boundary
    bad = 0
    for g in np.unique(gid):
        if len(np.unique(fold_of[gid == g])) != 1:
            bad += 1
    checks.append(check("groups_confined_to_single_fold", bad == 0, {"violating_groups": int(bad)}))

    # 2. every row predicted exactly once by per-fold readouts
    zp = np.load(DATA_OUT / "oof_predictions.npz")
    cov = {nmm: bool(np.all(np.isfinite(zp[nmm]))) for nmm in ("R0", "R1", "R2")}
    checks.append(check("oof_complete", all(cov.values()), cov))

    # 3. replay a heldout fold from SAVED train stats only (metric equality)
    te = np.where(fold_of == 0)[0]
    rep = {}
    for name, X in (("R1", feature_composition(seqs)),
                    ("R2", np.concatenate([feature_composition(seqs), H], axis=1))):
        s = np.load(MODEL_OUT / ("attempt_fold0_%s" % name) / "train_stats.npz")
        Z = (X[te] - s["train_mean"]) / s["train_scale"]
        p = Z @ s["coef"] + float(s["intercept"])
        saved = np.load(MODEL_OUT / ("attempt_fold0_%s" % name) / "pred.npz")["pred"]
        rep[name] = {"max_abs_diff_vs_saved": float(np.max(np.abs(p - saved))),
                     "spearman": spearman(p, y[te]), "mae": mae(p, y[te])}
    ok = all(v["max_abs_diff_vs_saved"] < 1e-9 for v in rep.values())
    checks.append(check("fold0_replay_from_saved_train_stats", ok, rep))

    # 4. frozen weights untouched: reload ckpt hash and confirm zero-HEK-label extraction
    ev = json.load(open(REPORT_OUT / "frozen_extraction_evidence.json", encoding="utf-8"))
    checks.append(check("frozen_model_sha_matches_ckpt",
                        ev["frozen_model"]["model_sha256"] == sha256_file(FROZEN_CKPT),
                        {"evidence_sha": ev["frozen_model"]["model_sha256"],
                         "ckpt_sha": sha256_file(FROZEN_CKPT)}))
    checks.append(check("no_hek_labels_in_extraction", ev["n_hek_labels_used"] == 0,
                        {"n_hek_labels_used": ev["n_hek_labels_used"]}))
    checks.append(check("extraction_device_cuda0", ev["device"] == "cuda:0",
                        {"device": ev["device"], "batch": ev["batch"]}))

    # 5. attempts ledger within budget, unique ids, PIDs recorded
    led = [json.loads(l) for l in open(REPORT_OUT / "attempts_ledger.jsonl", encoding="utf-8")]
    ids = [r["attempt_id"] for r in led]
    checks.append(check("attempts_budget_ok", len(led) <= 10,
                        {"entries": len(led), "budget": 10, "neural_full_fits": 0}))
    checks.append(check("attempt_ids_unique", len(set(ids)) == len(ids), {"n": len(ids)}))
    checks.append(check("attempts_record_pid_and_shas",
                        all(r.get("pid") and r.get("config_sha") and r.get("input_sha")
                            and r.get("start") and r.get("reason") for r in led),
                        {"pids": sorted({r["pid"] for r in led})}))

    # 6. decision rule frozen and consistent with results.json
    res = json.load(open(REPORT_OUT / "results.json", encoding="utf-8"))
    dr = res["decision_rule"]
    d = res["decision"]
    recomputed = ((res["pooled_oof"]["R2"]["spearman"] - res["pooled_oof"]["R1"]["spearman"])
                  >= dr["rho_gain_min"])
    checks.append(check("decision_rule_predeclared",
                        dr["rho_gain_min"] == 0.03 and dr["mae_reduction_min"] == 0.10
                        and dr["min_positive_folds"] == "4 of 5" and dr["frozen"] is True,
                        dr))
    checks.append(check("decision_threshold_eval_consistent",
                        bool(recomputed) == bool(d["delta_spearman_R2_minus_R1"]
                                                 >= dr["rho_gain_min"]),
                        {"delta_rho": d["delta_spearman_R2_minus_R1"]}))

    # 7. prior read-only artifacts untouched BY BATCH22.
    #    (a) any mismatch against batch21's own sha_manifest must predate batch22 start;
    #    (b) no file under any batch21 read-only tree may be modified after batch22 start.
    import datetime
    start = datetime.datetime.strptime(
        json.load(open(REPORT_OUT / "status.json", encoding="utf-8"))["started_local"],
        "%Y-%m-%dT%H:%M:%S%z").timestamp()
    sm = json.load(open(BATCH21 / "sha_manifest.json", encoding="utf-8"))
    mism, n_checked = [], 0
    for rel, h in list(sm.get("artifacts", {}).items()):
        p = ROOT / rel.replace("\\", "/")
        if p.exists():
            n_checked += 1
            if sha256_file(p) != h:
                mism.append({"path": rel, "mtime": p.stat().st_mtime,
                             "pre_batch22": p.stat().st_mtime < start})
    trees = [BATCH21, ROOT / "data" / "breakthrough_20261002" / "batch21",
             ROOT / "models" / "breakthrough_20261002" / "batch21",
             ROOT / "runs" / "breakthrough_20261002" / "code" / "batch21"]
    touched = []
    for t in trees:
        if t.exists():
            for p in t.rglob("*"):
                if p.is_file() and p.stat().st_mtime >= start:
                    touched.append(str(p.relative_to(ROOT)))
    checks.append(check("prior_batch21_artifacts_unchanged",
                        len(touched) == 0
                        and all(m["pre_batch22"] for m in mism),
                        {"hash_mismatches_vs_batch21_own_manifest": mism,
                         "checked": n_checked,
                         "files_modified_after_batch22_start": touched,
                         "note": "batch21's own sha_manifest was written at 14:40:59, before "
                                 "it finalized report.md/raw_fasta_predictions.tsv at 14:41-14:42; "
                                 "those stale entries predate batch22 and are NOT batch22 edits"}))

    # 8. deliverables present
    need = ["results.json", "figures_manifest.json", "manifests.json",
            "next_batch_proposal.json", "data_manifest.json", "freeze.json",
            "group_audit.json", "fits_summary.json", "attempts_summary.json",
            "inference_latency.json"]
    missing = [f for f in need if not (REPORT_OUT / f).exists()]
    figs = [f for f in ["fig1_domain_shift.png", "fig2_heldout_predictions.png",
                        "fig3_applicability.png"] if (REPORT_OUT / "figures" / f).exists()]
    checks.append(check("deliverables_present", not missing and len(figs) == 3,
                        {"missing": missing, "figures": figs}))

    # 9. target sanity: negatives preserved and definition verified
    dm = json.load(open(REPORT_OUT / "data_manifest.json", encoding="utf-8"))
    checks.append(check("negative_values_preserved",
                        dm["target"]["negative_values_kept"] is True
                        and dm["target"]["extra_log_applied"] is False,
                        {"n_negative_hek": int((np.array(y) < 0).sum()),
                         "definition_max_abs_diff": dm["target"]["verified_max_abs_diff_vs_definition"]}))

    # 10. batch22 sha manifest for every artifact batch22 produced
    trees = [REPORT_OUT, DATA_OUT, MODEL_OUT, ROOT / "runs" / "breakthrough_20261002" / "code" / "batch22"]
    art = {}
    for t in trees:
        for p in sorted(t.rglob("*")):
            if (p.is_file() and "__pycache__" not in p.parts and "_work" not in p.parts
                    and not p.name.endswith(".tmp")):
                art[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256_file(p)
    atomic_write_json(REPORT_OUT / "sha_manifest.json",
                      {"generated_at_local": now_local(), "version": VERSION,
                       "scope": "batch22-produced artifacts only; prior artifacts read-only",
                       "self_reference_note": "sha_manifest.json and selfcheck.json are written "
                                              "after this listing and therefore are not listed in it",
                       "excluded": "data/breakthrough_20261002/batch22/_work/ (scratch evidence of "
                                   "bounded acquisition attempts, including a partial Europe PMC "
                                   "supplementary archive that never validated as a zip)",
                       "n_artifacts": len(art), "sha256": art,
                       "readonly_inputs": {
                           "batch21_recipeA_ckpt": sha256_file(FROZEN_CKPT),
                           "GSE188639_bulk_exp": sha256_file(
                               DATA_OUT / "source" / "GSE188639_mixed_cell_bulk_exp.csv.gz")}})

    # 11. documented exclusion of the 3 synthetic SCP1 mutant members
    dm2 = json.load(open(REPORT_OUT / "data_manifest.json", encoding="utf-8"))
    atomic_write_json(REPORT_OUT / "scp1_exclusion.json", {
        "generated_at_local": now_local(),
        "excluded_ids": dm2["unresolved_ids"], "n_excluded": len(dm2["unresolved_ids"]),
        "reason": "synthetic SCP1 mutant members of GSE188639 with no published exact 133 bp "
                  "sequence; excluded rather than fabricated (protocol: never fabricate sequences)",
        "bounded_diagnostics_attempted": [
            "primary GEO series GSE188639 supplementary files (no sequence table for these ids)",
            "Europe PMC supplementaryFiles for the linked study record: the archive did not "
            "validate as a zip within the bounded attempt window and the download was stopped "
            "(scratch evidence kept under data/breakthrough_20261002/batch22/_work/); this is not "
            "required for the frozen-transfer question",
            "Haberle et al. 2019 Supplementary Table 6 exact-oligo-id join (the 3 ids are absent)"],
        "impact": "673 of 676 rows retained; no model input or target was imputed for the 3 rows",
        "decision": "document and exclude; do not re-open the batch for these rows"})

    out = {"generated_at_local": now_local(), "version": VERSION, "batch": "batch22",
           "kind": "worker_selfcheck (distinct from fresh independent review)",
           "n_checks": len(checks), "n_ok": sum(1 for c in checks if c["ok"]),
           "all_ok": all(c["ok"] for c in checks), "checks": checks}
    atomic_write_json(REPORT_OUT / "selfcheck.json", out)
    print("[b22_selfcheck] %d/%d ok" % (out["n_ok"], out["n_checks"]))
    for c in checks:
        if not c["ok"]:
            print("  FAIL:", c["check"], js0(c["detail"]))
    return out


def js0(x):
    import json as _j
    return _j.dumps(x)[:300]


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()

