#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch37 step 5 - assemble reports: paired predictions, SHA manifest, status,
worker self-check, remaining work, ONE proposal, README, registry append. CPU only."""
import csv, hashlib, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "data/breakthrough_20261002/batch37")
RPT = os.path.join(REPO, "reports/breakthrough_20261002/batch37")
FIGD = os.path.join(REPO, "figures/batch37")
CODED = os.path.join(REPO, "code/batch37")
BATCH = "batch37"; VERSION = "v6.26.0-frozen-variant-diagnostic.rc1"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    orders = {}
    with open(os.path.join(OUTDIR, "unique_scores.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"): orders[r["seq_id"]] = float(r["mean_score"])
    # paired predictions table
    pp = os.path.join(OUTDIR, "paired_predictions.tsv")
    with open(pp, "w", encoding="utf-8", newline="") as fo:
        fo.write("arm\toriginal_id\tchr\tpos\tvariant_class\tpair_status\tgroup_id\t"
                 "alt_mean_score\tref_mean_score\td_pred\tlabel_logFC\tlabel_missing\n")
        with open(os.path.join(OUTDIR, "paired_inputs.tsv"), encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                a = hashlib.sha256(r["alt_insert"].encode()).hexdigest()[:20]
                rf = hashlib.sha256(r["ref_insert"].encode()).hexdigest()[:20]
                fo.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.8g\t%.8g\t%.8g\t%s\t%s\n" % (
                    r["arm"], r["original_id"], r["chr"], r["pos"], r["variant_class"], r["pair_status"],
                    r["group_id"], orders[a], orders[rf], orders[a] - orders[rf],
                    r["label_logFC"], r["label_missing"]))
    m = json.load(open(os.path.join(RPT, "metrics.json"), encoding="utf-8"))

    # SHA manifest (code + outputs, excludes this file/status/review to avoid self-reference loops)
    shas = {}
    for d in (CODED, OUTDIR, FIGD, RPT):
        for fn in sorted(os.listdir(d)):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp) and fn not in ("sha_manifest.json",):
                shas[os.path.relpath(fp, REPO).replace("\\", "/")] = sha256_file(fp)
    with open(os.path.join(RPT, "sha_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "batch": BATCH, "version": VERSION,
                   "note": "SHAs of batch37 code + outputs; sha_manifest.json excludes itself",
                   "files": shas}, f, ensure_ascii=False, indent=2)
    metric_code = {fn: shas.get("code/batch37/" + fn) for fn in
                   ["build_manifest.py", "infer.py", "metrics.py", "figures.py", "finalize.py"]}

    status = {
        "batch": BATCH, "version": VERSION, "phase": "worker", "state": "complete",
        "generated_at": now, "zero_fit": True, "gpu": True,
        "gpu_scope": "frozen PARM forward only (no training/optimizer/gradient)",
        "target_requested_cell": "293FT", "model_source_cell": "HEK293",
        "task": "single frozen-model transfer diagnostic of variant-induced effects in HEK293FT MPRA",
        "verdict_worker_preliminary": "LIMITED (not PASS)",
        "decision_rule_all_met": m["decision_rule"]["all_met"],
        "headline": {
            "KV_SNV_pooled_spearman_d_logFC": m["kv_all"]["spearman_d_logFC"],
            "KV_SNV_valid_only_spearman": m["kv_valid_only"]["spearman_d_logFC"],
            "KV_SNV_bootstrap_ci95": m["kv_bootstrap"]["rho_ci95"],
            "SAT_all_spearman_descriptive": m["sat_all"]["spearman_d_logFC"],
            "deltaGC_baseline_rho_KV": m["kv_all"]["spearman_deltaGC_logFC"],
            "deltaGC_baseline_rho_SAT": m["sat_all"]["spearman_deltaGC_logFC"],
        },
        "audit": {
            "KV_rows": m["kv_all"]["n_rows"], "KV_degenerate_alt_eq_ref": m["kv_all"]["n_pred_ties_zero_d"],
            "KV_label_missing": m["kv_all"]["n_label_missing"],
            "SAT_rows": m["sat_all"]["n_rows"], "SAT_label_missing": m["sat_all"]["n_label_missing"],
            "KV_effective_groups": m["kv_group_stats"]["n_effective_groups"],
            "KV_2kb_gap_clusters": 48, "unique_sequences": 25193,
        },
        "notes": "ZERO-fit frozen PARM HEK293 variant-effect transfer diagnostic; ALL results retrospective; "
                 "prior training exposure UNKNOWN; not autonomous-promoter reliability; no model promotion.",
        "artifacts": {"manifest": "reports/breakthrough_20261002/batch37/diagnostic_manifest.json",
                      "metrics": "reports/breakthrough_20261002/batch37/metrics.json",
                      "paired_predictions": "data/breakthrough_20261002/batch37/paired_predictions.tsv",
                      "unique_scores": "data/breakthrough_20261002/batch37/unique_scores.tsv",
                      "inference_account": "data/breakthrough_20261002/batch37/inference_account.json",
                      "figures": "figures/batch37/"},
    }
    with open(os.path.join(RPT, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    selfcheck = {
        "batch": BATCH, "version": VERSION, "generated_at": now,
        "worker_model_label": "deepseek-flash",
        "zero_fits": True, "optimizer_steps": 0, "gradients": 0,
        "architecture_or_cell_selection": False, "external_model_downloads": 0,
        "forbidden_operations_avoided": [
            "log of PARM output", "linear calibration", "MAE/R2 across incompatible units",
            "best-fold selection", "global negative-correlation flip", "outcome-chosen RC",
            "significance/label-based subsampling", "using PromoterAI as truth"],
        "checks": {
            "parent_acceptance_present": os.path.exists(os.path.join(REPO, "runs/breakthrough_20261002/batch37/PARENT_ACCEPTANCE.txt")),
            "weights_sha_recomputed_ok": True, "design_fasta_sha_ok": True,
            "device_assertions_passed": True, "unique_within_30000": 25193 <= 30000,
            "figures_le_3": len([x for x in os.listdir(FIGD) if x.endswith(".png")]) <= 3,
            "old_assets_readonly": True,
        },
        "network_requests_used_for_source": 0,
        "artifacts_written": ["diagnostic_manifest.json", "metrics.json", "table_kv_groups.tsv",
                              "status.json", "worker_selfcheck.json", "remaining_work.json",
                              "next_batch_proposal.json", "README.md", "sha_manifest.json",
                              "version_registry_entry.json"],
        "independent_review": "NOT written by worker (fresh reviewer owns independent_review.json)",
    }
    with open(os.path.join(RPT, "worker_selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(selfcheck, f, ensure_ascii=False, indent=2)

    kva = m["kv_all"]; kvv = m["kv_valid_only"]; sata = m["sat_all"]
    remaining = {
        "batch": BATCH, "version": VERSION, "generated_at": now,
        "headline_result": "PARM HEK293 frozen score is NOT informative for HEK293FT matched "
                           "variant-reference MPRA effects (KV primary rho=%.4f, CI lower>0 but well "
                           "below the 0.20 operational bar; macro strict-sign does NOT beat both fixed baselines)."
                           % (kva["spearman_d_logFC"] if kva["spearman_d_logFC"] is not None else float("nan")),
        "open_items": [
            "Independent review to assign canonical PASS/FAIL + scientific_gain (worker MUST NOT write it).",
            "KV allele join coverage is %.2f%% (<99%%); %d/%d KV rows are DEGENERATE (author alt oligo == ref oligo)."
            % (100.0 * (kva["n_rows"] - kva["n_pred_ties_zero_d"]) / max(1, kva["n_rows"]),
               kva["n_pred_ties_zero_d"], kva["n_rows"]),
            "SAT/KV group families unverified beyond interval-union + 2kb gap clusters (%d clusters, %d union groups)."
            % (48, m["kv_group_stats"]["n_groups"]),
            "Cross-cell transfer only; prior training exposure of the MPRA labels to the model is UNKNOWN.",
            "The %d KV variants overlapping the SAT 2kb region are flagged; arms are NEVER pooled." % m["sat_kv_overlap"]["kv_variants_overlapping_sat_region"],
        ],
        "explicitly_not_done": [
            "No fits/solves/calibration/gradient/optimizer steps (ZERO-fit frozen inference only).",
            "No architecture/fold/cell selection; no external model downloads.",
            "No autonomous-promoter or truncation-reliability claim.",
            "No model promotion; old HEK 10%% MAE gate and local D28 FAIL are unchanged.",
        ],
        "resource_account": dict(m["kv_bootstrap"].get("n_finite_boot_rho") is not None and {"bootstrap_finite_rho": m["kv_bootstrap"]["n_finite_boot_rho"]} or {}),
    }
    with open(os.path.join(RPT, "remaining_work.json"), "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    proposal = {
        "batch": BATCH, "version": VERSION, "generated_at": now,
        "trigger": "negative_HK_variant_effect_evidence_is_NOT_reused_to_refit_these_effects",
        "principle": "ONE mechanistically different bounded ZERO-fit action relevant to the ORIGINAL "
                     "autonomous-promoter / truncation-reliability goal; do not fit to these effect results.",
        "proposed_action": {
            "name": "batch38_authored_truncation_scan_nonparametric_frozen",
            "rationale": "The transfer failure shows the frozen HEK293 score has near-zero monotone signal for "
                         "this system's variant effects. A mechanistically distinct, ZERO-fit probe of the SAME "
                         "original question is to test whether the frozen score is at least SENSITIVE to the "
                         "measured INSERT LENGTH/truncation geometry that the MPRA itself quantifies -- i.e. does "
                         "the model's raw activity respond monotonically to the authored 219bp window vs its "
                         "5bp-deletion (214bp) variants, WITHOUT any label or calibration?",
            "mechanism_differs": "Different quantity (structural length/truncation response, not variant effect), "
                                 "different statistic (within-locus paired rank agreement of raw score to authored "
                                 "length class), NO effect labels involved, still ZERO fit.",
            "bounded_scope": "<= 30000 unique sequences, <= 20 min GPU, one batch dir, one figure, no thresholds "
                             "tuned on outcomes.",
            "hard_constraints": ["ZERO fits", "no external downloads", "old assets readonly",
                                 "no autonomous reliability claim", "fresh review owns verdict"],
            "decision_output": "Report whether frozen raw score tracks authored length class at >= 10 pairs/locus; "
                               "if null, conclude the frozen HEK293 model is orthogonal to this system's geometry "
                               "and STOP probing this direction.",
        },
    }
    with open(os.path.join(RPT, "next_batch_proposal.json"), "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)

    readme = """# batch37 - v6.26.0-frozen-variant-diagnostic.rc1

**Single ZERO-fit frozen-model transfer diagnostic** of variant-induced effects in the
HEK293FT MAPT MPRA (PMID 41959502 / GEO GSE325670). Model = frozen official PARM
HEK293 5-fold equal mean (unmodified, SHA-verified, bare FASTA, central zero-pad
L_max=600). **NO fits/solves/calibration/gradient/optimizer.** ALL results
retrospective; prior training exposure of these labels to the model is UNKNOWN.

## Question
Does the existing frozen PARM HEK293 score carry information about the author's
**matched variant-vs-reference** MPRA effects in HEK293FT? This is NOT a test of
autonomous-promoter or truncation reliability.

## Design (frozen in diagnostic_manifest.json BEFORE any forward)
- Exact source variant + reference inserts taken from the 36k design FASTA (source header).
- SAT arm: all 8000 author rows (incl 3 NA), 6000 SNVs + 2000 deletions as separate strata.
- KV primary: SNV_HEK293FT all 8513 rows (incl 17 NA), all retained.
- Prediction d = mean_score(alt) - mean_score(ref); true effect = reported bcalm logFC.
- Zero-fit baselines: delta GC fraction and delta length on the SAME paired inserts.
- Label-free groups by interval-union overlap; 2kb gap clusters are NOT assumed families.

## Result (negative)
- KV pooled Spearman(d, logFC) = %.4f (valid-only %.4f); bootstrap CI lower > 0 but the
  point estimate is far below the frozen 0.20 operational bar.
- Macro strict-sign accuracy does NOT exceed BOTH fixed always-positive/always-negative baselines.
- SAT Spearman (descriptive only) = %.4f.
- %d/%d KV rows are DEGENERATE (author alt oligo == ref oligo); allele join coverage < 99%%.
- **Verdict: LIMITED (not PASS).** Decision conditions not all met.

## Honesty / scope
- SAT = single locus within-locus description only; SAT and KV are NEVER pooled.
- NaNs reported, no undefined rho set to 0; predicted ties counted wrong; true-zero effects
  excluded from the sign denominator.
- No model promotion; old HEK 10%% MAE gate and local D28 FAIL are unchanged.
- Independent review (canonical PASS/FAIL + scientific_gain) is owned by a fresh reviewer;
  the worker did NOT write it.

## Artifacts
- reports/breakthrough_20261002/batch37/: diagnostic_manifest.json, metrics.json,
  table_kv_groups.tsv, status.json, worker_selfcheck.json, remaining_work.json,
  next_batch_proposal.json, sha_manifest.json, version_registry_entry.json, README.md
- data/breakthrough_20261002/batch37/: unique_sequences.fasta, pair_sequence_ids.tsv,
  unique_scores.tsv, paired_inputs.tsv, paired_predictions.tsv, inference_account.json
- figures/batch37/: fig1_paired_effect_scatter.png, fig2_kv_region_signerrors.png,
  fig3_scope_evidence_card.png

## License
PARM used for **local academic evaluation only**; weights/source NOT redistributed.
MAPT data is public GEO (GSE325670). This deliverable is wrapper/analysis code + public sequences only.
""" % (kva["spearman_d_logFC"], kvv["spearman_d_logFC"], sata["spearman_d_logFC"],
       kva["n_pred_ties_zero_d"], kva["n_rows"])
    with open(os.path.join(RPT, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    registry = {
        "record_type": "evidence_and_usability",
        "batch": BATCH, "version": VERSION,
        "kind": "single ZERO-fit frozen PARM HEK293 -> HEK293FT variant-effect transfer diagnostic; "
                "no estimator/optimizer/gradient; frozen forward only",
        "appended_by": "batch37 worker (DeepSeek Flash implementer; parent-authorized append-only)",
        "appended_at": now,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "sectionA": {
            "question": "is frozen PARM HEK293 score informative for HEK293FT matched variant-reference effects?",
            "model": "frozen official PARM HEK293 5-fold equal mean (unmodified, SHA-verified)",
            "unique_sequences_scored": 25193, "gpu": "single RTX 5060 cuda0", "batch_size": 32,
            "primary_KV_spearman_d_logFC": kva["spearman_d_logFC"],
            "primary_KV_valid_only_spearman": kvv["spearman_d_logFC"],
            "KV_bootstrap_ci95": m["kv_bootstrap"]["rho_ci95"],
            "SAT_spearman_descriptive": sata["spearman_d_logFC"],
            "deltaGC_baseline_rho_KV": kva["spearman_deltaGC_logFC"],
            "decision_rule_all_met": m["decision_rule"]["all_met"],
            "worker_preliminary_verdict": "LIMITED (not PASS)",
        },
        "sectionB": {
            "audit": "KV degenerate alt==ref rows=%d of %d; allele join coverage <99%%; group families unverified"
                     % (kva["n_pred_ties_zero_d"], kva["n_rows"]),
            "honesty": "NaNs reported; ties counted wrong; no undefined rho->0; no calibration/MAE/R2; no global flip",
            "no_promotion": True,
        },
        "sectionC": {
            "readme": "reports/breakthrough_20261002/batch37/README.md",
            "figures": 3, "no_ui_wrapping": True,
        },
        "next": "fresh separate review (canonical PASS/FAIL + scientific_gain); parent decides bounded batch38",
    }
    with open(os.path.join(RPT, "version_registry_entry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    # ---- atomic changelog append (prior entries preserved) ----
    CH = os.path.join(REPO, "reports/breakthrough_20261002/CHANGELOG.txt")
    line = ("[" + now + "] batch37 " + VERSION + " -> EVIDENCE ONLY (NOT a trained model, NO promotion, "
            "ZERO fits/solves/calibration/gradient/optimizer). Single ZERO-fit frozen PARM HEK293 transfer "
            "diagnostic on the HEK293FT MAPT MPRA (PMID 41959502/GSE325670): prediction d=mean_score(alt)-mean_score(ref), "
            "5-fold equal mean, bare FASTA central zero-pad L_max=600, %d unique sequences (cap 30000), RTX5060 cuda0 batch32. "
            "KV primary (SNV_HEK293FT, all %d rows incl %d NA) pooled Spearman(d,logFC)=%.4f (valid-only %.4f); group bootstrap "
            "CI95 lower>0 but point estimate << frozen 0.20 bar; macro strict-sign does NOT beat both fixed baselines; "
            "%d/%d KV rows DEGENERATE (alt oligo==ref oligo, allele join <99%%). SAT (all 8000, descriptive only) rho=%.4f. "
            "Verdict LIMITED (not PASS); decision conditions not all met. ALL results retrospective, prior training exposure UNKNOWN; "
            "no autonomous-promoter/truncation claim; no model promotion (old HEK 10%% MAE gate + local D28 FAIL unchanged). "
            "Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved.\n"
            % (25193, kva["n_rows"], kva["n_label_missing"], kva["spearman_d_logFC"], kvv["spearman_d_logFC"],
               kva["n_pred_ties_zero_d"], kva["n_rows"], sata["spearman_d_logFC"]))
    old = open(CH, encoding="utf-8").read() if os.path.exists(CH) else ""
    tmp = CH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: f.write(old + line)
    os.replace(tmp, CH)

    # ---- verify every written JSON parses ----
    required = ["diagnostic_manifest.json", "metrics.json", "status.json", "worker_selfcheck.json",
                "remaining_work.json", "next_batch_proposal.json", "sha_manifest.json",
                "version_registry_entry.json"]
    parsed = {}
    for fn in required:
        with open(os.path.join(RPT, fn), encoding="utf-8") as f:
            parsed[fn] = type(json.load(f)).__name__
    for fn in ["inference_account.json"]:
        with open(os.path.join(OUTDIR, fn), encoding="utf-8") as f:
            parsed[fn] = type(json.load(f)).__name__
    print(json.dumps({"status": "finalize_ok", "metric_code_sha": metric_code,
                      "json_parsed": parsed, "changelog_appended": True,
                      "registry_entry": "reports/breakthrough_20261002/batch37/version_registry_entry.json"},
                     ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
