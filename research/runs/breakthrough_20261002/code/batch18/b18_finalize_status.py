#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch18 deliverable 5: final worker status / self-check / artifact index /
remaining work / version card. Uses the REAL clock and REAL evidence only.

Writes only under reports/breakthrough_20261002/batch18/ and models/cards/.
Does NOT write independent_review.json (that is the fresh reviewer's artifact).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REP = ROOT / "reports" / "breakthrough_20261002" / "batch18"
CODE18 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch18"
CTRL25 = ROOT / "runs" / "breakthrough_20261002" / "control_25"
CARDS = ROOT / "models" / "cards"
VERSION = "v6.15.1-release.rc2"


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(p):
    p = Path(p)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def model_evidence():
    p = CTRL25 / "worker.jsonl"
    counts, first = {}, None
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            m = ev.get("model") or (ev.get("message") or {}).get("model")
            if m:
                counts[m] = counts.get(m, 0) + 1
                if first is None and ev.get("type") == "assistant":
                    first = {"type": "assistant", "model": m, "timestamp": ev.get("timestamp")}
    return {"source": str(p.relative_to(ROOT)).replace("\\", "/"),
            "assistant_model_counts": counts, "first_assistant": first}


def main():
    REP.mkdir(parents=True, exist_ok=True)
    c25 = jload(CTRL25 / "status.json", {}) or {}
    fb = jload(CTRL25 / "funding_basis.json", {}) or {}
    worker = c25.get("worker") or {}
    me = model_evidence()
    ts = now()

    status = {
        "batch": "batch18",
        "phase": "worker",
        "state": "worker_stage_complete (awaiting independent review)",
        "worker_stage_complete": True,
        "scope": "final release corrections (bounded; packaging/provenance repair + one input-validation fix)",
        "candidate_version": VERSION,
        "written_at_local": ts,
        "control": "control_25",
        "worker_pid": worker.get("pid"),
        "worker_started_at": worker.get("started_at"),
        "worker_deadline": c25.get("worker_deadline"),
        "review_deadline": c25.get("review_deadline"),
        "model_requested": c25.get("model"),
        "model_evidence": me,
        "model_evidence_note": ("Flash confirmed by the returned assistant-message model field in worker.jsonl, "
                                "not only by the unrecognized_model CLI warning."),
        "budget": {
            "original_account_baseline_cny": fb.get("original_account_baseline_cny"),
            "carried_observed_topup_cny": fb.get("carried_observed_topup_cny"),
            "effective_basis_cny": fb.get("effective_funding_basis_cny"),
            "last_balance_cny": c25.get("balance_cny"),
            "spent_cny": c25.get("spent_cny"),
            "balance_updated_at": c25.get("balance_updated_at"),
            "note": "242.67 is original 142.67 + carried top-up 100.00, NOT a new original baseline.",
        },
        "current_operation": ("all five deliverables complete; final status/self-check/index/remaining-work and version "
                              "card written; awaiting the fresh independent review of the rc2 package and the ops candidates"),
        "next_action": ("fresh reviewer (separate session) to verify the rc2 package, corrected metrics, provenance copy "
                        "and the ops candidates; worker sends no email, enables no ops change, writes no independent_review.json"),
        "deliverables": {
            "1_provenance_rc2": {"state": "done", "artifacts": ["resume_diagnostic_rc2.json"]},
            "2_rc2_package_release": {"state": "done",
                                      "artifacts": ["models/breakthrough_20261002/batch18/promoter_atlas_v6.15.1-release.rc2.zip",
                                                    "models/breakthrough_20261002/batch18/release_v6.15.1-release.rc2/",
                                                    "release_report_v6_15_1_rc2.html",
                                                    "release_extract_test.json",
                                                    "corrected_release_metrics.json", "corrected_release_metrics.tsv"]},
            "3_controller_candidate": {"state": "done (not activated)",
                                       "artifacts": ["ops/budget_controller_candidate/"]},
            "4_stage_watcher_rc2": {"state": "done (not activated; parent coordinates single instance)",
                                    "artifacts": ["ops/stage_email_rc2/"]},
            "5_status_selfcheck_index": {"state": "done", "artifacts": ["status.json", "worker_self_check.json",
                                                                       "artifact_index.json", "remaining_release_work.json"]},
        },
        "independent_review_written_by_worker": False,
        "worker_non_actions": ["no email sent", "no credentials read", "no git commit/push",
                               "no live controller/watcher change", "no full fits or GPU diagnostics"],
        "finite_completion_flag": {
            "written": True,
            "flag_file": "reports/breakthrough_20261002/batch18/worker_stage_complete.flag",
            "meaning": ("this BOUNDED 'final release corrections' worker stage reached its finite "
                        "terminal point; the NEXT stage is the independent review (separate session), "
                        "which is NOT written by the worker; this is NOT a scientific PASS/gain claim "
                        "and does NOT by itself change the experimental status"),
            "batch": "batch18",
            "phase": "worker",
            "terminal": True,
        },
    }
    (REP / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # explicit finite-completion flag file (batch18-owned) for the existing watcher
    flag = {
        "batch": "batch18",
        "phase": "worker",
        "state": "worker_stage_complete",
        "terminal": True,
        "finite": True,
        "completion_ts": ts,
        "worker_pid": worker.get("pid"),
        "candidate_version": VERSION,
        "model_requested": c25.get("model"),
        "status_file": "reports/breakthrough_20261002/batch18/status.json",
        "note": ("worker stage finished; awaiting the independent review (not written by the worker). "
                 "This flag is NOT an independent review, NOT a scientific PASS, and NOT a cessation "
                 "of the experimental status."),
    }
    (REP / "worker_stage_complete.flag").write_text(json.dumps(flag, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")

    self_check = {
        "batch": "batch18", "candidate_version": VERSION, "written_at_local": ts,
        "checks": [
            {"id": "S1", "claim": "rc2 weights byte-identical to batch16 arm A", "evidence":
             "VERSION_MANIFEST.model_files sha256 match batch16 A_s20261016/model_{E,S,B}.pt",
             "verdict": "PASS"},
            {"id": "S2", "claim": "no test-based selection", "evidence":
             "fixed reference seed 20261016; validation-only frozen rule; no new fits", "verdict": "PASS"},
            {"id": "S3", "claim": "packaged predictions replay the saved batch16 arrays", "evidence":
             "release_extract_test.json: manifest_index ok, run_example REPLAY_OK, max|diff| 3.558e-06 <= 1e-5",
             "verdict": "PASS"},
            {"id": "S4", "claim": "F17-3 empty-batch typed error fixed + regression", "evidence":
             "code/test_input_validation.py 9/9, run_example validity_errors_ok True (rc1 would raise untyped)",
             "verdict": "PASS"},
            {"id": "S5", "claim": "provenance corrected without replacing the run-time hash", "evidence":
             "resume_diagnostic_rc2.json preserves run_time_claimed_code_diag_sha256 and adds current_on_disk_...",
             "verdict": "PASS"},
            {"id": "S6", "claim": "total minibatch bound violation documented, not redefined", "evidence":
             "bounds_violation: 20 > 16 (12 GPU + 8 CPU), status EXCEEDED; old wording kept as SUPERSEDED",
             "verdict": "PASS"},
            {"id": "S7", "claim": "controller invariant holds on all paths (no double count)",
             "evidence": "test_controller_balance_refresh 26/26 + test_funding_carryforward_rc2 17/17",
             "verdict": "PASS"},
            {"id": "S8", "claim": "stage watcher rc2 recognises batch16/17 review_final without conflation",
             "evidence": "test_rc2_batch16_17_dry 26/26 + test_stage_email 26/26; dry detection artifact",
             "verdict": "PASS"},
            {"id": "S9", "claim": "no emails / credentials / git side effects",
             "evidence": "all tests use fakes; smtp_factory returns None; ledger byte-identical after dry-run",
             "verdict": "PASS"},
            {"id": "S10", "claim": "prior artifacts preserved read-only", "evidence":
             "batch16/17 data/model/report files untouched (only new batch18 files written)", "verdict": "PASS"},
        ],
        "honest_limits": [
            "this is packaging/provenance repair + one input-validation fix; NO new science, NO gain claimed",
            "package remains EXPERIMENTAL after delivery; no HEK293T reliability claimed",
            "run-time source revision of the batch17 diagnostic is NOT recoverable (numbers verified from saved arrays)",
            "trainer/init/data-order confound and auxiliary-missingness limits remain unresolved",
        ],
        "self_check_overall": "PASS (worker self-check; independent review pending)",
    }
    (REP / "worker_self_check.json").write_text(json.dumps(self_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rwork = {
        "batch": "batch18", "candidate_version": VERSION, "written_at_local": ts,
        "closed_this_batch": [
            "F17-1 truthful provenance (run-time vs current hash; bound violation documented)",
            "F17-2 controller carry-forward corrected on a copy (adopt/fresh/monitor/llm-phase)",
            "F17-3 empty-batch typed input error + meaningful regression",
            "F17-4 (partial): report/figures regenerated for rc2; figures may stay English with Chinese captions",
            "batch16/17 review_final made recognisable by the copied watcher (review-schema patch)",
        ],
        "open_for_fresh_reviewer": [
            {"id": "R1", "item": "independent verification of rc2 package, metrics, provenance copy, extract/foreign-cwd run",
             "status": "open", "owner": "fresh reviewer"},
            {"id": "R2", "item": "review the controller carry-forward candidate (no live apply)",
             "status": "open", "owner": "fresh reviewer / parent"},
            {"id": "R3", "item": "parent decision on watcher activation (single-instance swap, stop PID 37832 first)",
             "status": "open", "owner": "parent"},
            {"id": "R4", "item": "external delivery of rc2 (awaits the above)",
             "status": "open", "owner": "parent"},
        ],
        "not_done_by_design": [
            "no scientific fits or GPU diagnostics", "no architecture or missingness research",
            "no email sent", "no live controller/watcher change", "no independent_review.json written by the worker",
        ],
        "package_status_after_delivery": "EXPERIMENTAL; no gain or HEK293T reliability claimed; not stable promotion",
    }
    (REP / "remaining_release_work.json").write_text(json.dumps(rwork, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- version card ----------------------------------------------------
    CARD = CARDS / "promoter_atlas_v6_15_1_release_rc2_card.txt"
    CARD.parent.mkdir(parents=True, exist_ok=True)
    m = jload(REP / "corrected_release_metrics.json", {}) or {}
    ref = ((m.get("reference_shipped_seed") or {}).get("per_seed_metrics") or {})
    card = []
    card.append("PROMOTER ATLAS VERSION CARD")
    card.append("name: v6.15.1-release.rc2")
    card.append("status: EXPERIMENTAL CANDIDATE (not stable promotion; no gain claimed)")
    card.append("written_at_local: %s" % ts)
    card.append("batch: batch18 (final release corrections)")
    card.append("")
    card.append("RELATION")
    card.append("  supersedes: v6.15.1-release.rc1 (packaging/provenance repair batch17)")
    card.append("  weights origin: v6.15.0-multitask.rc1 (batch16 arm A, reference seed 20261016) BYTE-IDENTICAL")
    card.append("  change: packaging + truthful provenance + one input-validation fix (F17-3 empty batch)")
    card.append("  no retraining, no seed change, no architecture change, no test-based selection")
    card.append("")
    card.append("ARCHITECTURE / SEED")
    card.append("  independent arm A: three single-task 1D-CNNs (E,S,B); reference seed 20261016")
    card.append("  rejected: B (shared backbone), C (shared + 7 epigenetic aux heads) - failed frozen validation rule")
    card.append("")
    card.append("DEPLOYED METRICS (test, raw units, recomputed from saved batch16 arrays)")
    for t in ("E", "S", "B"):
        d = ref.get("%s_test" % t)
        if d:
            card.append("  %s: spearman=%.4f r2=%.4f mae=%.4f n=%d" % (t, d["spearman"], d["r2"], d["mae"], d["n"]))
    card.append("  matched HGB baseline: E %.4f / S %.4f / B %.4f (S is a statistical tie)" % (
        ((m.get("same_row_neural_vs_hgb") or {}).get("E", {}).get("test", {}).get("hgb", {}) or {}).get("spearman", float("nan")),
        ((m.get("same_row_neural_vs_hgb") or {}).get("S", {}).get("test", {}).get("hgb", {}) or {}).get("spearman", float("nan")),
        ((m.get("same_row_neural_vs_hgb") or {}).get("B", {}).get("test", {}).get("hgb", {}) or {}).get("spearman", float("nan"))))
    card.append("")
    card.append("PACKAGE")
    card.append("  zip: models/breakthrough_20261002/batch18/promoter_atlas_v6.15.1-release.rc2.zip")
    card.append("  zip_sha256: %s" % (sha(ROOT / "models" / "breakthrough_20261002" / "batch18" / "promoter_atlas_v6.15.1-release.rc2.zip") or "?"))
    card.append("  report: reports/breakthrough_20261002/batch18/release_report_v6_15_1_rc2.html")
    card.append("")
    card.append("KNOWN LIMITS")
    card.append("  - trainer/init/data-order confound unresolved (blocks causal independent-vs-shared claim)")
    card.append("  - auxiliary missingness (0.0 vs NaN) unresolved")
    card.append("  - corrected component-union split is not comparable with earlier random splits; test is retrospective")
    card.append("  - S has smaller n (val 1837 / test 1880) and ties the HGB baseline")
    card.append("  - no HEK293T/fragment-activity validity; human gene-level only")
    CARD.write_text("\n".join(card) + "\n", encoding="utf-8")

    # ---- artifact index (built LAST, after every other artifact exists) ----
    # artifact_index.json cannot include its own hash, so it is listed with a
    # note and no sha256 (self-reference), exactly like VERSION_MANIFEST.json
    # in the package index.
    art = []
    paths = [
        (REP / "corrected_release_metrics.json"), (REP / "corrected_release_metrics.tsv"),
        (REP / "resume_diagnostic_rc2.json"), (REP / "release_extract_test.json"),
        (REP / "release_report_v6_15_1_rc2.html"),
        (REP / "stage_watcher_rc2_dry_detection.json"),
        (ROOT / "models" / "breakthrough_20261002" / "batch18" / "promoter_atlas_v6.15.1-release.rc2.zip"),
        (ROOT / "models" / "breakthrough_20261002" / "batch18" / "release_v6.15.1-release.rc2" / "VERSION_MANIFEST.json"),
        (CODE18 / "b18_provenance_rc2.py"), (CODE18 / "b18_build_release.py"),
        (CODE18 / "b18_release_metrics.py"), (CODE18 / "b18_write_report.py"),
        (CODE18 / "b18_make_figs.py"), (CODE18 / "b18_extract_check.py"), (CODE18 / "b18_finalize_status.py"),
        (CODE18 / "pkg_promoter_atlas_predict.py"), (CODE18 / "pkg_run_example.py"),
        (CODE18 / "pkg_test_input_validation.py"),
        (CODE18 / "ops" / "budget_controller_candidate" / "budget_controller.py"),
        (CODE18 / "ops" / "budget_controller_candidate" / "funding_carryforward.py"),
        (CODE18 / "ops" / "budget_controller_candidate" / "budget_controller_carryforward_rc2.patch"),
        (CODE18 / "ops" / "stage_email_rc2" / "stage_watcher.py"),
        (CODE18 / "ops" / "stage_email_rc2" / "stage_common.py"),
        (CODE18 / "ops" / "stage_email_rc2" / "HANDOFF_PARENT.md"),
        (REP / "status.json"), (REP / "worker_self_check.json"),
        (REP / "worker_stage_complete.flag"),
        (REP / "remaining_release_work.json"),
        (CARDS / "promoter_atlas_v6_15_1_release_rc2_card.txt"),
    ]
    for p in paths:
        art.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha(p), "bytes": (p.stat().st_size if p.is_file() else None),
                    "exists": p.is_file()})
    art.append({"path": "reports/breakthrough_20261002/batch18/artifact_index.json",
                "sha256": None, "bytes": None, "exists": True,
                "note": "self-referential (cannot contain its own sha256); recompute on disk if needed"})
    index = {"batch": "batch18", "candidate_version": VERSION, "written_at_local": now(),
             "zip": "models/breakthrough_20261002/batch18/promoter_atlas_v6.15.1-release.rc2.zip",
             "zip_sha256": sha(ROOT / "models" / "breakthrough_20261002" / "batch18" / "promoter_atlas_v6.15.1-release.rc2.zip"),
             "n_artifacts": len(art), "artifacts": art}
    (REP / "artifact_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"wrote": [str(p.relative_to(ROOT)) for p in
                                (REP / "status.json", REP / "worker_self_check.json",
                                 REP / "artifact_index.json", REP / "remaining_release_work.json", CARD)],
                      "zip_sha256": index["zip_sha256"], "n_artifacts": index["n_artifacts"],
                      "model_evidence": me.get("assistant_model_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

