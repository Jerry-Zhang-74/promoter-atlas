#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 notification compatibility repair artifacts.

Generates, under reports/breakthrough_20261002/batch12/mail_compat/:
  batch11_review_preview.txt / .html   (Chinese preview of the batch11 review mail)
  ready_to_switch.json                 (entrypoint/args/runtimepath + old watcher
                                        identity + new code hashes + test/review refs)
  code_shas.json                       (SHA256 of the v3 watcher files)

This is the OPERATIONAL repair deliverable. It does NOT launch any watcher,
does NOT kill the old watcher, does NOT send email, and does NOT read credentials.
It only READS the existing batch11 review artifact + control_17 status and the
v3 code, and WRITES the mail_compat previews/manifests.
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V3_DIR = REPO / "runs" / "breakthrough_20261002" / "code" / "stage_email_v3"
sys.path.insert(0, str(V3_DIR))

import stage_common as sc  # noqa: E402  (v3)
import stage_watcher as sw  # noqa: E402  (v3)

MAIL_COMPAT = (REPO / "reports" / "breakthrough_20261002" / "batch12"
               / "mail_compat")

OLD_WATCHER = {
    "pid": 44488,
    "created_local": "2026-10-03T02:23:03.972883+08:00",
    "wrapper_pid": 44300,
    "note": "current live reviewed watcher v2; do NOT kill in worker; parent switches only",
}
BATCH11_REVIEWER = {
    "controller": "control_17",
    "reviewer_pid": 5204,
    "review_completed_local": "2026-10-03T07:33:21+08:00",
    "worker_prelim_accepted_local": "2026-10-03T07:25:52.945452+08:00",
}

V3_FILES = ["stage_common.py", "stage_mailer.py", "stage_watcher.py",
            "test_stage_email.py", "__init__.py"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    MAIL_COMPAT.mkdir(parents=True, exist_ok=True)

    # 1. Build the real batch11 review evidence with the v3 nested-verdict path.
    cstatus = sc.read_json(
        REPO / "runs" / "breakthrough_20261002" / "control_17" / "status.json",
        default={})
    review_path = (REPO / "reports" / "breakthrough_20261002" / "batch11"
                   / "independent_review.json")
    review_json = sc.read_json(review_path, default=None)
    review_info = sw.detect_review("batch11", cstatus, review_json, review_path)
    assert review_info is not None, "v3 must detect the nested batch11 review"
    assert review_info["verdict_kind"] == "nested"
    evidence = sw.build_review_evidence(
        "batch11", cstatus, review_json, review_info,
        running_now="batch11 独立审查已结束；worker 阶段 preliminary 邮件已于 07:25:52 接受。",
        next_step="batch12（v6.13.0-siface.rc1）冻结接口已交付；等待独立审查。",
        estimated_handoff="2026-10-03T10:00:00+08:00",
        balance_obs=sw._latest_balance_obs(
            cstatus, REPO / "runs" / "breakthrough_20261002"),
        stage_summary=None)

    subject = sc.build_subject(evidence)
    body = sc.build_body(evidence)

    (MAIL_COMPAT / "batch11_review_preview.txt").write_text(
        "Subject: " + subject + "\n\n" + body, encoding="utf-8")
    html_doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
                "<title>batch11 审查邮件预览（v3 嵌套 verdict 修复）</title></head>"
                "<body style='font-family:sans-serif;max-width:900px;margin:auto'>"
                "<h1>batch11 审查邮件预览</h1><pre style='white-space:pre-wrap'>" +
                html.escape("Subject: " + subject) + "\n\n" +
                html.escape(body) + "</pre></body></html>")
    (MAIL_COMPAT / "batch11_review_preview.html").write_text(html_doc, encoding="utf-8")

    # 2. Code hashes.
    code_shas = {f: sha256_file(V3_DIR / f) for f in V3_FILES}
    (MAIL_COMPAT / "code_shas.json").write_text(
        json.dumps(code_shas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 3. ready_to_switch.json.
    ready = {
        "candidate_version": "v6.13.0-siface.rc1",
        "switch_kind": "stage_email_v3_nested_verdict_repair",
        "entrypoint": str(V3_DIR / "stage_watcher.py"),
        "args": ["--watch", "--interval", "45"],
        "runtime_path": str(V3_DIR),
        "runtime_notes": [
            "run with: python -X utf8 -u <entrypoint> --watch --interval 45",
            "CPU not applicable to this watcher (no model inference); poll 30-60s",
            "uses the SHARED ledger/state/lock under reports/breakthrough_20261002/stage_email",
            "never reads email_key / credentials into any model or log",
        ],
        "old_watcher": OLD_WATCHER,
        "old_watcher_verification_required_by_parent": [
            "check actual PID 44488 identity (creation 02:23:03.972883+08:00) before stop",
            "stop old watcher ONLY, then start new foreground persistent watcher",
        ],
        "batch11_repair_scope": {
            "problem": "nested protocol_compliance.verdict=PASS + scientific_acceptance.verdict=FAIL, no top-level verdict; v2 detect_review only normalized top-level verdict",
            "fix": "detect_review now classifies nested schema separately; integrity PASS + scientific FAIL preserved, never conflated; ambiguous/conflicting schemas blocked",
            "expected_event_id": "batch11:review_final",
            "expected_send_once": True,
            "no_old_batch01_03_flood": True,
        },
        "new_code_hashes": code_shas,
        "tests": {
            "file": str(V3_DIR / "test_stage_email.py"),
            "result": "26/26 PASS (MockSMTP only, no real SMTP / no credential read)",
            "coverage": ["old top-level schema", "nested schema", "scientific negative",
                         "missing/ambiguous/conflicting verdict blocked",
                         "premature worker artifact", "accepted/attempting/unknown no-resend",
                         "missing/corrupt ledger fail-closed", "stable event ID",
                         "accepted-before-QUIT", "low-balance ledger unchanged",
                         "Chinese batch11 review preview"],
        },
        "review_evidence": {
            "batch11_review_artifact": str(review_path),
            "batch11_review_sha256": sha256_file(review_path),
            "detected_verdict": review_info["verdict"],
            "detected_scientific_verdict": review_info["scientific_verdict"],
            "detected_verdict_kind": review_info["verdict_kind"],
        },
        "parent_action_after_fresh_review_pass": (
            "launch v3 watcher foreground persistent, verify SMTP ledger gets "
            "batch11:review_final exactly once, then keep v3 as successor watcher"),
        "do_not_change_budget_controller_or_automation_prompt": True,
        "generated_at_local": sc.now_iso() if hasattr(sc, "now_iso") else "",
    }
    (MAIL_COMPAT / "ready_to_switch.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "mail_compat": str(MAIL_COMPAT),
        "preview_txt": str(MAIL_COMPAT / "batch11_review_preview.txt"),
        "preview_html": str(MAIL_COMPAT / "batch11_review_preview.html"),
        "ready_to_switch": str(MAIL_COMPAT / "ready_to_switch.json"),
        "code_shas": str(MAIL_COMPAT / "code_shas.json"),
        "detected": {"verdict": review_info["verdict"],
                     "scientific_verdict": review_info["scientific_verdict"],
                     "kind": review_info["verdict_kind"]},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

