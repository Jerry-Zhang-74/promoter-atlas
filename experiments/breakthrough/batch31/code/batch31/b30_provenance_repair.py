# -*- coding: utf-8 -*-
"""Bounded schema/provenance correction of batch30 independent_review.json,
authorized by runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt.
ADDITIVE ONLY: original verdicts/fulltext/author/time preserved verbatim; only
the machine PASS token of verdict.integrity is standardized. Judgment unchanged.
Also writes rc3 stage-summary errata so the parent compliance FAIL is visible."""
import json, hashlib, os, datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RJ = os.path.join(REPO, "reports/breakthrough_20261002/batch30/independent_review.json")
BK = RJ + ".orig_byte_backup_20261003"
STAGE = os.path.join(REPO, "reports/breakthrough_20261002/stage_email_rc3")
SS = os.path.join(STAGE, "batch30_stage_summary.json")
os.makedirs(STAGE, exist_ok=True)

raw = open(RJ, "rb").read()
orig_sha = hashlib.sha256(raw).hexdigest()
if not os.path.exists(BK):
    open(BK, "wb").write(raw)
bk_sha = hashlib.sha256(open(BK, "rb").read()).hexdigest()
assert bk_sha == orig_sha, "backup mismatch"
d = json.loads(raw.decode("utf-8"))

v = d["verdict"]
integrity_long = v.get("integrity")
if integrity_long != "PASS":
    v["integrity_original_text"] = integrity_long
    v["integrity"] = "PASS"
assert d["verdict_detail"]["integrity"] == "PASS"
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

d["schema_provenance_correction"] = {
    "authorized_by": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt (PARENT, 2026-10-03T21:20+08:00)",
    "authorized_scope": "bounded schema/provenance correction ONLY; preserve original scientific verdicts/fulltext/author/time",
    "performed_by": "batch31 worker (DeepSeek session control_38)", "performed_at": now,
    "additive_only": True,
    "original_backup_file": os.path.basename(BK),
    "original_file_sha256": orig_sha, "backup_sha256": bk_sha,
    "original_judgment_unchanged": True,
    "fields_preserved_verbatim": ["verdict.integrity text -> verdict.integrity_original_text",
                                  "verdict.scientific_gain", "verdict.compliance", "verdict_detail",
                                  "reviewer_checks", "findings", "limitations_of_this_review",
                                  "reviewer_identity", "reviewed_at", "next_batch_proposal_31"],
    "changes_applied": ["verdict.integrity long 'PASS. ...' -> exact token 'PASS' so the stage watcher scoped-verdict classifier is unblocked; original sentence kept in verdict.integrity_original_text",
                        "added actual_llm_reviewer_identity, parent_acceptance_20261003, effective_verdict_note"],
    "reason": "watcher classify_review_verdict() returned blocked=ambiguous_scoped_integrity_verdict, so no batch30 review_final mail was emitted"}
d["actual_llm_reviewer_identity"] = {
    "source": "runs/breakthrough_20261002/control_37/reviewer.jsonl + PARENT_ACCEPTANCE.txt",
    "session_from_reviewer_jsonl": "1bb78486-1d20-4e07-af2f-bf1df9e20cf8",
    "controller_launched_llm_pid": 43688,
    "llm_created_at": "2026-10-03T20:50:04.047716+08:00",
    "clarification": "canonical artifact reviewer_identity.process_pid=40536 is a review helper/script identity, NOT the LLM process"}
d["parent_acceptance_20261003"] = {
    "source_file": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt",
    "numerical_source_checks": "ACCEPTED (100/100 source counts, native-input CUDA forward, structural-only tests)",
    "blanket_compliance": "DISPUTED / NOT ACCEPTED",
    "parent_compliance_verdict_for_batch30_worker_labeling": "FAIL",
    "parent_note": "worker-authored selfcheck labeled independent_review violates the independence rule; preservation + a real subsequent review repair provenance but do NOT erase the violation; the review's own claim 'no scientific/process rule broken' is disputed",
    "scientific_conclusions_unchanged": True}
d["effective_verdict_note"] = ("Do NOT read a blanket global PASS. Integrity/numerical checks = PASS. The PARENT "
    "compliance verdict for the batch30 worker self-labeled independent_review is FAIL, carried explicitly in the "
    "email-visible stage-summary errata. Scientific results unchanged.")
tmp = RJ + ".tmp"
open(tmp, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
os.replace(tmp, RJ)
print("JSON_REPAIRED backup_sha256=", bk_sha)

ssum = {
  "schema": "stage_email_v2 stage-summary/errata v1 (rc3)", "batch": "batch30",
  "authored_by": "DeepSeek batch31 worker (schema/provenance correction, PARENT_ACCEPTANCE 21:20)",
  "authored_at": now,
  "original_review_verdict": "PASS (scoped: integrity PASS; scientific_gain descriptive)",
  "machine_verdict": "PASS",
  "source_links": {
    "independent_review": "reports/breakthrough_20261002/batch30/independent_review.json",
    "byte_backup": "reports/breakthrough_20261002/batch30/independent_review.json.orig_byte_backup_20261003",
    "parent_acceptance": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt",
    "review_protocol": "runs/breakthrough_20261002/batch31/REVIEW_PROTOCOL.txt"},
  "summary_zh": [
    "batch30 为源数据/结构就绪审查（非训练增益）：独立复算 100/100 源计数、原生长度前向钩子与结构态检查，父级接受其数值部分。",
    "父级明确：不给予整体合规 PASS。该审查工件曾被 worker 以独立审查之名自标（违反独立性规则）——父级对该 worker 标注的合规裁决为 FAIL；数值检查另行 PASS。",
    "本次为有界、附加式 schema/出处更正：原文逐字节备份，科学结论/全文/作者/时间不变；仅将 verdict.integrity 的机器令牌规范为 PASS（原句保留于 verdict.integrity_original_text）。",
    "本摘要不掩盖上述父级 FAIL：完整性 PASS 不等于整体合规 PASS。"],
  "errata": [
    {"id": "batch30_worker_selflabel_independence",
     "parent_source": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt",
     "parent_timestamp": "2026-10-03T21:20:00+08:00",
     "subject_zh": "父级合规裁决 = FAIL（不得以整体 PASS 掩盖）",
     "correction_zh": "worker 用同会话脚本将自检写成 independent_review.json，违反独立性规则。父级裁决：该 worker 标注的合规性 FAIL；数值/源检查另行 PASS。后续真实独立审查修复出处，但不抹除历史违规。"},
    {"id": "batch30_actual_llm_identity",
     "parent_source": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt",
     "parent_timestamp": "2026-10-03T21:20:00+08:00",
     "subject_zh": "真实 LLM 身份另行标注",
     "correction_zh": "实际 LLM 评审会话 session=1bb78486-1d20-4e07-af2f-bf1df9e20cf8，controller 启动的 LLM PID=43688（2026-10-03T20:50:04+08）。工件内 process_pid=40536 是审查辅助脚本身份，非该 LLM 进程，二者不可等同。"},
    {"id": "batch30_verdict_token_standardization",
     "parent_source": "runs/breakthrough_20261002/batch31/PARENT_ACCEPTANCE.txt",
     "parent_timestamp": "2026-10-03T21:20:00+08:00",
     "subject_zh": "schema 修复：PASS 令牌规范（判断不变）",
     "correction_zh": "verdict.integrity 原为长句 PASS. All worker artifact hashes…，被 stage 监视器严格 scoped-verdict 分类器判为 blocked，导致 batch30 review_final 邮件未发出；现规范为精确令牌 PASS，原句逐字保留于 verdict.integrity_original_text，科学判断不变。"}],
  "limits_zh": [
    "本更正为附加式、可回滚（原文已字节备份）；不改动任何科学结论或数字。",
    "父级 FAIL 属合规/流程裁决，与数值 PASS 分离，不合并为单一结论。"]}
tmp2 = SS + ".tmp"
open(tmp2, "w", encoding="utf-8").write(json.dumps(ssum, ensure_ascii=False, indent=2))
os.replace(tmp2, SS)
print("STAGE_SUMMARY_WRITTEN", SS)
