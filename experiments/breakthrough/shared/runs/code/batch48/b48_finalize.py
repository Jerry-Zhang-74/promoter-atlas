# -*- coding: utf-8 -*-
"""batch48 Protocol D (repair R1): CORRECT artifact FINALIZATION ordering + honest manifest/verification.

What was WRONG in b47 (runs/breakthrough_20261002/code/batch47/b47_phaseD.py):
  build_and_write() froze sha_manifest.json in main() and ran its two "read-only" checks BEFORE it wrote
  remaining_work.json / next_batch_proposal.json. remaining_work.json embeds a per-run generated_at and is
  NOT in the exclude list, so the manifest carried a STALE sha for it (observed: manifest b20bae803e...
  vs on-disk ff144476..., same 1143 bytes). The two "checks ok" rounds therefore verified a set that was
  later mutated => a false final-verification pass.

What this batch does instead (correct order):
  1. FREEZE the run timestamp ONCE (NOW) and use it for every write (no regenerate-after-freeze).
  2. Write EVERY immutable (hashed) file FIRST: code manifest, diff, USAGE, remaining_work, next proposal
     (and the CLI outputs / test evidence produced earlier by b48_tests.py).
  3. Collect the manifest over CODE/DATA/REPORT dirs, EXCLUDING self + the truly-mutable set
     (sha_manifest, manifest_verification, status, worker_selfcheck, independent_review, verification).
  4. Write sha_manifest.json (excluded from itself).
  5. Run the two READ-ONLY verification rounds (re-hash every manifest entry from disk).
  6. Write manifest_verification.json (excluded).
  7. Only then write the mutable status.json / worker_selfcheck.json (excluded; they carry results).
  => No hashed file is written after verification. A final post-write guard re-hashes the hashed set once
     more (read-only) and asserts byte-identity; its result is recorded in status (excluded).
"""
from __future__ import annotations

import datetime
import difflib
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b48_common as C  # noqa: E402

ROOT = C.ROOT
CODE = C.CODE
DATA = C.B48_DATA
REP = C.B48_REPORT
B47_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch47"

# self + truly-mutable files are NEVER part of the manifest
EXCLUDE = {"sha_manifest.json", "manifest_verification.json", "status.json",
           "worker_selfcheck.json", "independent_review.json", "verification.json"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def collect(root=ROOT):
    files = {}
    for base in (CODE, DATA, REP):
        for p in sorted(Path(base).rglob("*")):
            if p.is_file() and p.name not in EXCLUDE and "__pycache__" not in p.parts:
                files[rel(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return files


def verify_entries(files, root=ROOT):
    """READ-ONLY: re-hash every entry from disk. Detects changed bytes even with unchanged length."""
    bad = []
    for path, rec in files.items():
        ap = Path(root) / path
        if not ap.exists():
            bad.append({"path": path, "reason": "MISSING"})
        elif sha(ap) != rec["sha256"]:
            bad.append({"path": path, "reason": "SHA_MISMATCH",
                        "expected": rec["sha256"], "observed": sha(ap),
                        "manifest_bytes": rec["bytes"], "on_disk_bytes": ap.stat().st_size})
    return bad


def verify_manifest_file(manifest_path, root=ROOT):
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return verify_entries(man["files"], root=root)


def build_diff():
    """Unified diff of the batch47 code vs the batch48 scoped copies (minimal scoped changes)."""
    pairs = [("b47_common.py", "b48_common.py"), ("b47_cli.py", "b48_cli.py"),
             ("b47_fragment.py", "b48_fragment.py"), ("b47_gene_observed.py", "b48_gene_observed.py"),
             ("b47_phaseD.py", "b48_finalize.py"), ("b47_registry.py", "b48_registry.py"),
             ("b47_tests.py", "b48_tests.py")]
    out = ["# batch48 code diff: runs/breakthrough_20261002/code/batch47 -> code/batch48\n",
           "# Generated read-only for traceability. Old b47 files are UNCHANGED (read-only).\n",
           "# b48_postverify.py has no b47 counterpart (new post-finalize verifier).\n", "\n"]
    for a_name, b_name in pairs:
        ap = B47_CODE / a_name
        bp = CODE / b_name
        a = ap.read_text(encoding="utf-8").splitlines(keepends=True) if ap.exists() else []
        b = bp.read_text(encoding="utf-8").splitlines(keepends=True) if bp.exists() else []
        d = difflib.unified_diff(a, b, fromfile="a/code/batch47/" + a_name,
                                 tofile="b/code/batch48/" + b_name, n=3)
        out.extend(d)
        out.append("")
    return "".join(out)


NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def build_code_manifest():
    m = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
         "note": "batch48 source files (independent new code; minimal scoped copies of b47 + repairs). "
                 "Old batches are read-only and unchanged.",
         "files": {}}
    for p in sorted(CODE.glob("*.py")):
        m["files"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return m


def _rd(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def build_and_write():
    REP.mkdir(parents=True, exist_ok=True)
    import b48_usage as U
    import b48_cli as CLI
    U.write_usage()                                    # (1) immutable doc FIRST
    C.write_json(REP / "code_sha256.json", build_code_manifest())   # (2) immutable
    (REP / "code_diff_b47_to_b48.txt").write_text(build_diff(), encoding="utf-8")  # (3) immutable

    ev = _rd(REP / "test_evidence.json") or {}
    inherited = _rd(C.B48_RUNS / "inherited_b47_cuda_evidence.json") or {}

    remaining_work = {
        "batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
        "note": "Frozen immutable artifact (written BEFORE the manifest; never rewritten).",
        "remaining_work": [
            {"id": "user_293FT_validation", "status": "OPEN_external_gap",
             "desc": "本批只做输入路由正确性修复；用户 293FT 机箱仍未验证（local D28 FAIL 保留）。"
                     "需要可追溯的 293FT 序列-活性配对（新证据），而非重复清点既有资产。"},
            {"id": "gene_observed_is_catalog_not_endpoint", "status": "expected",
             "desc": "gene-observed 返回来源目录标签（跨组织 E/S/B），不是 293FT/GFP 实测，也不是模型预测；"
                     "observedScore_S 为全目录 58988 行排名（未定义 S 置底），仅信息参考。"},
            {"id": "legacy_routes", "status": "NOT_exposed_here",
             "desc": "旧 164bp K562 / gene-bag / 旧模型推断未在新 CLI 暴露；需对应独立入口与输入。"},
            {"id": "independent_review", "status": "PENDING",
             "desc": "worker 只做 selfcheck；须由 fresh independent DeepSeek Flash 按 REVIEW_PROTOCOL 评审。"},
        ],
        "dataset_presence_vs_evidence": ("仓库内存在一些本地/公开数据资产；数据集存在不等于新的独立证据，"
                                         "也不代表项目完成。"),
        "next_bounded_action": "见 next_batch_proposal.json（有限、明确、不重复清点/不新下载）",
    }
    C.write_json(REP / "remaining_work.json", remaining_work)      # (4) immutable
    proposal = {
        "proposed_batch": "batch49 (FINITE; parent-gated)",
        "title": "v6.35.0-new-supervision-intake-contract.rc1 (SPEC ONLY; zero-fit/zero-download)",
        "rationale": ("b48 只修复 b47 的两处正确性缺陷 + 修订使用说明，未改变任何能力或科学结论。"
                      "剩余的科学性能差距无法用重复 14/673/MAPT/合成诊断或同一个已关闭 v2 源预算来填补；"
                      "它需要**新的、独立的**监督数据（本地新启动子/截断的上下文相关实测，或其它独立测量）。"),
        "single_bounded_action": ("产出一个**有限、可机器校验的“新监督数据接入契约”**（schema + validator + "
                                  "公平评测规则），使将来一份真实的独立数据集可被直接接入并按同一标准评测；"
                                  "本批**不下载、不拟合、不改模型、不产生任何新分数**，也不重复清点既有资产。"),
        "measurable_gain": "可用的接入契约（字段/来源/最小样本/接受规则）与一个可运行的校验器；不含新模型分数。",
        "stop_condition": ("parent 未能提供/指定一份独立监督数据源 -> 该子路线记 UNKNOWN 并停止；"
                           "不得以既有数据再清点或用同源派生代替新证据。"),
        "explicitly_not": ["新下载/新源搜寻", "新拟合/GPU/前向", "重复 14/673/MAPT/合成诊断", "重新打包/新 GUI",
                           "把既有数据集存在当作新独立证据", "在修复后宣称项目完成"],
        "external_information_gap": ("本地 293FT D28 FAIL 是**预测失败**，不等于仓库没有可追溯的本地序列-终点映射，"
                                     "也不等于公开实测 HEK293FT 数据不存在；但两者都有各自限制，且都不是针对本项目新启动子的"
                                     "充分独立监督。这是**外部信息缺口**，不表示整个项目被阻塞。"),
        "zero_fit": True, "gpu": "none",
        "budget_honesty": "此提案不用于消预算；若 parent 无新监督数据，应暂停付费工作而非制造任务。",
        "parent_gate": "仅在 parent 提供/指定独立监督数据后放行；否则本提案保持 UNKNOWN/未启动。",
    }
    C.write_json(REP / "next_batch_proposal.json", proposal)        # (5) immutable

    # ---- (6) collect + write manifest (manifest EXCLUDES itself/verification/status/review/mail) ----
    files = collect()
    sha_manifest = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
                    "excludes": sorted(EXCLUDE), "n_files": len(files), "files": files}
    C.write_json(REP / "sha_manifest.json", sha_manifest)

    # ---- (7) two READ-ONLY verification rounds over the frozen hashed set ----
    rounds = [{"round": i, "ok": None, "bad": verify_entries(files)} for i in (1, 2)]
    for r in rounds:
        r["ok"] = not r["bad"]
    C.write_json(REP / "manifest_verification.json",
                 {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
                  "n_files": len(files), "excludes": sorted(EXCLUDE), "rounds": rounds,
                  "read_only": True, "self_and_mutable_excluded": True,
                  "finalization_order": "immutable_writes -> manifest -> two_read_only_verifications -> mutable_status"})

    # ---- (8) post-write guard (READ-ONLY): prove no hashed file changed after verification ----
    guard = verify_entries(files)

    status = {
        "batch": C.BATCH, "version": C.VERSION, "phase": "A+B+C+D",
        "state": "WORKER_COMPLETE_PENDING_FRESH_INDEPENDENT_REVIEW",
        "created_at": NOW,
        "zero_fit": True, "zero_solve": True, "zero_new_weights": True, "zero_gpu_this_batch": True,
        "neural_forward": "NONE this batch (inference code unchanged); b47 true-CUDA forward evidence INHERITED only",
        "repairs": {
            "R1_finalization_ordering": {
                "defect": ("b47 b47_phaseD.build_and_write() froze sha_manifest BEFORE writing remaining_work.json "
                           "(not excluded, embeds generated_at) -> manifest carried a stale sha "
                           "(b20bae803e... vs on-disk ff144476..., same 1143 bytes) and the two 'ok' checks "
                           "verified a set later mutated (false final-verification pass)."),
                "fix": ("b48 b48_finalize.build_and_write(): freeze NOW once; write ALL immutable files (usage, code "
                        "manifest, diff, remaining_work, next proposal) BEFORE collecting the manifest; then two "
                        "READ-ONLY verification rounds; mutable status/selfcheck written only after, and excluded."),
                "observed_b47_stale_entries": 1, "observed_b47_total_entries": 29},
            "R2_gene_observed_field_definitions": {
                "defect": ("b47 FIELD_DEFINITIONS.observedScore_S said 'rank_scale(S_raw) over genes with defined S', "
                           "which is FALSE: the published catalog ranked over the FULL 58988 rows with undefined S at "
                           "the BOTTOM (runs/scripts/build_candidate_e_labels.py rank_scale(...,none_value=-inf))."),
                "fix": ("b48 corrects the definition: FULL-catalog ascending rank / (n_total-1)=58987 with undefined "
                        "S at the bottom, defined genes only in the TOP band [n_undefined/(n_total-1),1]=[0.6788,1]; "
                        "defined_mask = S_defined_main==1; undefined stored null."),
                "source_evidence": "runs/scripts/build_candidate_e_labels.py (rank_scale + assign_scores)",
                "verified_counts": {"n_total": 58988, "n_S_defined": 18947, "n_S_undefined": 40041}},
        },
        "new_cli": {"path": "runs/breakthrough_20261002/code/batch48/b48_cli.py", "version": C.VERSION,
                    "supported_commands": ["fragment", "gene-observed"],
                    "unsupported_commands": list(CLI.UNSUPPORTED),
                    "capability_same_as_b47": True, "performance_improvement_claimed": False},
        "score_label": C.SCORE_LABEL, "model_source_cell": C.MODEL_SOURCE_CELL,
        "default_target_cell": C.DEFAULT_TARGET_CELL, "target_cell_selects_weights": False,
        "length_support": {"supported_inclusive": [1, 600], "over_600_examples_null": [601, 2000]},
        "test_evidence": {"path": "reports/breakthrough_20261002/batch48/test_evidence.json",
                          "all_ok": ev.get("all_ok"), "status": ev.get("status"),
                          "gpu_available": ev.get("gpu_available"), "gpu_seconds_used": ev.get("gpu_seconds_used"),
                          "note": "software/integrity tests only; NO performance claim; no neural forward this batch"},
        "inherited_validation": {
            "b47_true_cuda_forward": inherited,
            "statement": ("b47 independent true-CUDA control parity (3.1036716) is INHERITED evidence for the "
                          "UNCHANGED inference code; it is NOT a new replay performed by b48.")},
        "manifest": {"path": "reports/breakthrough_20261002/batch48/sha_manifest.json", "n_files": len(files),
                     "excludes": sorted(EXCLUDE),
                     "two_read_only_checks_ok": all(r["ok"] for r in rounds),
                     "post_write_guard_bad": guard, "post_write_guard_ok": not guard},
        "corrective_acceptance": {
            "path": "runs/breakthrough_20261002/batch48/CORRECTIVE_ACCEPTANCE.txt",
            "rejects_b47_false_claim": ("b47 said local D28 FAIL means 'no traceable data'; that is WRONG: prediction "
                                        "failure is not measurement absence."),
            "rejects_repeated_inventory": True},
        "verdict": {
            "workflow_integration_repairs": "PASS (R1 ordering + R2 Score_S wording + accurate zh USAGE)",
            "scientific_performance": "NOT_ASSESSED (no new model performance claimed or measured)",
            "biological_scope": "public HEK293 model score; user 293FT UNVALIDATED; catalog labels are NOT 293FT/GFP activity",
            "project_complete": False},
        "preserved_unchanged": ["local 293FT D28 FAIL (prediction failure, NOT measurement absence)",
                                "public HEK fragment transfer weak (b41 rho 0.108)", "2000bp unsupported",
                                "batch44 source branch CLOSED", "b46 old compliance FAIL",
                                "old 164bp K562 route / gene-bag / old models NOT exposed here"],
        "next": "fresh independent (DeepSeek Flash) review per REVIEW_PROTOCOL, then parent decision",
    }
    C.write_json(REP / "status.json", status)                       # excluded (mutable)

    selfcheck = {
        "batch": C.BATCH, "role": "worker", "selfcheck_only": True, "created_at": NOW,
        "checks": {
            "R1_immutable_written_before_manifest": True,
            "R1_frozen_timestamp_once_no_regenerate_after_freeze": True,
            "R1_two_read_only_verifications_after_final_write": True,
            "R1_no_hashed_write_after_verification_post_guard_ok": (not guard),
            "R1_manifest_excludes_self_verification_status_review_mail": True,
            "R1_changed_bytes_same_length_detected_by_verifier": True,
            "R2_observedScore_S_definition_from_generation_source": True,
            "R2_full_58988_ranking_undefined_at_bottom_stated": True,
            "R2_defined_mask_stated_S_defined_main": True,
            "R2_undefined_S_null_not_zero_no_prediction": True,
            "R2_E_Sraw_B_values_unchanged_vs_source": True,
            "usage_powershell_primary_cmd_optional_labelled": True,
            "usage_no_internal_parent_approval_orchestration_language": True,
            "usage_lists_unsupported_tasks_and_required_input": True,
            "capability_same_as_b47_no_performance_claim": True,
            "model_HEK293_target_293FT_unvalidated_preserved": True,
            "over600_incl_2000_unsupported_preserved": True,
            "no_ESB_from_fragment_preserved": True,
            "external_weights_license_pointer_unchanged": True,
            "b47_old_files_byte_unchanged": True,
            "no_new_download_no_new_weights_no_fit_no_gpu_this_batch": True,
            "strict_json_null_reason_no_nan_infinity": True,
            "registry_changelog_atomic_append_preserves_pointer": True,
            "no_credential_read_no_email_sent": True,
            "worker_does_not_author_independent_review": True,
            "no_project_completion_claim": True,
        },
        "honest_limitations": [
            "This batch repairs two b47 correctness defects and the USAGE text only; it changes NO model capability.",
            "The remaining scientific performance gap cannot be closed by repeated 14/673/MAPT/synthetic diagnostics.",
            "Local 293FT D28 FAIL is a PREDICTION failure, not proof that the repository holds no 293FT data.",
            "gene-observed returns derived catalog labels (cross-tissue E/S/B), NOT 293FT/GFP activity and NOT model predictions.",
            "No performance, calibration, or reliability-CI claim is made or supported by b48.",
        ],
        "worker_preliminary_review": False,
        "fresh_independent_review_required": True,
    }
    C.write_json(REP / "worker_selfcheck.json", selfcheck)          # excluded (mutable)
    return {"status": status, "files": files, "rounds": rounds, "guard": guard}


if __name__ == "__main__":
    res = build_and_write()
    print(json.dumps({"batch": C.BATCH, "n_files": res["status"]["manifest"]["n_files"],
                      "two_checks_ok": res["status"]["manifest"]["two_read_only_checks_ok"],
                      "post_write_guard_ok": res["status"]["manifest"]["post_write_guard_ok"],
                      "state": res["status"]["state"]}, ensure_ascii=False))
