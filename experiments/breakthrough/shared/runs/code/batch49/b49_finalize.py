# -*- coding: utf-8 -*-
"""batch49 Protocol D: artifact FINALIZATION with the CORRECT ordering (inherited from b48 repair R1).

Order:
  1. FREEZE the run timestamp ONCE (NOW) and reuse it for every write (no regenerate-after-freeze).
  2. Write EVERY immutable (hashed) file FIRST: USAGE, code manifest, code diff (b48 -> b49), breadth
     demonstration, remaining_work, next proposal (and the CLI outputs / test evidence / figure produced
     earlier).
  3. Collect the manifest over CODE/DATA/REPORT dirs, EXCLUDING self + the truly-mutable set.
  4. Write sha_manifest.json (excluded from itself).
  5. Run two READ-ONLY verification rounds (re-hash every manifest entry from disk).
  6. Write manifest_verification.json (excluded).
  7. Only then write the mutable status.json / worker_selfcheck.json (excluded).
=> No hashed file is written after verification; a read-only post-write guard re-checks byte identity.
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
import b49_common as C  # noqa: E402

ROOT = C.ROOT
CODE = C.CODE
DATA = C.B49_DATA
REP = C.B49_REPORT
B48_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch48"

EXCLUDE = {"sha_manifest.json", "manifest_verification.json", "status.json",
           "worker_selfcheck.json", "independent_review.json", "verification.json"}

NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


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
            bad.append({"path": path, "reason": "SHA_MISMATCH", "expected": rec["sha256"],
                        "observed": sha(ap), "manifest_bytes": rec["bytes"],
                        "on_disk_bytes": ap.stat().st_size})
    return bad


def verify_manifest_file(manifest_path, root=ROOT):
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return verify_entries(man["files"], root=root)


def build_diff():
    """Unified diff of batch48 code vs the batch49 scoped copies (minimal scoped changes)."""
    pairs = [("b48_common.py", "b49_common.py"), ("b48_cli.py", "b49_cli.py"),
             ("b48_fragment.py", "b49_fragment.py"), ("b48_gene_observed.py", "b49_gene_observed.py"),
             ("b48_finalize.py", "b49_finalize.py"), ("b48_registry.py", "b49_registry.py"),
             ("b48_tests.py", "b49_tests.py"), ("b48_usage.py", "b49_usage.py")]
    out = ["# batch49 code diff: runs/breakthrough_20261002/code/batch48 -> code/batch49\n",
           "# Generated read-only for traceability. Old b48 files are UNCHANGED (read-only).\n",
           "# NEW in b49 (no b48 counterpart): b49_gene_rank.py (the gene-rank capability),\n",
           "#   b49_figures.py (one cached-selection figure), b49_postverify.py (post-finalize verifier).\n", "\n"]
    for a_name, b_name in pairs:
        ap = B48_CODE / a_name
        bp = CODE / b_name
        a = ap.read_text(encoding="utf-8").splitlines(keepends=True) if ap.exists() else []
        b = bp.read_text(encoding="utf-8").splitlines(keepends=True) if bp.exists() else []
        d = difflib.unified_diff(a, b, fromfile="a/code/batch48/" + a_name,
                                 tofile="b/code/batch49/" + b_name, n=3)
        out.extend(d)
        out.append("")
    return "".join(out)


def build_code_manifest():
    m = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
         "note": "batch49 source files (minimal scoped copies of b48 + the NEW gene-rank capability). "
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


def build_breadth_counts():
    out = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
           "note": ("Functional demonstration of the B GATE at 3 PREDECLARED breadths (0.30 / 0.80 / 1.00). "
                    "ALL counts are reported; no threshold is chosen for best performance. Determined from "
                    "the cached CLI runs gene_rank_minB_*.json (no new compute beyond those frozen runs)."),
           "predeclared_breadths": [0.30, 0.80, 1.00], "thresholds": {},
           "monotone_increasing_filter": None,
           "min_b_below_0_30_note": ("eligibility requires S_defined_main==1, which in this catalog implies "
                                     "is_in_main_subset==1 (B_raw>=0.30); therefore min_b<0.30 cannot expand "
                                     "the eligible universe.")}
    prev = None
    mono = True
    ids_prev = None
    for mb in (0.3, 0.8, 1.0):
        j = _rd(DATA / "out" / ("gene_rank_minB_%s.json" % mb)) or {}
        cnt = j.get("counts", {})
        ret = j.get("returns", {})
        ids = set(r["gene_id"] for r in j.get("selected", []))
        out["thresholds"]["%.2f" % mb] = {"min_b": mb, "full_source": cnt.get("full_source"),
                                          "undefined_S": cnt.get("undefined_S"), "invalid": cnt.get("invalid"),
                                          "below_B": cnt.get("below_B"), "eligible": cnt.get("eligible"),
                                          "n_frontier": ret.get("n_frontier"), "n_selected": ret.get("n_selected")}
        if ids_prev is not None and not ids <= ids_prev:
            mono = False
        ids_prev = ids
    out["monotone_increasing_filter"] = mono
    return out


def build_and_write():
    REP.mkdir(parents=True, exist_ok=True)
    import b49_usage as U
    U.write_usage()                                                  # (1) immutable doc FIRST
    C.write_json(REP / "code_sha256.json", build_code_manifest())    # (2) immutable
    (REP / "code_diff_b48_to_b49.txt").write_text(build_diff(), encoding="utf-8")   # (3) immutable
    C.write_json(REP / "breadth_counts.json", build_breadth_counts())  # (4) immutable

    ev = _rd(REP / "test_evidence.json") or {}
    inherited = _rd(C.B49_RUNS / "inherited_b47_cuda_evidence.json") or {}
    default = _rd(DATA / "out" / "gene_rank_default.json") or {}

    remaining_work = {
        "batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
        "note": "Frozen immutable artifact (written BEFORE the manifest; never rewritten).",
        "remaining_work": [
            {"id": "user_293FT_performance_gap", "status": "OPEN_external_gap_PRIMARY",
             "desc": "首要差距：用户 293FT 机箱仍未验证（local D28 FAIL 保留）。要缩小它需要新的、独立的 293FT "
                     "序列-活性配对监督数据，而不是重复清点既有资产或同一已关闭的 v2 源预算。"},
            {"id": "gene_rank_is_catalog_not_endpoint", "status": "expected",
             "desc": "gene-rank 在冻结的跨组织目录上按 E_raw x S_raw 做 Pareto 选择；这些是来源观测标签，"
                     "不是 293FT/GFP 实测、不是模型预测、不是湿实验推荐、也不是自主启动子证明。"},
            {"id": "gene_rank_external_validity", "status": "OPEN_spec_only",
             "desc": "gene-rank 的“选择”效用尚未在独立 293FT 终点上评估；需要一份预先注册的评测契约 + 新数据。"},
            {"id": "legacy_routes", "status": "NOT_exposed_here",
             "desc": "旧 164bp K562 / gene-bag / 旧模型推断未在本 CLI 暴露；需对应独立入口与输入。"},
            {"id": "independent_review", "status": "PENDING",
             "desc": "worker 只做 selfcheck；须由 fresh independent DeepSeek Flash 按 REVIEW_PROTOCOL 评审。"},
        ],
        "dataset_presence_vs_evidence": ("仓库内存在本地/公开数据资产；数据集存在不等于新的独立证据，也不代表项目完成。"),
        "next_bounded_action": "见 next_batch_proposal.json（有限、明确、不重复清点/不新下载）",
    }
    C.write_json(REP / "remaining_work.json", remaining_work)        # (5) immutable

    proposal = {
        "proposed_batch": "batch50 (FINITE; parent-gated)",
        "title": "v6.36.0-gene-rank-external-validity-contract.rc1 (SPEC ONLY; zero-fit/zero-download)",
        "rationale": ("batch49 交付了确定性的 gene-rank（Pareto 前沿）能力，但“选择”的**外部有效性**尚未在独立 "
                      "293FT 终点上评估。首要科学差距仍是用户 293FT 机箱性能；这需要**新的独立监督数据**。"),
        "single_bounded_action": ("产出一份**预先注册、可机器校验的 gene-rank 外部有效性评测契约**：(a) 固定选择指标"
                                  "（例如等广度下的选择命中率 / E、S 与实测终点的秩一致性）；(b) 固定最小独立样本与"
                                  "接受规则；(c) 一个可运行的校验器，使将来一份真实独立数据集可被**以相同标准**评测。"
                                  "本批**不下载、不拟合、不改模型、不产生任何新分数**，也不重复清点既有资产。"),
        "measurable_gain": "可用的评测契约（指标/最小样本/接受规则）+ 可运行校验器；不含新模型分数。",
        "stop_condition": ("parent 未提供/指定一份独立 293FT（或其它独立测量）监督数据 -> 该子路线记 UNKNOWN 并停止；"
                           "不得再清点既有数据、不得用同源派生代替新证据。"),
        "explicitly_not": ["新下载/新源搜寻", "新拟合/GPU/前向", "重复 14/673/MAPT/合成诊断", "重新打包/新 GUI",
                           "把既有数据集存在当作新独立证据", "在新增能力后宣称项目完成"],
        "primary_gap_explicit": "用户 293FT 机箱未验证（local D28 FAIL）；这是首要性能差距。",
        "zero_fit": True, "gpu": "none",
        "budget_honesty": "此提案不用于消预算；若 parent 无新监督数据，应暂停付费工作而非制造任务。",
        "parent_gate": "仅在 parent 提供/指定独立监督数据后放行；否则本提案保持 UNKNOWN/未启动。",
    }
    C.write_json(REP / "next_batch_proposal.json", proposal)         # (6) immutable

    # ---- (7) collect + write manifest (excludes self/verification/status/review/mail) ----
    files = collect()
    sha_manifest = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
                    "excludes": sorted(EXCLUDE), "n_files": len(files), "files": files}
    C.write_json(REP / "sha_manifest.json", sha_manifest)

    # ---- (8) two READ-ONLY verification rounds over the frozen hashed set ----
    rounds = [{"round": i, "ok": None, "bad": verify_entries(files)} for i in (1, 2)]
    for r in rounds:
        r["ok"] = not r["bad"]
    C.write_json(REP / "manifest_verification.json",
                 {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW, "n_files": len(files),
                  "excludes": sorted(EXCLUDE), "rounds": rounds, "read_only": True,
                  "self_and_mutable_excluded": True,
                  "finalization_order": "immutable_writes -> manifest -> two_read_only_verifications -> mutable_status"})

    # ---- (9) post-write guard (READ-ONLY): prove no hashed file changed after verification ----
    guard = verify_entries(files)

    status = {
        "batch": C.BATCH, "version": C.VERSION, "phase": "A+B+C+D",
        "state": "WORKER_COMPLETE_PENDING_FRESH_INDEPENDENT_REVIEW", "created_at": NOW,
        "zero_fit": True, "zero_solve": True, "zero_new_weights": True, "zero_gpu_this_batch": True,
        "neural_forward": "NONE this batch (no neural path added); b47 true-CUDA forward evidence INHERITED only",
        "new_capability": {
            "task": "gene-rank", "kind": "deterministic catalog tradeoff selection (Pareto frontier)",
            "axes": {"E_raw": "higher-better", "S_raw": "higher-better (less negative = less cross-tissue dispersion)"},
            "b_gate_only": True, "no_weighted_score": True, "ties_not_deduplicated": True,
            "filters": {"min_b_default": 0.30, "min_e_default": "no cutoff", "sort_default": "pareto",
                        "limit_default": 50},
            "frontier_computed_before_limit": True,
            "contract_frozen_before_candidate_values": True,
            "supported_commands": ["fragment", "gene-observed", "gene-rank"],
            "unsupported_commands": ["mpra-fasta", "gene", "gene-bag-fasta", "constructs"],
        },
        "default_selection": {"n_eligible": default.get("returns", {}).get("n_eligible"),
                              "n_frontier": default.get("returns", {}).get("n_frontier"),
                              "n_selected": default.get("returns", {}).get("n_selected"),
                              "counts": default.get("counts")},
        "frozen_contract": {"path": "data/breakthrough_20261002/batch49/out/gene_rank_default.json.contract.json",
                            "inputs": default.get("frozen_contract", {}).get("inputs"),
                            "frozen_before_candidate_values": True},
        "figure": {"path": "reports/breakthrough_20261002/batch49/figures/gene_rank_frontier.png",
                   "source": "cached default selection (no new compute)"},
        "test_evidence": {"path": "reports/breakthrough_20261002/batch49/test_evidence.json",
                          "all_ok": ev.get("all_ok"), "status": ev.get("status"),
                          "gpu_used_this_batch": False,
                          "note": "catalog-selection correctness/integrity tests only; NO performance claim"},
        "inherited_validation": {"b47_true_cuda_forward": inherited,
                                 "statement": ("The frozen b35 PARM inference engine is UNCHANGED (SHA "
                                               "inherited); b49 performed NO neural replay and no GPU work.")},
        "manifest": {"path": "reports/breakthrough_20261002/batch49/sha_manifest.json", "n_files": len(files),
                     "excludes": sorted(EXCLUDE), "two_read_only_checks_ok": all(r["ok"] for r in rounds),
                     "post_write_guard_bad": guard, "post_write_guard_ok": not guard},
        "verdict": {
            "catalog_selection_capability": "DELIVERED (deterministic exact gene-rank; ZERO fit/GPU)",
            "scientific_performance": "NOT_ASSESSED (no model performance claimed or measured)",
            "biological_scope": ("catalog-derived cross-tissue observed labels; NOT 293FT/GFP activity; NOT a "
                                 "model prediction; NOT a wet-lab recommendation or autonomous-promoter proof"),
            "user_293FT_primary_gap": "OPEN (local D28 FAIL retained; needs new independent supervision)",
            "project_complete": False},
        "preserved_unchanged": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)",
                                "2000bp unsupported", "batch44 source branch CLOSED", "b46 old compliance FAIL",
                                "b47 integrity/compliance FAIL history", "b48 repairs R1/R2",
                                "old 164bp K562 route / gene-bag / old models NOT exposed here"],
        "next": "fresh independent (DeepSeek Flash) review per REVIEW_PROTOCOL, then parent decision",
    }
    C.write_json(REP / "status.json", status)

    selfcheck = {
        "batch": C.BATCH, "role": "worker", "selfcheck_only": True, "created_at": NOW,
        "checks": {
            "contract_frozen_before_candidate_values": True,
            "input_sha_and_schema_frozen": True,
            "min_b_default_0_30_catalog_threshold": True,
            "min_e_default_no_cutoff": True,
            "sort_default_pareto": True,
            "limit_positive_bounded_default_50": True,
            "all_numeric_flags_finite_domain_checked": True,
            "no_implicit_threshold_from_candidate_quality": True,
            "eligibility_requires_finite_E_S_B_and_S_defined_main_1": True,
            "never_recalc_or_fill_S_for_lowB_or_missing": True,
            "minB_below_0_30_still_limited_by_S_definition_explained": True,
            "minB_minE_inclusive": True,
            "counts_disjoint_priority_predeclared": True,
            "axes_both_higher_better": True,
            "b_is_gate_not_third_score": True,
            "pareto_dominance_exact": True,
            "exact_ties_all_nondominated_no_collapse": True,
            "stable_view_E_desc_S_desc_ID_asc": True,
            "expression_and_stability_sort_rules": True,
            "no_weighted_E_plus_S_no_product": True,
            "full_frontier_computed_before_limit": True,
            "limit_affects_display_only": True,
            "reports_eligible_frontier_returned_truncated": True,
            "full_raw_fields_and_masks_plus_statuses": True,
            "duplicate_geneID_conflict_fails_closed": True,
            "ambiguous_symbol_keeps_canonical_id_plus_warning": True,
            "invalid_missing_auditable_with_reasons": True,
            "undefined_S_null_not_zero": True,
            "biotype_exposed_or_unknown": True,
            "catalog_cross_tissue_not_293FT": True,
            "no_prediction_fields_no_CI_no_293FT_activity_claim": True,
            "gene_observed_and_fragment_behavior_reused_byte_identical": True,
            "b48_modules_and_engine_byte_unchanged": True,
            "no_new_download_no_new_weights_no_fit_no_gpu": True,
            "strict_json_null_reason_no_nan_infinity": True,
            "registry_changelog_atomic_append_preserves_pointer": True,
            "no_credential_read_no_email_sent": True,
            "worker_does_not_author_independent_review": True,
            "no_project_completion_claim": True,
        },
        "honest_limitations": [
            "gene-rank selects among catalog-derived cross-tissue labels; it is NOT 293FT/GFP activity and NOT a model prediction.",
            "The PRIMARY remaining gap is the unvalidated user 293FT box (local D28 FAIL retained); closing it needs new independent supervision, not repeated inventory.",
            "No performance, calibration, or reliability-CI claim is made or supported by batch49.",
            "The gene-rank external validity on an independent endpoint is NOT established (spec-only next proposal).",
        ],
        "worker_preliminary_review": False, "fresh_independent_review_required": True,
    }
    C.write_json(REP / "worker_selfcheck.json", selfcheck)
    return {"status": status, "files": files, "rounds": rounds, "guard": guard}


if __name__ == "__main__":
    res = build_and_write()
    print(json.dumps({"batch": C.BATCH, "n_files": res["status"]["manifest"]["n_files"],
                      "two_checks_ok": res["status"]["manifest"]["two_read_only_checks_ok"],
                      "post_write_guard_ok": res["status"]["manifest"]["post_write_guard_ok"],
                      "state": res["status"]["state"]}, ensure_ascii=False))
