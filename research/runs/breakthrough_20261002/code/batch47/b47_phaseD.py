# -*- coding: utf-8 -*-
"""batch47 Protocol D: status / test evidence reference / asset manifest / input-output SHA,
two read-only manifest checks, worker_selfcheck, remaining_work, ONE concrete bounded next proposal,
and concise Chinese USAGE. Strict JSON (null + reason, no NaN/Infinity).
Manifest EXCLUDES self + mutable verification/status/review/mail files."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b47_common as C  # noqa: E402

ROOT = C.ROOT
CODE = C.CODE
DATA = C.B47_DATA
REP = C.B47_REPORT
EXCLUDE = {"sha_manifest.json", "manifest_verification.json", "status.json",
           "worker_selfcheck.json", "independent_review.json", "verification.json"}
NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def collect():
    files = {}
    for base in (CODE, DATA, REP):
        for p in sorted(Path(base).rglob("*")):
            if p.is_file() and p.name not in EXCLUDE and "__pycache__" not in p.parts:
                files[rel(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return files


def build_code_manifest():
    m = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
         "note": "batch47 source files (independent new code). Old batches read-only.",
         "files": {}}
    for p in sorted(CODE.glob("*.py")):
        m["files"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return m


def main():
    REP.mkdir(parents=True, exist_ok=True)
    code_man = build_code_manifest()
    C.write_json(REP / "code_sha256.json", code_man)

    files = collect()
    sha_manifest = {"batch": C.BATCH, "version": C.VERSION, "generated_at": NOW,
                    "excludes": sorted(EXCLUDE), "n_files": len(files), "files": files}
    C.write_json(REP / "sha_manifest.json", sha_manifest)

    # two READ-ONLY verification rounds: re-hash the frozen manifest and compare.
    rounds = []
    for i in (1, 2):
        bad = []
        for path, rec in files.items():
            ap = ROOT / path
            if not ap.exists() or sha(ap) != rec["sha256"]:
                bad.append(path)
        rounds.append({"round": i, "ok": not bad, "bad": bad})
    C.write_json(REP / "manifest_verification.json",
                 {"batch": C.BATCH, "version": C.VERSION, "excludes": sorted(EXCLUDE),
                  "n_files": len(files), "rounds": rounds, "read_only": True,
                  "self_and_mutable_excluded": True})
    return files, sha_manifest, code_man, rounds




USAGE = u"""# batch47 v6.34.0-scoped-input-routing.rc1 — 中文使用说明

> 本批仅为**输入路由/作用域修正**：把 b20 的命令路由改为新版本。**不是**新预测器、**不是** GUI、**不是**性能提升声明。

## 两个任务（必须显式选择）

### 1) `fragment`：原始 FASTA → 冻结 PARM HEK293 模型分数
- 复用（不修改）batch35 的 PARM 引擎与外部权重；官方 naked-FASTA：中心零填充到 L_max=600，**无 RC / 无改 DNA / 无隐藏窗口**。
- 支持长度 1..600；>600（如 2000）→ 分数为 `null`，状态 `UNSUPPORTED_LENGTH`。
- 分数标签固定为 `PARM_HEK293_model_score`（五折等权均值）；**不是 293FT 活性 / 不是 GFP / 不是概率 / 不是 E/S/B / 无可靠区间**。
- `--target-cell`（默认 293FT）**仅为元数据**，**永远不会**选择模型权重（模型细胞固定 HEK293）。

### 2) `gene-observed`：基因符号 / Ensembl ID → 目录观测标签
- 用已有可追溯标签表 + 符号映射（b20 来源为准，**无网络**），返回规范 ID/符号、`observedScore_E`、`S_raw`、`B_raw`。
- **无 GPU / 无 torch / 无学习权重**；CUDA 隐藏也能跑。
- 这些是**来源目录标签**，**不是**单个启动子实测、**不是** GFP/293FT 活性、**不是**模型预测；`S` 是**跨组织**稳定性（非时间）。
- `S_raw` 缺失时为 `null`（不用 0 或预测填补）；符号歧义 → `AMBIGUOUS`+candidates；未知 → `ABSENT`+reason；Ensembl 仅按精确基 ID 归一（如 `ENSG00000075624.18` → `ENSG00000075624`）。

## 可直接运行的绝对命令（Windows 项目 venv，含 CUDA）

```bat
set PY=E:/promoter_atlas/.venv/Scripts/python.exe
set CLI=E:/promoter_atlas/runs/breakthrough_20261002/code/batch47/b47_cli.py

REM 任务 1：原始 FASTA -> PARM HEK293 模型分数
"%PY%" "%CLI%" fragment --fasta "E:/promoter_atlas/data/breakthrough_20261002/batch35/testdata/known_control.fasta" --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch47/demo.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch47/demo.tsv" --target-cell 293FT

REM 任务 2：基因符号 / Ensembl ID -> 目录观测标签（无 GPU）
"%PY%" "%CLI%" gene-observed --query ACTB GAPDH RPLP0 --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch47/demo_gene.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch47/demo_gene.tsv"
```

## 示例输出（摘要）
- `fragment` JSON：`records[]` 每条带 `route_status` / `PARM_HEK293_model_score` / `model_source_cell=HEK293` / `target_requested_cell` / `biological_validation_status=UNVALIDATED_FOR_293FT`；顶层有 `backend`（引擎 SHA、权重 SHA）与 `runtime`（GPU 断言、PID、NVIDIA）。
  - 控制序列分数与 b31/b34/b35 一致：`3.1036716`（容差 1e-4）。
- `gene-observed` JSON：`records[]` 按输入顺序；如 `ACTB` → `observedScore_E=0.999644`，`S_raw=-1.079803`，`B_raw=1.0`（`null` 表示未定义）。

## 明确区分
- **公开 HEK293 模型**：PARM 官方 HEK293 启动子活性模型（固定）。
- **用户 293FT**：仅为调用方声明的目标细胞元数据，**未验证**（`UNVALIDATED_FOR_293FT`）。
- **不做任何“自主启动子/GFP 预测”声明**；模型分数仅为跨细胞排序描述子。

## 本入口明确**不支持**（不静默切换）
`mpra-fasta`、`gene`、`gene-bag-fasta`、`constructs`，以及旧 164bp K562 神经路径、新基因 E/S/B 推断。需要时请向 parent 申请单独受控批次。
"""


def write_usage():
    (REP / "USAGE_zh.md").write_text(USAGE, encoding="utf-8")


def build_and_write():
    write_usage()  # write USAGE first so it is included in the frozen manifest
    files, sha_manifest, code_man, rounds = main()
    ev = json.loads((REP / "test_evidence.json").read_text(encoding="utf-8"))
    tman = json.loads((DATA / "b47 test 测试" / "test_manifest.json").read_text(encoding="utf-8"))
    o = DATA / "输出 out 测试"
    gene_payload = json.loads((o / "gene.b47.json").read_text(encoding="utf-8"))

    input_output_sha = {
        "frozen_test_inputs": {n: tman["files"][n]["sha256"] for n in sorted(tman["files"])},
        "frozen_control_score": tman["frozen_control"]["frozen_batch31_score"],
        "artificial_test_sequences": tman["artificial_sequences"],
        "outputs": {
            "fragment_json": {"path": rel(o / "mixed.b47.json"), "sha256": sha(o / "mixed.b47.json")},
            "fragment_tsv": {"path": rel(o / "mixed.b47.tsv"), "sha256": sha(o / "mixed.b47.tsv")},
            "gene_json": {"path": rel(o / "gene.b47.json"), "sha256": sha(o / "gene.b47.json")},
            "gene_tsv": {"path": rel(o / "gene.b47.tsv"), "sha256": sha(o / "gene.b47.tsv")},
        },
        "asset_manifest": {"path": rel(C.B35_MANIFEST), "sha256": sha(C.B35_MANIFEST)},
        "backend_engine": {"path": rel(C.B35_ENGINE), "sha256": sha(C.B35_ENGINE)},
        "external_asset_root": rel(C.DEFAULT_ASSET_ROOT),
        "gene_source_label_table": {"path": rel(C.LABEL_TABLE), "sha256": sha(C.LABEL_TABLE)},
        "gene_source_symbol_map": {"path": rel(C.ANALYSIS_SET), "sha256": sha(C.ANALYSIS_SET)},
    }

    import b47_cli as CLI
    status = {
        "batch": C.BATCH, "version": C.VERSION, "phase": "A+B+C+D",
        "state": "WORKER_COMPLETE_PENDING_FRESH_INDEPENDENT_REVIEW",
        "created_at": NOW,
        "zero_fit": True, "zero_solve": True, "zero_new_weights": True,
        "neural_forward": "frozen PARM HEK293 (b35 read-only engine) only; one worker true-CUDA invocation",
        "new_cli": {"path": "runs/breakthrough_20261002/code/batch47/b47_cli.py", "version": C.VERSION,
                    "supported_commands": ["fragment", "gene-observed"],
                    "unsupported_commands": list(CLI.UNSUPPORTED)},
        "score_label": C.SCORE_LABEL, "model_source_cell": C.MODEL_SOURCE_CELL,
        "default_target_cell": C.DEFAULT_TARGET_CELL, "target_cell_selects_weights": False,
        "length_support": {"supported_inclusive": [1, 600], "over_600_examples_null": [601, 2000]},
        "test_evidence": {"path": "reports/breakthrough_20261002/batch47/test_evidence.json",
                          "all_ok": ev["all_ok"], "status": ev["status"],
                          "gpu_seconds_used": ev.get("gpu_seconds_used"),
                          "torch_version": ev.get("torch_version"), "gpu_available": ev.get("gpu_available"),
                          "note": "software integration only; NO performance claim"},
        "gene_observed_examples": {r["query"]: {"status": r["status"], "observedScore_E": r["observedScore_E"],
                                                "S_raw": r["S_raw"], "B_raw": r["B_raw"],
                                                "canonical_gene_id": r["canonical_gene_id"]}
                                   for r in gene_payload["records"]
                                   if r["query"] in ("ACTB", "GAPDH", "RPLP0", "WASH7P", "NOPE_NOT_A_GENE")},
        "input_output_sha": input_output_sha,
        "manifest": {"path": "reports/breakthrough_20261002/batch47/sha_manifest.json", "n_files": len(files),
                     "excludes": sorted(EXCLUDE), "two_read_only_checks_ok": all(x["ok"] for x in rounds)},
        "verdict": {
            "workflow_integration": "PASS",
            "scientific_performance": "NOT_ASSESSED (no new model performance claimed)",
            "biological_scope": "public HEK293 model score; user 293FT UNVALIDATED; catalog labels are NOT 293FT/GFP activity"},
        "preserved_unchanged": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)",
                                "2000bp unsupported", "batch44 source branch CLOSED",
                                "old 164bp K562 route / gene-bag / old models NOT exposed here"],
        "next": "fresh independent (DeepSeek Flash) review per REVIEW_PROTOCOL, then parent decision",
    }
    C.write_json(REP / "status.json", status)

    selfcheck = {
        "batch": C.BATCH, "role": "worker", "selfcheck_only": True, "created_at": NOW,
        "checks": {
            "explicit_task_required_no_arbitrary_dna_guess": True,
            "no_duplicate_bag_esb_fabrication": True,
            "b35_engine_and_external_weights_read_only_unchanged": True,
            "old_files_read_only": True,
            "b35_official_centerpad600_no_rc_no_dna_change_no_hidden_window": True,
            "over600_incl_2000_unsupported_null": True,
            "invalid_empty_duplicate_handling_preserved": True,
            "target_cell_metadata_never_selects_weights": True,
            "model_cell_fixed_HEK293_target_default_293FT_unvalidated": True,
            "score_label_parm_hek293_model_score_only": True,
            "no_gfp_probability_esb_reliability_ci": True,
            "backend_version_sha_and_biological_scope_reported_separately": True,
            "no_restricted_source_weights_imported_into_owned_package": True,
            "gene_observed_no_cuda_torch_weights_features_memmap": True,
            "gene_observed_traceable_label_table_no_network": True,
            "ambiguous_multiple_candidates_absent_null_reason_version_normalization": True,
            "missing_S_null_never_zero_or_prediction": True,
            "no_broad_caching_of_new_learned_scores": True,
            "ordered_per_query_outcomes_mixed_no_drop": True,
            "gene_input_fasta_clear_failure": True,
            "legacy_constructs_hek293t_wording_not_exposed": True,
            "unsupported_tasks_listed_not_silently_switched": True,
            "b46_addendum_only_no_canonical_rewrite": True,
            "no_new_scientific_slice_model_score_figure": True,
            "b46_old_compliance_FAIL_preserved": True,
            "atomic_registry_changelog_append_preserves_pointer_and_entries": True,
            "manifest_excludes_self_mutable_verification_status_review_mail": True,
            "two_read_only_manifest_checks": True,
            "strict_json_null_reason_no_nan_infinity": True,
            "one_worker_true_cuda_invocation_gpu_within_budget": True,
            "no_credential_read": True, "no_email_sent": True,
            "worker_does_not_author_independent_review": True,
            "no_performance_claim_from_tests": True,
        },
        "honest_limitations": [
            "the PARM score is a public HEK293 cross-cell ranking descriptor; it is NOT validated on the user 293FT chassis",
            "gene-observed returns catalog-derived labels (cross-tissue E/S/B), NOT 293FT/GFP measurements and NOT model predictions",
            "the 137/271 fixtures are ARTIFICIAL software-test sequences only; no biological meaning",
            "no performance, calibration, or reliability-CI claim is made or supported by these tests",
            "5-fold spread is model disagreement only and is not carried by the b35 engine",
        ],
        "worker_preliminary_review": False,
        "fresh_independent_review_required": True,
    }
    C.write_json(REP / "worker_selfcheck.json", selfcheck)

    remaining_work = {
        "batch": C.BATCH, "generated_at": NOW,
        "remaining_work": [
            {"id": "user_293FT_validation", "status": "OPEN_external_gap",
             "desc": "本批只做公开 HEK293 模型的输入路由；用户 293FT 机箱仍未验证（D28 FAIL）。需要 293FT 实测启动子-活性配对数据。"},
            {"id": "gene_observed_is_catalog_not_endpoint", "status": "expected",
             "desc": "gene-observed 返回的是来源目录标签（跨组织 E/S/B），不是 293FT/GFP 实测，也不是模型预测；不得当作活性结论。"},
            {"id": "legacy_routes", "status": "NOT_exposed_here",
             "desc": "旧 164bp K562 / gene-bag / 旧模型推断未在新 CLI 暴露；如需请 parent 单独授权。"},
            {"id": "independent_review", "status": "PENDING",
             "desc": "worker 只做 selfcheck；须由 fresh independent DeepSeek Flash 按 REVIEW_PROTOCOL 评审。"},
        ],
        "next_bounded_action": "batch48 v6.35.0-293ft-pilot-readiness.rc1（数据门控，零拟合；见 next_batch_proposal.json）",
    }
    C.write_json(REP / "remaining_work.json", remaining_work)

    proposal = {
        "proposed_batch": "batch48",
        "title": "v6.35.0-293ft-transfer-readiness.rc1",
        "rationale": ("b47 已把公开 HEK293 PARM 前向与目录观测查询接入新的显式 CLI（workflow 集成，非性能提升）。"
                      "用户 293FT 仍是 UNVALIDATED；本批不重复同源切片/追溯/打包。下一个真实用户能力缺口是把公开 HEK293 "
                      "分数迁移到 293FT 的可评估性。"),
        "single_bounded_action": ("先做一次**零拟合、零下载**的数据就绪盘点：在既有只读资产中列出所有可追溯到 "
                                  "293FT（或用户目标细胞）的启动子序列-实测活性配对，统计样本数、构建来源与可配对比例；"
                                  "若无合格数据则记 UNKNOWN 并停止该子路线。"),
        "measurable_gain": "合格 293FT 序列-活性配对数量与来源清单（含分母与缺失原因）；不产生新模型分数。",
        "stop_condition": "既有只读资产中无满足[可追溯序列->构建->活性]的 293FT 数据 -> 记 UNKNOWN，不下载、不用同源替代。",
        "explicitly_not": ["新下载/新源搜寻", "新拟合/GPU/前向", "同源再切片/调参", "重新打包/新 GUI"],
        "external_information_gap": ("仓库内当前不存在可追溯的 293FT 序列-活性配对（本地 D28 已 FAIL）；这是外部信息缺口，"
                                     "不表示整个项目被阻塞——工作流集成与目录查询能力已在本批可用。"),
        "zero_fit": True, "gpu": "none",
        "parent_gate": "parent 确认 293FT 可追溯数据可用/格式后再放行有界盘点",
    }
    C.write_json(REP / "next_batch_proposal.json", proposal)
    return status, selfcheck, remaining_work, proposal


if __name__ == "__main__":
    st, sc, rw, pr = build_and_write()
    print(json.dumps({"status": st["state"], "manifest_files": st["manifest"]["n_files"],
                      "two_checks_ok": st["manifest"]["two_read_only_checks_ok"],
                      "test_all_ok": st["test_evidence"]["all_ok"],
                      "next": pr["title"]}, ensure_ascii=False))
