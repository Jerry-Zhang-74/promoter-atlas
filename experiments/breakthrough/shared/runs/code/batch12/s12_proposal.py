#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 next-batch proposal (batch13 draft).

Writes reports/breakthrough_20261002/batch12/next_batch_proposal.json and
BATCH13_DRAFT.txt. The proposal addresses ONE concrete remaining cellline
applicability question with exact grouping/comparisons/resources (no vague
leave-one-identity-group-out, no >=10 split sweep pretending new prospective
validation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s12_common import (  # noqa: E402
    BATCH, CANDIDATE_VERSION, REPORT_OUT, atomic_write_json, atomic_write_text,
    now_local)

NEXT_BATCH = "batch13"
NEXT_VERSION = "v6.14.0-scope14.rc1"
NEXT_TRACK = "scope14"


def build_proposal():
    return {
        "from_batch": BATCH,
        "from_candidate_version": CANDIDATE_VERSION,
        "next_batch": NEXT_BATCH,
        "next_candidate_version_suggestion": NEXT_VERSION,
        "track_suggestion": NEXT_TRACK,
        "chosen_question": (
            "Characterize the cellline applicability boundary of the frozen gene-level "
            "raw-S interface using ONLY the already-verified 14 measured D28 constructs: "
            "(a) per-family gene-level Spearman between observed/predicted raw-S and the "
            "family median relative GFP for ACTB (n=6) and GAPDH (n=6) separately (no "
            "cross-family pooling); (b) within-family fragment-length trend (Spearman "
            "length vs relative GFP) for ACTB and GAPDH separately; (c) a confirmatory "
            "'out-of-scope' check that CMVfull (highest measured GFP) has NO gene-level S "
            "input. Descriptive association only; no new model, no prospective claim."),
        "question_type": "cellline applicability boundary (descriptive, existing data only)",
        "why_distinct": (
            "batch12 delivered a usable frozen gene-level S interface but explicitly does NOT "
            "predict construct activity. The single most valuable open question is the honest "
            "applicability boundary against the 14 already-measured constructs: does gene-level "
            "dispersion (S) carry ANY descriptive signal for fragment activity within a gene "
            "family, and where does it structurally break (CMV has no gene context). This is "
            "concrete and predeclared, unlike a new activity model."),
        "exact_grouping": {
            "n_constructs_total": 14,
            "families": ["ACTB", "GAPDH", "CMV"],
            "per_family_counts": {"ACTB": 6, "GAPDH": 6, "CMV": 2},
            "construct_ids": ["A1","A2","A3","A4","A5","A20",
                              "G1","G2","G3","G4","G5","G20",
                              "CMVcore","CMVfull"],
            "endpoint": "D28_second_sort_primary_diagnostic_endpoint (relative_gfp)",
            "replicate_note": "n=3 same-tube technical re-measures -> NO cross-construct significance tests",
        },
        "comparisons_predeclared": [
            "Spearman(observed_raw_S, per-family median relative GFP) within ACTB (n=6)",
            "Spearman(observed_raw_S, per-family median relative GFP) within GAPDH (n=6)",
            "Spearman(predicted_raw_S, per-family median relative GFP) within ACTB (n=6)",
            "Spearman(predicted_raw_S, per-family median relative GFP) within GAPDH (n=6)",
            "Spearman(fragment_length_bp, relative GFP) within ACTB (n=6)",
            "Spearman(fragment_length_bp, relative GFP) within GAPDH (n=6)",
            "out_of_scope check: CMVfull/CMVcore have no Ensembl gene ID -> no gene-level S input",
        ],
        "explicitly_not_claimed": [
            "no HEK293T activity model", "no new metrics", "no cross-family pooled correlation",
            "no cross-construct significance (technical replicates only)",
            "no prospective / causal / calibration claims",
        ],
        "resources": {
            "cpu_threads_max": 4, "gpu": "zero", "new_training": "none",
            "budget_minutes_target": 30,
            "data": "existing verified construct_endpoints_D28.tsv + batch12 catalog/trace only",
        },
        "parent_decision_required": True,
    }


def build_draft_text(p):
    L = []
    A = L.append
    A("=" * 78)
    A("BATCH13 DRAFT（batch12 之后的下一个独立有用问题）")
    A("=" * 78)
    A("候选标识建议：%s（track=%s）。" % (p["next_candidate_version_suggestion"], p["track_suggestion"]))
    A("")
    A("1. 唯一问题（与 batch12 明显不同）")
    A("batch12 交付了可用的冻结基因级 raw-S 推理接口，并明确该接口不预测构建体活性。")
    A("下一个最小有用方向是：仅用已核实的 14 个 D28 已测构建体，诚实刻画该接口的")
    A("细胞系适用边界——基因级散布 S 是否在“同一基因家族内”对片段活性携带任何描述性信号，")
    A("以及在结构上哪里失效（CMV 无基因上下文）。这是描述性关联，不是新活性模型。")
    A("")
    A("2. 精确分组")
    A("- 总数 14；家族 ACTB=6 / GAPDH=6 / CMV=2。")
    A("- 构建体：A1-A5,A20 / G1-G5,G20 / CMVcore,CMVfull。")
    A("- 终点：D28_second_sort_primary_diagnostic_endpoint（relative_gfp）。")
    A("- 重复口径：n=3 为同管复测，不做跨构建体显著性检验。")
    A("")
    A("3. 预声明比较（全部描述性，不新训练、不新指标）")
    A("- 家族内 Spearman(observed_raw_S, 家族中位 relative GFP)：ACTB、GAPDH 各一。")
    A("- 家族内 Spearman(predicted_raw_S, 家族中位 relative GFP)：ACTB、GAPDH 各一。")
    A("- 家族内 Spearman(片段长度, relative GFP)：ACTB、GAPDH 各一。")
    A("- 越界确认：CMVfull/CMVcore 无 Ensembl 基因 ID，故无基因级 S 输入（结构失效点）。")
    A("")
    A("4. 明确不声称")
    A("- 不建 HEK293T 活性模型；不加新指标；不做跨家族合并相关；")
    A("- 不做跨构建体显著性（技术重复）；不做前瞻/因果/校准声明。")
    A("")
    A("5. 资源：CPU<=4、无 GPU、无新训练、预算<=30 分钟、仅用已核实 construct_endpoints_D28.tsv "
      "+ batch12 catalog/trace。")
    A("生成时间（本地）：%s" % now_local())
    return "\n".join(L)


def main():
    p = build_proposal()
    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", p)
    atomic_write_text(REPORT_OUT / "BATCH13_DRAFT.txt", build_draft_text(p))
    print(json.dumps(p, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

