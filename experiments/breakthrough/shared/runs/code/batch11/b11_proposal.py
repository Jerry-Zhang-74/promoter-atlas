#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 successor proposal writer (next_batch_proposal.json + BATCH12_DRAFT.txt).

Proposes ONE bounded distinct useful next question supported by batch11 results,
preferring a reliability/applicability extension or a usable inference interface
with explicit input domain rather than further narrow feature tuning. Reads the
actual batch11 metrics to cite the factual contrast numbers.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (REPORT_OUT, MODEL_OUT, CANDIDATE_VERSION, BATCH,
                        atomic_write_json, atomic_write_text)  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    metrics = load(REPORT_OUT / "metrics.json")
    pri = metrics["primary_conditional_increment"]["contrasts"]
    tm = metrics["test_metrics"]

    ridge_FA = pri["ridge_full_minus_A"]["delta"]
    ridge_FB = pri["ridge_full_minus_B"]["delta"]
    hgb_FA = pri["hgb_full_minus_A"]["delta"]
    hgb_FB = pri["hgb_full_minus_B"]["delta"]

    proposal = {
        "from_batch": "batch11",
        "from_candidate_version": CANDIDATE_VERSION,
        "next_batch": "batch12",
        "next_candidate_version_suggestion": "v6.13.0-siface.rc1",
        "track_suggestion": "siface",
        "chosen_question": (
            "Build and reliability-test a frozen, usable gene-level raw-S inference "
            "interface with an explicit input domain: a single fixed predictor "
            "(frozen features + a chosen learner) exposed with documented "
            "preprocessing/eligibility, and quantify (a) prediction stability under "
            "the repeated/adaptive test-split concern via component-based repeated "
            "holdout, and (b) near-homology leakage sensitivity via leave-one-"
            "identity-group-out evaluation. This extends reliability/applicability "
            "instead of continuing narrow feature tuning."
        ),
        "question_type": "reliability / usable inference interface",
        "why_distinct": (
            "batch09-batch11 established target, feature, and ablation results "
            "(full-vs-A/B deltas: Ridge %.4f/%.4f, HGB %.4f/%.4f). Four narrow "
            "representation batches are enough; the next useful step is a usable "
            "interface with explicit input domain and quantified reliability under "
            "the known repeated-test and near-homology limitations."
        ) % (ridge_FA, ridge_FB, hgb_FA, hgb_FB),
        "explicit_input_domain": {
            "inputs": "canonical human protein-coding (or transcript-level) gene identifier that is (i) S-defined under the batch09 eligibility rule AND (ii) present in the frozen 57827-gene promoter-feature universe",
            "features": "frozen batch06 45 composition/count features + frozen batch10 1024 positional-sequence features (or the chosen single fixed subset), no on-the-fly re-extraction",
            "output": "predicted continuous raw S (cross-tissue dispersion) in raw units, with a documented not-for-HEK293T / not-for-prospective-use caveat",
            "eligibility_interface": "explicit 178-gene exclusion and non-feature-covered rejection path",
        },
        "acceptance_criterion": (
            "reliability is measured by component-level repeated holdout Spearman "
            "stability (e.g. mean and spread of test Spearman across >=10 fixed-seed "
            "component resampling splits) and a leave-one-identity-group-out leakage "
            "check; the interface is 'usable' if the frozen predictor reproduces its "
            "stored outputs bit-for-bit and documents every preprocessing step. No "
            "promotion threshold for the main v6.3/cellline v6.4 models."
        ),
        "resources": {
            "cpu_threads_max": 4,
            "gpu": "zero",
            "new_training": "bounded (fixed learner only; repeated holdout is evaluation, not tuning)",
            "budget_minutes_target": 60,
        },
        "explicitly_not_chosen": [
            "further narrow k-mer / feature-stack tuning",
            "interpreting absence of batch11 conditional increment as biological ceiling or redundancy proof",
            "prospective / causal / HEK293T / synthetic promoter claims",
            "promotion of main v6.3 or cellline v6.4",
        ],
        "parent_decision_required": True,
    }

    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", proposal)

    lines = []
    A = lines.append
    A("=" * 78)
    A("BATCH12 DRAFT（batch11 之后的下一个独立有用问题）")
    A("=" * 78)
    A("候选标识建议：v6.13.0-siface.rc1（track=siface）。")
    A("")
    A("1. 唯一问题（与 batch11 明显不同）")
    A("batch11 已量化四个条件增量：Ridge full-A=%.4f、full-B=%.4f；HGB full-A=%.4f、full-B=%.4f。"
      % (ridge_FA, ridge_FB, hgb_FA, hgb_FB))
    A("连续四个窄表示批次（batch09 目标、batch10 特征、batch11 消融）已足够。")
    A("下一个最小有用方向是：构建并做可靠性验证的冻结、可用基因级 raw S 推理接口，")
    A("显式给出输入域；并量化 (a) 在已知重复/自适应测试切分顾虑下的组件级重复留出稳定性，")
    A("(b) 近同源泄漏敏感性（按同一性组留一）。这扩展的是可靠性与可用性，而不是继续窄特征调参。")
    A("")
    A("2. 显式输入域")
    A("- 输入：满足 batch09 合格规则的 canonical 人类基因标识，且存在于冻结的 57827 基因")
    A("  启动子特征宇宙；178 个排除基因与非特征覆盖基因给出显式拒绝路径。")
    A("- 特征：冻结 batch06 45 维 + 冻结 batch10 1024 维（或选定的单一固定子集），不做")
    A("  在线重提取。")
    A("- 输出：raw 单位的连续 S 预测，附带“不用于 HEK293T / 前瞻 / 合成启动子”的声明。")
    A("")
    A("3. 判据（可靠性/可用性，非主模型提升阈值）")
    A("- 可靠性：>=10 个固定 seed 组件重采样留出下测试 Spearman 的均值与离散度；")
    A("  以及按同一性组留一的近同源泄漏检查。")
    A("- 可用性：冻结预测器能逐位复现存储输出，并文档化每个预处理步骤。")
    A("")
    A("4. 明确不选")
    A("- 不继续窄 k-mer/特征堆叠调参；不把 batch11 无显著条件增量解释为生物天花板或冗余；")
    A("  不做前瞻/因果/HEK293T/合成启动子声明；不提升主模型 v6.3/cellline v6.4。")
    A("")
    A("5. 资源：CPU<=4、无 GPU、有界新训练（固定学习器；重复留出是评估不是调参）、预算<=60 分钟。")
    A("生成时间（本地）：%s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    atomic_write_text(REPORT_OUT / "BATCH12_DRAFT.txt", "\n".join(lines) + "\n")
    print("wrote next_batch_proposal.json / BATCH12_DRAFT.txt")


if __name__ == "__main__":
    main()

