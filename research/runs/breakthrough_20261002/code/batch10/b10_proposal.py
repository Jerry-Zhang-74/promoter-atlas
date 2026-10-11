#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 -> BATCH11 proposal writer (next_batch_proposal.json + BATCH11_DRAFT.txt).

Chooses ONE distinct next step (mechanism question): an ablation that decomposes the
batch10 positional-sequence gain into its two frozen components -- the 4 local-position
trinucleotide bins (512 features) vs the full-window tetranucleotide frequencies
(512 features) -- for each of Ridge and HGB. Read-only reuse of frozen features/models;
no new target, no feature pile-up, no equivalent rescaling, no noise-ceiling claim.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b10_common import (REPORT_OUT, CANDIDATE_VERSION, BATCH, atomic_write_json,
                        atomic_write_text)  # noqa: E402


def main():
    draft = (
        "==============================================================================\n"
        "BATCH11 DRAFT：定位 batch10 增益来源的机制消融（组件归因）\n"
        "==============================================================================\n"
        "候选标识建议：v6.12.0-sfeatablate.rc1（track=sfeatablate）。这是一个机制/诊断批次，\n"
        "不是新目标、不是新特征堆叠、不做任何等价重标定。只读复用 batch10 冻结的 1024 维特征、\n"
        "batch09 冻结基线模型，以及同一 frozen raw S 目标与切分。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "1. 唯一问题（ONE question，与 batch10 明显不同）\n"
        "------------------------------------------------------------------------------\n"
        "batch10 已确认：在 batch09 45 维组成/计数基线上，新增 1024 维“位置序列特征”带来\n"
        "显著测试 Spearman 增益（Ridge +0.0765, CI [0.0359,0.1197]；HGB +0.0655,\n"
        "CI [0.0326,0.1015]）。本批次的唯一问题：这 1024 维增益来自哪个成分？\n"
        "  (A) 4 个局部位置三核苷酸 bin（sym400 4x64 + up2000 4x64 = 512 维，捕捉“在哪里\n"
        "      出现什么 3-mer”），还是\n"
        "  (B) 全窗口四核苷酸频率（sym400 256 + up2000 256 = 512 维，捕捉“整体 4-mer 内容”），\n"
        "  或两者各自独立贡献？\n"
        "回答方式：分别拟合 Ridge / HGB 于 45+512(A) 与 45+512(B)，与 batch09 45 维基线做\n"
        "配对组件 bootstrap 共主比较，报告两成分各自的点估计与 CI，并以 batch10 全量 1069 维\n"
        "作为上界参照。不改目标、不改切分、不改超参、不加特征、不调参。\n"
        "为什么是这一步：batch10 给出的是“总和”证据；机制上需要把“位置”与“4-mer 长度”解耦，\n"
        "才能判断 S 预测增益来自局部位置上下文还是更长窗口序列内容，这是下一个最小、最直接、\n"
        "无需新数据的科学问题。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "2. 证据支撑（只读复用，无新拟合的基线）\n"
        "------------------------------------------------------------------------------\n"
        "- batch09 冻结：45 维 Ridge-raw 测试 Spearman 0.3701、HGB-raw 0.4089。\n"
        "- batch10 冻结：1069 维 Ridge-aug 0.4466、HGB-aug 0.4744；增益点估计/CI 如上；\n"
        "  features1024.npz、models.joblib、X_scaler.joblib、transforms.json、predictions.tsv\n"
        "  全部冻结可重放（replay 匹配已确认）。\n"
        "- 组件可精确拆分：特征字典顺序保证 0..255 为 sym400 tri 4bin、256..511 为 up2000\n"
        "  tri 4bin、512..767 为 sym400 tetra、768..1023 为 up2000 tetra，可直接切片构造\n"
        "  A/B 两臂。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "3. 冻结设计（草案要点）\n"
        "------------------------------------------------------------------------------\n"
        "- 目标：连续 raw S（frozen，沿用 batch10 transforms；不再跑 raw/rank 双目标）。\n"
        "- 表示：A 臂 = 45 基线 + 512 三核苷酸位置特征；B 臂 = 45 基线 + 512 四核苷酸全窗口特征。\n"
        "  每臂仍用 train-only StandardScaler；模型超参与 batch10 完全一致：\n"
        "  Ridge(alpha=0.1)、HGB(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,\n"
        "  l2_regularization=0.1, early_stopping=False, random_state=20261011)。\n"
        "- 共主比较：组件臂减 batch09 基线（测试 Spearman），配对组件 bootstrap n=2000、\n"
        "  seed=20261011、97.5% 双侧百分位 [1.25, 98.75]；报告 A 臂与 B 臂各自点估计/CI；\n"
        "  batch10 全量臂作为上界参照。\n"
        "- 结论判据（机制归因，不是提升推广）：报告哪个成分校正下界>0、是否两成分都有独立\n"
        "  贡献、以及 A+B 相对各自单独成分的关系；不做“该成分是唯一机制”的强因果声明。\n"
        "- 资源：CPU<=4 核、无 GPU、无新外部数据、预算<=60 分钟。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "4. 明确不选（保留为后续独立方向，避免本批次混淆）\n"
        "------------------------------------------------------------------------------\n"
        "- 不选“rank S 重跑 / 目标重标定”：协议已冻结 raw S；且属于等价重标定，不回答机制。\n"
        "- 不选“k=5 / motif 库 / 更多特征”：在尚未知道现有增益来源前堆特征，缺少证据方向。\n"
        "- 不选“prospective/causal/HEK293T 验证”：无此类数据，且超出本仓库条件性回顾范围。\n"
        "- 不做任何“噪声天花板”声明：本仓库没有测量的生物学重复噪声数据。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "5. 交付物\n"
        "------------------------------------------------------------------------------\n"
        "组件臂预测、配对 bootstrap 结果、归因结论中文报告（HTML+TXT）、组件模型与变换、\n"
        "next_batch_proposal.json + BATCH12_DRAFT.txt、SHA 注册表与不可变尝试台账。\n"
        "==============================================================================\n"
    )

    proposal = {
        "from_batch": BATCH,
        "from_candidate_version": CANDIDATE_VERSION,
        "next_batch": "batch11",
        "next_candidate_version_suggestion": "v6.12.0-sfeatablate.rc1",
        "track_suggestion": "sfeatablate",
        "chosen_question": (
            "Which component of the 1024 added positional-sequence features carries the "
            "batch10 S-prediction gain: (A) the 4 local-position trinucleotide bins "
            "(512 features: sym400 4x64 + up2000 4x64) vs (B) the full-window tetranucleotide "
            "frequencies (512 features: sym400 256 + up2000 256), for each of Ridge and HGB?"
        ),
        "question_type": "mechanism (component attribution / ablation)",
        "why_distinct": (
            "batch10 established the aggregate gain (Ridge +0.0765 CI [0.0359,0.1197]; "
            "HGB +0.0655 CI [0.0326,0.1015]); this decomposes it to decouple local-position "
            "context from longer-window 4-mer content, using only frozen features/models and "
            "no new target or data."
        ),
        "acceptance_criterion": (
            "mechanism attribution (NOT a promotion threshold): for each component arm, "
            "paired component bootstrap of (component-model minus batch09 baseline) test "
            "Spearman, n=2000, seed=20261011, 97.5% two-sided percentile [1.25,98.75]; "
            "report point/CI for A and B arms with batch10 full-1069 as the upper reference; "
            "answer which component(s) have adjusted lower bound > 0."
        ),
        "resources": {"cpu_threads_max": 4, "gpu": "zero", "new_training": "two fits per model "
                     "(tri-only, tetra-only); no new target/scaler/tuning",
                     "budget_minutes_target": 60},
        "explicitly_not_chosen": [
            "rank S re-run / target rescaling (frozen raw S only; equivalent rescaling, not mechanism)",
            "k=5 / motif-library / extra features (pile-up before the gain source is known)",
            "prospective/causal/HEK293T validation (no such data; out of scope)",
            "any noise-ceiling claim (no measured biological replicate noise data)",
        ],
        "parent_decision_required": True,
    }

    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", proposal)
    atomic_write_text(REPORT_OUT / "BATCH11_DRAFT.txt", draft)
    print("wrote next_batch_proposal.json / BATCH11_DRAFT.txt")


if __name__ == "__main__":
    main()
