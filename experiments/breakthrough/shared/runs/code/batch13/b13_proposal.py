#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 -> BATCH14 提案书写器 (next_batch_proposal.json + BATCH14_DRAFT.txt)。

选择一个唯一、具体、与 batch13 明显不同的下一步问题：近同源性泄漏敏感性（单一、
预先声明的重分块实验）。明确排除“基因 S 与片段活性的族内常数相关”与“无限机械扫描”。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import (REPORT_OUT, CANDIDATE_VERSION, BATCH, atomic_write_json,
                        atomic_write_text)  # noqa: E402


def main():
    draft = (
        "==============================================================================\n"
        "BATCH14 草案：近同源性泄漏敏感性（单一、预先声明的重分块实验）\n"
        "==============================================================================\n"
        "候选标识建议：v6.15.0-nhblock.rc1（track=nhblock）。这是 batch13 稳健性结论的\n"
        "一个直接、具体的泄漏敏感性诊断，不是新特征/新目标/接口批次；不做任何超参/特征/目标搜索。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "1. 唯一问题（ONE question，与 batch13 明显不同）\n"
        "------------------------------------------------------------------------------\n"
        "在预先声明的“近同源性重分块”下，batch13 观察到的 pooled OOF 全-基线 Spearman 增益\n"
        "（0.0539，95% CI [0.0447,0.0628]）是否仍然保持（仍 delta>=0.03 且 95% 下界>0 且\n"
        ">=4/5 折为正）？\n"
        "动机：batch05 的 split 只保证“精确/RC + 已检测近同源”不分折；协议已明确“精确/RC\n"
        "组件分组对近同源性仍不完整”。若残留在不同折之间的近同源启动子造成训练/held-out\n"
        "泄漏，batch13 的交叉拟合增益可能被高估。本批次量化这一具体通道的敏感性。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "2. 证据支撑（只读复用，无新拟合前先冻结设计）\n"
        "------------------------------------------------------------------------------\n"
        "- batch13：18769 合格基因 / 18400 组件；五折交叉拟合 pooled delta=0.0539，\n"
        "  95% CI [0.0447,0.0628]，5/5 折为正；HGB base45/full1069 配置已冻结。\n"
        "- batch05：near_identity_pairs.tsv（>=90% identity 且 >=90% coverage，400bp，\n"
        "  双向，gapless 对角比对确认）约 206 万行启动子对；split_plan.json 记录检测方法。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "3. 冻结设计（草案要点）\n"
        "------------------------------------------------------------------------------\n"
        "- 基因级近同源图 G：节点=18769 合格基因；边=(g_a,g_b) 当且仅当 near_identity_pairs\n"
        "  中存在跨基因对（gene_a != gene_b，两侧基因均合格，忽略同基因自身对）。\n"
        "- 连通分量=“近同源超组件”；跨基因近同源边把两个基因强制分入同一超组件。\n"
        "- 五折分配：对超组件用与 batch13 相同的确定性算法（salt=batch14|20261014|，按\n"
        "  合格基因数递减、再按 hash、再按 ID，贪心最小折，平手最小折号）。标签不参与。\n"
        "- 拟合：恰好 10 个 HGB（每折 base45/full1069），配置/seed 与 batch13 完全一致；\n"
        "  每折 X 缩放与目标均值/std 仅在其余四折拟合；两种表示独立缩放、配对折内共享目标。\n"
        "- 主对比：pooled OOF 全-基线 Spearman 差值 + 配对超组件 bootstrap（2000，\n"
        "  seed=20261014，95% [2.5,97.5]）。接受判据同 batch13：delta>=0.03 且下界>0 且\n"
        "  >=4/5 折为正。\n"
        "- 输出：超组件清单与规模分布、五折 manifest（拟合前冻结）、10 次拟合、per-fold 与\n"
        "  pooled 指标、bootstrap、与 batch13 的诚实描述性对照。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "4. 诚实边界（不得越界）\n"
        "------------------------------------------------------------------------------\n"
        "- 这是单一近同源性重分块的回顾性敏感性，不是“同源性无关”证明；近同源检测本身有界\n"
        "  （minhash + gapless 比对 + >=90%/90% 阈值），残留在阈值外/未检索到的近同源仍可能存在。\n"
        "- 若 delta 保持：支持“增益对已检测近同源分块不敏感”；若 delta 明显下降：量化该通道\n"
        "  的泄漏膨胀，batch13 结论需相应降级。两种结果都诚实报告，不做选择/提升。\n"
        "- 不做构建活性/HEK293T/时间稳定性/因果声明；不提供新序列预测接口；主模型指针不变。\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        "5. 交付物与资源\n"
        "------------------------------------------------------------------------------\n"
        "中文报告（HTML+TXT）、config/超组件与五折 manifest/SHA、全 OOF 预测、per-fold 与\n"
        "pooled 指标/bootstrap、10 个模型/scaler 与重放、attempts ledger、status、测试与\n"
        "独立评审、next_batch_proposal.json + BATCH15_DRAFT.txt。\n"
        "资源：CPU<=4、无 GPU、恰好 10 次拟合、数值预算<=60 分钟、无搜索。\n"
        "==============================================================================\n"
    )

    proposal = {
        "from_batch": BATCH,
        "from_candidate_version": CANDIDATE_VERSION,
        "next_batch": "batch14",
        "next_candidate_version_suggestion": "v6.15.0-nhblock.rc1",
        "track_suggestion": "nhblock",
        "chosen_question": (
            "Under a predeclared near-homology re-blocking (single experiment), does the "
            "batch13 pooled OOF full-minus-base Spearman gain (0.0539, 95% CI "
            "[0.0447,0.0628]) survive (still delta>=0.03, 95% lower>0, >=4/5 positive folds)?"
        ),
        "question_type": "near-homology leakage sensitivity (retrospective robustness)",
        "why_distinct": (
            "batch13 showed the 45->1069 gain survives component cross-fitting, but the "
            "batch05 exact/RC component grouping is incomplete for near homology. This "
            "batch quantifies the sensitivity of that gain to a specific, predeclared "
            "near-homology re-blocking channel, using the existing near_identity_pairs.tsv."
        ),
        "exact_grouping": {
            "nodes": "18769 eligible genes",
            "edges": "cross-gene pairs (gene_a != gene_b) in batch05 near_identity_pairs.tsv "
                     "with both genes eligible (>=90% identity over >=90% of 400bp, either orientation)",
            "ignore_self_pairs": True,
            "super_components": "connected components of the gene-level near-identity graph",
            "fold_algorithm": "same as batch13 (salt=batch14|20261014|, decreasing size then hash then id, greedy smallest fold)",
        },
        "comparisons_predeclared": [
            "pooled OOF full-minus-base Spearman delta under near-homology re-blocking",
            "paired super-component bootstrap 2000 seed=20261014 95% [2.5,97.5]",
            "per-fold Spearman deltas (all 5 shown)",
            "descriptive comparison to batch13 pooled delta (0.0539) and CI",
        ],
        "acceptance_criterion": (
            "robustness retained iff pooled delta >= 0.03 AND 95% lower > 0 AND "
            "positive fold Spearman delta in >= 4 of 5 folds; both outcomes reported honestly"
        ),
        "model_config": {
            "model": "HistGradientBoostingRegressor",
            "max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15,
            "l2_regularization": 0.1, "early_stopping": False, "random_state": 20261011,
            "n_fits": 10,
            "arms": "base45 vs full1069",
        },
        "explicitly_not_claimed": [
            "not a homology-free proof (detection bounded at 90%/90% gapless)",
            "no construct activity / HEK293T / temporal stability / causal claim",
            "no new sequence prediction interface",
            "no indefinite mechanical sweeps",
            "no constant gene-S within-fragment-family correlation (rejected as undefined)",
        ],
        "resources": {
            "cpu_threads_max": 4,
            "gpu": "zero",
            "n_fits": 10,
            "budget_minutes_target": 60,
            "data": "existing near_identity_pairs.tsv + batch13 fold/eligible inputs only",
        },
        "parent_decision_required": True,
    }

    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", proposal)
    atomic_write_text(REPORT_OUT / "BATCH14_DRAFT.txt", draft)
    print("wrote next_batch_proposal.json / BATCH14_DRAFT.txt")


if __name__ == "__main__":
    main()
