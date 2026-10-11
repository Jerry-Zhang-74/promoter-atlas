#!/usr/bin/env python3
"""Batch07 -> Batch08 proposal: ONE different mechanism (Score_S target granularity)."""
import sys
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import REPORT, atomic_write_json, CANDIDATE  # noqa: E402


def main():
    now = datetime.datetime.now().astimezone().isoformat()

    proposal = {
        "batch_id": "batch08",
        "proposed_candidate_id": "v6.8.0-labelgranularity.rc1",
        "parent_batch": "batch07",
        "parent_result": "v6.8.0-covergate.rc1 NEGATIVE (90% coverage gate reduced test MAE by only 1.40%, below the 5% threshold)",
        "question": "Does the current Score_S rank_scale (ranked over ALL 58,988 genes with undefined S placed at the bottom) collapse the 18,947 S-defined genes into a compressed [0.679, 1.000] dynamic range, and does a main-subset-only rank_scale restore the full [0,1] range without changing the S ordering?",
        "question_type": "label target granularity / rank-compression diagnostic (NO new model training, NO coverage gate, NO threshold grid)",
        "why_different_from_batch07": "batch07 tested a train-only feature-coverage gate on the frozen Score_E HGB predictor; batch08 diagnoses a label-construction defect in the Score_S target (rank compression) that is independent of any predictor or coverage rule.",
        "evidence_backing": {
            "observed_Score_S_compression": {"n_defined": 18947, "min": 0.6788, "max": 1.0, "std": 0.0927, "mean": 0.8394},
            "contrast": {"Score_E": {"min": 0.0, "max": 1.0, "std": 0.2887}, "Score_B": {"min": 0.0, "max": 1.0, "std": 0.2887}},
            "source": "runs/scripts/build_candidate_e_labels.py:330 rank_scale(gene_rows, 'S_raw', 'Score_S', none_value=float('-inf'))",
            "note": "undefined S placed at -inf over 58,988 rows => S-defined genes occupy only the top 18,947 ranks => [0.679, 1.000]",
        },
        "frozen_design_summary": {
            "inputs": "read-only labels/build/gene_expression_label_table.tsv (and source E_raw/S_raw/B_raw columns already present)",
            "method": "recompute Score_S as rank_scale over main-subset rows only (n=18,947), undefined stays missing; compare dynamic range/ordering to the current Score_S",
            "outputs": "corrected Score_S column (new file, never overwrite), dynamic-range/variance before-after table, ordering-invariance check, provenance SHA",
            "no_training": True,
            "no_gpu": True,
        },
        "resource_budget": {"gpu": "0 minutes", "cpu": "<= 4 cores, <= 30 minutes", "notes": "no topup; preserve all prior artifacts; no model refit"},
        "deliverables": [
            "Chinese report HTML+text",
            "Score_S rank-compression before/after table",
            "ordering-invariance check (Spearman between current and corrected Score_S = 1.0 expected)",
            "frozen config + SHA registry",
            "next_batch_proposal.json + BATCH09_DRAFT.txt",
        ],
        "unsupported_scope": [
            "does NOT retrain any predictor",
            "does NOT claim a new model or a better Score_E predictor",
            "does NOT change main v6.3 / v6.4 pointers",
            "does NOT re-run any coverage gate or hyperparameter search",
        ],
    }
    atomic_write_json(REPORT / "next_batch_proposal.json", proposal)

    draft = []
    draft.append("=" * 78)
    draft.append("BATCH08 DRAFT（v6.8.0-labelgranularity.rc1）：修复 Score_S 目标粒度 / 排序压缩")
    draft.append("=" * 78)
    draft.append("状态：草案（待父级审批/冻结）")
    draft.append("来源：batch07 覆盖门控负结果之后，转向一个不同的、证据支撑的标签层问题。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("1. 问题（ONE question，与 batch07 不同）")
    draft.append("-" * 78)
    draft.append("当前 Score_S 的 rank_scale 在全部 58,988 个基因上做，未定义 S 的基因被置于底部")
    draft.append("（none_value=-inf）。实测 S 有定义的 18,947 个基因只占据 [0.6788, 1.0000]，")
    draft.append("std=0.0927，而 Score_E / Score_B 都铺满 [0,1]、std=0.2887。")
    draft.append("问题：把 Score_S 的 rank_scale 改为只在 main subset 内部排名，能否恢复完整 [0,1]")
    draft.append("动态范围，同时不改变 S 的排序？")
    draft.append("")
    draft.append("与 batch07 的差异：batch07 在冻结 Score_E 预测器上测试覆盖门控（结论：负）；")
    draft.append("batch08 诊断标签构建缺陷（S 排序压缩），与预测器/覆盖门控完全无关。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("2. 证据支撑（只读，无新训练）")
    draft.append("-" * 78)
    draft.append("labels/build/gene_expression_label_table.tsv 实测：")
    draft.append("  Score_S: n_defined=18947 min=0.6788 max=1.0 std=0.0927")
    draft.append("  Score_E: min=0.0 max=1.0 std=0.2887; Score_B: min=0.0 max=1.0 std=0.2887")
    draft.append("  runs/scripts/build_candidate_e_labels.py:330  rank_scale(..., 'S_raw', 'Score_S', none_value=-inf)")
    draft.append("结论：S 定义基因被压缩到 [0.679,1.0]，方差被压到 E/B 的约 1/3，这是实现细节的")
    draft.append("下游症状（历史轮次发现的 S 方差塌缩）。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("3. 冻结设计（草案要点）")
    draft.append("-" * 78)
    draft.append("- 输入：只读标签表（E_raw/S_raw/B_raw 已存在），不重建矩阵。")
    draft.append("- 方法：Score_S_corrected = rank_scale(S_raw, rows=main_subset 仅 18947)，未定义保持缺失。")
    draft.append("- 判据：")
    draft.append("    1) 动态范围恢复（min→0, max→1, std 显著提升）；")
    draft.append("    2) 排序不变（current 与 corrected 的 Spearman = 1.0，证明只是重定标而非改排序）。")
    draft.append("- 产出：新列写入新文件，绝不覆盖原标签表；SHA 登记；不训练模型。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("4. 资源与恢复")
    draft.append("-" * 78)
    draft.append("- GPU：0 分钟；CPU <= 4 核、<= 30 分钟；纯标签重定标。")
    draft.append("- 不 topup/不浪费；保留全部既有工件；不改变主 v6.3 / v6.4 指针。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("5. 交付物")
    draft.append("-" * 78)
    draft.append("中文报告 HTML+text；压缩前后对比表；排序不变性校验；冻结配置 + SHA 登记；")
    draft.append("next_batch_proposal.json + BATCH09_DRAFT.txt。")
    draft.append("")
    draft.append("-" * 78)
    draft.append("6. unsupported_scope")
    draft.append("-" * 78)
    draft.append("- 不得声称改进任何预测器或 Score_E 排序；")
    draft.append("- 不得重跑覆盖门控或任何超参/阈值扫描；")
    draft.append("- 不得改变主模型 v6.3 或细胞系 pilot v6.4 指针。")
    draft.append("=" * 78)
    (REPORT / "BATCH08_DRAFT.txt").write_text("\n".join(draft) + "\n", encoding="utf-8")

    print("next_batch_proposal.json and BATCH08_DRAFT.txt written")


if __name__ == "__main__":
    main()
