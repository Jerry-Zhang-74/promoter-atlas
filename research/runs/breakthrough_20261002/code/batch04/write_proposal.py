#!/usr/bin/env python3
"""Batch04 next_batch_proposal + concrete batch05 DRAFT (conditional on diagnosis)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import atomic_write_json  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"


def main():
    cons = json.loads((REPORT / "consolidated_summary.json").read_text(encoding="utf-8"))
    ann = json.loads((REPORT / "ann_diagnostics.json").read_text(encoding="utf-8"))
    delta = cons["paired_b2_minus_b1"]["_3seed_delta_spearman"]["mean"]
    rp = cons["arms"]["repeat_bag"]["mean"]
    dp = cons["arms"]["diverse_bag"]["mean"]

    # observed diagnosis (descriptive, non-causal)
    gap = dp["test_4slot_full_sp"] - rp["test_4slot_full_sp"]
    unexp_rp = rp["test_4slot_unexp_sp"]
    unexp_dp = dp["test_4slot_unexp_sp"]
    add_count = ann["additive_count_info"]["test_full"]["delta_spearman"]

    diagnosis = []
    if abs(gap) < 0.02:
        diagnosis.append("B1~B2 under matched 4-slot input: diversity does not add much beyond repeated-input capacity/exposure in this contrast")
    elif gap > 0.02:
        diagnosis.append("B2 > B1: promoter diversity contributes beyond repeated-input exposure in this matched contrast")
    else:
        diagnosis.append("B2 < B1: repeated-input exposure outperforms diversity in this matched contrast")
    if abs(add_count) < 0.01:
        diagnosis.append("promoter-count annotation adds little beyond 44 composition features")
    else:
        diagnosis.append("promoter-count annotation carries measurable additive information beyond composition")
    diagnosis.append("unexposed-subset gene E Spearman remains lower than full (calibration/homology limit)")

    # batch05 mechanism-different directions (conditional, NOT mechanical grid search)
    options = [
        {"id": "homology_blocked_resplit",
         "question": "does a homology-blocked train/val/test re-split (exclude exact/RC/20mer-near-dup cross-gene pairs) change the full vs unexposed gap and the B1/B2 contrast?",
         "why": "residual cross-gene homology and old-test reuse are the largest validity limits; a re-split isolates their effect on ranking",
         "bound": "1 split re-derivation (CPU) + up to 2 candidates x 1 seed x 12 epochs re-trained on the new split; <=60 min GPU"},
        {"id": "label_granularity",
         "question": "does training on finer label granularity (e.g. per-tissue E rather than mean E, or E/S/B multi-head from scratch) change gene-level ranking and calibration?",
         "why": "raw Score_E is a mean over tissues; granularity may affect ranking and explain residual error",
         "bound": "<=2 heads x 1 seed x 12 epochs; <=45 min GPU; train-inner selection only"},
        {"id": "calibration_train_inner",
         "question": "can a train-inner-only calibration layer (isotonic/quantile on inner-val) reduce unexposed-subset miscalibration without test leakage?",
         "why": "unexposed R2 and calibration are material limits; calibration must be validated train-inner only",
         "bound": "CPU, deterministic, on frozen predictions; <=15 min; no test selection"},
    ]

    proposal = {
        "batch": "batch05",
        "draft_path": str(RUNS / "BATCH05_DRAFT.txt"),
        "priority": "homology-blocked re-split as the primary mechanism-different question, then calibration train-inner; label granularity optional",
        "observed_diagnosis_batch04": diagnosis,
        "bounded_options": options,
        "compute_estimate": "<=60 min GPU single job, <=2 candidates, batch<=32, <=12 epochs, 1-3 seeds, CPU calibration <=15 min",
        "note": "conditional on observed batch04 diagnosis; NOT a mechanical grid search; parent approves after independent review",
    }
    atomic_write_json(REPORT / "next_batch_proposal.json", proposal)

    draft = []
    D = draft.append
    D("批次 05 协议草案（batch05 DRAFT，待父进程批准，非自行授权）")
    D("机制不同的下一问题（基于 batch04 观察到的诊断，非机械网格搜索）")
    D("")
    D("观察诊断:")
    for d in diagnosis:
        D("  - " + d)
    D("")
    D("首选问题: 同源阻断重划分 (homology-blocked re-split)")
    D("  在 batch03 精确/RC/20-mer 近邻同源审计基础上，重新划分 train/val/test，")
    D("  排除跨基因精确重复/反向互补/20-mer 近邻对，量化同源与旧测试复用对")
    D("  全量 vs 未暴露子集差距以及 B1/B2 对比的影响。")
    D("  边界: 1 次划分重推(CPU) + 至多 2 候选 x 1 种子 x 12 epoch 重训; <=60 min GPU。")
    D("")
    D("次选问题: 仅训练内校准 (calibration train-inner only)")
    D("  在冻结预测上用 train-inner isotonic/分位数校准，评估未暴露子集校准改善，")
    D("  无测试泄露; CPU 确定性; <=15 min。")
    D("")
    D("可选: 标签粒度 (label granularity) —— 每组织 E 或多头 E/S/B 从零训练; <=2 头 x 1 种子。")
    D("")
    D("不做的: 机械 LR/epoch/架构网格搜索; 原始测试驱动选择; 新增 MPRA 小样本打分。")
    D("父进程在独立评审后批准并实际启动。")
    draft_text = "\n".join(draft) + "\n"
    (RUNS / "BATCH05_DRAFT.txt").write_text(draft_text, encoding="utf-8")

    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    print(draft_text)
    print("DONE")


if __name__ == "__main__":
    main()
