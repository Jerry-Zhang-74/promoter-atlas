#!/usr/bin/env python3
"""Batch05 next_batch_proposal + concrete BATCH06_DRAFT (one evidence-driven question).

Chooses ONE mechanism-different question from the batch05 observed diagnosis:
  * if the simple composition+count ridge still ties/beats B2 (within 0.03),
    the chosen question is annotation-matched residual target (does the CNN learn
    signal beyond simple composition+count features?).
  * otherwise the chosen question is target granularity (per-tissue E).
Bounded compute/configs only; the parent approves and actually launches batch06.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import atomic_write_json  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"


def main():
    cons = json.loads((REPORT / "consolidated_summary_v5.json").read_text(encoding="utf-8"))
    ann = json.loads((REPORT / "ann_diagnostics_v5.json").read_text(encoding="utf-8"))

    ab = cons["annotation_baseline_comparison"]
    gap = ab["b2_minus_best_baseline_spearman"]
    ev = cons["evidence"]
    d = cons["paired_b2_minus_b1"]["test"]["delta_spearman"]

    diagnosis = []
    diagnosis.append("冻结划分下 B2 测试Spearman=%.4f, B1=%.4f, B2-B1=%.4f (组件bootstrap CI[%.4f, %.4f])"
                     % (ab["b2_test_spearman"], ab["b1_test_spearman"], d["base"], d["ci_low"], d["ci_high"]))
    diagnosis.append("最优训练内选注释基线测试Spearman=%.4f; B2 与其差=%.4f (负值=基线占优)"
                     % (ab["best_baseline_test_spearman"], gap))
    diagnosis.append("单种子; 回顾性(复用已知数据集/靶标); 划分非因果/非前瞻; 近邻检索有界非穷尽。")

    if gap < 0.03:
        chosen = {
            "id": "annotation_matched_residual_target",
            "question": "在简单组成+计数 ridge 预测残差上训练序列 CNN，能否学到超出序列组成/启动子数的信号？",
            "why": "batch05 显示简单组成+计数 ridge 仍与 B2 并列/占优，说明当前 Score_E 排序大部分被组成与计数解释；残差目标直接分离非组成信号，是本批次最主要的混淆。",
            "bound": "复用冻结划分; ridge 残差目标(训练内定义); <=2 候选 x 1 种子 x 12 epoch; batch<=32; CPU ridge 拟合 <=15min; 总 GPU <=60min; 不做架构/超参搜索",
        }
    else:
        chosen = {
            "id": "target_granularity",
            "question": "用逐组织 E(而非均值 Score_E)训练序列模型，能否改善基因级排序与校准？",
            "why": "batch05 显示 B2 相对简单基线的边际收益大于 0.03，目标粒度(均值 vs 逐组织)是剩余误差的候选解释。",
            "bound": "逐组织 E 标签(只读); <=2 头 x 1 种子 x 12 epoch; batch<=32; 训练内选择; 总 GPU <=60min",
        }

    proposal = {
        "batch": "batch06",
        "draft_path": str(RUNS / "BATCH06_DRAFT.txt"),
        "chosen_question": chosen,
        "observed_diagnosis_batch05": diagnosis,
        "compute_estimate": "single GPU, <=2 candidates x 1 seed x 12 epochs, batch<=32, CPU<=4, CPU ridge <=15min, total GPU <=60min",
        "note": "evidence-driven single question; NOT a mechanical grid search; parent approves after independent review of batch05 and actually launches batch06",
    }
    atomic_write_json(REPORT / "next_batch_proposal.json", proposal)

    D = []
    ap = D.append
    ap("批次 06 协议草案（BATCH06 DRAFT，待父进程批准，非自行授权）")
    ap("机制不同的下一问题（基于 batch05 观察到的诊断，非机械网格搜索）")
    ap("")
    ap("观察诊断:")
    for d in diagnosis:
        ap("  - " + d)
    ap("")
    ap("首选问题: " + chosen["id"])
    ap("  " + chosen["question"])
    ap("  理由: " + chosen["why"])
    ap("  边界: " + chosen["bound"])
    ap("")
    ap("不做的: 机械 LR/epoch/架构网格搜索; 原始测试驱动选择; 新增 MPRA 小样本打分; 自行启动批次。")
    ap("父进程在独立评审后批准并实际启动 batch06。")
    text = "\n".join(D) + "\n"
    (RUNS / "BATCH06_DRAFT.txt").write_text(text, encoding="utf-8")

    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    print(text)
    print("DONE")


if __name__ == "__main__":
    main()
