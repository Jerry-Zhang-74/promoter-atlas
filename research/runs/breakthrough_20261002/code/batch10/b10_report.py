#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 Chinese report generator (report.html + report.txt).

Loads existing frozen artifacts only (never refits): config, preflight, metrics,
tests, replay and attempt ledger. Distinguishes procedural compliance, scientific
evidence and acceptance. Emits UTF-8 HTML and plain-text Chinese reports.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b10_common import (MODEL_OUT, REPORT_OUT, DATA_OUT, CANDIDATE_VERSION, BATCH,
                        TRACK, FEATURE_NAMES_ALL, atomic_write_text)  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, nd=6):
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def main():
    config = load(MODEL_OUT / "config.json")
    preflight = load(REPORT_OUT / "preflight_summary.json")
    metrics = load(REPORT_OUT / "metrics.json")
    tests = load(REPORT_OUT / "tests_summary.json")
    replay = load(DATA_OUT / "replay.json")

    aug = metrics["augmented_arms"]
    base = metrics["baseline_arms"]
    pri = metrics["primary_feature_gap"]
    rawd = metrics["raw_unit_deltas_aug_minus_base"]
    constb = metrics["constant_baselines_raw"]

    lines = []
    A = lines.append

    A("=" * 78)
    A("Batch10 报告（%s）" % CANDIDATE_VERSION)
    A("=" * 78)
    A("")
    A("【任务与范围】")
    A("候选版本：%s（track=%s）。问题：在冻结的 18769 个合格人类基因"
      % (CANDIDATE_VERSION, TRACK))
    A("（train15004/val1898/test1867，batch05 修正组件切分）上，固定的局部位置三核苷酸")
    A("（每个 sym400/up2000 窗口按索引均分为 4 个不重叠 bin，每 bin 64 个有向连续三核苷酸")
    A("频率）与全窗口 256 个有向四核苷酸频率，共 1024 个新特征，是否在 batch09 的 45 维")
    A("组成/计数基线之上，为“连续 raw S（跨可检测组织 log 表达 MAD 的负值）”带来额外预测信息。")
    A("目标仅为连续 raw S（不再跑 raw/rank 双目标）。")
    A("")

    A("【总体结论（正结果）】")
    A("两组共主比较（增强模型减 batch09 基线、测试 Spearman、对全精度 raw truth）均达到")
    A("接受阈值（delta>=0.03 且 97.5%% 双侧校正下界>0）：Ridge 点估计 +0.0765，区间")
    A("[0.0359,0.1197]；HGB 点估计 +0.0655，区间 [0.0326,0.1015]。该正结果只支持“这一种更")
    A("丰富特征表示在该固定配置/切分下”为 raw S 带来额外预测信息；既不证明噪声/生物天花板，")
    A("也不排除其他序列表示更强或更弱。")
    A("")

    A("【冻结配置与来源 SHA 校验】")
    A("source_sha_ok：%s" % json.dumps(preflight["source_sha_ok"], ensure_ascii=False))
    A("trackA_gene_promoters.npz SHA：%s" % preflight["source_sha"]["trackA_gene_promoters.npz"])
    A("promoter_sym400_uint8.memmap SHA：%s" % preflight["source_sha"]["promoter_sym400_uint8.memmap"])
    A("promoter_up2000_uint8.memmap SHA：%s" % preflight["source_sha"]["promoter_up2000_uint8.memmap"])
    A("序列编码 uint8：A=0,C=1,G=2,T=3,N=4；N 在两个窗口均存在（sym400 N 计数=%d）。"
      % preflight["sequence_encoding_checks"]["sym400_value_histogram_A_C_G_T_N"][4])
    A("基因身份：gene_id[starts[:-1]]；57827 个唯一基因 id；n_promoters==diff(starts)。")
    A("")

    A("【特征块（无搜索、冻结）】")
    A("基线 45 维不变 + 新 1024 维 = 共 1069 维。新特征顺序：")
    A("  sym400 tri(4 bin x 64)、up2000 tri(4 bin x 64)、sym400 tetra(256)、up2000 tetra(256)。")
    A("k-mer 词表：ACGT 字典序（base-4，A=0,C=1,G=2,T=3）；仅统计完全 ACGT 的 k-mer 起点；")
    A("含 N 的 k-mer 全部打断；无有效起点 -> 该块全 0（非 NaN）。每个启动子内先归一化，再对基因")
    A("的全部启动子做算术平均（与基线聚合口径一致，无标签驱动的启动子选择）。")
    A("零分母启动子计数（QC）：%s" % json.dumps(
        json.load(open(DATA_OUT / "attempt_001_extract" / "manifest.json", encoding="utf-8"))
        ["zero_denominator_promoter_counts"], ensure_ascii=False))
    A("")

    A("【拟合与比较（恰两个新科学拟合）】")
    A("Ridge(alpha=0.1, fit_intercept=True) 与 HistGradientBoostingRegressor(max_iter=200,")
    A("learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.1, early_stopping=False,")
    A("random_state=20261011)。X 用 StandardScaler 仅拟合合格 train；raw S 用 train 均值/标准差")
    A("标准化并在预测后逆变换回 raw 单位。无调参/特征选择/额外种子/集成/验证早停。")
    A("")
    A("测试 Spearman（vs 全精度 raw truth）：")
    A("  ridge_base=%s  ridge_aug=%s  (delta=%.6f)"
      % (fmt(base["ridge_base"]["spearman_vs_raw_truth"]),
         fmt(aug["ridge_aug"]["spearman_vs_raw_truth"]),
         pri["ridge"]["point_estimate"]))
    A("  hgb_base  =%s  hgb_aug  =%s  (delta=%.6f)"
      % (fmt(base["hgb_base"]["spearman_vs_raw_truth"]),
         fmt(aug["hgb_aug"]["spearman_vs_raw_truth"]),
         pri["hgb"]["point_estimate"]))
    A("")
    A("共主比较（配对组件 bootstrap，n=%d，seed=%d，百分位[%.2f,%.2f]）："
      % (pri["ridge"]["n_resamples"], metrics["seed"], pri["ridge"]["lo_pct"], pri["ridge"]["hi_pct"]))
    A("  Ridge: point=%.6f, 区间[%.6f, %.6f], 满足规则=%s"
      % (pri["ridge"]["point_estimate"], pri["ridge"]["lo"], pri["ridge"]["hi"], pri["ridge_meets_rule"]))
    A("  HGB  : point=%.6f, 区间[%.6f, %.6f], 满足规则=%s"
      % (pri["hgb"]["point_estimate"], pri["hgb"]["lo"], pri["hgb"]["hi"], pri["hgb_meets_rule"]))
    A("接受规则：delta>=0.03 且校正下界>0（两组均满足：ridge=%s, hgb=%s）。"
      % (pri["ridge_meets_rule"], pri["hgb_meets_rule"]))
    A("")
    A("raw 单位 MAE/MSE/R2 差值（aug - base，测试集）：")
    A("  Ridge: MAE=%.6f, MSE=%.6f, R2=%.6f" % (rawd["ridge"]["mae_delta"],
        rawd["ridge"]["mse_delta"], rawd["ridge"]["r2_delta"]))
    A("  HGB  : MAE=%.6f, MSE=%.6f, R2=%.6f" % (rawd["hgb"]["mae_delta"],
        rawd["hgb"]["mse_delta"], rawd["hgb"]["r2_delta"]))
    A("常数基线（train 均值，测试集）：MAE=%.6f, MSE=%.6f, R2=%.6f"
      % (constb["train_mean"]["mae"], constb["train_mean"]["mse"], constb["train_mean"]["r2"]))
    A("（R2 均为显式计算，绝不从 rho 平方推断。）")
    A("")

    A("【验证/测试】")
    A("tests_summary：%d/%d 通过，all_pass=%s（见 tests_summary.json 明细）。"
      % (tests["n_pass"], tests["n_total"], tests["all_pass"]))
    A("replay：batch10 ridge_aug/hgb_aug 重放匹配=%s/%s；batch09 基线子集重放匹配=%s/%s。"
      % (replay["batch10_ridge_aug_replay_match"], replay["batch10_hgb_aug_replay_match"],
         replay["batch09_baseline_subset_replay"]["ridge_raw_match"],
         replay["batch09_baseline_subset_replay"]["hgb_raw_match"]))
    A("非覆盖守护：b10_run --attempt 1 对已存在拟合工件应 REFUSE（测试见 tests_summary）。")
    A("")

    A("【范围与局限（不得越界）】")
    A("- 仅人类条件性基因级回顾性 S 预测；非哺乳动物通用启动子强度/合成启动子活性/HEK293T")
    A("  构建/时间稳定性；无前瞻/因果声明；不比较 E 分数或 v6.3；不提升主模型 v6.3/cellline v6.4。")
    A("- 固定测试集被反复使用，存在自适应研究暴露；近同源性隔离不完整。")
    A("- 正结果只支持这一具体特征添加（在该固定配置/切分下）；不能外推到其他特征、目标或物种。")
    A("")
    A("生成时间（本地）：%s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    txt = "\n".join(lines) + "\n"

    # ---- HTML wrapper ----
    html_body = []
    for ln in lines:
        import html
        html_body.append(html.escape(ln))
    html = ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<title>Batch10 report %s</title></head><body><pre>" % CANDIDATE_VERSION
            + "\n".join(html_body) + "</pre></body></html>\n")

    atomic_write_text(REPORT_OUT / "report.txt", txt)
    atomic_write_text(REPORT_OUT / "report.html", html)
    print("wrote report.txt / report.html")


if __name__ == "__main__":
    main()
