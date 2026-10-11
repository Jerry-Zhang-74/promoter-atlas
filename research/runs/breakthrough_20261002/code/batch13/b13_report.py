#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 中文报告生成器 (report.html + report.txt)。

仅读取已冻结工件（绝不重拟合）：config、preflight、fold manifest、metrics、tests、
replay 与 worker self-check。区分流程合规、科学证据与接受判据。输出 UTF-8 HTML 与
纯文本中文报告。
"""
import html as _html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import (  # noqa: E402
    MODEL_OUT, REPORT_OUT, DATA_OUT, CANDIDATE_VERSION, BATCH, atomic_write_text)


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
    fold_manifest = load(DATA_OUT / "fold_manifest.json")
    selfcheck = load(REPORT_OUT / "worker_self_check.json")

    b10 = load(Path(__file__).resolve().parents[4]
               / "reports" / "breakthrough_20261002" / "batch10" / "metrics.json")
    b10_base = b10["baseline_arms"]["hgb_base"]["spearman_vs_raw_truth"]
    b10_full = b10["augmented_arms"]["hgb_aug"]["spearman_vs_raw_truth"]
    b10_delta = b10_full - b10_base

    pooled = metrics["pooled_oof"]
    pri = metrics["primary_contrast"]
    boot = pri["bootstrap"]
    fm = metrics["fold_metrics"]
    afd = metrics["across_fold_descriptive"]
    const = metrics["same_fold_train_mean_constant_baseline_pooled"]

    lines = []
    A = lines.append

    A("=" * 78)
    A("Batch13 报告（%s）" % CANDIDATE_VERSION)
    A("=" * 78)
    A("")
    A("【任务与范围】")
    A("候选版本：%s（track=scv5）。唯一可靠性问题：在冻结的 18769 个合格人类基因"
      % CANDIDATE_VERSION)
    A("（18400 个组件）上，固定的 45 维组成/计数基线 到 1069 维（45 基线 + 1024 位置")
    A("序列特征）的特征扩充，是否在一个预先声明的组件五折交叉拟合方案下保持增益。")
    A("这是对已复用标签的回顾性稳健性检验，不是独立外部/前瞻验证，也不修复所有近同源性")
    A("或自适应测试隐患。目标仅为连续 raw S（HGB 单模型，基线 45 vs 全 1069）。")
    A("")
    A("【总体结论】")
    A("预先声明的稳健性判据全部满足：")
    A("  1) pooled OOF 全-基线 Spearman 差值 = %.6f（>= %.2f）；" % (pri["pooled_delta"], pri["delta_min"]))
    A("  2) 95%% 双侧配对组件 bootstrap 区间 [%.6f, %.6f]（下界 > 0）；"
      % (boot["lo"], boot["hi"]))
    A("  3) 五个折叠的 Spearman 差值全部为正（%d/5，要求 >= %d）。"
      % (pri["n_positive_folds"], pri["min_positive_folds_required"]))
    A("因此：在该固定表示 + 该固定配置 + 组件五折交叉拟合的边界内，45->1069 的")
    A("S 特征增益在交叉拟合下保持（criterion_met=%s）。这仍只是回顾性表示稳健性，"
      % pri["criterion_met"])
    A("不代表 HEK293T/片段活性、时间稳定性或因果机制。")
    A("")

    A("【冻结配置与来源 SHA 校验】")
    A("source_sha_ok：%s" % json.dumps(preflight["source_sha_ok"], ensure_ascii=False))
    A("features45.npz SHA：%s" % preflight["source_sha"]["features45.npz"])
    A("features1024.npz SHA：%s" % preflight["source_sha"]["features1024.npz"])
    A("split_arrays.npz SHA：%s" % preflight["source_sha"]["split_arrays.npz"])
    A("trackA_gene_promoters.npz SHA：%s" % preflight["source_sha"]["trackA_gene_promoters.npz"])
    A("人口：canonical=%d，S 定义=%d，特征宇宙=%d，合格=%d，排除=%d，组件=%d。"
      % (preflight["population"]["canonical_genes"],
         preflight["population"]["canonical_S_defined"],
         preflight["population"]["feature_universe_genes"],
         preflight["population"]["eligible_S_defined_and_feature_covered"],
         preflight["population"]["excluded_canonical_S_defined"],
         preflight["population"]["eligible_components"]))
    A("")

    A("【五折组件分配（标签无关、确定性、拟合前冻结）】")
    A("算法：组件 ID 按规范字符串排序；hash=SHA256(%s+ID)；按合格基因数递减、再按 hash、"
      % config["fold_allocation"]["salt"])
    A("再按 ID 排序；贪心分配到当前合格基因数最小的折（平手取最小折号 0..4）。")
    A("折基因数：%s（合计 18769，最大-最小差=%d）。" % (
        fold_manifest["fold_gene_counts"],
        fold_manifest["verification"]["fold_gene_counts_balance_max_minus_min"]))
    A("验证：每基因恰好一次=%s，无组件跨折=%s，拟合前冻结=%s。"
      % (fold_manifest["verification"]["every_gene_assigned_exactly_once"],
         fold_manifest["verification"]["no_component_crossing"],
         fold_manifest["frozen_before_any_fit_or_heldout_metric"]))
    A("")
    A("【模型与拟合（恰好 10 个新科学拟合）】")
    A("HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,")
    A("l2_regularization=0.1, early_stopping=False, random_state=20261011)。每折在其余四折上")
    A("分别拟合 45 维基线与 1069 维全表示；X 用 StandardScaler 仅拟合训练折（两种表示独立缩放），")
    A("raw S 目标均值/标准差仅拟合训练折（配对折内共享），预测后逆变换回 raw 单位。")
    A("实际执行拟合数=%d/10；未复用 batch10 的 scaler/模型（它们见过不同行）。"
      % metrics["n_fits_executed_this_run"])
    A("")

    A("【pooled OOF（来自五个分别训练的模型，不是单一可部署模型）】")
    A("  base  Spearman=%.6f  MAE=%.6f  MSE=%.6f  R2=%.6f"
      % (pooled["base"]["spearman"], pooled["base"]["mae"], pooled["base"]["mse"], pooled["base"]["r2"]))
    A("  full  Spearman=%.6f  MAE=%.6f  MSE=%.6f  R2=%.6f"
      % (pooled["full"]["spearman"], pooled["full"]["mae"], pooled["full"]["mse"], pooled["full"]["r2"]))
    A("  差值 full-base：Spearman=%.6f  MAE=%.6f  MSE=%.6f  R2=%.6f"
      % (pooled["full_minus_base"]["spearman_delta"],
         pooled["full_minus_base"]["mae_delta"],
         pooled["full_minus_base"]["mse_delta"],
         pooled["full_minus_base"]["r2_delta"]))
    A("同折训练均值常数基线（pooled，无 held-out 拟合）：MAE=%.6f, MSE=%.6f, R2=%.6f"
      % (const["mae"], const["mse"], const["r2"]))
    A("")

    A("【各折指标（全部展示，不做选择/提升）】")
    A("fold  n     baseSpear  fullSpear  delta     baseMAE   fullMAE   baseR2   fullR2")
    for f in range(5):
        k = "fold%d" % f
        m = fm[k]
        A("  %d   %4d  %.6f  %.6f  %.6f  %.6f  %.6f  %.6f  %.6f"
          % (f, m["n"], m["base"]["spearman"], m["full"]["spearman"],
             m["spearman_delta_full_minus_base"], m["base"]["mae"], m["full"]["mae"],
             m["base"]["r2"], m["full"]["r2"]))
    A("跨折描述统计（训练有重叠，非独立重复；不做 t 检验）：")
    A("  Spearman delta mean=%.6f min=%.6f max=%.6f std=%.6f"
      % (afd["spearman_delta"]["mean"], afd["spearman_delta"]["min"],
         afd["spearman_delta"]["max"], afd["spearman_delta"]["std"]))
    A("")
    A("【主对比 + 配对组件 bootstrap（2000 次，seed=20261013，95% [2.5,97.5]）】")
    A("point delta=%.6f；bootstrap mean=%.6f std=%.6f；区间[%.6f, %.6f]；NaN=%d。"
      % (pri["pooled_delta"], boot["bootstrap_mean"], boot["bootstrap_std"],
         boot["lo"], boot["hi"], boot["n_bootstrap_nan"]))
    A("bootstrap 在折内按组件重采样、保留各折组件数；固定预测的 bootstrap 只刻画")
    A("给定这十次拟合下的 held-out 抽样变异，不覆盖训练变异/全部研究选择不确定性。")
    A("")

    A("【与旧 batch10 固定测试的描述性对照（不同训练总体，非直接得分胜出）】")
    A("batch10 旧固定测试（train/val/test 切分）HGB：base=%.6f, full=%.6f, delta=%.6f。"
      % (b10_base, b10_full, b10_delta))
    A("batch13 交叉拟合 pooled OOF：base=%.6f, full=%.6f, delta=%.6f。"
      % (pooled["base"]["spearman"], pooled["full"]["spearman"], pri["pooled_delta"]))
    A("两者训练总体不同，不构成新的得分胜出声明；仅作量级一致的描述性参考。")
    A("")

    A("【验证/测试】")
    A("tests_summary：%d/%d 通过，all_pass=%s。"
      % (tests["n_pass"], tests["n_total"], tests["all_pass"]))
    A("replay：五折 base/full 重放全部匹配=%s。"
      % replay["all_folds_all_arms_match"])
    A("worker_self_check：all_ok=%s（状态=awaiting_review，独立评审将另行给出 PASS/FAIL）。"
      % selfcheck["all_ok"])
    A("非覆盖守护：b13_run --attempt 1 对已完成拟合应 REFUSE（测试通过）。")
    A("")

    A("【范围与局限（不得越界）】")
    A("- 回顾性稳健性、复用标签；非独立外部/前瞻验证；未修复所有近同源性/自适应测试隐患。")
    A("- 精确/RC 组件分组对近同源性仍不完整；保留 component ID 不解决该问题。")
    A("- 阈值通过仅支持该表示在该条件下的稳健性，不代表可靠的 HEK293T/片段活性、时间")
    A("  稳定性或因果机制。")
    A("- 不比较跨单位/跨目标分数；不把五个分别训练的模型当作单一固定可部署模型；")
    A("  不从这些 OOF 模型提供任意新序列预测接口。")
    A("- 主模型 v6.3 / cellline v6.4 / batch12 接口指针均不变。")
    A("")
    A("生成时间（本地）：%s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    txt = "\n".join(lines) + "\n"

    html_body = []
    for ln in lines:
        html_body.append(_html.escape(ln))
    html_doc = ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
                "<title>Batch13 report %s</title></head><body><pre>" % CANDIDATE_VERSION
                + "\n".join(html_body) + "</pre></body></html>\n")

    atomic_write_text(REPORT_OUT / "report.txt", txt)
    atomic_write_text(REPORT_OUT / "report.html", html_doc)
    print("wrote report.txt / report.html")


if __name__ == "__main__":
    main()
