#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 Chinese report generator (report.html + report.txt).

Loads existing frozen artifacts only (never refits): config, preflight, metrics,
tests, replay and attempt ledger. Distinguishes procedural compliance, scientific
evidence and acceptance. Emits UTF-8 HTML and plain-text Chinese reports.
"""
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (MODEL_OUT, REPORT_OUT, DATA_OUT, CANDIDATE_VERSION, BATCH,
                        TRACK, atomic_write_text)  # noqa: E402


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

    tm = metrics["test_metrics"]
    pri = metrics["primary_conditional_increment"]
    sec = metrics["secondary_descriptive_vs_baseline"]
    ret = metrics["retention_descriptive"]
    constb = metrics["constant_baselines_raw"]
    contrasts = pri["contrasts"]

    lines = []
    A = lines.append

    A("=" * 78)
    A("Batch11 报告（%s）" % CANDIDATE_VERSION)
    A("=" * 78)
    A("")
    A("【任务与范围】")
    A("候选版本：%s（track=%s）。唯一问题：在 batch10 确认的 1024 维位置序列特征增益上，"
      % (CANDIDATE_VERSION, TRACK))
    A("(A) 局部位置三核苷酸 bin（sym400 4x64 + up2000 4x64 = 512 维）与 (B) 全窗口四核苷酸频率")
    A("（sym400 256 + up2000 256 = 512 维）两组特征，是否在另一组特征之上（条件地）提供额外预测")
    A("信息。这是预测性消融（predictive ablation），不是因果机制识别。A 与 B 在 k 与位置聚合两")
    A("方面同时不同，不能把差异唯一归因于“位置”。")
    A("")
    A("【总体结论】")
    for key, label in [("ridge_full_minus_A", "Ridge full-A（B 对 A 的条件增量）"),
                       ("ridge_full_minus_B", "Ridge full-B（A 对 B 的条件增量）"),
                       ("hgb_full_minus_A", "HGB full-A（B 对 A 的条件增量）"),
                       ("hgb_full_minus_B", "HGB full-B（A 对 B 的条件增量）")]:
        c = contrasts[key]
        b = c["bootstrap"]
        A("  %s: delta=%.6f, 区间[%.6f, %.6f], 满足 delta>=0.01 且校正下界>0 = %s"
          % (label, c["delta"], b["lo"], b["hi"], c["meets_rule"]))
    A("接受规则：scoped 条件增量要求 delta>=0.01 且校正下界>0；四个共主比较全部展示，无事后选择。")
    A("full 模型仅作参照，不是数学上界；单独的 A/B 相对 baseline 的增益不能证明独立/可加贡献。")
    A("未检出显著条件增量不等于“等价”或“冗余”的证明。")
    A("")
    A("【冻结配置与来源 SHA 校验】")
    A("source_sha_ok：%s" % json.dumps(preflight["source_sha_ok"], ensure_ascii=False))
    A("batch10_features1024.npz SHA：%s" % preflight["source_sha"]["batch10_features1024.npz"])
    A("特征按名字校验：A=tri 索引 0..511、B=tetra 索引 512..1023，均与冻结 feature_dictionary 一致。")
    A("人口：合格 18769 = train15004/val1898/test1867，组件总数 %d；排除 178（canonical S 定义但不在特征宇宙）。"
      % preflight["eligible_split"]["total_components"])
    A("")
    A("【拟合与比较（恰四个新科学拟合）】")
    A("A 臂 = 45 基线 + 512 位置三核苷酸；B 臂 = 45 基线 + 512 全窗口四核苷酸（各 557 维）。")
    A("模型：Ridge(alpha=0.1, fit_intercept=True)；HistGradientBoostingRegressor(max_iter=200,")
    A("learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.1, early_stopping=False,")
    A("random_state=20261011)。X 用 StandardScaler 仅拟合合格 train（A/B 各自独立 scaler）；")
    A("raw S 用 train 均值/标准差标准化并在预测后逆变换回 raw 单位。full（batch10 1069 维）与")
    A("baseline（batch09 45 维）全部加载复用，绝不重训。")
    A("")
    A("测试 Spearman（vs 全精度 raw truth）：")
    for k, lab in [("full_ridge", "full_ridge"), ("A_ridge", "A_ridge"), ("B_ridge", "B_ridge"),
                   ("base_ridge", "base_ridge"), ("full_hgb", "full_hgb"), ("A_hgb", "A_hgb"),
                   ("B_hgb", "B_hgb"), ("base_hgb", "base_hgb")]:
        A("  %-10s spearman=%.6f  MAE=%.6f  MSE=%.6f  R2=%.6f"
          % (lab, tm[k]["spearman_vs_raw_truth"], tm[k]["mae"], tm[k]["mse"], tm[k]["r2"]))
    A("")
    A("共主比较（配对组件 bootstrap，n=%d，seed=%d，百分位[%.3f,%.3f]，98.75%% 双侧、四比较 Bonferroni 风格）："
      % (pri["contrasts"]["ridge_full_minus_A"]["bootstrap"]["n_resamples"], metrics["seed"],
         pri["contrasts"]["ridge_full_minus_A"]["bootstrap"]["lo_pct"],
         pri["contrasts"]["ridge_full_minus_A"]["bootstrap"]["hi_pct"]))
    for key, label in [("ridge_full_minus_A", "Ridge full-A"),
                       ("ridge_full_minus_B", "Ridge full-B"),
                       ("hgb_full_minus_A", "HGB full-A"),
                       ("hgb_full_minus_B", "HGB full-B")]:
        c = contrasts[key]; b = c["bootstrap"]
        A("  %-14s delta=%.6f, 区间[%.6f, %.6f], 下界>0=%s"
          % (label, c["delta"], b["lo"], b["hi"], b["lo"] > 0))
    A("")
    A("【次要描述性比较（不是额外的确认性发现）】")
    for key, lab in [("ridge_A_minus_base", "Ridge A-baseline"),
                     ("ridge_B_minus_base", "Ridge B-baseline"),
                     ("hgb_A_minus_base", "HGB A-baseline"),
                     ("hgb_B_minus_base", "HGB B-baseline")]:
        s = sec[key]; b = s["bootstrap"]
        A("  %-18s delta=%.6f, 区间[%.6f, %.6f]" % (lab, b["point_estimate"], b["lo"], b["hi"]))
    A("保留率（retention=(arm-baseline)/(full-baseline)，描述性，分母不稳定有保护）：")
    for k in ("ridge_A", "ridge_B", "hgb_A", "hgb_B"):
        r = ret[k]
        A("  %-8s %s" % (k, json.dumps(r, ensure_ascii=False)))
    A("常数基线（train 均值，测试集）：MAE=%.6f, MSE=%.6f, R2=%.6f"
      % (constb["train_mean"]["mae"], constb["train_mean"]["mse"], constb["train_mean"]["r2"]))
    A("（R2 均为显式计算，绝不从 rho 平方推断。）")
    A("")
    A("【验证/测试】")
    A("tests_summary：%d/%d 通过，all_pass=%s（见 tests_summary.json 明细）。"
      % (tests["n_pass"], tests["n_total"], tests["all_pass"]))
    A("replay：batch11 A/B 重放匹配=%s；full(batch10) 旧预测精确匹配=%s；baseline(batch09) 旧预测精确匹配=%s。"
      % (json.dumps(replay["batch11_replay_match"]),
         json.dumps(replay["full_batch10_exact_old_prediction_match"]),
         json.dumps(replay["baseline_batch09_exact_old_prediction_match"])))
    A("非覆盖守护：b11_run --attempt 1 对已存在拟合工件应 REFUSE（测试见 tests_summary）。")
    A("")
    A("【范围与局限（不得越界）】")
    for s in config["scope_limits"]:
        A("- %s" % s)
    A("")
    A("【后端 provenance（配置 vs 观测，无秘密）】")
    A("- 控制器 deepseek_launcher._build_child_env 配置 ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic，")
    A("  ANTHROPIC_MODEL=deepseek-v4-pro。CLI 的 unrecognized_model 警告（stderr）仅表示该模型名不在本地")
    A("  注册表，不构成 provider/model 回退的证据；请求经 ANTHROPIC_BASE_URL 路由。")
    A("")
    A("【修复/偏差披露（attempt ledger）】")
    ledger = DATA_OUT / "attempts.jsonl"
    repairs = []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "repair_test":
                repairs.append(rec)
    if repairs:
        for r in repairs:
            A("  - [%s] %s: %s" % (r["ts_local"], r["target"], r["reason"]))
            A("    影响：%s" % r["impact"])
    else:
        A("  （无修复记录）")
    A("生成时间（本地）：%s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    txt = "\n".join(lines) + "\n"

    html_body = [html.escape(ln) for ln in lines]
    html_doc = ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
                "<title>Batch11 report %s</title></head><body><pre>" % CANDIDATE_VERSION
                + "\n".join(html_body) + "</pre></body></html>\n")

    atomic_write_text(REPORT_OUT / "report.txt", txt)
    atomic_write_text(REPORT_OUT / "report.html", html_doc)
    print("wrote report.txt / report.html")


if __name__ == "__main__":
    main()

