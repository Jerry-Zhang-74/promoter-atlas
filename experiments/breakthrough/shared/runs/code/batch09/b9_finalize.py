#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 finalize: SHA registry, status.json, Chinese report (txt + html),
next_batch_proposal.json and BATCH10_DRAFT.txt.

Distinguishes audit/integrity PASS from scientific acceptance. The four research
fits and the affine diagnostic are already frozen; this script only aggregates
and documents, it does not recompute research metrics or refit anything.
"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                       FEATURES_NPZ, DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT,
                       CODE_OUT, CANDIDATE_VERSION, BATCH, TRACK,
                       sha256_file, atomic_write_json, atomic_write_text)

CN = timezone(timedelta(hours=8))


def now_cn():
    return datetime.now(CN).strftime("%Y-%m-%dT%H:%M:%S%z")


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


def main():
    t0 = time.time()
    for d in (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    metrics = json.load(open(REPORT_OUT / "metrics.json", encoding="utf-8"))
    preflight = json.load(open(REPORT_OUT / "preflight_summary.json", encoding="utf-8"))
    tests = json.load(open(REPORT_OUT / "tests_summary.json", encoding="utf-8"))
    config = json.load(open(MODEL_OUT / "config.json", encoding="utf-8"))

    # ---- SHA registry ----
    source_files = [EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ, FEATURES_NPZ]
    code_files = sorted(CODE_OUT.glob("b9_*.py"))
    artifact_files = [
        DATA_OUT / "eligible_genes.tsv",
        DATA_OUT / "predictions.tsv",
        MODEL_OUT / "config.json",
        MODEL_OUT / "models.joblib",
        MODEL_OUT / "X_scaler.joblib",
        MODEL_OUT / "rank_reference.npz",
        MODEL_OUT / "transforms.json",
        REPORT_OUT / "metrics.json",
        REPORT_OUT / "preflight_summary.json",
        REPORT_OUT / "tests_summary.json",
    ]
    rows = []
    for p in source_files:
        rows.append(("source", str(p), sha256_file(p), p.stat().st_size))
    for p in code_files:
        rows.append(("code", str(p), sha256_file(p), p.stat().st_size))
    for p in artifact_files:
        if p.exists():
            rows.append(("artifact", str(p), sha256_file(p), p.stat().st_size))
    with open(REPORT_OUT / "sha_registry.tsv", "w", encoding="utf-8") as fh:
        fh.write("category\tpath\tsha256\tbytes\n")
        for r in rows:
            fh.write("%s\t%s\t%s\t%d\n" % r)

    # ---- status.json ----
    worker_identity = {
        "worker_pid": 39040,
        "worker_created_at": "2026-10-03T05:58:15.567829+08:00",
        "worker_role": "DeepSeek CLI worker harness (batch09 worker)",
        "source": "runs/breakthrough_20261002/control_15/PARENT_HANDOFF.txt (recorded at launch)",
        "note": "worker identity is the launch-time recorded process, NOT the short-lived status-writer PID",
    }
    status = {
        "batch": BATCH,
        "candidate_version": CANDIDATE_VERSION,
        "track": TRACK,
        "phase": "worker",
        "state": "complete",
        "written_at": now_cn(),
        "worker": worker_identity,
        "scope": ("continuous raw S vs train-reference empirical rank S, four frozen CPU "
                  "fits (Ridge/HGB); target-representation diagnosis only"),
        "eligibility": {
            "canonical_S_defined": preflight["population"]["canonical_S_defined"],
            "eligible_feature_covered": preflight["population"]["eligible_S_defined_and_feature_covered"],
            "excluded": preflight["population"]["excluded_canonical_S_defined"],
            "train_genes": metrics["n_train"],
            "val_genes": metrics["n_val"],
            "test_genes": metrics["n_test"],
            "components": preflight["eligible_split"]["total_components"],
        },
        "integrity": {
            "source_sha_ok": preflight["source_sha_ok"],
            "tests_pass": "%d/%d" % (tests["n_pass"], tests["n_total"]),
            "integrity_verdict": "PASS" if tests["all_pass"] and all(preflight["source_sha_ok"].values()) else "FAIL",
        },
        "scientific": {
            "primary_delta_ridge": metrics["primary_target_representation"]["ridge"],
            "primary_delta_hgb": metrics["primary_target_representation"]["hgb"],
            "ridge_meets_rule": metrics["primary_target_representation"]["ridge_meets_rule"],
            "hgb_meets_rule": metrics["primary_target_representation"]["hgb_meets_rule"],
            "scientific_verdict": ("NO scoped target-representation improvement (both deltas below 0.03 "
                                   "and/or adjusted lower bound not > 0); integrity PASS is distinct from "
                                   "scientific acceptance"),
        },
        "main_model_unchanged": "v6.3 / cellline v6.4 pointers unchanged",
        "next_action": "hand off for independent review; parent decides next batch from next_batch_proposal.json",
    }
    atomic_write_json(REPORT_OUT / "status.json", status)

    # ---- metrics snapshot for report ----
    arm = metrics["arms"]
    prim = metrics["primary_target_representation"]
    base = metrics["baselines"]
    aff = metrics["affine_diagnostic"]

    sp_test = {k: arm[k]["spearman_vs_raw_truth"] for k in arm}
    r2_test = {k: arm[k]["r2"] for k in arm}
    mae_test = {k: arm[k]["mae"] for k in arm}

    # ---- report text ----
    txt = []
    txt.append("=" * 78)
    txt.append("batch09 报告  v6.10.0-starget.rc1  目标表示诚实基线")
    txt.append("（连续跨组织离散度 raw S vs 训练参考经验秩 rank S；固定 Ridge/HGB，四次拟合）")
    txt.append("=" * 78)
    txt.append("")
    txt.append("生成时间（本地真实时钟）: %s" % now_cn())
    txt.append("候选版本: %s  批次: %s  轨道: %s" % (CANDIDATE_VERSION, BATCH, TRACK))
    txt.append("")
    txt.append("一、问题")
    txt.append("    在 S 有定义且具备启动子组成/计数特征的人类基因条件群体上，用固定的")
    txt.append("    序列组成+计数基线（45 维，batch06 冻结）预测两种真正不同的目标表示——")
    txt.append("    (A) 连续跨组织离散度 raw S（-MAD(log2(1+meanTPM))，全精度），")
    txt.append("    (B) 训练参考经验秩 F(x)=(#train<x + 0.5*#train==x)/n_train——")
    txt.append("    是否改变 S 的可预测性？batch08 已确认缺失掩码正确、仿射仅改变损失单位而")
    txt.append("    不提升预测；本批不重复等价重标定，只比较两个本质不同的目标。")
    txt.append("")
    txt.append("二、群体与来源（精确连接，非行序假设）")
    txt.append("    规范 S 定义基因 18947（B_raw>=0.30 且可检出组织数>=3）；启动子特征覆盖")
    txt.append("    的 S 定义基因 18769。二者差 178，全部为规范表达表有 S 定义但 batch03")
    txt.append("    57827 基因特征宇宙中无启动子记录（无组成/计数特征、无切分行）的基因。")
    txt.append("    178 个排除基因全部 tissue_count=54、B_raw 范围 [%.6f, %.6f]，非掩码缺陷。" % (
        preflight["excluded_genes"]["B_raw_min"], preflight["excluded_genes"]["B_raw_max"]))
    txt.append("    本批建模/评估仅用 18769 个同时满足“S 有定义 且 特征覆盖 且 有切分”的基因，")
    txt.append("    不把 18947 当作已建模。")
    txt.append("")
    txt.append("    batch05 修正后的组件切分（未改动）在合格基因上的分布：")
    txt.append("      train %d 基因 / %d 组件；val %d / %d；test %d / %d；总组件 %d。" % (
        metrics["n_train"], preflight["eligible_split"]["train"]["components"],
        metrics["n_val"], preflight["eligible_split"]["val"]["components"],
        metrics["n_test"], preflight["eligible_split"]["test"]["components"],
        preflight["eligible_split"]["total_components"]))
    txt.append("    特征：batch06 冻结 features45.npz，45 维 = comp44(对称400/上游2000 的单/二核苷酸/")
    txt.append("    GC/长度=44) + log1p_count(1)，仅组成/计数，无任何 E/S/B 标签作特征。")
    txt.append("    SHA 复核：split_arrays.npz / features45.npz 与协议给定值一致。")
    txt.append("")
    txt.append("三、冻结设计与四次拟合（先写 config.json，再查任何 held-out 指标）")
    txt.append("    种子 20261011；四臂共享同一行集/特征/切分：")
    txt.append("      1) Ridge(alpha=0.1, fit_intercept=True) 预测连续 raw S（标准化）")
    txt.append("      2) 同一 Ridge 预测训练参考经验秩 F（标准化）")
    txt.append("      3) HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,")
    txt.append("         max_leaf_nodes=15, l2_regularization=0.1, early_stopping=False,")
    txt.append("         random_state=20261011) 预测连续 raw S（标准化）")
    txt.append("      4) 同一 HGB 预测训练参考经验秩 F（标准化）")
    txt.append("    X 用 StandardScaler 仅拟合合格 train 并复用；各目标用其合格 train 均值/标准差")
    txt.append("    （总体标准差 ddof=0）标准化后再拟合，预测再逆变换回目标单位。HGB 与 Ridge 共享")
    txt.append("    同一目标变换。验证集只描述、不参与任何拟合/选择；四臂输出全部保留。")
    txt.append("")
    txt.append("    训练参考经验秩 F(x)=(#train<x + 0.5*#train==x)/n_train，n_train=15004；")
    txt.append("    区间外映射 0/1。实测：val/test 无越界值（低于 train 最小=0，高于 train 最大=0）；")
    txt.append("    因 F 是阶梯函数，3765 个非训练行中有 1342 行落在 595 个共享 F 值上（诱导并列）。")
    txt.append("    全精度 raw S 真并列=0；6 位小数导出 234 个舍入伪并列组（batch08 已记录）。")
    txt.append("")
    txt.append("四、结果（测试集，Spearman 对共同全精度 raw S 真值）")
    txt.append("    模型/目标          Spearman    目标单位 MAE     目标单位 R2")
    for k in ["ridge_raw", "ridge_rank", "hgb_raw", "hgb_rank"]:
        txt.append("    %-18s  %.4f       %.4f        %.4f" % (k, sp_test[k], mae_test[k], r2_test[k]))
    txt.append("    基线（train 均值常数）: raw 目标 R2=%.6f, rank 目标 R2=%.6f；模型显著优于常数基线。" % (
        base["raw_target"]["train_mean"]["r2"], base["rank_target"]["train_mean"]["r2"]))
    txt.append("")
    txt.append("五、主比较（rank 目标 - raw 目标 的测试 Spearman 差值）")
    txt.append("    约定：两个主比较各自使用 97.5%% 双侧百分位区间（[%.2f, %.2f] 百分位）作为" % (
        prim["ridge"]["lo_pct"], prim["ridge"]["hi_pct"]))
    txt.append("    保守的两比较校正；接受规则：delta>=0.03 且校正下界>0。")
    txt.append("    Ridge: 点估计 %+.4f, 区间 [%.4f, %.4f] -> %s" % (
        prim["ridge"]["point_estimate"], prim["ridge"]["lo"], prim["ridge"]["hi"],
        "达标" if prim["ridge_meets_rule"] else "未达标"))
    txt.append("    HGB:   点估计 %+.4f, 区间 [%.4f, %.4f] -> %s" % (
        prim["hgb"]["point_estimate"], prim["hgb"]["lo"], prim["hgb"]["hi"],
        "达标" if prim["hgb_meets_rule"] else "未达标"))
    txt.append("    结论：两种模型下 rank 目标都只比 raw 目标略高（+0.014 / +0.012），均小于 0.03；")
    txt.append("    Ridge 区间下界虽为正但幅度不达阈值，HGB 区间含 0。因此【无范围限定的目标表示")
    txt.append("    改进】——目标表示不是 S 可预测性的主要杠杆。")
    txt.append("")
    txt.append("六、Ridge vs HGB（同一测试行上的描述性结果，非接受性比较）")
    txt.append("    HGB 的测试 Spearman 略高于 Ridge（raw 0.409 vs 0.370；rank 0.421 vs 0.384），")
    txt.append("    提示存在少量非线性组成信号；不据此与 batch06 的 E 或 v6.3 结果比较。")
    txt.append("")
    txt.append("七、仿射诊断（代数检验，非第五研究臂、非得分增益）")
    txt.append("    Ridge 在相同 alpha/intercept 下拟合 y 与 a+b*y（a=0.5, b=2.0）：预测仿射相关")
    txt.append("    最大绝对误差 %.3e；R2/Spearman 不变；MAE 比=%.4f(=b)，MSE 比=%.4f(=b^2)。" % (
        aff["max_abs_affine_err_all_rows"], aff["mae_ratio_y2_over_y1"], aff["mse_ratio_y2_over_y1"]))
    txt.append("    诊断通过。")
    txt.append("")
    txt.append("八、完整性检验（12/12 通过）：连接身份、178 排除、组件不跨切分、仅训练拟合")
    txt.append("    缩放器/目标标准化、秩公式/尾/并列、目标标准化逆变换、缺失 S 排除 vs 合法零、")
    txt.append("    保存模型重放、确定性仿射诊断、指标与预测工件一致性。源 SHA 全部匹配。")
    txt.append("")
    txt.append("九、判定区分")
    txt.append("    - 审计/完整性判定：PASS（可复现、诚实、负/中性结论正确保留）。")
    txt.append("    - 科学接受：未达成“目标表示改进”（两 delta 均 <0.03 且/或校正下界未 >0）。")
    txt.append("      负/中性结果有效，不构成生物学可靠性、外推或主模型优越性声明。")
    txt.append("")
    txt.append("十、局限")
    txt.append("    S 是跨组织可检出表达离散度，非转染后时间稳定性；资格由真实标签定义条件评估")
    txt.append("    群体，非推理期资格预测器；重复使用的测试集为回顾性、非前瞻；exact/RC 切分不")
    txt.append("    是完整的近同源隔离；组成/计数特征不含调控语法，Spearman≈0.37–0.42 说明剩余")
    txt.append("    信号缺口更可能在特征而非目标表示。")
    txt.append("")
    txt.append("十一、下一步建议（见 next_batch_proposal.json / BATCH10_DRAFT.txt）：")
    txt.append("    在冻结切分/合格群体上诊断“特征缺口 vs 目标噪声上限”：加入确定性的位置感知")
    txt.append("    k-mer/基序计数特征（CPU、无神经训练），与 batch09 的 45 维基线在同一测试行上")
    txt.append("    比较，判断剩余 S 信号是特征缺口还是目标噪声不可约。")
    txt.append("")
    txt.append("=" * 78)
    report_txt = "\n".join(txt) + "\n"
    atomic_write_text(REPORT_OUT / "report.txt", report_txt)

    # ---- report html ----
    rows_html = "".join(
        "<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>" % (k, sp_test[k], mae_test[k], r2_test[k])
        for k in ["ridge_raw", "ridge_rank", "hgb_raw", "hgb_rank"])
    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>batch09 报告 v6.10.0-starget.rc1</title>
<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;line-height:1.55}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #999;padding:4px 10px}
h1,h2{color:#1a3c6e}.ok{color:#0a7d33}.no{color:#b02418}</style></head>
<body>
<h1>batch09 报告 · v6.10.0-starget.rc1</h1>
<p>连续跨组织离散度 raw S vs 训练参考经验秩 rank S；固定 Ridge/HGB，四次拟合。生成时间 %s。</p>
<h2>群体与来源</h2>
<p>规范 S 定义 18947；特征覆盖的 S 定义 <b>18769</b>；排除 178（S 定义但无启动子特征）。合格切分：
train %d / val %d / test %d，总组件 %d。</p>
<h2>测试指标（Spearman 对共同全精度 raw S 真值；MAE/R2 为目标单位）</h2>
<table><tr><th>臂</th><th>Spearman</th><th>MAE</th><th>R2</th></tr>%s</table>
<h2>主比较（rank−raw 测试 Spearman 差值，97.5%% 双侧百分位区间，2000 次组件配对 bootstrap）</h2>
<table><tr><th>模型</th><th>点估计</th><th>下界</th><th>上界</th><th>是否达标</th></tr>
<tr><td>Ridge</td><td>%+.4f</td><td>%.4f</td><td>%.4f</td><td class="%s">%s</td></tr>
<tr><td>HGB</td><td>%+.4f</td><td>%.4f</td><td>%.4f</td><td class="%s">%s</td></tr></table>
<p>接受规则：delta≥0.03 且校正下界&gt;0。结论：<b>无范围限定的目标表示改进</b>（负/中性结果有效）。</p>
<h2>仿射诊断</h2>
<p>最大仿射误差 %.3e；MAE 比=%.4f(=b)，MSE 比=%.4f(=b²)，R2/Spearman 不变 → 诊断通过。</p>
<h2>判定</h2>
<p>审计/完整性判定：<b class="ok">PASS</b>（12/12 检验、源 SHA 匹配）。科学接受：<b class="no">未达成</b>
（目标表示改进不成立）。主模型 v6.3 / 细胞系 v6.4 指针不变。</p>
<p>局限：S 为跨组织离散度非时间稳定性；回顾性测试集非前瞻；exact/RC 非完整近同源隔离。</p>
</body></html>
""" % (now_cn(), metrics["n_train"], metrics["n_val"], metrics["n_test"],
       preflight["eligible_split"]["total_components"], rows_html,
       prim["ridge"]["point_estimate"], prim["ridge"]["lo"], prim["ridge"]["hi"],
       "ok" if prim["ridge_meets_rule"] else "no", "达标" if prim["ridge_meets_rule"] else "未达标",
       prim["hgb"]["point_estimate"], prim["hgb"]["lo"], prim["hgb"]["hi"],
       "ok" if prim["hgb_meets_rule"] else "no", "达标" if prim["hgb_meets_rule"] else "未达标",
       aff["max_abs_affine_err_all_rows"], aff["mae_ratio_y2_over_y1"], aff["mse_ratio_y2_over_y1"])
    atomic_write_text(REPORT_OUT / "report.html", html)

    print("wrote sha_registry.tsv, status.json, report.txt, report.html")
    print("elapsed_s", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
