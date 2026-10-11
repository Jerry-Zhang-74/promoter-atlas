#!/usr/bin/env python3
"""Batch07 report generator: Chinese text + HTML report with figures/tables/SHAs."""
import sys
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    REPORT, MODEL_DIR, sha256_file, CANDIDATE,
    HGB_SHA, FINALRIDGE_SHA, FEATURES45_SHA, SPLIT_ARRAYS_SHA,
    PRIMARY_QUANTILE, DIAGNOSTIC_QUANTILES, MIN_TEST_COVERAGE,
    MIN_REL_MAE_REDUCTION, BOOT_N, BOOT_SEED, RANDOM_MASK_N, RANDOM_MASK_SEED,
)


def load():
    evalj = json.loads((REPORT / "eval_covergate_b7.json").read_text(encoding="utf-8"))
    pre = json.loads((REPORT / "preflight_b7.json").read_text(encoding="utf-8"))
    geom = json.loads((MODEL_DIR / "covergate_geometry.json").read_text(encoding="utf-8"))
    return evalj, pre, geom


def pct(x):
    return x * 100.0


def build_text(v):
    t = v["by_split"]["test"]
    prim = t["hgb_primary_gate"]
    acc = v["acceptance"]
    diag = t["hgb_diagnostic_quantiles"]
    boot = t["hgb_component_bootstrap_mae"]
    rmb = t["random_mask_baseline"]
    same = t["same_row_retained"]
    strata = t["hgb_promoter_count_strata"]
    gc = t["hgb_gc_tertile_strata"]

    L = []
    L.append("=" * 78)
    L.append("Batch07 报告 —— 覆盖门控（v6.8.0-covergate.rc1）选择性误差诊断")
    L.append("=" * 78)
    L.append("生成时间: " + datetime.datetime.now().astimezone().isoformat())
    L.append("候选版本: " + CANDIDATE)
    L.append("")
    L.append("【一句话结论】")
    L.append("在一个已冻结、被反复复用的 train/val/test 划分上，仅用 outer-train 45 维特征")
    L.append("拟合的固定覆盖门控（马氏距离中心性 <= train 90% 分位），对冻结 HGB 预测器在")
    L.append("test 上仅带来 1.40% 的相对 MAE 下降，未达到预注册的 5% 阈值；因此主接受判定为")
    L.append("NEGATIVE（acceptance_threshold_met=false）。这是一个回顾性/探索性的适用性诊断，")
    L.append("不是模型改进，也不构成置信度/校准保证。")
    L.append("")
    L.append("【科学诚实性（重要）】")
    L.append("1) 现有划分与 test 被反复复用 => 回顾性/探索性，绝不宣称前瞻性或全新独立验证；")
    L.append("2) train-only 阈值并不能抹除先前由结果驱动(项目已见过这些分数)的决策；")
    L.append("3) 保留子集 vs 全样本是不同人群，子集相关性更高不等于模型更好；")
    L.append("4) 特征中心性(马氏距离)不是完整密度/OOD 保证，多模态与数据内部标注偏差仍存在；")
    L.append("5) 门控未命中不等于生物学上的分布外，也不是确认不可靠或零活性。")
    L.append("")
    L.append("-" * 78)
    L.append("问题与范围")
    L.append("-" * 78)
    L.append("问题: 一个预注册的 train-only 特征覆盖规则，能否在冻结 HGB 模型上、在高覆盖率下")
    L.append("降低选择性预测误差？")
    L.append("范围: 仅基因级 Score_E 排序；不支持启动子/构建体/HEK293T 活性。")
    L.append("")
    L.append("-" * 78)
    L.append("冻结设计")
    L.append("-" * 78)
    L.append("划分 SHA: " + SPLIT_ARRAYS_SHA)
    L.append("特征45 SHA: " + FEATURES45_SHA)
    L.append("HGB 模型 SHA: " + HGB_SHA)
    L.append("finalridge 模型 SHA: " + FINALRIDGE_SHA)
    L.append("几何工件 SHA: " + v["geometry_sha256"])
    L.append("几何: StandardScaler 仅用 outer-train 特征拟合, 再对标准化 train 特征拟合 sklearn")
    L.append("      LedoitWolf 协方差; 覆盖分数 = 平方马氏距离 (x-mu)^T precision (x-mu)。")
    L.append("主门控: 距离 <= train 第 " + str(int(PRIMARY_QUANTILE * 100)) + " 百分位 (线性插值分位), ")
    L.append("阈值 = " + ("%.4f" % v["primary_gate"]["threshold"]) + "。")
    L.append("诊断覆盖点: 50% / 70% / 90% / 100% train 分位, 一并报告, 不做事后择优。")
    L.append("")
    return L


def build_text2(L, v):
    t = v["by_split"]["test"]
    prim = t["hgb_primary_gate"]
    acc = v["acceptance"]
    diag = t["hgb_diagnostic_quantiles"]
    boot = t["hgb_component_bootstrap_mae"]
    rmb = t["random_mask_baseline"]
    same = t["same_row_retained"]
    strata = t["hgb_promoter_count_strata"]
    gc = t["hgb_gc_tertile_strata"]

    L.append("-" * 78)
    L.append("主接受判定 (HGB, test)")
    L.append("-" * 78)
    L.append("覆盖率 = %.2f%%  (要求 >= %.0f%% => %s)" %
             (pct(acc["coverage"]), pct(MIN_TEST_COVERAGE), "达标" if acc["coverage_ge_80pct"] else "未达标"))
    L.append("相对 MAE 下降 = %.3f%%  (要求 >= %.0f%% => %s)" %
             (pct(acc["relative_mae_reduction"]), pct(MIN_REL_MAE_REDUCTION), "达标" if acc["relative_mae_reduction_ge_5pct"] else "未达标"))
    L.append("组件 bootstrap 95%% 下界 = %.6f  (要求 > 0 => %s)" %
             (acc["component_bootstrap_lb"], "达标" if acc["component_bootstrap_lb_gt_0"] else "未达标"))
    L.append("acceptance_threshold_met = %s" % acc["acceptance_threshold_met"])
    L.append("")
    L.append("主门控指标 (HGB, test):")
    L.append("  全样本: n=%d MAE=%.5f R2=%.4f Spearman=%.4f 高误差率=%.4f" %
             (prim["full"]["n"], prim["full"]["mae"], prim["full"]["r2"],
              prim["full"]["spearman"], prim["full"]["high_error_rate_gt_020"]))
    L.append("  保留:   n=%d MAE=%.5f R2=%.4f Spearman=%.4f 高误差率=%.4f" %
             (prim["retained"]["n"], prim["retained"]["mae"], prim["retained"]["r2"],
              prim["retained"]["spearman"], prim["retained"]["high_error_rate_gt_020"]))
    L.append("  拒绝:   n=%d MAE=%.5f R2=%.4f Spearman=%.4f 高误差率=%.4f" %
             (prim["rejected"]["n"], prim["rejected"]["mae"], prim["rejected"]["r2"],
              prim["rejected"]["spearman"], prim["rejected"]["high_error_rate_gt_020"]))
    L.append("  目标范围: min=%.4f max=%.4f std=%.4f" %
             (prim["full"]["target_range"]["min"], prim["full"]["target_range"]["max"],
              prim["full"]["target_range"]["std"]))
    L.append("  预测范围: min=%.4f max=%.4f std=%.4f" %
             (prim["full"]["pred_range"]["min"], prim["full"]["pred_range"]["max"],
              prim["full"]["pred_range"]["std"]))
    L.append("  组件 bootstrap: 点估计 diff=%.6f, 95%% CI=[%.6f, %.6f], 退化(0保留)=%d, n_boot=%d, n_components=%d" %
             (boot["point_diff"], boot["bootstrap_diff_ci_low"], boot["bootstrap_diff_ci_high"],
              boot["n_degenerate_zero_retained"], boot["n_boot"], boot["n_components"]))
    L.append("")
    L.append("-" * 78)
    L.append("诊断覆盖点 (HGB, test) —— 全部报告, 不择优")
    L.append("-" * 78)
    for q in ["0.50", "0.70", "0.90", "1.00"]:
        d = diag[q]
        rel = (d["full"]["mae"] - d["retained"]["mae"]) / d["full"]["mae"]
        L.append("  q=%s: 覆盖率=%5.1f%%  保留MAE=%.5f  相对下降=%5.2f%%  保留Spearman=%.4f  保留R2=%.4f" %
                 (q, pct(d["coverage"]), d["retained"]["mae"], pct(rel),
                  d["retained"]["spearman"], d["retained"]["r2"]))
    L.append("")
    L.append("-" * 78)
    L.append("同行比较 (相同保留行上, test)")
    L.append("-" * 78)
    for k in ["hgb", "finalridge", "b2_original", "combined_fixedmean"]:
        d = same[k]
        L.append("  %-18s n=%d Spearman=%.4f R2=%.4f MAE=%.5f 高误差率=%.4f" %
                 (k, d["n"], d["spearman"], d["r2"], d["mae"], d["high_error_rate_gt_020"]))
    L.append("")
    L.append("-" * 78)
    L.append("随机掩码基线 (test, 与主门控保留数严格一致)")
    L.append("-" * 78)
    L.append("  保留数 K = %d" % rmb["target_retained_count"])
    L.append("  500 个随机掩码 MAE: mean=%.5f std=%.6f p50=%.5f [p2.5=%.5f, p97.5=%.5f]" %
             (rmb["random_mae_mean"], rmb["random_mae_std"], rmb["random_mae_p50"],
              rmb["random_mae_p2_5"], rmb["random_mae_p97_5"]))
    L.append("  门控保留 MAE = %.5f  (位于随机掩码分布的第 %.1f 百分位)" %
             (rmb["gate_mae"], rmb["gate_percentile_vs_random"]))
    L.append("  说明: 仅作诊断, 不构成新的接受/调参。门控选择了非随机的更低误差子集, 但幅度远低于 5%% 阈值。")
    L.append("")
    L.append("-" * 78)
    L.append("描述性分层 (HGB, test) —— 诊断, 不是可部署门控")
    L.append("-" * 78)
    for k in ["n=1", "n=2-3", "n>=4"]:
        d = strata[k]
        L.append("  启动子数 %s: n=%d Spearman=%.4f MAE=%.5f" % (k, d["n"], d["spearman"], d["mae"]))
    for k in ["gc_tertile_1", "gc_tertile_2", "gc_tertile_3"]:
        d = gc[k]
        L.append("  %s: n=%d Spearman=%.4f MAE=%.5f" % (k, d["n"], d["spearman"], d["mae"]))
    L.append("")
    L.append("-" * 78)
    L.append("结论与局限")
    L.append("-" * 78)
    L.append("主接受判定 NEGATIVE。覆盖门控在 90%% 覆盖率下未能带来 >=5%% 的相对 MAE 下降;")
    L.append("选择性误差下降在低覆盖率(50%%/70%%)时才明显, 这本质是覆盖率-误差权衡, 而非模型改进;")
    L.append("相关性提升不能覆盖 MAE/覆盖率失败。特征中心性门控不等于完整密度/OOD, 门控未命中")
    L.append("不等于不可靠。结果如实保留, 不重扫, 不事后调整阈值。")
    L.append("")
    L.append("审查完整性 PASS 与科学门控接受是两回事: 本报告描述科学门控接受(NEGATIVE);")
    L.append("审查完整性由独立审查者单独判定。")
    L.append("=" * 78)
    return L


def build_html(v, pre):
    t = v["by_split"]["test"]
    prim = t["hgb_primary_gate"]
    acc = v["acceptance"]
    diag = t["hgb_diagnostic_quantiles"]
    boot = t["hgb_component_bootstrap_mae"]
    rmb = t["random_mask_baseline"]
    same = t["same_row_retained"]
    strata = t["hgb_promoter_count_strata"]
    gc = t["hgb_gc_tertile_strata"]

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    diag_rows = ""
    for q in ["0.50", "0.70", "0.90", "1.00"]:
        d = diag[q]
        rel = (d["full"]["mae"] - d["retained"]["mae"]) / d["full"]["mae"]
        diag_rows += (
            "<tr><td>%s</td><td>%.1f%%</td><td>%.5f</td><td>%.2f%%</td>"
            "<td>%.4f</td><td>%.4f</td></tr>" % (
                q, pct(d["coverage"]), d["retained"]["mae"], pct(rel),
                d["retained"]["spearman"], d["retained"]["r2"]))

    same_rows = ""
    for k in ["hgb", "finalridge", "b2_original", "combined_fixedmean"]:
        d = same[k]
        same_rows += (
            "<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td>"
            "<td>%.5f</td><td>%.4f</td></tr>" % (
                esc(k), d["n"], d["spearman"], d["r2"], d["mae"], d["high_error_rate_gt_020"]))

    strata_rows = ""
    for k in ["n=1", "n=2-3", "n>=4"]:
        d = strata[k]
        strata_rows += "<tr><td>启动子数 %s</td><td>%d</td><td>%.4f</td><td>%.5f</td></tr>" % (
            esc(k), d["n"], d["spearman"], d["mae"])
    for k in ["gc_tertile_1", "gc_tertile_2", "gc_tertile_3"]:
        d = gc[k]
        strata_rows += "<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.5f</td></tr>" % (
            esc(k), d["n"], d["spearman"], d["mae"])

    html = []
    html.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    html.append("<title>Batch07 覆盖门控报告 v6.8.0-covergate.rc1</title>")
    html.append("<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;line-height:1.5}"
                "table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #ccc;padding:6px 10px}"
                "th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd;margin:8px 0}"
                ".neg{color:#b00020;font-weight:bold}.ok{color:#0a7a0a;font-weight:bold}</style></head><body>")
    html.append("<h1>Batch07 覆盖门控报告 —— v6.8.0-covergate.rc1</h1>")
    html.append("<p>生成时间: " + esc(datetime.datetime.now().astimezone().isoformat()) + "</p>")

    html.append("<h2>结论</h2>")
    html.append("<p class='neg'>主接受判定 NEGATIVE (acceptance_threshold_met=false)。</p>")
    html.append("<p>冻结 HGB 预测器 + train-only 马氏距离覆盖门控(90%% 分位)在 test 上覆盖率 "
                "%.2f%%，但相对 MAE 下降仅 %.3f%%，未达到预注册的 5%% 阈值。这是回顾性/探索性的适用性诊断, "
                "不是模型改进, 也不构成置信度/校准保证。</p>" %
                (pct(acc["coverage"]), pct(acc["relative_mae_reduction"])))

    html.append("<h2>科学诚实性</h2><ul>")
    for s in ["现有划分与 test 被反复复用 => 回顾性/探索性, 绝不宣称前瞻性",
              "train-only 阈值不能抹除先前由结果驱动的项目决策",
              "保留子集 vs 全样本是不同人群, 子集相关性更高不等于模型更好",
              "特征中心性(马氏距离)不是完整密度/OOD 保证",
              "门控未命中不等于生物学分布外, 也不是确认不可靠/零活性"]:
        html.append("<li>" + esc(s) + "</li>")
    html.append("</ul>")

    html.append("<h2>主接受判定 (HGB, test)</h2><table><tr><th>指标</th><th>值</th><th>判定</th></tr>")
    html.append("<tr><td>覆盖率</td><td>%.2f%%</td><td class='%s'>%s</td></tr>" %
                (pct(acc["coverage"]), "ok" if acc["coverage_ge_80pct"] else "neg",
                 "达标" if acc["coverage_ge_80pct"] else "未达标"))
    html.append("<tr><td>相对 MAE 下降</td><td>%.3f%%</td><td class='%s'>%s</td></tr>" %
                (pct(acc["relative_mae_reduction"]), "ok" if acc["relative_mae_reduction_ge_5pct"] else "neg",
                 "达标" if acc["relative_mae_reduction_ge_5pct"] else "未达标"))
    html.append("<tr><td>组件 bootstrap 95%% 下界</td><td>%.6f</td><td class='%s'>%s</td></tr>" %
                (acc["component_bootstrap_lb"], "ok" if acc["component_bootstrap_lb_gt_0"] else "neg",
                 "达标" if acc["component_bootstrap_lb_gt_0"] else "未达标"))
    html.append("</table>")

    html.append("<h2>主门控指标</h2><table><tr><th>集合</th><th>n</th><th>MAE</th><th>R2</th><th>Spearman</th><th>高误差率</th></tr>")
    for name, d in [("全样本", prim["full"]), ("保留", prim["retained"]), ("拒绝", prim["rejected"])]:
        html.append("<tr><td>%s</td><td>%d</td><td>%.5f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>" %
                    (name, d["n"], d["mae"], d["r2"], d["spearman"], d["high_error_rate_gt_020"]))
    html.append("</table>")

    html.append("<h2>诊断覆盖点 (全部报告, 不择优)</h2><table><tr><th>分位</th><th>覆盖率</th><th>保留MAE</th><th>相对下降</th><th>保留Spearman</th><th>保留R2</th></tr>")
    html.append(diag_rows)
    html.append("</table>")

    html.append("<h2>同行比较 (相同保留行)</h2><table><tr><th>预测器</th><th>n</th><th>Spearman</th><th>R2</th><th>MAE</th><th>高误差率</th></tr>")
    html.append(same_rows)
    html.append("</table>")

    html.append("<h2>随机掩码基线</h2>")
    html.append("<p>保留数 K=%d; 500 个固定种子随机掩码 MAE mean=%.5f, p50=%.5f; 门控保留 MAE=%.5f "
                "(位于随机掩码分布第 %.1f 百分位)。仅诊断, 不构成接受/调参。</p>" %
                (rmb["target_retained_count"], rmb["random_mae_mean"], rmb["random_mae_p50"],
                 rmb["gate_mae"], rmb["gate_percentile_vs_random"]))

    html.append("<h2>描述性分层 (诊断, 非可部署门控)</h2><table><tr><th>分层</th><th>n</th><th>Spearman</th><th>MAE</th></tr>")
    html.append(strata_rows)
    html.append("</table>")

    html.append("<h2>图</h2>")
    for f in ["coverage_selective_risk.png", "coverage_rel_mae_reduction.png",
              "mahalanobis_distance.png", "random_mask_baseline.png"]:
        html.append("<img src='figures/" + f + "' alt='" + f + "'>")

    html.append("<h2>冻结工件 SHA</h2><table><tr><th>工件</th><th>SHA256</th></tr>")
    for name, sha in [("划分 split_arrays.npz", SPLIT_ARRAYS_SHA),
                      ("特征45 features45.npz", FEATURES45_SHA),
                      ("HGB hgb_baseline.joblib", HGB_SHA),
                      ("finalridge finalridge.npz", FINALRIDGE_SHA),
                      ("几何 covergate_geometry.npz", v["geometry_sha256"])]:
        html.append("<tr><td>%s</td><td><code>%s</code></td></tr>" % (esc(name), esc(sha)))
    html.append("</table>")

    html.append("<h2>审查说明</h2>")
    html.append("<p>审查完整性 PASS 与科学门控接受是两回事; 本报告描述科学门控接受(NEGATIVE)。</p>")
    html.append("</body></html>")
    return "\n".join(html)


def main():
    v, pre, geom = load()
    L = build_text(v)
    L = build_text2(L, v)
    text = "".join(x + "\n" for x in L)
    (REPORT / "report.txt").write_text(text, encoding="utf-8")
    (REPORT / "report.html").write_text(build_html(v, pre), encoding="utf-8")
    print("report.txt and report.html written")
    print("acceptance_threshold_met =", v["acceptance"]["acceptance_threshold_met"])


if __name__ == "__main__":
    main()
