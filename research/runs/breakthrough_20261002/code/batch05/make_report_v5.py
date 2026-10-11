#!/usr/bin/env python3
"""Batch05 report generator (Chinese HTML + text)."""
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(x, nd=4):
    if x is None:
        return "nan"
    if isinstance(x, float):
        return "%.*f" % (nd, x)
    return str(x)


def mstr(m):
    return "n=%s sp=%s r2=%s mae=%s" % (m.get("n"), fmt(m.get("spearman")), fmt(m.get("r2")), fmt(m.get("mae")))


def build_text(manifest, split, cons, ann, resume, fracs):
    A = []
    ap = A.append
    arms = cons["arms"]
    rp = arms["repeat_bag"]
    dp = arms["diverse_bag"]
    rv, rt = rp["val"], rp["test"]
    dv, dt = dp["val"], dp["test"]
    pb = cons["paired_b2_minus_b1"]

    ap("=" * 78)
    ap("Batch05 实验报告：冻结序列-组件划分下的匹配对比与简单基线")
    ap("=" * 78)
    ap("候选: B1=v6.6.0-scratchgene.rc6(repeat_bag)  B2=v6.6.0-scratchgene.rc7(diverse_bag)")
    ap("注释基线: v6.6.0-annridge.rc2 (count/44组成/44组成+log1pcount ridge)")
    ap("")
    ap("【问题】batch04 的 B1/B2 受控对比与“简单注释基线几乎并列”的结论，在冻结序列-组件划分下是否保持。")
    ap("")
    ap("【冻结划分】%s" % split["near_identity_method"]["split_name"])
    asg = split["assignment"]
    ap("  基因数 train=%d val=%d test=%d (目标 80/10/10)；组件数=%d，最大组件基因数=%d"
      % (asg["realized_gene_counts"]["train"], asg["realized_gene_counts"]["val"],
         asg["realized_gene_counts"]["test"], asg["n_components"], asg["largest_component_genes"]))
    ap("  E 标签定义: %s" % split["E_label_definition"]["definition"])
    ap("  精确/RC 跨划分重叠唯一序列数: train_val=%d train_test=%d val_test=%d (必须为0)"
      % (split["audit"]["exact_rc_cross_split_overlap_unique_keys"]["train_val"],
         split["audit"]["exact_rc_cross_split_overlap_unique_keys"]["train_test"],
         split["audit"]["exact_rc_cross_split_overlap_unique_keys"]["val_test"]))
    rr = split["retrieval_report"]
    ap("  近邻同源检索(有界,非穷尽): 候选(>=2band)=%d, minhash预筛后=%d, 确认近邻对=%d, 跨基因边=%d"
      % (rr["n_candidates_share_ge2_bands"], rr["n_candidates_after_minhash_prefilter"],
         rr["n_confirmed_near_identity_pairs"], rr["n_confirmed_cross_gene_edges"]))
    rs = split["audit"]["near_identity_residual_sampling"]
    ap("  残余近邻抽样(跨划分随机对): %d/%d 命中 (有界核查, 非全面同源保证)"
      % (rs["n_hits"], rs["n_sampled_cross_split_pairs"]))
    ap("")

    ap("【主结果 · 4槽匹配推理 (冻结epoch=0轮换)】")
    ap("  验证 gene E: B1 sp=%.4f r2=%.4f mae=%.4f n=%d | B2 sp=%.4f r2=%.4f mae=%.4f n=%d"
      % (rv["full"]["spearman"], rv["full"]["r2"], rv["full"]["mae"], rv["full"]["n"],
         dv["full"]["spearman"], dv["full"]["r2"], dv["full"]["mae"], dv["full"]["n"]))
    ap("  测试 gene E: B1 sp=%.4f r2=%.4f mae=%.4f n=%d | B2 sp=%.4f r2=%.4f mae=%.4f n=%d"
      % (rt["full"]["spearman"], rt["full"]["r2"], rt["full"]["mae"], rt["full"]["n"],
         dt["full"]["spearman"], dt["full"]["r2"], dt["full"]["mae"], dt["full"]["n"]))
    ap("")
    ap("【B2-B1 配对差异 (组件级bootstrap, 整组件重采样)】")
    for sn in ["val", "test"]:
        d = pb[sn]["delta_spearman"]
        ap("  %s: dSpearman=%.4f 95%%CI[%.4f, %.4f] dR2=%.4f n基因=%d n组件=%d"
          % (sn, d["base"], d["ci_low"], d["ci_high"], pb[sn]["delta_r2"]["base"],
             pb[sn]["n_genes"], pb[sn]["n_components"]))
    ap("  非独立说明: 整组件重采样保留组件内基因依赖; 跨组件残余近邻/精确重复未校正; CI为设计条件型。")
    ap("")
    mc = cons["matched_design_checks"]
    ap("【匹配设计核查】初始权重 B1==B2: %s  参数量相等: %s (B1=%d B2=%d)"
      % (mc["initial_weights_match_B1_B2"], mc["n_params_match"],
         mc["n_params"]["repeat_bag"], mc["n_params"]["diverse_bag"]))
    ap("")
    ap("【注释基线 (ridge, 组保持训练内折选alpha, 训练内缩放)】")
    for vname in ["count_only", "comp44", "comp44_count"]:
        v = ann["variants"][vname]
        t = v["test"]["full"]
        ap("  %-12s alpha=%s test sp=%.4f r2=%.4f mae=%.4f n=%d"
          % (vname, v["selected_alpha"], t["spearman"], t["r2"], t["mae"], t["n"]))
    ab = cons["annotation_baseline_comparison"]
    ap("  最优训练内选基线=%s (inner_val sp=%.4f, alpha=%s); 其测试 sp=%.4f r2=%.4f"
      % (ab["best_inner_selected_baseline"]["variant"],
         ab["best_inner_selected_baseline"]["inner_val_spearman"],
         ab["best_inner_selected_baseline"]["selected_alpha"],
         ab["best_baseline_test_spearman"], ab["best_baseline_test_r2"]))
    ap("  B2 测试 sp=%.4f 与最优基线差=%.4f (负值=基线占优, 不隐藏基线损失)"
      % (ab["b2_test_spearman"], ab["b2_minus_best_baseline_spearman"]))
    ap("")
    ev = cons["evidence"]
    ap("【证据阈值】B2-B1 测试 dSpearman=%.4f, 组件bootstrap CI[%.4f, %.4f], 达到>=.03且下界>0: %s"
      % (ev["b2_minus_b1_test_spearman"], ev["component_bootstrap_ci_low"],
         ev["component_bootstrap_ci_high"], ev["meets_improvement_threshold"]))
    ap("  (单种子=>种子稳健性未解决; 这是批次决策阈值, 不代表可靠构建活性, 不停止研究)")
    ap("")
    ap("【恢复验证】权重哈希一致=%s 最大权重绝对差=%s 首batch一致=%s"
      % (resume["weights_sha_match"], resume["max_weight_absdiff"], resume["first_batch_epoch1_match"]))
    ap("")
    ap("【启动子数比例 (全基因)】n=1: %.1f%%  n<4: %.1f%%  n>=4: %.1f%%"
      % (100 * fracs["n1"], 100 * fracs["n_lt4"], 100 * fracs["n_ge4"]))
    ap("")
    ap("【限制】负结果全部保留; 单种子; 回顾性(复用已知数据集/靶标); 划分非因果/非前瞻验证; 近邻检索有界非穷尽;")
    ap("  相似臂不证明容量因果, 基线并列不证明注释丰富度因果; inference 入口拒绝单构建/合成活动预测; 无主指针切换。")
    ap("")
    ap("【batch04 勘误(不改其已接受文件)】GC上阈 .57 非精确训练三分位 .5725; 区分相同 n1 输入与输出; 报告启动子数比例;")
    ap("  区分 status 写入者 PID 与真实 worker PID。")
    ap("")
    ap("【SHA】manifest=%s  split_plan=%s  split_arrays=%s"
      % (manifest.get("manifest_sha256", "n/a"),
         manifest["split_artifact_hashes"]["split_plan.json"],
         manifest["split_artifact_hashes"]["split_arrays.npz"]))
    return "\n".join(A) + "\n"


def build_html(manifest, split, cons, ann, resume, fracs):
    def esc(x):
        return html.escape(str(x))

    arms = cons["arms"]
    rp, dp = arms["repeat_bag"], arms["diverse_bag"]
    rv, rt = rp["val"], rp["test"]
    dv, dt = dp["val"], dp["test"]
    pb = cons["paired_b2_minus_b1"]
    asg = split["assignment"]
    rr = split["retrieval_report"]
    rs = split["audit"]["near_identity_residual_sampling"]
    mc = cons["matched_design_checks"]
    ab = cons["annotation_baseline_comparison"]
    ev = cons["evidence"]

    h = []
    h.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<style>body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:920px;margin:auto;padding:12px;line-height:1.5}")
    h.append("table{border-collapse:collapse;width:100%;margin:8px 0}th,td{border:1px solid #ccc;padding:6px;font-size:12px}")
    h.append("th{background:#f2f2f2}h1{font-size:20px}h2{font-size:16px;margin-top:16px}.warn{color:#a33}</style></head><body>")
    h.append("<h1>Batch05 实验报告</h1>")
    h.append("<p><b>冻结序列-组件划分下的匹配对比与简单基线</b> (B1=rc6, B2=rc7, 注释基线=annridge.rc2)</p>")

    h.append("<h2>冻结划分</h2>")
    h.append("<p>名称: %s</p>" % esc(split["near_identity_method"]["split_name"]))
    h.append("<p>基因 train/val/test = %d/%d/%d; 组件=%d; 最大组件基因=%d</p>"
             % (asg["realized_gene_counts"]["train"], asg["realized_gene_counts"]["val"],
                asg["realized_gene_counts"]["test"], asg["n_components"], asg["largest_component_genes"]))
    h.append("<p>E 标签: %s</p>" % esc(split["E_label_definition"]["definition"]))
    ov = split["audit"]["exact_rc_cross_split_overlap_unique_keys"]
    h.append("<p>精确/RC 跨划分重叠(必须0): train_val=%d train_test=%d val_test=%d</p>"
             % (ov["train_val"], ov["train_test"], ov["val_test"]))
    h.append("<p>近邻检索(有界): 候选=%d, 预筛后=%d, 确认近邻对=%d, 跨基因边=%d; 残余抽样命中=%d/%d</p>"
             % (rr["n_candidates_share_ge2_bands"], rr["n_candidates_after_minhash_prefilter"],
                rr["n_confirmed_near_identity_pairs"], rr["n_confirmed_cross_gene_edges"],
                rs["n_hits"], rs["n_sampled_cross_split_pairs"]))

    h.append("<h2>主结果 (4槽匹配推理)</h2>")
    h.append("<table><tr><th>集合</th><th>指标</th><th>B1</th><th>B2</th></tr>")
    for name, a, b in [("验证", rv["full"], dv["full"]), ("测试", rt["full"], dt["full"])]:
        for k, lbl in [("spearman", "Spearman"), ("r2", "R2"), ("mae", "MAE"), ("n", "n")]:
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (name, lbl, fmt(a[k]), fmt(b[k])))
    h.append("</table>")

    h.append("<h2>B2-B1 组件级 bootstrap 差异</h2>")
    h.append("<table><tr><th>集合</th><th>dSpearman</th><th>95%CI</th><th>dR2</th><th>n基因/n组件</th></tr>")
    for sn in ["val", "test"]:
        d = pb[sn]["delta_spearman"]
        h.append("<tr><td>%s</td><td>%s</td><td>[%s, %s]</td><td>%s</td><td>%d/%d</td></tr>"
                 % (sn, fmt(d["base"]), fmt(d["ci_low"]), fmt(d["ci_high"]),
                    fmt(pb[sn]["delta_r2"]["base"]), pb[sn]["n_genes"], pb[sn]["n_components"]))
    h.append("</table><p class='warn'>整组件重采样保留组件内基因依赖; 跨组件残余近邻/精确重复未校正; CI为设计条件型。</p>")

    h.append("<h2>匹配设计核查</h2>")
    h.append("<p>初始权重 B1==B2: %s; 参数量 B1=%d B2=%d 相等=%s</p>"
             % (mc["initial_weights_match_B1_B2"], mc["n_params"]["repeat_bag"],
                mc["n_params"]["diverse_bag"], mc["n_params_match"]))

    h.append("<h2>注释基线 (ridge)</h2>")
    h.append("<table><tr><th>变体</th><th>alpha</th><th>测试Spearman</th><th>R2</th><th>MAE</th><th>n</th></tr>")
    for vname in ["count_only", "comp44", "comp44_count"]:
        v = ann["variants"][vname]
        t = v["test"]["full"]
        h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (esc(vname), fmt(v["selected_alpha"]), fmt(t["spearman"]), fmt(t["r2"]), fmt(t["mae"]), t["n"]))
    h.append("</table>")
    h.append("<p>最优训练内选基线=%s (测试 sp=%s); B2 测试 sp=%s, 差=%s (负值=基线占优, 不隐藏基线损失)</p>"
             % (esc(ab["best_inner_selected_baseline"]["variant"]), fmt(ab["best_baseline_test_spearman"]),
                fmt(ab["b2_test_spearman"]), fmt(ab["b2_minus_best_baseline_spearman"])))

    h.append("<h2>证据阈值</h2>")
    h.append("<p>B2-B1 测试 dSpearman=%s, 组件bootstrap 95%%CI [%s, %s]; 达到>=.03且下界>0: %s</p>"
             % (fmt(ev["b2_minus_b1_test_spearman"]), fmt(ev["component_bootstrap_ci_low"]),
                fmt(ev["component_bootstrap_ci_high"]), ev["meets_improvement_threshold"]))
    h.append("<p class='warn'>单种子=>种子稳健性未解决; 这是批次决策阈值, 不代表可靠构建活性, 不停止研究。</p>")

    h.append("<h2>测试集分层 (Spearman / n)</h2>")
    h.append("<table><tr><th>分层</th><th>B1</th><th>B2</th><th>最优基线</th></tr>")
    strata_names = [("n=1", "fixed_n_strata", "n=1"), ("n=2-3", "fixed_n_strata", "n=2-3"),
                    ("n>=4", "fixed_n_strata", "n>=4"),
                    ("gc_low", "gc_strata", "gc_low"), ("gc_mid", "gc_strata", "gc_mid"),
                    ("gc_high", "gc_strata", "gc_high"),
                    ("count_low", "count_strata", "count_low"),
                    ("count_mid", "count_strata", "count_mid"),
                    ("count_high", "count_strata", "count_high")]
    best_var = ann["best_inner_selected_baseline"]["variant"]
    for disp, grp, key in strata_names:
        r1 = rt[grp][key]
        r2 = dt[grp][key]
        rb = ann["variants"][best_var]["test"][grp][key]
        h.append("<tr><td>%s</td><td>%s (n=%s)</td><td>%s (n=%s)</td><td>%s (n=%s)</td></tr>"
                 % (disp, fmt(r1["spearman"]), r1["n"], fmt(r2["spearman"]), r2["n"],
                    fmt(rb["spearman"]), rb["n"]))
    h.append("</table>")
    h.append("<p>GC 三分位(训练定义): [%s, %s]; 计数三分位(训练定义): [%s, %s]</p>"
             % (fmt(ann["strata_thresholds"]["gc_tertile_edges_train"][0], 5),
                fmt(ann["strata_thresholds"]["gc_tertile_edges_train"][1], 5),
                fmt(ann["strata_thresholds"]["count_tertile_edges_train"][0], 3),
                fmt(ann["strata_thresholds"]["count_tertile_edges_train"][1], 3)))

    h.append("<h2>恢复验证</h2>")
    h.append("<p>权重哈希一致=%s; 最大权重绝对差=%s; 首batch一致=%s</p>"
             % (resume["weights_sha_match"], resume["max_weight_absdiff"], resume["first_batch_epoch1_match"]))

    h.append("<h2>启动子数比例</h2>")
    h.append("<p>n=1: %.1f%%  n<4: %.1f%%  n>=4: %.1f%%</p>"
             % (100 * fracs["n1"], 100 * fracs["n_lt4"], 100 * fracs["n_ge4"]))

    h.append("<h2>限制</h2><p>负结果全部保留; 单种子; 回顾性(复用已知数据集/靶标); 划分非因果/非前瞻; 近邻检索有界非穷尽; 相似臂不证明容量因果; 无主指针切换。</p>")
    h.append("<h2>batch04 勘误</h2><p>GC上阈 .57 非精确训练三分位 .5725; 区分相同 n1 输入与输出; 报告启动子数比例; 区分 status 写入者 PID 与真实 worker PID。</p>")
    h.append("<p>manifest SHA: %s; split_plan SHA: %s</p>"
             % (esc(manifest.get("manifest_sha256", "n/a")),
                esc(manifest["split_artifact_hashes"]["split_plan.json"])))
    h.append("</body></html>")
    return "\n".join(h)


def main():
    manifest = jload(RUNS / "frozen_manifest.json")
    split = jload(DATA05 / "split_plan.json")
    cons = jload(REPORT / "consolidated_summary_v5.json")
    ann = jload(REPORT / "ann_diagnostics_v5.json")
    resume = jload(MODEL_DIR / "resume_check_v5.json")

    import pandas as pd
    assign = pd.read_csv(DATA05 / "split_assignment.tsv", sep="\t")
    n = assign["n_promoters"].to_numpy()
    fracs = {
        "n1": float((n == 1).mean()),
        "n_lt4": float((n < 4).mean()),
        "n_ge4": float((n >= 4).mean()),
    }

    text = build_text(manifest, split, cons, ann, resume, fracs)
    html_out = build_html(manifest, split, cons, ann, resume, fracs)
    (REPORT / "report.txt").write_text(text, encoding="utf-8")
    (REPORT / "report.html").write_text(html_out, encoding="utf-8")
    print("wrote report.txt and report.html")
    print(text[:2000])
    print("DONE")


if __name__ == "__main__":
    main()
