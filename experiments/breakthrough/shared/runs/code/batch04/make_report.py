#!/usr/bin/env python3
"""Batch04 report generator (Chinese HTML + text)."""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(x, nd=4):
    if x is None:
        return "nan"
    if isinstance(x, float):
        return "%.*f" % (nd, x)
    return str(x)


def build_text(manifest, cons, ann, resume, cell):
    arms = cons["arms"]
    rp = arms["repeat_bag"]
    dp = arms["diverse_bag"]
    rp_m = rp["mean"]
    dp_m = dp["mean"]

    L = []
    A = L.append
    A("=" * 78)
    A("Batch04 实验报告：匹配输入/读出/更新次数下的基因-启动子多样性对比")
    A("=" * 78)
    A("候选: B1=v6.6.0-scratchgene.rc4(repeat_bag)  B2=v6.6.0-scratchgene.rc5(diverse_bag)")
    A("注释基线: v6.6.0-annridge.rc1")
    A("")
    A("【问题】匹配输入槽数/网络头/参数数/更新次数下，B1(同一启动子重复4次) vs B2(最多4个不同启动子)。")
    A("        这是受控对比，用于分离重复输入暴露/表征容量与启动子多样性；不做因果断言、不做提分搜索。")
    A("")
    A("【主结果 · 4槽匹配推理(epoch=0 固定轮换, 3种子均值)】")
    A("  测试 gene E Spearman: B1=%.4f  B2=%.4f  差异=%.4f"
      % (rp_m["test_4slot_full_sp"], dp_m["test_4slot_full_sp"],
         dp_m["test_4slot_full_sp"] - rp_m["test_4slot_full_sp"]))
    A("  测试 gene E R2:       B1=%.4f  B2=%.4f"
      % (rp_m["test_4slot_full_r2"], dp_m["test_4slot_full_r2"]))
    A("  测试未暴露 Spearman:  B1=%.4f  B2=%.4f"
      % (rp_m["test_4slot_unexp_sp"], dp_m["test_4slot_unexp_sp"]))
    A("  验证 gene E Spearman: B1=%.4f  B2=%.4f"
      % (rp_m["val_4slot_full_sp"], dp_m["val_4slot_full_sp"]))
    A("")
    A("  各种子测试 Spearman:")
    for s in [20261011, 20261012, 20261013]:
        A("    seed %d: B1=%.4f  B2=%.4f" % (s, rp["seeds"][str(s)]["test_4slot_full_sp"],
                                            dp["seeds"][str(s)]["test_4slot_full_sp"]))
    A("")
    pb = cons["paired_b2_minus_b1"]
    A("【B2-B1 配对差异(测试集, 4槽主指标)】")
    for s in [20261011, 20261012, 20261013]:
        d = pb[str(s)]["test_delta_spearman"]
        A("    seed %d: dSpearman=%.4f  (配对bootstrap 95%%CI [%.4f, %.4f])"
          % (s, d["base"], d["ci_low"], d["ci_high"]))
    ds = pb["_3seed_delta_spearman"]
    A("    3种子 dSpearman 均值=%.4f, 符号一致=%s" % (ds["mean"], ds["consistent_sign"]))
    A("    非独立说明: 配对基因残余跨基因同源/精确重复, CI 为设计条件型, 非独立基因显著性检验。")
    A("")
    mc = cons["matched_design_checks"]
    A("【匹配设计核查】")
    A("    初始权重 B1==B2: %s" % mc["initial_weights_match_B1_B2"])
    A("    参数量 B1=%d B2=%d 相等: %s" % (mc["n_params"]["repeat_bag"],
                                          mc["n_params"]["diverse_bag"], mc["n_params_match"]))
    A("")
    A("【注释丰富度诊断(ridge, 训练内选alpha, 描述性非因果)】")
    for vname in ["count_only", "comp44", "comp44_count"]:
        v = ann["variants"][vname]
        t = v["test"]["full"]
        A("    %-12s alpha=%s  test Spearman=%.4f  R2=%.4f  MAE=%.4f  n=%d"
          % (vname, v["selected_alpha"], t["spearman"], t["r2"], t["mae"], t["n"]))
    add = ann["additive_count_info"]
    A("    加性启动子数信息(test full): dSpearman=%.4f dR2=%.4f"
      % (add["test_full"]["delta_spearman"], add["test_full"]["delta_r2"]))
    A("    NN vs 注释基线: diverse_bag NN 测试 sp=%.4f 与 comp44_count ridge sp=%.4f 几乎并列"
      % (dp_m["test_4slot_full_sp"], ann["variants"]["comp44_count"]["test"]["full"]["spearman"]))
    A("                    (NN R2=%.4f 反而低于 ridge R2=%.4f)；NN 相对简单计数+组成特征的边际排序收益有限。"
      % (dp_m["test_4slot_full_r2"], ann["variants"]["comp44_count"]["test"]["full"]["r2"]))
    A("")
    A("【恢复验证】")
    A("    权重哈希一致: %s  最大权重绝对差: %s  首batch一致: %s"
      % (resume["weights_sha_match"], resume["max_weight_absdiff"], resume["first_batch_epoch1_match"]))
    A("")
    A("【细胞系14构建诊断】")
    A("    construct_LOO MAE_log10=%.4f R2=%.4f  family_LOO MAE_log10=%.4f R2=%.4f"
      % (cell["construct_LOO_MAE_log10"], cell["construct_LOO_R2"],
         cell["family_LOO_MAE_log10"], cell["family_LOO_R2"]))
    A("    高误差构建(回顾性): %s" % ", ".join(cell["high_error_constructs"]))
    A("    弃权: 新单构建/不支持应用返回 unsupported_scope; 支持规则仅基于训练输入覆盖, 显式未验证。")
    A("")
    A("【限制】负结果全部保留; 相似臂不证明容量因果, 分层关联不证明注释丰富度因果; 未暴露子集 R2 仍为限制。")
    A("    inference 入口拒绝单构建活动预测; 无主指针切换。")
    A("")
    A("【SHA】manifest=%s  data_npz=%s" % (manifest.get("manifest_sha256", "n/a"),
                                           manifest["source_hashes"]["data_npz"]))
    return "\n".join(L) + "\n"


def build_html(manifest, cons, ann, resume, cell):
    arms = cons["arms"]
    rp = arms["repeat_bag"]
    dp = arms["diverse_bag"]
    rp_m = rp["mean"]
    dp_m = dp["mean"]
    pb = cons["paired_b2_minus_b1"]

    def esc(x):
        return html.escape(str(x))

    h = []
    h.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<style>body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:900px;margin:auto;padding:12px;line-height:1.5}")
    h.append("table{border-collapse:collapse;width:100%;margin:8px 0}th,td{border:1px solid #ccc;padding:6px;font-size:13px}")
    h.append("th{background:#f2f2f2}h1{font-size:20px}h2{font-size:16px;margin-top:16px}.warn{color:#a33}</style></head><body>")
    h.append("<h1>Batch04 实验报告</h1>")
    h.append("<p><b>匹配输入/读出/更新次数下的基因-启动子多样性对比</b></p>")
    h.append("<p>B1=v6.6.0-scratchgene.rc4(repeat_bag)；B2=v6.6.0-scratchgene.rc5(diverse_bag)；注释基线=v6.6.0-annridge.rc1</p>")

    h.append("<h2>主结果（4槽匹配推理，3种子均值）</h2>")
    h.append("<table><tr><th>指标</th><th>B1</th><th>B2</th><th>B2-B1</th></tr>")
    h.append("<tr><td>测试 E Spearman(全量)</td><td>%s</td><td>%s</td><td>%s</td></tr>"
             % (fmt(rp_m["test_4slot_full_sp"]), fmt(dp_m["test_4slot_full_sp"]),
                fmt(dp_m["test_4slot_full_sp"] - rp_m["test_4slot_full_sp"])))
    h.append("<tr><td>测试 E R2(全量)</td><td>%s</td><td>%s</td><td>%s</td></tr>"
             % (fmt(rp_m["test_4slot_full_r2"]), fmt(dp_m["test_4slot_full_r2"]),
                fmt(dp_m["test_4slot_full_r2"] - rp_m["test_4slot_full_r2"])))
    h.append("<tr><td>测试 E Spearman(未暴露)</td><td>%s</td><td>%s</td><td>%s</td></tr>"
             % (fmt(rp_m["test_4slot_unexp_sp"]), fmt(dp_m["test_4slot_unexp_sp"]),
                fmt(dp_m["test_4slot_unexp_sp"] - rp_m["test_4slot_unexp_sp"])))
    h.append("<tr><td>验证 E Spearman(全量)</td><td>%s</td><td>%s</td><td>%s</td></tr>"
             % (fmt(rp_m["val_4slot_full_sp"]), fmt(dp_m["val_4slot_full_sp"]),
                fmt(dp_m["val_4slot_full_sp"] - rp_m["val_4slot_full_sp"])))
    h.append("</table>")

    h.append("<h2>各种子测试 Spearman</h2><table><tr><th>seed</th><th>B1</th><th>B2</th></tr>")
    for s in [20261011, 20261012, 20261013]:
        h.append("<tr><td>%d</td><td>%s</td><td>%s</td></tr>"
                 % (s, fmt(rp["seeds"][str(s)]["test_4slot_full_sp"]), fmt(dp["seeds"][str(s)]["test_4slot_full_sp"])))
    h.append("</table>")

    h.append("<h2>B2-B1 配对差异</h2><table><tr><th>seed</th><th>dSpearman</th><th>95%CI</th></tr>")
    for s in [20261011, 20261012, 20261013]:
        d = pb[str(s)]["test_delta_spearman"]
        h.append("<tr><td>%d</td><td>%s</td><td>[%s, %s]</td></tr>"
                 % (s, fmt(d["base"]), fmt(d["ci_low"]), fmt(d["ci_high"])))
    h.append("</table><p class='warn'>配对基因残余跨基因同源/精确重复，CI 为设计条件型，不是独立基因显著性检验。</p>")

    h.append("<h2>注释丰富度诊断（ridge，描述性非因果）</h2><table><tr><th>变体</th><th>alpha</th><th>Spearman</th><th>R2</th><th>MAE</th><th>n</th></tr>")
    for vname in ["count_only", "comp44", "comp44_count"]:
        v = ann["variants"][vname]
        t = v["test"]["full"]
        h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (esc(vname), fmt(v["selected_alpha"]), fmt(t["spearman"]), fmt(t["r2"]), fmt(t["mae"]), t["n"]))
    h.append("</table>")
    h.append("<p class='warn'>NN vs 注释基线: diverse_bag NN 测试 sp=%s 与 comp44_count ridge sp=%s 几乎并列（NN R2=%s 反而低于 ridge R2=%s）；NN 相对简单计数+组成特征的边际排序收益有限。</p>"
             % (fmt(dp_m["test_4slot_full_sp"]), fmt(ann["variants"]["comp44_count"]["test"]["full"]["spearman"]),
                fmt(dp_m["test_4slot_full_r2"]), fmt(ann["variants"]["comp44_count"]["test"]["full"]["r2"])))

    h.append("<h2>恢复验证</h2><table><tr><th>项</th><th>结果</th></tr>")
    h.append("<tr><td>权重哈希一致</td><td>%s</td></tr>" % resume["weights_sha_match"])
    h.append("<tr><td>最大权重绝对差</td><td>%s</td></tr>" % resume["max_weight_absdiff"])
    h.append("<tr><td>首 batch 顺序一致</td><td>%s</td></tr>" % resume["first_batch_epoch1_match"])
    h.append("</table>")

    h.append("<h2>细胞系14构建诊断</h2>")
    h.append("<p>construct_LOO MAE_log10=%.4f / family_LOO MAE_log10=%.4f</p>"
             % (cell["construct_LOO_MAE_log10"], cell["family_LOO_MAE_log10"]))
    h.append("<p class='warn'>高误差为回顾性标记，非可部署弃权分类器；n=14/3家族不支持新构建置信；不支持应用返回 unsupported_scope。</p>")

    h.append("<h2>限制</h2><p>负结果全部保留；相似臂不证明容量因果，分层关联不证明注释丰富度因果；未暴露子集 R2 仍为限制；无主指针切换。</p>")
    h.append("<p>manifest SHA: %s</p>" % esc(manifest.get("manifest_sha256", "n/a")))
    h.append("</body></html>")
    return "\n".join(h)


def main():
    manifest = jload(RUNS / "selection_run_manifest.json")
    cons = jload(REPORT / "consolidated_summary.json")
    ann = jload(REPORT / "ann_diagnostics.json")
    resume = jload(MODEL_DIR / "resume_check.json")
    cell = jload(REPORT / "cellline_14construct_diagnostic_table.json")

    text = build_text(manifest, cons, ann, resume, cell)
    html_out = build_html(manifest, cons, ann, resume, cell)
    (REPORT / "report.txt").write_text(text, encoding="utf-8")
    (REPORT / "report.html").write_text(html_out, encoding="utf-8")
    print("wrote report.txt and report.html")
    print(text[:1500])
    print("DONE")


if __name__ == "__main__":
    main()
