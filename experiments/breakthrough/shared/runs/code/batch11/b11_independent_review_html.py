#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate independent_review.html from independent_review.json (Chinese UTF-8)."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import REPORT_OUT  # noqa: E402


def esc(s):
    return html.escape(str(s))


def main():
    r = json.load(open(REPORT_OUT / "independent_review.json", encoding="utf-8"))
    pc = r["protocol_compliance"]
    sa = r["scientific_acceptance"]
    out = []
    A = out.append
    A("<!DOCTYPE html>")
    A("<html lang='zh-CN'><head><meta charset='utf-8'>")
    A("<title>Batch11 独立复核 %s</title>" % esc(r["candidate_version"]))
    A("<style>body{font-family:sans-serif;max-width:1000px;margin:24px auto;line-height:1.55;color:#111}"
      "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #bbb;padding:6px 10px;text-align:left}"
      ".pass{color:#0a6d0a;font-weight:bold}.fail{color:#b00000;font-weight:bold}"
      "pre{white-space:pre-wrap;background:#f5f5f5;padding:10px;border:1px solid #ddd}</style></head><body>")
    A("<h1>Batch11 独立复核（%s）</h1>" % esc(r["candidate_version"]))
    A("<p><strong>复核者身份</strong>：reviewer PID=%s，进程创建时间=%s（Windows GetProcessTimes 验证）。"
      % (esc(r["reviewer_pid"]), esc(r["reviewer_process_create_time_local"])))
    A("控制器 started_at=%s；瞬时 writer PID=%s（与复核者区分）。</p>"
      % (esc(r["controller_started_at"]), esc(r["transient_writer_pid"])))
    A("<p><strong>复核时间</strong>：%s</p>" % esc(r["review_timestamp_local"]))
    A("<h2>结论</h2>")
    pc_cls = "pass" if pc["verdict"] == "PASS" else "fail"
    sa_cls = "pass" if sa["verdict"] == "PASS" else "fail"
    A("<p>协议合规（protocol_compliance）：<span class='%s'>%s</span>（%d/%d 检查通过）</p>"
      % (pc_cls, esc(pc["verdict"]), pc["n_pass"], pc["n_total"]))
    A("<p>科学接受（scientific_acceptance）：<span class='%s'>%s</span>（满足规则的共主比较数=%d）</p>"
      % (sa_cls, esc(sa["verdict"]), sa["n_contrasts_meeting_rule"]))
    A("<p>%s</p>" % esc(sa["meaning"]))
    A("<h2>四个共主条件增量比较</h2>")
    A("<p>接受规则：delta>=0.01 且校正下界&gt;0。配对组件 bootstrap n=2000 seed=20261011，"
      "98.75%% 双侧区间 [0.625, 99.375]（四比较 Bonferroni 风格）。</p>")
    A("<table><tr><th>比较</th><th>学习器</th><th>delta</th><th>校正区间下界</th><th>校正区间上界</th>"
      "<th>meets</th></tr>")
    for k, c in sa["contrasts"].items():
        m = "是" if c["meets_rule"] else "否"
        A("<tr><td>%s</td><td>%s</td><td>%.6f</td><td>%.6f</td><td>%.6f</td><td>%s</td></tr>"
          % (esc(k), esc(c["learner"]), c["delta"], c["adjusted_lo"], c["adjusted_hi"], m))
    A("</table>")
    A("<h2>摘要（中文）</h2><pre>%s</pre>" % esc(r["summary_zh"]))
    A("<h2>协议合规检查明细</h2>")
    A("<table><tr><th>检查</th><th>结果</th><th>说明</th></tr>")
    for c in pc["checks"]:
        cls = "pass" if c["pass"] else "fail"
        lab = "PASS" if c["pass"] else "FAIL"
        A("<tr><td>%s</td><td class='%s'>%s</td><td>%s</td></tr>"
          % (esc(c["name"]), cls, lab, esc(c["detail"])))
    A("</table>")
    A("<h2>独立复算测试集指标（vs 全精度 raw S truth）</h2>")
    A("<table><tr><th>臂</th><th>Spearman</th><th>MAE</th><th>MSE</th><th>R2</th></tr>")
    for k in ("full_ridge", "full_hgb", "A_ridge", "A_hgb", "B_ridge", "B_hgb", "base_ridge", "base_hgb"):
        m = r["recomputed_test_metrics"][k]
        A("<tr><td>%s</td><td>%.6f</td><td>%.6f</td><td>%.6f</td><td>%.6f</td></tr>"
          % (k, m["spearman_vs_raw_truth"], m["mae"], m["mse"], m["r2"]))
    A("</table>")
    A("<h2>与 worker 结果一致性</h2>")
    A("<p>metrics_agree=%s；contrasts_agree=%s；max_abs_spearman_diff=%.3g</p>"
      % (r["worker_agreement"]["metrics_agree"], r["worker_agreement"]["contrasts_agree"],
         r["worker_agreement"]["max_abs_spearman_diff"]))
    A("<h2>证据 SHA256（独立校验）</h2><pre>%s</pre>"
      % esc(json.dumps(r["evidence_sha256"], ensure_ascii=False, indent=2)))
    A("<h2>局限</h2><ul>")
    for lim in r["limitations"]:
        A("<li>%s</li>" % esc(lim))
    A("</ul>")
    A("<h2>后端 provenance</h2><pre>%s</pre>"
      % esc(json.dumps(r["backend_provenance"], ensure_ascii=False, indent=2)))
    if "successor_proposal_assessment" in r:
        sa2 = r["successor_proposal_assessment"]
        A("<h2>后继提案评估</h2>")
        A("<p>建议候选：%s（track=%s）；与 batch11 不同：%s；证据支持：%s</p>"
          % (esc(sa2.get("next_candidate_suggestion")), esc(sa2.get("track_suggestion")),
             esc(sa2.get("distinct_from_batch11")), esc(sa2.get("evidence_supported"))))
        A("<p>%s</p>" % esc(sa2.get("note")))
    A("</body></html>")
    html_doc = "\n".join(out) + "\n"
    (REPORT_OUT / "independent_review.html").write_text(html_doc, encoding="utf-8")
    print("wrote independent_review.html bytes=%d" % len(html_doc.encode("utf-8")))


if __name__ == "__main__":
    main()

