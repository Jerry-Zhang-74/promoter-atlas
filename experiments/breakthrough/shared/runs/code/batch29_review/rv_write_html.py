# -*- coding: utf-8 -*-
import json, html
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
B = ROOT/"reports/breakthrough_20261002/batch29"
d = json.loads((B/"independent_review.json").read_text(encoding="utf-8"))
def esc(x): return html.escape(str(x))
rows = []
rows.append("<h1>batch29 独立审查 (v6.20.1-sureproxy.rc1)</h1>")
rows.append("<p><b>审查时间</b>: %s &nbsp; <b>审查员</b>: %s</p>" % (esc(d["reviewed_at"]), esc(d["reviewer"])))
rows.append("<p><b>结论(完整性/协议)</b>: %s</p><p><b>结论(科学)</b>: %s</p><p><b>结论(合规)</b>: %s</p>"
            % (esc(d["verdict"]["integrity"]), esc(d["verdict"]["scientific_gain"]), esc(d["verdict"]["compliance"])))
rows.append("<p><b>判定依据</b>: %s</p>" % esc(d["integrity_basis"]))
rows.append("<h2>分离判定</h2><ul>%s</ul>" % "".join("<li>%s: %s</li>"%(esc(k),esc(v)) for k,v in d["verdict_detail"].items()))
rows.append("<h2>发现</h2><ul>%s</ul>" % "".join("<li>[%s/%s/%s] %s</li>"%(esc(f["id"]),esc(f["severity"]),esc(f["status"]),esc(f["title"])) for f in d["findings"]))
rows.append("<h2>执行槽/GPU 审计</h2><p>已用槽位 %d/%d；reserved-before-compute=%s；真实GPU=%s (cuda:0, n_params %s)</p>"
            % (d["execution_gate_audit"]["slots_consumed"], d["execution_gate_audit"]["slots_declared"],
               esc(d["execution_gate_audit"]["reserved_before_compute"]), esc(d["gpu_telemetry_audit"]["true_gpu"]), esc(d["gpu_telemetry_audit"]["n_params"])))
rows.append("<h2>数值复现(零拟合)</h2><ul><li>E1 ridge: spearman %s / gbMAE %s (diff %s)</li><li>E2 cnn: spearman %s / gbMAE %s (diff %s)</li></ul>"
            % (d["numeric_replay"]["E1_ridge"]["spearman"], d["numeric_replay"]["E1_ridge"]["group_balanced_mae"], d["numeric_replay"]["E1_ridge"]["abs_diff_vs_reported"],
               d["numeric_replay"]["E2_cnn_true_seed20261101"]["spearman"], d["numeric_replay"]["E2_cnn_true_seed20261101"]["group_balanced_mae"], d["numeric_replay"]["E2_cnn_true_seed20261101"]["abs_diff_vs_reported"]))
sv = d["source_verification"]
rows.append("<h2>数据源独立重算</h2><p>源SHA %s（3次一致）；全表计数与 90 条目标值独立重算 max|dy|=%s，计数不符 %s</p>"
            % (esc(sv["source_sha256"][:24]+"..."), esc(sv["row_recompute"]["max_abs_y_diff"]), esc(sv["row_recompute"]["count_mismatches"])))
rows.append("<h2>修复计划(下一有界批次)</h2><ul>%s</ul>" % "".join("<li><b>%s</b>: %s</li>"%(esc(k),esc(v)) for k,v in d["repair_plan"].items()))
rows.append("<h2>限制</h2><ul>%s</ul>" % "".join("<li>%s</li>"%esc(x) for x in d["limitations_of_this_review"]))
rows.append("<p>零拟合；未发送邮件；未读凭据；未启动控制器。Test 未打开，无测试分数选择。</p>")
doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>batch29 independent review</title>"
       "<style>body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:24px auto;line-height:1.5}"
       "h1{font-size:20px}h2{font-size:16px;margin-top:18px}code{background:#f4f4f4}</style></head><body>"
       + "".join(rows) + "</body></html>")
(B/"independent_review.html").write_text(doc, encoding="utf-8")
assert "\ufffd" not in doc, "replacement char!"
print("html bytes:", (B/"independent_review.html").stat().st_size, "no U+FFFD")
