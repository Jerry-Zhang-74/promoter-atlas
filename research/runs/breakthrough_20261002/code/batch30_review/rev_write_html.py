# -*- coding: utf-8 -*-
"""Emit a concise Chinese+English HTML of the canonical batch30 review from the JSON."""
import html, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
REP30 = ROOT / "reports/breakthrough_20261002/batch30"
d = json.load(open(REP30 / "independent_review.json", encoding="utf-8"))
e = lambda s: html.escape(str(s), quote=False)
rows = "".join("<tr><td>%s</td><td class='%s'>%s</td></tr>" % (e(k), "ok" if v else "bad", e("PASS" if v else "FAIL"))
               for k, v in d["checks_ok"].items())
find = "".join("<li><b>%s</b> [%s] %s</li>" % (e(f["id"]), e(f["severity"]), e(f["title"])) for f in d["findings"])
h = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>batch30 独立复核 / independent review</title>
<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:960px;margin:24px auto;line-height:1.5}
.ok{color:#0a7d00;font-weight:700}.bad{color:#b00000;font-weight:700}table{border-collapse:collapse;width:100%%}
td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}code{background:#f4f4f4;padding:0 3px}</style></head><body>
<h1>batch30 独立复核 / fresh independent review</h1>
<p><b>复核者</b>：独立复核会话（与 worker 不同进程/会话）。<b>实际时钟</b>：%s。<b>零拟合</b>：fits=0, optimizer_steps=0；
所有神经前向仅在 <code>cuda:0</code>（RTX 5060），前向总耗时 %.3f s（预算 300 s）。</p>
<h2>结论 / verdict</h2><ul>
<li>数据/数值完整性 integrity：<span class="ok">%s</span></li>
<li>协议合规 compliance：<span class="ok">%s</span>（两项已披露偏差，见 finding F1/F2）</li>
<li>科学增益 scientific_gain：<b>%s</b>（就绪批次，非模型、非性能提升、无晋升）</li>
<li>结构化就绪 structural readiness：%s；训练续训经验等价性：%s</li>
<li>生产指针：%s</li></ul>
<h2>独立核验项 / independent checks (%d/%d)</h2><table><tr><th>check</th><th>result</th></tr>%s</table>
<h2>关键发现 / findings</h2><ul>%s</ul>
<h2>已修正的历史断言 / corrections</h2><ul>%s</ul>
<h2>下一批次提案 batch31（父级授权后方可执行，本批复核不授权）</h2>
<p>恰好 <b>4</b> 次估计器执行：E1 ridge342 α100；E2 legnet64 真标签 seed20261101（<b>一次真实拟合</b>，含计划内首轮结束暂停止与恢复，同一 exec_id）；
E3 legnet64 seed20261102；E4 legnet64 标签打乱 seed20261101。崩溃/重置重启另计一次。≤3 GPUh（含≤5 min 复核前向），batch ≤32（训练与评估），CPU ≤4。
接受标准保持不作放宽；TEST 仅在 selection_manifest 冻结后开启一次；必须披露 F3/F4。</p>
<p style="color:#666;font-size:12px">worker 自查文件已原字节保留为 worker_preliminary_review.json（sha256 %s）。本文件由 JSON 自动生成。</p>
</body></html>""" % (
    e(d["reviewed_at"]), float(d["gpu_forward_wall_s"]),
    e(d["verdict_detail"]["integrity"]), e(d["verdict_detail"]["compliance"]), e(d["verdict_detail"]["scientific_gain"]),
    e(d["verdict_detail"]["structural_readiness"]), e(d["verdict_detail"]["empirical_training_continuation"]),
    e(d["verdict_detail"]["production_pointer"]), d["n_checks_ok"], d["n_checks"], rows, find,
    "".join("<li>%s</li>" % e(v) for v in d["corrections_carried_forward"].values()),
    e(d["worker_preliminary_review"]["sha256"]))
(REP30 / "independent_review.html").write_text(h, encoding="utf-8")
print("html bytes", len(h.encode("utf-8")))
