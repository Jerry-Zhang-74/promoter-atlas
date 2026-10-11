#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Chinese HTML + text report for batch08 S label audit (read-only sources)."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
R = ROOT / "reports" / "breakthrough_20261002" / "batch08"
M = ROOT / "models" / "breakthrough_20261002" / "batch08"

def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))

def main():
    s = load("audit_summary.json")
    meta = json.loads((M / "transform_metadata.json").read_text(encoding="utf-8"))
    sci = load("scale_invariance_results.json")
    pc = s["pipeline_counts"]
    ta = s["tie_audit_summary"]
    es = s["es_sensitivity"]
    bt = s["B_tie_artifact"]
    tr = s["split_rows"]
    aff = meta["affine_transform"]

    fig_path = R / "fig_ES_conventions.png"
    fig_b64 = ""
    if fig_path.exists():
        fig_b64 = base64.b64encode(fig_path.read_bytes()).decode("ascii")

    ms_rows = ""
    for x in s["mask_stats"]:
        ms_rows += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%.6f</td><td>%.6f</td>"
                    "<td>%.6f</td><td>%.6f</td><td>%.6f</td></tr>") % (
                        x["split"], x["label"], x["n_genes"], x["min"], x["max"],
                        x["range"], x["sd"], x["variance"])

    split_rows_html = ""
    for x in tr:
        split_rows_html += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                            "<td>%s</td><td>%s</td><td>%s</td></tr>") % (
            x["split"], x["records"], x["labeled_E"], x["unlabeled"], x["s_defined"],
            x["s_missing_labeled"], x["unique_genes_total"], x["unique_genes_S"])

    sci_checks = ""
    for c in sci["checks"]:
        sci_checks += ("<tr><td>%s</td><td>%s</td><td>%s</td></tr>") % (
            ("通过" if c["pass"] else "失败"), c["test"], c["detail"])

    html_head = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>batch08 Score_S 标签审计报告 (v6.9.0-labelaudit.rc1)</title>
<style>
body{{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;max-width:980px;margin:0 auto;padding:16px;line-height:1.6;color:#1a1a1a}}
h1{{font-size:1.5em;border-bottom:2px solid #2c5aa0;padding-bottom:6px}}
h2{{font-size:1.2em;color:#2c5aa0;margin-top:1.4em;border-left:4px solid #2c5aa0;padding-left:8px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:0.92em}}
th,td{{border:1px solid #ccc;padding:5px 8px;text-align:left}}
th{{background:#eef3fa}}
.tag{{display:inline-block;background:#2c5aa0;color:#fff;border-radius:4px;padding:1px 8px;font-size:0.85em;margin-right:6px}}
.warn{{background:#fdf3e7;border-left:4px solid #e69a3a;padding:8px 12px;margin:8px 0}}
.ok{{background:#eef9ee;border-left:4px solid #3aa03a;padding:8px 12px;margin:8px 0}}
img{{max-width:100%;border:1px solid #ddd}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
</style></head><body>

<h1>batch08 · 遗留 Score_S 标签审计报告</h1>
<p><span class="tag">candidate_version = v6.9.0-labelaudit.rc1</span>
<span class="tag">track = label_audit_S</span>
<span class="tag">日期 = 2026-10-03</span></p>

<div class="ok"><b>结论（一句话）：</b>
遗留 <code>Score_S</code> 的定义、缺失掩码、并列/精度约定均已核对并复算一致；
<code>Score_S</code> 是<b>跨组织可检出表达离散度</b>（S_raw = -MAD(log2(1+TPM))，仅在可检出组织上计算，B_raw&gt;=0.30 才定义），
<b>不是</b>纵向/时序稳定性；它在全体 58988 个基因上做全局稳定序秩（未定义者按 -inf 沉底后清空），
因此已定义值落在 <b>0.6788-1.0</b>。真实训练中未定义的 S 目标被 <code>m_esb[:,1]=0</code> 掩码，
<b>不会</b>影响梯度。仿射重标定只改变损失尺度/梯度权重与 E*S 复合几何，<b>不改变排序</b>。
本批次<b>未</b>做任何预测器训练或覆盖率门控，也<b>无</b>新生物学验证声明。</div>

<h2>1. 唯一有界问题与回答</h2>
<p><b>问题：</b>遗留 <code>Score_S</code> 的定义、缺失掩码、并列/精度约定与损失尺度后果是什么？
未来 from-scratch S 实验可用哪些向后兼容、全程可追溯的替代标签？</p>
<ul>
<li><b>定义：</b><code>E_raw = mean(log2(1+TPM))</code>（54 组织）；<code>B_raw = 可检出组织占比（TPM&gt;=1）</code>；
<code>S_raw = -MAD(log2(1+TPM))</code>，仅在可检出组织（TPM&gt;=1）上计算；
S 仅当 <code>B_raw &gt;= 0.30</code> 且 <code>可检出组织数 &gt;= 3</code> 时定义（54 组织下 B&gt;=0.30 使可检出&gt;=17，故“&gt;=3”条款实际冗余）。</li>
<li><b>归一化：</b><code>gene_id</code> 取“.”前的 base；基因x组织重复按“存在零与非零时取最大非零，否则取均值”折叠。</li>
<li><b>缺失掩码：</b>未定义基因在规范表 <code>Score_S</code> 列为空（<code>S_defined_main=0</code>）；v6 输入中 <code>label_mask</code> 以 E 是否缺失为据，
真实 S 掩码 <code>m_S = (label_mask==1) 且 Score_S 非空</code>；损失用 <code>m_S</code> 乘方差，未定义行贡献严格为 0。</li>
<li><b>并列/精度：</b>全精度 S_raw（来自 6 位小数 mean_tpm 的浮点重算）<b>无任何真并列</b>（0 个全精度并列组）；
规范表 6 位小数 S_raw 的 {ta['sixdec_tie_groups']} 个并列组（{ta['sixdec_tie_genes']} 个基因）全部是<b>舍入产物</b>，不参与 <code>Score_S</code> 排序。</li>
<li><b>损失尺度：</b>若目标与预测都做仿射 T(x)=(x-a)/b，则 MAE'=MAE/b、MSE'=MSE/b^2、R^2 与 Spearman 不变；
只改目标不改预测则 MAE/MSE/R^2 均改变（Spearman 仍不变，因 b&gt;0 单调）。</li>
</ul>

<h2>2. 遗留 Score_S 精确定义（已核对源脚本）</h2>
<table><tr><th>项</th><th>值</th></tr>
<tr><td>源脚本</td><td><code>runs/scripts/build_candidate_e_labels.py</code></td></tr>
<tr><td>S_raw 公式</td><td>-MAD(log2(1+mean_tpm))，仅在可检出组织（TPM&gt;=1）上</td></tr>
<tr><td>S 资格</td><td>B_raw&gt;=0.30 且 可检出组织&gt;=3</td></tr>
<tr><td>Score_S</td><td>全体 58988 基因上的稳定升序序秩/(58988-1)；未定义 S_raw 以 -inf 沉底后清空</td></tr>
<tr><td>并列约定</td><td>稳定排序（等值按 gene_id 字典序），非平均秩</td></tr>
<tr><td>Score_HKG</td><td>(Score_E x Score_S x Score_B)^(1/3)；仅作为遗留排序列，本批次不采用为新目标</td></tr>
</table>

<h2>3. 关键数值</h2>
<table><tr><th>量</th><th>值</th></tr>
<tr><td>规范基因数 N</td><td>{s['N']}</td></tr>
<tr><td>已定义 Score_S 基因数 M</td><td>{s['M']}</td></tr>
<tr><td>a = (N-M)/(N-1)</td><td>{s['a']:.15f}</td></tr>
<tr><td>b = (M-1)/(N-1)</td><td>{s['b']:.15f}</td></tr>
<tr><td>Score_S 已定义区间</td><td>[{s['score_s_defined_range'][0]:.6f}, {s['score_s_defined_range'][1]:.6f}]</td></tr>
<tr><td>S_raw 已定义区间</td><td>[{s['s_raw_defined_range'][0]:.6f}, {s['s_raw_defined_range'][1]:.6f}]</td></tr>
<tr><td>S_affine 区间</td><td>[{s['affine_range'][0]:.2e}, {s['affine_range'][1]:.6f}]</td></tr>
<tr><td>Score_S 复算最大误差</td><td>{s['max_score_s_recompute_err']:.3e}（6 位小数舍入容差）</td></tr>
<tr><td>S_raw 6 位小数最大误差</td><td>{s['s_raw_max_6dec_err']:.3e}</td></tr>
</table>

<h2>4. 缺失掩码：缺失 vs 合法零（真实 loader 证据）</h2>
<p>用真实 <code>AtlasDataset</code>（非 mock）对几条实际映射行做了 CPU 探针：</p>
<table><tr><th>情形</th><th>label_mask</th><th>m_E</th><th>m_S</th><th>m_B</th><th>y_S</th></tr>
<tr><td>已定义 S 行</td><td>1</td><td>1</td><td>1</td><td>1</td><td>实际值 0.819621</td></tr>
<tr><td>未定义 S 行（E 有值、S 空）</td><td>1</td><td>1</td><td><b>0</b></td><td>1</td><td>0.0（被掩码）</td></tr>
<tr><td>label_mask=0 行</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr>
<tr><td>合法零（Score_E=0.0 已定义）</td><td>1</td><td>1</td><td>-</td><td>-</td><td>-（m_E=1, y_E=0.0，被计入）</td></tr>
<tr><td>合法零（Score_B=0.0 已定义）</td><td>1</td><td>-</td><td>-</td><td>1</td><td>-（m_B=1, y_B=0.0，被计入）</td></tr>
</table>
<div class="ok"><b>结论：</b>区分“缺失”与“合法零”靠的是 <code>m_esb</code> 掩码（缺失 m=0 / 合法零 m=1、y=0），
而非 y==0 哨兵值。未定义 S 行的 (pred-target)^2 x m_S 恒为 0，<b>不可能</b>推动 Score_S 头的梯度。</div>
"""
    html_parts = [html_head]
    print("head built", len(html_head))

    html_mid = f"""
<h2>5. 管线分母追溯（不同分母，明确列出）</h2>
<table><tr><th>层级</th><th>记录/基因数</th><th>S 相关</th></tr>
<tr><td>规范基因表</td><td>{pc['canonical_genes']} 基因</td><td>S 已定义 {pc['canonical_S_defined']}</td></tr>
<tr><td>v6 输入 · 人类记录</td><td>{pc['human_records_v6_input']} 条</td><td>有 E 标签 {pc['human_labeled_E_records']}；无标签 {pc['human_unlabeled_records']}</td></tr>
<tr><td>v6 输入 · 人类 S 掩码</td><td>S 已定义记录 {pc['human_S_defined_records']}；有标签但 S 缺失 {pc['human_S_missing_among_labeled_records']}</td><td>唯一 S 基因 {pc['human_unique_S_genes']}（规范 18947 中 178 个无人类启动子记录）</td></tr>
<tr><td>附带参考（symbol 标注清单）</td><td>gene_split_manifest {pc['gene_split_manifest_genes']} 基因</td><td>S 已定义 {pc['gene_split_manifest_S_defined']}</td></tr>
</table>
<p>按 split 的记录/基因计数：</p>
<table><tr><th>split</th><th>记录</th><th>有 E</th><th>无标签</th><th>S 已定义</th><th>有标签但 S 缺失</th><th>唯一基因总数</th><th>唯一 S 基因</th></tr>
{split_rows_html}</table>

<h2>6. 精度与并列审计</h2>
<table><tr><th>量</th><th>值</th></tr>
<tr><td>全精度并列组</td><td>{ta['full_precision_tie_groups']}（真并列 = 0）</td></tr>
<tr><td>全精度并列基因</td><td>{ta['full_precision_tie_genes']}</td></tr>
<tr><td>6 位小数并列组</td><td>{ta['sixdec_tie_groups']}（全部为舍入产物）</td></tr>
<tr><td>6 位小数并列基因</td><td>{ta['sixdec_tie_genes']}</td></tr>
<tr><td>6 位小数唯一值（18947 中）</td><td>{ta['unique_sixdec_values_among_defined']}</td></tr>
</table>
<div class="warn"><b>重要：</b>6 位小数的 <code>S_raw</code> 列<b>无法</b>重建原始全精度排序——若对 6 位小数做平均秩，
与遗留 Score_S 排序相比有 247 个基因改变相对次序（见 scale_invariance 测试）。
因此不宣称“舍入后的 S_raw 秩复现原始排序”。<b>保留排序的参考是遗留 Score_S 本身</b>，不是 6 位小数 S_raw。</div>

<h2>7. 仿射兼容标签 S_affine 与 S_defined_rank</h2>
<ul>
<li><b>S_affine = (Score_S - a)/b</b>，其中 a=(N-M)/(N-1)={s['a']:.6f}，b=(M-1)/(N-1)={s['b']:.6f}；
逆变换 <b>Score_S = a + b x S_affine</b>。仅对 18947 个已定义基因计算，未定义保持缺失。</li>
<li>因 Score_S 以 6 位小数存储，最小已定义值 0.678811 略高于精确 a={s['a']:.15f}，
故 S_affine 最小值为 {s['affine_range'][0]:.2e}（端点舍入容差，非强制精确）。</li>
<li><b>S_defined_rank</b>：在 M=18947 个已定义基因内对<b>全精度</b> S_raw 做显式平均秩并列，quantile=(rank-1)/(M-1) 取值 [0,1]；
与遗留“稳定 gene_id 序”并列约定不同；M&gt;1 故无除零。</li>
<li>因为全精度 S_raw <b>无真并列</b>，三种 S 约定（遗留稳定秩 / S_affine / S_defined_rank）彼此严格单调，两两 Spearman = 1.0（排序完全一致）；它们只差<b>尺度</b>。</li>
</ul>
<p>全部 58988 行（含空白/理由列）已写入 <code>data/breakthrough_20261002/batch08/label_audit_all_genes.tsv</code>，
新列：<code>S_raw_full / S_raw_6dec / S_raw_6dec_err / S_affine / S_affine_exact / S_affine_inverse_check / S_defined_rank_int / S_defined_rank / S_eligibility_reason</code>。
未覆盖 <code>Score_S/E/B/Score_HKG</code> 列。绘制主图仅用 18947 个已定义基因，<b>不</b>宣称全体基因均有有效 S。</p>

<h2>8. 损失尺度后果（数值测试，{sci['n_pass']}/{sci['n_total']} 通过）</h2>
<p>在 paired 变换（目标与预测同时 T(x)=(x-a)/b，b={s['b']:.6f}）下：</p>
<ul>
<li>MAE' = MAE / b（约 1/{s['b']:.4f} 倍）</li>
<li>MSE' = MSE / b^2（约 1/{s['b']**2:.4f} 倍）</li>
<li>R^2 不变；Spearman 不变</li>
</ul>
<p>只改目标不改预测时：MAE/MSE/R^2 均改变（R^2 甚至变负），Spearman 仍不变（b&gt;0 单调）。</p>
<table><tr><th>结果</th><th>测试</th><th>细节</th></tr>
{sci_checks}</table>
<div class="warn"><b>注意：</b>“S 的 SD 从 ~0.0926 变为 ~0.2884”是<b>标准差</b>变化，<b>方差</b>从 ~0.00858 变为 ~0.08316（约 1/b^2 倍）。
不要把 SD 与方差混为一谈。范围扩大本身<b>不代表</b>科学收益；且本批次<b>未</b>改任何损失权重（真实损失 s_weight=2.0），
也<b>未见</b>梯度平衡修复的证据。</div>
"""
    html_parts.append(html_mid)
    print("mid built", len(html_mid))

    html_tail = f"""
<h2>9. 各 split 掩码上的 raw vs affine 尺度（基因级去重）</h2>
<table><tr><th>split</th><th>标签</th><th>n_genes</th><th>min</th><th>max</th><th>range</th><th>SD</th><th>VARIANCE</th></tr>
{ms_rows}</table>

<h2>10. E x S 排序敏感性（以 B_raw 为资格门控）</h2>
<ul>
<li>在 {es['n_S_defined_for_ES']} 个 S 已定义基因上，E x S_legacy 与 E x S_affine 的 Spearman = <b>{es['spearman_ES_legacy_vs_affine']:.4f}</b>，
top-{es['topk']} 重叠 = <b>{es['topk_overlap_legacy_vs_affine']}/{es['topk']}</b>。</li>
<li>E x S_legacy 与 E x S_defined_rank 的 Spearman = {es['spearman_ES_legacy_vs_rank']:.4f}，top-{es['topk']} 重叠 = {es['topk_overlap_legacy_vs_rank']}/{es['topk']}。</li>
<li>这说明 S 重标定会改变 E x S 的目标权重与候选排序，<b>不是</b>生物学验证或“更优候选”的证据。</li>
</ul>

<h2>11. B 并列产物（仅量化，本批次不改）</h2>
<p>B_raw 是 k/54 型分数，仅 {bt['B_raw_distinct_values']} 个不同值；Score_B 对相等 B_raw 用稳定 gene_id 序并列，
造成 {bt['B_tie_groups']} 个并列组、{bt['B_tied_genes_beyond_first']} 个基因“首位之外”的任意序，最大组 {bt['max_B_group_size']}。
遗留 Score_HKG=(E x S x B)^(1/3) 仅作遗留列保留，不作为新目标。</p>

<h2>12. 局限与结论</h2>
<ul>
<li>S 是<b>跨组织可检出表达离散度</b>，不是纵向/时序稳定性，也不是转染后稳定性。</li>
<li>标签审计与既往神经/覆盖率实验<b>互相独立</b>；本批次无预测器训练、无覆盖率门控、无新生物学验证。</li>
<li>S 仅人类使用；非人类物种只迁移 ortholog_E（Score_E），<b>不</b>迁移 S。</li>
<li>历史全表分位标签是<b>全局参考的固定目标</b>，非新独立验证；未来训练需用 batch05 修正的组件切分与 train-only 参考变换。</li>
<li>PASS 指<b>审计可复现</b>，不代表模型变好。</li>
</ul>

<h2>13. 图：E x S 两种坐标约定 + raw 迹线（仅 18947 已定义基因）</h2>
<img src="data:image/png;base64,{fig_b64}" alt="E x S conventions">

<h2>14. 输出清单与 SHA 注册表</h2>
<ul>
<li><code>data/breakthrough_20261002/batch08/label_audit_all_genes.tsv</code>（58988 行全量标签 + 新列）</li>
<li><code>data/breakthrough_20261002/batch08/tie_audit.tsv</code>（并列组审计）</li>
<li><code>data/breakthrough_20261002/batch08/pipeline_mask_counts.tsv</code>（分母追溯）</li>
<li><code>data/breakthrough_20261002/batch08/mask_loss_stats.tsv</code>（尺度统计）</li>
<li><code>models/breakthrough_20261002/batch08/transform_metadata.json / .tsv</code>（变换元数据 + 逆变换）</li>
<li><code>reports/breakthrough_20261002/batch08/audit_summary.json / scale_invariance_results.json / sha_registry.tsv</code></li>
<li><code>runs/breakthrough_20261002/code/batch08/*.py</code>（审计/探针/测试/报告脚本）</li>
</ul>
<p>只读源 SHA256 见 <code>sha_registry.tsv</code>。所有旧 <code>Score_S/E/B/Score_HKG</code> 列与旧图/模型<b>未</b>被覆盖。</p>

<hr><p style="color:#777;font-size:0.85em">本报告为标签审计（label audit），非模型提升声明。版本 v6.9.0-labelaudit.rc1；v6.3 / cellline v6.4 指针保持不变。</p>
</body></html>"""
    html_parts.append(html_tail)
    html = "".join(html_parts)
    (R / "report.html").write_text(html, encoding="utf-8")
    print("html written", len(html))

    txt_tpl = """batch08 Score_S 标签审计报告（v6.9.0-labelaudit.rc1，2026-10-03）
========================================================================

一、结论（一句话）
遗留 Score_S 的定义、缺失掩码、并列/精度约定均已核对并复算一致。Score_S 是跨组织可检出
表达离散度（S_raw = -MAD(log2(1+TPM))，仅在可检出组织上计算，B_raw>=0.30 才定义），不是
纵向/时序稳定性。它在全体 58988 基因上做全局稳定序秩（未定义按 -inf 沉底后清空），因此
已定义值落在 0.6788-1.0。真实训练中未定义 S 目标被 m_esb[:,1]=0 掩码，不会影响梯度。
仿射重标定只改变损失尺度/梯度权重与 E*S 复合几何，不改变排序。本批次无预测器训练、无
覆盖率门控、无新生物学验证声明。

二、关键数值
N={N}；M={M}；a={a}；b={b}
Score_S 已定义区间 [{sslo:.6f}, {sshi:.6f}]
S_raw 已定义区间 [{srlo:.6f}, {srhi:.6f}]
S_affine 区间 [{aflo:.2e}, {afhi:.6f}]
Score_S 复算最大误差 {err:.3e}（6 位小数舍入容差）

三、管线分母追溯
规范基因 {canonical_genes}（S 已定义 {canonical_S_defined}）
v6 输入人类记录 {human_records}（有 E 标签 {labeled}，无标签 {unlabeled}）
S 掩码：已定义记录 {s_defined}；有标签但 S 缺失 {s_missing}；唯一 S 基因 {uniq_s_genes}
split（记录/唯一 S 基因）：train {tr0r}/{tr0g}，val {tr1r}/{tr1g}，test {tr2r}/{tr2g}
附带参考 gene_split_manifest：{gsm_genes} 基因 / {gsm_s} S。

四、缺失 vs 合法零（真实 AtlasDataset 探针）
已定义 S 行 m_S=1 y_S=实际值；未定义 S 行 m_S=0 y_S=0（被掩码）；label_mask=0 行 m=0；
合法零(Score_E=0.0 / Score_B=0.0) m=1 y=0（被计入）。结论：掩码 m 而非 y==0 哨兵区分缺失与零，
未定义 S 行不可能推动 Score_S 头梯度。

五、精度与并列
全精度并列组 0（真并列）；6 位小数并列组 {sixdec_groups}（全为舍入）；6 位小数唯一值 {sixdec_uniq}。
6 位小数 S_raw 列无法重建原始全精度排序（平均秩下 247 基因相对次序改变），保留排序的参考是遗留 Score_S。

六、S_affine 与 S_defined_rank
S_affine=(Score_S-a)/b（a={a:.6f}, b={b:.6f}），逆 Score_S=a+b*S_affine；未定义保持缺失；
S_affine 最小值 {aflo:.2e}（端点舍入容差）。S_defined_rank 在 18947 已定义基因内对全精度 S_raw
做平均秩并列，quantile=(rank-1)/(M-1)。因无真并列，三种 S 约定两两 Spearman=1.0（排序一致），只差尺度。

七、损失尺度（{scipass}/{scitotal} 通过）
paired 变换：MAE'/MAE=1/b={invb:.6f}；MSE'/MSE=1/b^2={invb2:.6f}；R2 与 Spearman 不变。
只改目标：MAE/MSE/R2 均变，Spearman 不变。注意 ~0.0926->~0.2884 是 SD 变化，方差从 ~0.00858
变 ~0.08316（约 1/b^2），不要把 SD 与方差混淆；范围扩大不代表科学收益，未见梯度平衡修复证据。

八、E*S 敏感性（B_raw 门控）
E*S_legacy vs E*S_affine Spearman={spes:.4f}，top-{topk} 重叠 {ovlap}/{topk}。
S 重标定改变 E*S 目标权重，非生物学验证。

九、局限与结论
S 是跨组织可检出表达离散度，非纵向/时序稳定性；S 仅人类使用（非人类只迁 ortholog_E）。
历史全表分位标签是全局参考固定目标；未来训练需 batch05 修正组件切分 + train-only 参考变换。
PASS=审计可复现，非模型变好。

（详见 report.html；全量标签表 data/breakthrough_20261002/batch08/label_audit_all_genes.tsv；
只读源 SHA 见 sha_registry.tsv）"""
    txt = txt_tpl.format(
        N=s['N'], M=s['M'], a=s['a'], b=s['b'],
        sslo=s['score_s_defined_range'][0], sshi=s['score_s_defined_range'][1],
        srlo=s['s_raw_defined_range'][0], srhi=s['s_raw_defined_range'][1],
        aflo=s['affine_range'][0], afhi=s['affine_range'][1],
        err=s['max_score_s_recompute_err'],
        canonical_genes=pc['canonical_genes'], canonical_S_defined=pc['canonical_S_defined'],
        human_records=pc['human_records_v6_input'], labeled=pc['human_labeled_E_records'],
        unlabeled=pc['human_unlabeled_records'], s_defined=pc['human_S_defined_records'],
        s_missing=pc['human_S_missing_among_labeled_records'], uniq_s_genes=pc['human_unique_S_genes'],
        tr0r=tr[0]['records'], tr0g=tr[0]['unique_genes_S'],
        tr1r=tr[1]['records'], tr1g=tr[1]['unique_genes_S'],
        tr2r=tr[2]['records'], tr2g=tr[2]['unique_genes_S'],
        gsm_genes=pc['gene_split_manifest_genes'], gsm_s=pc['gene_split_manifest_S_defined'],
        sixdec_groups=ta['sixdec_tie_groups'], sixdec_uniq=ta['unique_sixdec_values_among_defined'],
        invb=1.0/s['b'], invb2=1.0/(s['b']**2),
        scipass=sci['n_pass'], scitotal=sci['n_total'],
        spes=es['spearman_ES_legacy_vs_affine'], topk=es['topk'],
        ovlap=es['topk_overlap_legacy_vs_affine'])
    (R / "report.txt").write_text(txt, encoding="utf-8")
    print("txt written", len(txt))

if __name__ == "__main__":
    main()
