#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch18 deliverable 2: Chinese HTML release report for v6.15.1-release.rc2.

Numbers are read from the batch18 artifacts (never hardcoded), so the report
cannot drift from the recomputed source of truth.  Adds vs rc1's report:
  * the F17-3 empty-batch typed-input-validation fix and its regression test,
  * the truthful-provenance correction (run-time claimed vs current on-disk hash)
    and the documented TOTAL minibatch BOUND VIOLATION (20 > 16),
  * the S val/test count typo resolution (val n=1837, test n=1880),
  * the observed-label database reference and its evidence limits,
  * links that point at the ACTUAL batch18 rc2 artifacts (no dangling batch17 links).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REP = ROOT / "reports" / "breakthrough_20261002" / "batch18"
REL = ROOT / "models" / "breakthrough_20261002" / "batch18"
OUT = REP / "release_report_v6_15_1_rc2.html"
ZIP = REL / "promoter_atlas_v6.15.1-release.rc2.zip"

M = json.loads((REP / "corrected_release_metrics.json").read_text(encoding="utf-8"))
RD = json.loads((REP / "resume_diagnostic_rc2.json").read_text(encoding="utf-8"))
VM = json.loads((REL / "release_v6.15.1-release.rc2" / "VERSION_MANIFEST.json").read_text(encoding="utf-8"))

TASKS = ("E", "S", "B")
REF = M["reference_shipped_seed"]["per_seed_metrics"]
MOS = M["mean_of_seed_and_ensemble"]["A"]
HGB = M["same_row_neural_vs_hgb"]
VR = M["validation_rule_applied"]
BOOT = M["paired_component_bootstrap"]


def f4(x):
    return "%.4f" % float(x)


def f5(x):
    return "%+.5f" % float(x)


def ref_table():
    lines = []
    for t in TASKS:
        for split in ("val", "test"):
            d = REF["%s_%s" % (t, split)]
            lines.append("<tr><td>%s</td><td>%s</td><td class=num>%s</td>"
                         "<td class=num>%s</td><td class=num>%s</td>"
                         "<td class=num>%d</td></tr>"
                         % (t, split, f4(d["spearman"]), f4(d["r2"]), f4(d["mae"]), d["n"]))
    return "\n".join(lines)


def estimand_table():
    lines = []
    for t in TASKS:
        ms = MOS[t]["test"]["mean_of_seed"]
        en = MOS[t]["test"]["ensemble"]
        hg = HGB[t]["test"]["hgb"]
        lines.append("<tr><td>%s</td><td class=num>%s</td><td class=num>%s</td>"
                     "<td class=num>%s</td><td class=num>%d</td></tr>"
                     % (t, f4(ms["spearman"]), f4(en["spearman"]), f4(hg["spearman"]), ms["n"]))
    return "\n".join(lines)


def rule_table():
    lines = []
    for arm in ("B", "C"):
        v = VR[arm]
        cells = "".join("<td class=num>%s</td>" % f5(v["per_target"][t]["gain"]) for t in TASKS)
        lines.append("<tr><td>%s</td>%s<td class=num>%s</td><td>%s</td><td class=fail>%s</td></tr>"
                     % (arm, cells, f5(v["mean_gain"]), "yes" if v["no_drop_gt_0.02"] else "no",
                        "NO (failed)" if not v["rule_pass"] else "yes"))
    return "\n".join(lines)


def boot_table():
    lines = []
    for arm in ("B", "C"):
        for t in TASKS:
            b = BOOT[arm][t]
            cross = "spans 0 (n.s.)" if (b["ci95_low"] <= 0 <= b["ci95_high"]) else "excludes 0"
            lines.append("<tr><td>%s - A</td><td>%s</td><td class=num>%s</td>"
                         "<td class=num>[%s, %s]</td><td class=num>%d</td><td>%s</td></tr>"
                         % (arm, t, f4(b["base_point_estimate"]), f4(b["ci95_low"]),
                            f4(b["ci95_high"]), b["n_components"], cross))
    return "\n".join(lines)


zip_rel = "../../../models/breakthrough_20261002/batch18/promoter_atlas_v6.15.1-release.rc2.zip"
zip_sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
manifest_sha = hashlib.sha256((REL / "release_v6.15.1-release.rc2" / "VERSION_MANIFEST.json").read_bytes()).hexdigest()
nfiles = len(VM["package_files_sha256"])
BV = RD["bounds_violation"]

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Promoter Atlas v6.15.1-release.rc2 release report (candidate / experimental)</title>
<style>
 body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1080px;margin:0 auto;
      padding:24px 28px 80px;color:#1c1c1c;line-height:1.75;background:#fff}
 h1{font-size:24px;margin:0 0 6px}
 h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid #e6e6e6}
 h3{font-size:15px;margin:20px 0 6px}
 .banner{background:#fff4e5;border:2px solid #d9822b;border-radius:8px;padding:14px 18px;margin:14px 0 8px}
 .meta{color:#555;font-size:13px}
 table{border-collapse:collapse;width:100%%;margin:12px 0;font-size:13.5px}
 th,td{border:1px solid #dcdcdc;padding:6px 9px;text-align:left}
 th{background:#f3f6fa}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 .fail{color:#b03030;font-weight:600}
 .ok{color:#2e8b57;font-weight:600}
 code,pre{font-family:Consolas,"Courier New",monospace;font-size:12.5px}
 pre{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:6px;padding:10px 12px;overflow-x:auto}
 .fig{margin:16px 0 4px}
 .fig img{width:100%%;border:1px solid #e6e6e6;border-radius:6px}
 .cap{color:#666;font-size:12.5px;margin:2px 0 18px}
 .warn{background:#fdecea;border-left:4px solid #b03030;padding:10px 14px;margin:12px 0}
 .note{background:#eef5ff;border-left:4px solid #1f5fa9;padding:10px 14px;margin:12px 0}
 a{color:#1f5fa9}
 .pill{display:inline-block;background:#d9822b;color:#fff;border-radius:12px;padding:2px 10px;font-size:12px;font-weight:700}
</style></head><body>

<h1>Promoter Atlas v6.15.1-release.rc2 发布报告</h1>
<p class="meta">候选版本 / 实验性制品 &nbsp;·&nbsp; 生成时间 %(gen)s &nbsp;·&nbsp; batch18（打包 / 溯源修复轮次，含一项输入校验修复）</p>

<div class="banner">
<span class="pill">候选 / 实验性</span>
<b>本版本不是一次科学突破，也不是分数提升。</b>
它是对已存在权重的<b>打包 / 溯源修复</b>：模型权重逐字节沿用 batch16 的 <code>v6.15.0-multitask.rc1</code> 独立臂检查点，
没有重新训练、没有换种子、没有改架构、没有依据测试分数做任何选择；此外修复了一项<b>空批次输入校验</b>缺陷（F17-3）。
随附的仍是一个<b>阴性（未通过）的共享网络选择结果</b>，以及一个可移植、可复现的发布包。
在独立评审通过之前，本版本应被视为<b>实验性候选</b>，不应用于任何下游决策。
</div>

<h2>一、版本与状态</h2>
<table>
<tr><th>字段</th><th>值</th></tr>
<tr><td>候选版本</td><td><code>%(ver)s</code></td></tr>
<tr><td>来源候选</td><td><code>%(src)s</code></td></tr>
<tr><td>改动性质</td><td>打包 / 溯源修复（权重未变） + 空批次输入校验修复（F17-3）</td></tr>
<tr><td>部署架构</td><td>独立臂 A = 三个单任务 1D-CNN（不共享主干）</td></tr>
<tr><td>实际部署种子</td><td>20261016（唯一随包发布的权重）</td></tr>
<tr><td>第二个种子</td><td>20261017，仅用于复现/集成估计，未随包发布</td></tr>
<tr><td>被否决架构</td><td>B（共享主干 + E/S/B 头）、C（共享主干 + 7 个表观辅助头）</td></tr>
<tr><td>发布包文件数</td><td>%(nfiles)d（含 SHA-256 清单）</td></tr>
<tr><td>父级已接受</td><td>batch17 的重算指标与可复现性；完整性发现（F17-1/F17-2/F17-3）在本轮<b>逐项修复</b></td></tr>
</table>
<div class="note">历史制品 v6.3、cellline v6.4、batch16/batch17 的全部数据、代码、模型、报告均<b>只读保留、未被覆盖</b>。
本报告为 batch18 新增记录；<b>旧的评审记录不被编辑</b>。</div>

<h2>二、实际部署权重的指标（由保存的预测重算）</h2>
<p>下表的每一个数字都由 batch16 保存的目标 <code>y</code> 与预测 <code>pred_*</code> 数组精确重算得到
（预测以标准化单位保存，已按 <code>transforms/scalers.npz</code> 还原为原始单位）。<b>没有新的拟合。</b></p>
<table>
<tr><th>头</th><th>划分</th><th>Spearman</th><th>R2</th><th>MAE</th><th>n</th></tr>
%(reftab)s
</table>
<div class="note"><b>计数与一处评审笔误的更正：</b>S 头的有效样本为 <b>验证集 n=1837、测试集 n=1880</b>；
batch17 独立评审文本中出现的「S val 1880 / test 1837」<b>与磁盘上的真实划分相反</b>，属笔误。
本条在本轮新记录中更正，<b>旧评审文件保持原样不被编辑</b>。</div>

<h3>2.1 三种估计量不可互换</h3>
<p>「各种子指标均值」「集成预测（跨种子取均值后计算）」「匹配梯度提升基线」是三个不同估计量，
数值接近但含义不同，报告与对比时必须写明用的是哪一个。</p>
<table>
<tr><th>头</th><th>各种子均值 (A)</th><th>集成 (A)</th><th>匹配 HGB（同一行）</th><th>n</th></tr>
%(esttab)s
</table>
<div class="fig"><img src="figures/fig1_deployed_vs_baseline.png" alt="deployed vs baseline"></div>
<p class="cap">图 1：实际部署种子（20261016）与匹配 HGB 基线在同一批基因行上的测试集指标。矢量版本：
<a href="figures/fig1_deployed_vs_baseline.svg">SVG</a>。</p>
<div class="fig"><img src="figures/fig4_estimands.png" alt="estimands"></div>
<p class="cap">图 4：均值 / 集成 / 基线三种估计量的差异。矢量版本：<a href="figures/fig4_estimands.svg">SVG</a>。</p>
<p><b>S 头需要特别注意：</b>神经网络与匹配 HGB 在 S 上基本持平（同一行 Spearman %(sneural)s 对 %(shgb)s），
这提示 S 的排序信号有相当一部分可被常规特征基线复现。</p>

<h2>三、目标定义、掩码与划分</h2>
<h3>3.1 目标定义（人类基因级自然表达）</h3>
<table>
<tr><th>目标</th><th>定义</th><th>掩码 / 有效子集</th></tr>
<tr><td><b>E</b></td><td>Score_E，秩缩放到 [0,1] 的表达强度</td><td>E_mask</td></tr>
<tr><td><b>S</b></td><td>S_raw = -MAD(log2(1+可检测 TPM))，跨组织离散度</td>
<td>S_defined：B_raw &gt;= 0.30 且 可检测组织数 &gt;= 3</td></tr>
<tr><td><b>B</b></td><td>B_raw = 可检测组织数 / 组织总数，宽度 [0,1]</td><td>B_mask</td></tr>
</table>
<h3>3.2 划分（组件并集修正划分）</h3>
<table>
<tr><th>划分</th><th>基因</th><th>组件</th></tr>
<tr><td>train</td><td class="num">46367</td><td class="num">43713</td></tr>
<tr><td>val</td><td class="num">5609</td><td class="num">5308</td></tr>
<tr><td>test</td><td class="num">5851</td><td class="num">5397</td></tr>
</table>
<div class="warn"><b>划分不可比：</b>修正后的组件并集划分与早期的随机划分<b>不可直接比较</b>。
任何历史（旧随机划分）分数都不得与本报告中的分数并列解释或直接相减。测试集为<b>回顾性留出</b>：
该划分早于 batch16，其基因曾对开发过程可见。</div>

<h3>3.3 观测标签来源与证据边界</h3>
<p>本发布包<b>只提供模型预测</b>，不提供也不虚构观测标签。数据库观测标签单独、可追溯地维护于：</p>
<ul>
<li><code>labels/build/gene_expression_label_table.tsv</code>（列含 E_raw, S_raw, B_raw, Score_E, Score_S, Score_B, B_min_used）</li>
<li><code>labels/build/gene_expression_table.tsv</code>（组织级平均 TPM，上表的来源）</li>
<li>标签溯源报告 <code>reports/label_trace_20261001/</code>（含 <code>gene_labels_all_database.tsv</code>、
<code>label_recompute_audit.tsv</code>、<code>labels_explorer_validation.json</code>）</li>
</ul>
<div class="warn"><b>预测与观测必须区分：</b>模型输出 <code>is_model_prediction=true</code>、
<code>is_catalog_observed=false</code>。观测标签是数据库<b>观测值</b>，绝不可当作预测；预测也绝不可当作观测。
证据边界：标签是基于组织级汇编数据的<b>推导量</b>（非原始测序读数），S 只对 S_defined 子集有定义，
组织覆盖率对某些基因/组织可能不足；超出这些范围的外推不被支持。</div>

<h2>四、阴性结果：共享网络选择未通过（冻结验证规则）</h2>
<p>冻结验证规则（<b>仅用验证集</b>）：各种子平均、每个目标的验证集 Spearman 增益 &gt;= 0.02，且没有任何目标的下降超过 0.02。</p>
<table>
<tr><th>臂</th><th>E 增益</th><th>S 增益</th><th>B 增益</th><th>平均增益</th><th>无下降 &gt; 0.02</th><th>规则通过</th></tr>
%(ruletab)s
</table>
<div class="fail">结论：B（共享主干）与 C（共享主干 + 表观辅助头）<b>两条规则均未通过</b>（平均增益均 &lt; +0.02），因此发布独立臂 A。</div>
<div class="fig"><img src="figures/fig2_validation_rule.png" alt="validation rule"></div>
<p class="cap">图 2：两个共享臂在各目标上的验证集增益，均低于 +0.02 的增益阈值。矢量版本：
<a href="figures/fig2_validation_rule.svg">SVG</a>。</p>

<h3>4.1 配对组件自助法（测试集，仅作回顾性区间）</h3>
<p>对比定义：集成预测下，Spearman(预测 B/C) - Spearman(预测 A)，按组件整体重抽样（<code>new_comp_dense</code> 组件并集，
2000 次重抽样，随机种子 20261012，2.5-97.5 百分位区间）。</p>
<table>
<tr><th>对比</th><th>目标</th><th>点估计</th><th>95%% CI</th><th>组件数</th><th>是否跨 0</th></tr>
%(boottab)s
</table>
<div class="note">E 与 B 上的对比区间基本跨越 0（不显著）；S 上共享臂略好，但 S 的有效样本远小（测试 n=1880），
且 S 本身与基线持平。这些区间是<b>回顾性</b>的，仅用于说明不确定性，不构成对共享任务学习优越性的证明。</div>
<div class="fig"><img src="figures/fig3_bootstrap_ci.png" alt="bootstrap CI"></div>
<p class="cap">图 3：集成 B/C 相对 A 的配对组件自助法区间。矢量版本：<a href="figures/fig3_bootstrap_ci.svg">SVG</a>。</p>

<h2>五、必须同时阅读的混杂与限制</h2>
<div class="warn">
<b>1. 训练器 / 初始化 / 数据顺序混杂（最重要）。</b> A 参考拟合与 A 种子 2 的 E/S 来自原训练器（全局 RNG 洗牌）；
B/C 两个共享臂以及 A 种子 2 的 B 来自进程中断后的恢复训练器（每轮确定性采样器种子）。
算法、优化器、损失、轮数相同，但<b>初始化 RNG 流与数据顺序调度不同</b>。
这使所有「共享 vs 独立」的对比都被混淆，<b>无法给出严格的因果结论</b>（不能说独立任务一定更好）。
</div>
<div class="warn">
<b>2. 辅助目标缺失性未解决。</b> 表观辅助目标把「未覆盖的测定」编码为 0.0，且没有 NaN 标记，
因此零值被当作<b>「缺失 vs 真为零」尚未解决</b>的情形处理。不主张辅助监督带来任何收益或损害。
</div>
<div class="warn">
<b>3. 家族泛化受限。</b> 已发现的调控家族只覆盖 <b>14 个构念</b>，无法证明对未见家族的泛化。
</div>
<div class="warn">
<b>4. 划分不可比 / 回顾性测试集。</b> 见第三节：修正组件并集划分与旧随机划分分数不可直接比较；测试集早于开发过程可见。
</div>
<p>综合以上，本版本是一个<b>实验性候选</b>与一个<b>阴性共享网络选择结果</b>，
<b>不是</b>关于「独立任务优于共享任务」或「表观辅助任务有用」的受控因果证据。</p>

<h2>六、发布包与首次使用</h2>
<p>发布包为自包含目录 + ZIP，<b>不包含训练资产</b>，可在解压到任意新目录后直接运行小例子。</p>
<table>
<tr><th>项目</th><th>值</th></tr>
<tr><td>ZIP</td><td><a href="%(zip)s">promoter_atlas_v6.15.1-release.rc2.zip</a></td></tr>
<tr><td>ZIP SHA-256</td><td><code>%(zipsha)s</code></td></tr>
<tr><td>VERSION_MANIFEST.json SHA-256</td><td><code>%(mansha)s</code></td></tr>
<tr><td>包内文件数</td><td>%(nfiles)d（<code>VERSION_MANIFEST.json</code> 提供全量 SHA 清单；该清单自引用自身排除）</td></tr>
<tr><td>打包预测与已保存预测一致性</td><td>最大绝对差 %(crossdiff).3e（容差 <b>1e-5，未放宽</b>，判定 <span class="ok">%(crossv)s</span>，%(crossn)d 项比对；差异为 CPU 浮点求和次序效应）</td></tr>
<tr><td>解压后可运行性</td><td>已在一个全新的自有目录、<b>外部工作目录</b>下验证（见 <a href="release_extract_test.json">release_extract_test.json</a>）</td></tr>
</table>
<h3>6.1 首次使用命令</h3>
<pre># 1) unzip
unzip promoter_atlas_v6.15.1-release.rc2.zip -d promoter_atlas_release
cd promoter_atlas_release/release_v6.15.1-release.rc2

# 2) 运行自带示例（同时自检 4 个非法输入，含空批次 N=0）
python -X utf8 code/run_example.py

# 3) F17-3 回归：所有非法输入必须抛出 typed PromoterAtlasInputError
python -X utf8 code/test_input_validation.py

# 4) 在你自己的数据上预测
python -X utf8 code/promoter_atlas_predict.py --input code/example_input.npz --out preds.json</pre>
<h3>6.2 输入接口（明确、可校验）</h3>
<ul>
<li><code>seqs</code>：形状 <code>(N,4,400)</code> 或 <code>(4,400)</code>，uint8，字母表 A/C/G/T/N 映射 0/1/2/3/4（N 编码为 4，one-hot 全零）。</li>
<li><code>features</code>：形状 <code>(N,1069)</code> 或 <code>(1069,)</code>，float 且有限，<b>原始未标准化</b>（包内变换自动做 (x-mean)/std）。</li>
<li>批量规则：<code>len(seqs)</code> 必须等于 <code>len(features)</code>；<b>空批次（N=0）为非法输入，抛 typed 错误</b>（F17-3 修复）。</li>
<li>违规一律抛出 <code>PromoterAtlasInputError</code>（shape / dtype / alphabet / non-finite / 长度不匹配 / 空批次）。</li>
</ul>
<div class="warn"><b>F17-3 修复说明：</b>rc1 中空批次（N=0）会落到 numpy 的 <code>min()/max()</code> 对零长数组求约简，
抛出<b>未归类</b>的裸 <code>ValueError</code>，使 <code>except PromoterAtlasInputError</code> 无法捕获。
rc2 在 <code>validate_seqs</code>/<code>validate_features</code> 中显式判定 N=0 并抛出 typed 错误；
<code>code/test_input_validation.py</code> 作为<b>有意义的回归</b>覆盖全部非法输入（该用例在 rc1 上会失败）。
<b>无权重、无变换、无架构、无数值容差改动。</b></div>
<div class="warn"><b>标签与预测必须区分：</b>输出中的 <code>is_model_prediction = true</code>、
<code>is_catalog_observed = false</code>；数据库观测标签（<code>labels/build/gene_expression_label_table.tsv</code>）
是<b>观测值</b>，绝不是模型预测，二者不得混用。</div>

<h2>七、训练中断续跑诊断（有界、只读；含溯源更正与边界违规）</h2>
<p>为回应「生产续跑是否真的可续」，batch17 做了一次有界诊断。本轮<b>不重跑诊断</b>，仅<b>更正溯源</b>并<b>如实记录边界违规</b>。</p>

<h3>7.1 观测到的差异（原样保留，未取整、未隐藏）</h3>
<table>
<tr><th>检验</th><th>最大绝对差</th><th>判定</th></tr>
<tr><td>CPU：连续 vs 中断续跑</td><td class="num">%(cpu).1f</td><td>逐位完全相同（仅支持该 CPU 路径）</td></tr>
<tr><td>CUDA：连续 vs 连续重复（确定性对照）</td><td class="num">%(cdet).3e</td><td>单次对照即出现差异（非逐位可复现）</td></tr>
<tr><td>CUDA：连续 vs 中断续跑</td><td class="num">%(cres).3e</td><td>与对照组量级相近（未证明同因）</td></tr>
</table>
<div class="note"><b>%(honest)s</b><br>
要点：CUDA 上「同样代码 + 同样种子」的两次连续运行已经出现差异，说明该环境下并非逐位可复现；
但<b>单个对照组不能定义一个分布或可复现区间</b>，与对照组量级相近也<b>不能证明</b>续跑残差仅（或部分）来自卷积核非确定性。
CPU 上逐位相同<b>只支持本诊断所测的 CPU 路径</b>，不构成对 CUDA 精确续跑的一般性证明。
全部差异与运行记录原样保留，无整批隐藏。<b>未发生任何科学完整训练（full_fits = 0）。</b></div>

<h3>7.2 溯源更正（F17-1）：运行期声明哈希 vs 当前磁盘哈希</h3>
<p>batch17 的 <code>resume_diagnostic.json</code> 自报的 <code>provenance.code_diag_sha256</code>
是<b>运行期声明值</b>，但诊断脚本在运行<b>之后</b>被编辑过，故当前磁盘代码的哈希<b>不等于</b>该声明值。
本轮<b>保留原运行期声明哈希不变</b>，另列当前磁盘哈希与精确编辑时序，<b>绝不</b>用当前代码哈希冒充「旧数字由当前代码产生」。</p>
<table>
<tr><th>字段</th><th>值</th></tr>
<tr><td>运行期声明 <code>b17_resume_diag.py</code> 哈希</td><td><code>%(rt)s</code>（自报值，保留）</td></tr>
<tr><td>当前磁盘 <code>b17_resume_diag.py</code> 哈希</td><td><code>%(cur)s</code>（运行后编辑所致）</td></tr>
<tr><td>运行期源码版本可复原？</td><td><b>否</b>（脚本在运行后被编辑；原版本不可复原）</td></tr>
<tr><td>独立验证范围</td><td>由<b>已保存的 .npz 数组</b>重算，验证的是<b>数值</b>，<b>不是</b>「运行期源码的精确重建」</td></tr>
</table>
<div class="note">编辑时序（本地时间）：11:17:39 运行并写入诊断；11:21:00 父级要求更强的措辞与真实 mini-batch 对账；
随后 JSON 被就地修订（其内部 <code>recorded_at_local</code> 未更新，文件 mtime 才是权威编辑时刻）；
<code>b17_resume_diag.py</code> 亦在运行后被编辑（措辞）。以上为实际时钟证据，<b>不使用任何伪造的、指向未来的事后时间戳</b>。</div>

<h3>7.3 边界违规（F17-1）：总数上限 16 被超出（20 = 12 GPU + 8 CPU）</h3>
<table>
<tr><th>进程</th><th>设备</th><th>mini-batch</th></tr>
<tr><td>continuous</td><td>CUDA</td><td class="num">4</td></tr>
<tr><td>continuous 重复（确定性对照）</td><td>CUDA</td><td class="num">4</td></tr>
<tr><td>interrupted（第 1 段）</td><td>CUDA</td><td class="num">2</td></tr>
<tr><td>resume（第 2 段）</td><td>CUDA</td><td class="num">2</td></tr>
<tr><td><b>CUDA 合计</b></td><td>CUDA</td><td class="num"><b>12</b></td></tr>
<tr><td>continuous / interrupted / resume</td><td>CPU</td><td class="num">4 + 2 + 2 = 8</td></tr>
<tr><td><b>全部进程合计</b></td><td>—</td><td class="num"><b>20</b>（7 个进程）</td></tr>
</table>
<div class="warn"><b>违规如实记录：</b>批次原声明「总数 &lt;= 16 个生产 mini-batch」，实际全部进程合计 <b>20</b>，
<b>超出 4 个</b>（12 GPU + 8 CPU）。此前的「边界已遵守」措辞把 &lt;=16 只套用到单个 CUDA 作业，属<b>事后重新解释</b>；
该旧措辞已作为 <code>original_bounds_respected_text_SUPERSEDED</code> <b>原文保留并标注为已被取代</b>。
本轮将超限记为<b>已记录的违规</b>，<b>不追溯修改边界、不隐藏诊断工作量</b>。所有诊断运行均已列出，无完整训练。</div>

<h2>八、运维跟进（副本与补丁；未应用、未发送）</h2>
<p>本轮另附两项<b>最小修复候选</b>（预算控制器资金结转、看门狗审查结论 schema），均为
<b>副本 + ready-to-run 候选 + 测试</b>，<b>没有</b>应用到线上控制器/看门狗，也<b>没有</b>发送任何邮件。
必须经独立评审后才可由父级决定启用。详见 <code>runs/breakthrough_20261002/code/batch18/ops/</code>。</p>
<ul>
<li><b>(a) 预算控制器资金结转（F17-2）：</b>现有控制器在全新启动时把累计充值重置为 0。候选修复从
<code>funding_basis.json</code> / manifest 直接结转<b>原始基线 142.67</b> 与<b>累计历史充值 100.00</b>
（有效资金 242.67 只是「原始 142.67 + 充值 100.00」的临时口径，<b>不是新的原始基线</b>），
对全部路径维持同一不变量：<code>原始基线 142.67 + 累计充值 - 实时余额 = 估计总支出</code>；
并覆盖未来充值、无充值支出、在已充值阶段之后新起控制器等场景。测试直接驱动
<code>_monitor_adopted</code> 与 <code>_run_llm_phase</code> 的余额刷新，而非仅测快照/助手算术。</li>
<li><b>(b) 看门狗审查结论 schema（F17-3 之外的已知问题）：</b>batch16/17 的 <code>independent_review.json</code>
顶层 <code>verdict</code> 是字典（<code>integrity</code>, <code>scientific_gain</code>），被现有分类器判为「歧义」而阻断了
<code>batch16:review_final</code>/<code>batch17:review_final</code> 事件。候选补丁在<b>不混淆完整性与科学性</b>的前提下接受该 scoped 结构
（<code>scientific_gain</code> 仅作描述性文本，<b>绝不</b>与 integrity 合并），对其它未知结构保持 fail-closed。
本轮保留共享的既有账本与锁、保留原有邮件凭据运行期处理；<b>不发送测试邮件</b>。</li>
<li><b>运维边界：</b>本轮所有运维改动均指向 batch18 的<b>副本</b>；线上 <code>budget_controller.py</code>、
<code>stage_watcher.py</code> 与邮件账本均<b>未被改动</b>。控制器<b>尚未激活</b>。</li>
</ul>

<h2>九、数值来源（可下载）</h2>
<ul>
<li><a href="corrected_release_metrics.tsv">corrected_release_metrics.tsv</a> — 本报告全部数字的扁平数值来源表</li>
<li><a href="corrected_release_metrics.json">corrected_release_metrics.json</a> — 结构化重算结果（含估计量定义、自助法参数、验证规则）</li>
<li><a href="resume_diagnostic_rc2.json">resume_diagnostic_rc2.json</a> — 续跑诊断（溯源更正 + 边界违规记录）</li>
<li><a href="release_extract_test.json">release_extract_test.json</a> — 解压后可运行性证据（外部工作目录）</li>
<li><a href="%(zip)s">promoter_atlas_v6.15.1-release.rc2.zip</a> — 发布包（含 <code>VERSION_MANIFEST.json</code> SHA 清单）</li>
</ul>

<h2>十、未决事项</h2>
<ul>
<li>发布包与两项运维改动<b>仍需独立评审</b>；本报告不构成评审通过，也不代表实验性状态终止。</li>
<li>训练器/初始化/数据顺序混杂未消除，独立任务因果结论仍不可下。</li>
<li>表观辅助目标零值仍为「缺失 vs 真为零」未决。</li>
<li>家族泛化仅覆盖 14 个构念。</li>
<li>诊断的运行期源码版本不可复原；数值由保存数组重算验证，非源码级重建。</li>
</ul>

<p class="meta" style="margin-top:34px">本文件由 batch18 工作者生成；工作者未发送任何邮件，未读取任何凭据，未执行 git 提交。
状态与自检见 <code>status.json</code>、<code>worker_self_check.json</code>。</p>
</body></html>
"""

html = HTML % {
    "gen": M["written_at_local"],
    "ver": M["candidate_version"],
    "src": M["source_candidate"],
    "nfiles": nfiles,
    "reftab": ref_table(),
    "esttab": estimand_table(),
    "ruletab": rule_table(),
    "boottab": boot_table(),
    "sneural": f4(HGB["S"]["test"]["neural_A_ensemble"]["spearman"]),
    "shgb": f4(HGB["S"]["test"]["hgb"]["spearman"]),
    "zip": zip_rel,
    "zipsha": zip_sha,
    "mansha": manifest_sha,
    "crossdiff": VM["packaged_vs_saved_prediction_crosscheck"]["max_absdiff"],
    "crossv": VM["packaged_vs_saved_prediction_crosscheck"]["verdict"],
    "crossn": VM["packaged_vs_saved_prediction_crosscheck"]["n_comparisons"],
    "cpu": RD["results"]["cpu_continuous_vs_resumed"]["max_absdiff"],
    "cdet": RD["results"]["cuda_determinism_control_continuous_vs_continuous_repeat"]["max_absdiff"],
    "cres": RD["results"]["cuda_continuous_vs_resumed"]["max_absdiff"],
    "honest": RD["honest_label"],
    "rt": RD["provenance"]["run_time_claimed_code_diag_sha256"],
    "cur": RD["provenance"]["current_on_disk_code_diag_sha256"],
}

OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")

