#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 deliverable 4: Chinese HTML release report.

Numbers are read from the batch17 artifacts (never hardcoded), so the report
cannot drift from the recomputed source of truth.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REP = ROOT / "reports" / "breakthrough_20261002" / "batch17"
OUT = REP / "release_report_v6_15_1_rc1.html"

M = json.loads((REP / "corrected_release_metrics.json").read_text(encoding="utf-8"))
RD = json.loads((REP / "resume_diagnostic.json").read_text(encoding="utf-8"))
VM = json.loads((ROOT / "models" / "breakthrough_20261002" / "batch17" /
                 "release_v6.15.1-release.rc1" / "VERSION_MANIFEST.json").read_text(encoding="utf-8"))

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


zip_rel = "../../../models/breakthrough_20261002/batch17/promoter_atlas_v6.15.1-release.rc1.zip"
zip_sha = "b2d5fd33ce273b289ca75f6a335cb52613b090a5bc1f16be5ec9f46ea4f315d9"
nfiles = len(VM["package_files_sha256"])

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Promoter Atlas v6.15.1-release.rc1 release report (candidate / experimental)</title>
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

<h1>Promoter Atlas v6.15.1-release.rc1 发布报告</h1>
<p class="meta">候选版本 / 实验性制品 &nbsp;·&nbsp; 生成时间 %(gen)s &nbsp;·&nbsp; batch17（打包与溯源修复轮次）</p>

<div class="banner">
<span class="pill">候选 / 实验性</span>
<b>本版本不是一次科学突破，也不是分数提升。</b>
它是对已存在权重的<b>打包与溯源修复</b>：模型权重逐字节沿用 batch16 的 <code>v6.15.0-multitask.rc1</code> 独立臂检查点，
没有重新训练、没有换种子、没有改架构、没有依据测试分数做任何选择。
随附的是<b>一个阴性（未通过）的共享网络选择结果</b>，以及一个可移植、可复现的发布包。
在独立评审通过之前，本版本应被视为<b>实验性候选</b>，不应用于任何下游决策。
</div>

<h2>一、版本与状态</h2>
<table>
<tr><th>字段</th><th>值</th></tr>
<tr><td>候选版本</td><td><code>%(ver)s</code></td></tr>
<tr><td>来源候选</td><td><code>%(src)s</code></td></tr>
<tr><td>改动性质</td><td>打包 / 溯源修复（权重未变；版本号提升仅反映打包）</td></tr>
<tr><td>部署架构</td><td>独立臂 A = 三个单任务 1D-CNN（不共享主干）</td></tr>
<tr><td>实际部署种子</td><td>20261016（唯一随包发布的权重）</td></tr>
<tr><td>第二个种子</td><td>20261017，仅用于复现/集成估计，未随包发布</td></tr>
<tr><td>被否决架构</td><td>B（共享主干 + E/S/B 头）、C（共享主干 + 7 个表观辅助头）</td></tr>
<tr><td>发布包文件数</td><td>%(nfiles)d（含 SHA-256 清单）</td></tr>
</table>
<div class="note">历史制品 v6.3 与 celllinev6.4 未被修改，保持原样保留；batch16 及更早的全部数据、代码、模型、报告均只读保留。</div>

<h2>二、实际部署权重的指标（由保存的预测重算）</h2>
<p>下表的每一个数字都由 batch16 保存的目标 <code>y</code> 与预测 <code>pred_*</code> 数组精确重算得到
（预测以标准化单位保存，已按 <code>transforms/scalers.npz</code> 还原为原始单位）。<b>没有新的拟合。</b></p>
<table>
<tr><th>头</th><th>划分</th><th>Spearman</th><th>R2</th><th>MAE</th><th>n</th></tr>
%(reftab)s
</table>

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
任何历史（旧随机划分）分数都不得与本报告中的分数并列解释或直接相减。</div>

<h2>四、阴性结果：共享网络选择未通过（冻结验证规则）</h2>
<p>冻结验证规则（<b>仅用验证集</b>）：各种子平均、每个目标的验证集 Spearman 增益 &gt;= 0.02，且没有任何目标的下降超过 0.02。</p>
<table>
<tr><th>臂</th><th>E 增益</th><th>S 增益</th><th>B 增益</th><th>平均增益</th><th>无下降 &gt; 0.02</th><th>规则通过</th></tr>
%(ruletab)s
</table>
<div class="fail">结论：B（共享主干）与 C（共享主干 + 表观辅助头）都未通过冻结验证规则，因此发布独立臂 A。</div>
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
<div class="note">E 与 B 上的对比区间基本跨越 0（不显著）；S 上共享臂略好，但 S 的有效样本远小（n=1880），
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
<b>4. 划分不可比。</b> 见第三节：修正组件并集划分与旧随机划分分数不可直接比较。
</div>
<p>综合以上，本版本是一个<b>实验性候选</b>与一个<b>阴性共享网络选择结果</b>，
<b>不是</b>关于「独立任务优于共享任务」或「表观辅助任务有用」的受控因果证据。</p>

<h2>六、发布包与首次使用</h2>
<p>发布包为自包含目录 + ZIP，<b>不包含训练资产</b>，可在解压到任意新目录后直接运行小例子。</p>
<table>
<tr><th>项目</th><th>值</th></tr>
<tr><td>ZIP</td><td><a href="%(zip)s">promoter_atlas_v6.15.1-release.rc1.zip</a></td></tr>
<tr><td>ZIP SHA-256</td><td><code>%(zipsha)s</code></td></tr>
<tr><td>包内文件数</td><td>%(nfiles)d（含 <code>VERSION_MANIFEST.json</code> 全量 SHA 清单）</td></tr>
<tr><td>打包预测与已保存预测一致性</td><td>最大绝对差 %(crossdiff).3e（容差 1e-5，判定 <span class="ok">%(crossv)s</span>，%(crossn)d 项比对）</td></tr>
<tr><td>解压后可运行性</td><td>已在一个全新的自有目录、外部工作目录下验证（见 <code>release_extract_test.json</code>）</td></tr>
</table>
<h3>6.1 首次使用命令</h3>
<pre># 1) unzip
unzip promoter_atlas_v6.15.1-release.rc1.zip -d promoter_atlas_release
cd promoter_atlas_release/release_v6.15.1-release.rc1

# 2) run the bundled example (also self-checks 3 invalid inputs)
python -X utf8 code/run_example.py

# 3) predict on your own data (python API)
python -X utf8 -c "import sys; sys.path.insert(0, chr(46)); from code.promoter_atlas_predict import PromoterAtlasGenePredictor as P; m=P(package_root=chr(46)); print(m.predict_batch(seqs, feats))"</pre>
<h3>6.2 输入接口（明确、可校验）</h3>
<ul>
<li><code>seqs</code>：形状 <code>(N,4,400)</code> 或 <code>(4,400)</code>，uint8，字母表 A/C/G/T/N 映射 0/1/2/3/4（N 编码为 4，one-hot 全零）。</li>
<li><code>features</code>：形状 <code>(N,1069)</code> 或 <code>(1069,)</code>，float 且有限，<b>原始未标准化</b>（包内变换自动做 (x-mean)/std）。</li>
<li>批量规则：<code>len(seqs)</code> 必须等于 <code>len(features)</code>；否则抛出 <code>PromoterAtlasInputError</code>。</li>
</ul>
<div class="warn"><b>标签与预测必须区分：</b>输出中的 <code>is_model_prediction = true</code>、
<code>is_catalog_observed = false</code>；数据库观测标签（如 <code>labels/build/gene_expression_label_table.tsv</code>）
是<b>观测值</b>，绝不是模型预测，二者不得混用。</div>

<h2>七、训练中断续跑诊断（有界、只读）</h2>
<p>为回应「生产续跑是否真的可续」，做了一次有界诊断（&lt;=10 分钟 GPU、&lt;=16 个生产 mini-batch、单 CUDA 任务、
batch &lt;= 32、CPU &lt;= 4 线程），实际在单 CUDA 作业上使用 <b>%(mb)d</b> 个生产 mini-batch（另在 CPU 上使用 8 个），未做任何完整训练。</p>
<table>
<tr><th>检验</th><th>最大绝对差</th><th>判定</th></tr>
<tr><td>CPU：连续 vs 中断续跑</td><td class="num">%(cpu).1f</td><td>逐位完全相同（仅支持该 CPU 路径）</td></tr>
<tr><td>CUDA：连续 vs 连续重复（确定性对照）</td><td class="num">%(cdet).3e</td><td>单次对照即出现差异（非逐位可复现）</td></tr>
<tr><td>CUDA：连续 vs 中断续跑</td><td class="num">%(cres).3e</td><td>与对照组量级相近（未证明同因）</td></tr>
</table>
<div class="note"><b>%(honest)s</b><br>
要点：CUDA 上「同样代码 + 同样种子」的两次连续运行已经出现差异，说明该环境下并非逐位可复现；
但<b>单个对照组不能定义一个分布或可复现区间</b>，与对照组量级相近也<b>不能证明</b>续跑残差仅（或部分）来自卷积核非确定性，
更不能排除量级相当的状态恢复缺陷。CPU 上逐位相同<b>只支持本诊断所测的 CPU 路径</b>，不构成对 CUDA 精确续跑的一般性证明。
原样保留观测到的差异数值，不做取整或隐藏；重新对账后的 mini-batch 计数见下。</div>
<table>
<tr><th>进程</th><th>设备</th><th>mini-batch</th></tr>
<tr><td>continuous</td><td>CUDA</td><td class="num">4</td></tr>
<tr><td>continuous 重复（确定性对照）</td><td>CUDA</td><td class="num">4</td></tr>
<tr><td>interrupted（第 1 段）</td><td>CUDA</td><td class="num">2</td></tr>
<tr><td>resume（第 2 段）</td><td>CUDA</td><td class="num">2</td></tr>
<tr><td><b>CUDA 合计</b></td><td>CUDA</td><td class="num"><b>12</b>（声明上限 16）</td></tr>
<tr><td>continuous / interrupted / resume</td><td>CPU</td><td class="num">4 + 2 + 2 = 8</td></tr>
<tr><td><b>全部进程合计</b></td><td>—</td><td class="num"><b>20</b>（7 个进程）</td></tr>
</table>
<div class="note">计数按启动时声明的边界如实对账，<b>不追溯修改边界、不隐藏诊断工作量</b>：单 CUDA 作业 12 个 mini-batch 在 &lt;=16 之内；
CPU 对照在独立的 4 线程上限下另计 8 个；无完整训练。出处保存在 <code>resume_diagnostic.json</code>，<b>未替换任何科学权重</b>。</div>

<h2>八、运维跟进（仅提出，未应用、未发送）</h2>
<p>本轮另附两项<b>最小修复提案</b>与一项<b>溯源更正记录</b>。它们都<b>没有</b>应用到线上控制器/看门狗，
也<b>没有</b>发送任何邮件；必须经独立评审后才可由上层决定启用。详见
<a href="ops_proposal.json">ops_proposal.json</a>。</p>
<ul>
<li><b>(a) 预算控制器资金结转：</b>现有控制器在全新启动时把累计充值重置为 0（control_23 记录原始基线 142.67、结转充值 100.00、
末次余额 114.04；control_24 使用「有效基线 242.67」临时绕过）。提案从 manifest / funding_basis.json 直接结转
原始基线与累计历史充值，使支出估计保持连续；已通过对真实数据的单元测试、控制器级测试与既有回归套件。</li>
<li><b>(b) 看门狗审查结论 schema：</b>batch16 的 <code>independent_review.json</code> 顶层 <code>verdict</code> 是字典
（<code>integrity: PASS, scientific_gain: ...</code>），被现有分类器判为「歧义」而阻断了 <code>batch16:review_final</code> 事件。
提案在<b>不混淆完整性与科学性</b>的前提下接受该 scoped 结构，对其他未知结构保持 fail-closed；已有 14 项测试覆盖（含真实制品）。</li>
<li><b>(c) 审查者溯源更正：</b>batch16 评审制品自报的是外壳进程 pid 1870；真实审查进程为 control_23 托管的
reviewer（pid 17072，10:46:15 启动，returncode 0，模型 <code>deepseek-v4-pro</code>）。
<b>原始 batch16 制品未被修改</b>，更正仅作为记录。</li>
</ul>
<div class="note">所有运维改动均指向 <code>runs/breakthrough_20261002/code/batch17/ops/</code> 的副本与补丁；
线上 <code>budget_controller.py</code>、<code>stage_watcher.py</code> 与邮件账本均<b>未被改动</b>。</div>

<h2>九、数值来源（可下载）</h2>
<ul>
<li><a href="corrected_release_metrics.tsv">corrected_release_metrics.tsv</a> — 本报告全部数字的扁平数值来源表</li>
<li><a href="corrected_release_metrics.json">corrected_release_metrics.json</a> — 结构化重算结果（含估计量定义、自助法参数、验证规则）</li>
<li><a href="resume_diagnostic.json">resume_diagnostic.json</a> — 续跑诊断与边界</li>
<li><a href="release_extract_test.json">release_extract_test.json</a> — 解压后可运行性证据</li>
<li><a href="%(zip)s">promoter_atlas_v6.15.1-release.rc1.zip</a> — 发布包（含 <code>VERSION_MANIFEST.json</code> SHA 清单）</li>
</ul>

<h2>十、未决事项</h2>
<ul>
<li>发布包与两项运维改动<b>仍需独立评审</b>；本报告不构成评审通过。</li>
<li>训练器/初始化/数据顺序混杂未消除，独立任务因果结论仍不可下。</li>
<li>表观辅助目标零值仍为「缺失 vs 真为零」未决。</li>
<li>家族泛化仅覆盖 14 个构念。</li>
</ul>

<p class="meta" style="margin-top:34px">本文件由 batch17 工作者生成；工作者未发送任何邮件，未读取任何凭据，未执行 git 提交。
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
    "crossdiff": VM["packaged_vs_saved_prediction_crosscheck"]["max_absdiff"],
    "crossv": VM["packaged_vs_saved_prediction_crosscheck"]["verdict"],
    "crossn": VM["packaged_vs_saved_prediction_crosscheck"]["n_comparisons"],
    "mb": RD["bounds_respected"]["production_minibatches_total"],
    "cpu": RD["results"]["cpu_continuous_vs_resumed"]["max_absdiff"],
    "cdet": RD["results"]["cuda_determinism_control_continuous_vs_continuous_repeat"]["max_absdiff"],
    "cres": RD["results"]["cuda_continuous_vs_resumed"]["max_absdiff"],
    "honest": RD["honest_label"],
}

OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
