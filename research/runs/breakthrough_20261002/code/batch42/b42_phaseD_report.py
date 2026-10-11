# -*- coding: utf-8 -*-
"""batch42 Phase D: <=3 figures + report.html + SHA manifest + registry/changelog
+ worker selfcheck/status/remaining/proposal.  Honest outcome: gate PASSED
(union grouping, 21 held-out groups), three fixed fits executed, DESCRIPTIVE
pilot only (21<30 groups -> no CI/bootstrap/significance), verdict LIMITED.
"""
from __future__ import annotations
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import b42_common as C

T0 = time.time()
REP = C.B42_REPORT
FIG = C.B42_FIG
REP.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

frozen = json.load(open(C.B42_DATA / "frozen_split.json", encoding="utf-8"))
src = json.load(open(C.B42_DATA / "source_validation.json", encoding="utf-8"))
hom = json.load(open(C.B42_DATA / "alignment_findings.json", encoding="utf-8"))
bg = json.load(open(C.B42_RUN / "phaseB_gate.json", encoding="utf-8"))
bf = json.load(open(C.B42_RUN / "phaseB_fits.json", encoding="utf-8"))
met = json.load(open(REP / "metrics.json", encoding="utf-8"))
g = frozen["gate"]

# group sizes by split (from groups.tsv)
sizes = {"TRAIN": [], "TEST": []}
with open(C.B42_DATA / "groups.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        lab, grp, canon, sp, *_ = line.rstrip("\n").split("\t")
        sizes[sp].append((grp, 1))
from collections import Counter
gcount = {"TRAIN": Counter(), "TEST": Counter()}
with open(C.B42_DATA / "groups.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        _, grp, _, sp, *_ = line.rstrip("\n").split("\t")
        gcount[sp][grp] += 1
tr_sz = sorted(gcount["TRAIN"].values(), reverse=True)
te_sz = sorted(gcount["TEST"].values(), reverse=True)

# ---- fig1: split + gate -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(range(1, len(tr_sz) + 1), tr_sz, marker="o", ms=3, label="TRAIN groups (%d)" % len(tr_sz))
ax[0].plot(range(1, len(te_sz) + 1), te_sz, marker="s", ms=3, label="TEST groups (%d)" % len(te_sz))
ax[0].set_yscale("log"); ax[0].set_xlabel("group rank"); ax[0].set_ylabel("members (log)")
ax[0].set_title("Union-group sizes (overlap U author-family U exact/RC U homology)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
labels = ["train_rows", "train_groups", "test_rows", "test_groups"]
have = [g["train_rows"], g["train_groups"], g["test_rows"], g["test_groups"]]
need = [C.MIN_TRAIN_ROWS, C.MIN_TRAIN_GROUPS, C.MIN_TEST_ROWS, C.MIN_TEST_GROUPS]
x = np.arange(len(labels)); ax[1].bar(x - .2, have, .4, label="observed")
ax[1].bar(x + .2, need, .4, label="required (amendment 10/5)")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=20, fontsize=8)
ax[1].set_yscale("log"); ax[1].set_title("Feasibility gate: PASS=%s" % g["PASS"]); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig1_split_gate.png", dpi=130); plt.close()

# ---- fig2: test metrics -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
names = ["R1", "R0", "Rnull", "cached_parm_mean", "cached_h1_mean", "gc"]
rho = [met["R1"]["pooled_spearman"], met["R0"]["pooled_spearman"], met["Rnull"]["pooled_spearman"],
       met["cached_batch41_same_testids"]["cached_parm_mean"]["spearman"],
       met["cached_batch41_same_testids"]["cached_h1_mean"]["spearman"],
       met["cached_batch41_same_testids"]["gc"]["spearman"]]
gmae = [met["R1"]["group_balanced_mae"], met["R0"]["group_balanced_mae"], met["Rnull"]["group_balanced_mae"],
        None, None, None]
cols = ["#c0392b", "#2e86c1", "#7f8c8d", "#8e44ad", "#16a085", "#f39c12"]
ax[0].bar(names, rho, color=cols); ax[0].axhline(.30, ls="--", c="k", lw=.8, label="R1 rho>=0.30 req")
ax[0].set_title("Held-out TEST pooled Spearman vs alpha"); ax[0].tick_params(axis="x", rotation=35, labelsize=8)
ax[0].legend(fontsize=7); ax[0].set_ylabel("Spearman rho")
gv = [v for v in gmae if v is not None]
ax[1].bar(["R1", "R0", "Rnull"], gv, color=["#c0392b", "#2e86c1", "#7f8c8d"])
ax[1].axhline(met["zero_fit_train_weighted_mean_baseline"]["group_balanced_mae"], ls="--", c="k", lw=.8,
              label="zero-fit baseline")
ax[1].set_title("Held-out group-balanced MAE (lower better)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "fig2_test_metrics.png", dpi=130); plt.close()

# ---- fig3: near-homology ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar([0, 1], [hom["n_ed_le_27"], hom["n_ed_le_40"] - hom["n_ed_le_27"]],
          color=["#27ae60", "#e67e22"])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["ed<=27 (merge)", "28..40"])
ax[0].set_title("Cross-group near-homology: min ed=%d, merges=%d" % (hom["min_edit_distance_found"], hom["n_confirmed_merges"]))
ax[0].set_ylabel("cross-group pairs")
ax[1].axis("off")
ax[1].text(0.02, 0.95, "Phase A freeze\n  frozen table sha ok: %s\n  bio rows: %d / controls: %d\n  upper270 ok: %d/%d\n"
          "  author families: %d   b41 components: %d\n  union groups: %d   candidates scanned: %d\n"
          "Split (once, SHA)\n  TRAIN %d rows / %d groups\n  TEST  %d rows / %d groups\n  cross-split exact/RC: %s\n"
          "Fits (gate PASS, 3/3 consumed)\n  R0 comp342 alpha100\n  R1 H1-1280 alpha1\n  Rnull H1 shuffled seed %d\n"
          "Verdict: %s (no operational PASS; 21<30 groups -> no CI)" % (
              src["frozen_table_sha_ok"], src["bio_rows"], src["control_rows"], src["upper270_ok"],
              src["upper270_ok"] + src["upper270_bad"], frozen["union_sources"]["n_author_families"],
              frozen["union_sources"]["n_components_batch41"], frozen["counts"]["groups"],
              hom["n_candidate_pairs_scanned"], frozen["counts"]["TRAIN"], frozen["counts"]["groups_TRAIN"],
              frozen["counts"]["TEST"], frozen["counts"]["groups_TEST"], frozen["cross_split_exact_or_rc"],
              C.NULL_SEED, met["decision"]["verdict"]), va="top", family="monospace", fontsize=8.5)
plt.tight_layout(); plt.savefig(FIG / "fig3_freeze_summary.png", dpi=130); plt.close()
print("3 figures written")

# ---- report.html ------------------------------------------------------------
d = met["decision"]; t10 = met["top10"]; cb = met["cached_batch41_same_testids"]
fits_short = "R0 comp342 alpha=100 | R1 H1-1280 alpha=1 | Rnull H1 shuffled once seed=20261043"
rows = ""
for k, v in (("R1", met["R1"]), ("R0", met["R0"]), ("Rnull", met["Rnull"])):
    rows += "<tr><td>%s</td><td>%.3f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>" % (
        k, v["pooled_spearman"], v["r2"], v["mae"], v["group_balanced_mae"])
base = met["zero_fit_train_weighted_mean_baseline"]
rows += "<tr><td>zero-fit const (train w-mean)</td><td>n/a</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>" % (
    C.r2(np.full(met["n_test_rows"], base["const"]), None) if False else float("nan"), base["mae"], base["group_balanced_mae"])
crow = ""
for nm, key in (("cached batch41 PARM", "cached_parm_mean"), ("cached batch41 H1", "cached_h1_mean"), ("GC content", "gc")):
    crow += "<tr><td>%s</td><td>%.3f</td><td>%.3f</td></tr>" % (
        nm, cb[key]["spearman"], cb[key]["top10"]["precision"])
trow = ""
for nm in ("R1", "R0", "Rnull"):
    trow += "<tr><td>%s</td><td>%.4f</td><td>%.3f</td></tr>" % (nm, t10[nm]["precision"], t10[nm]["enrichment"])
st = met["strata"]
html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>batch42 v6.29.0 有监督学习试点</title>
<style>body{font-family:sans-serif;max-width:1000px;margin:20px auto;line-height:1.65}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #999;padding:4px 9px}
.neg{color:crimson;font-weight:bold}code{background:#f2f2f2;padding:1px 4px}
h2{border-bottom:2px solid #ddd;padding-bottom:3px}</style></head><body>
<h1>batch42 v6.29.0-hekft-supervised-pilot.rc1</h1>
<p><b>问题：</b>在本 assay（公开 HEK293FT 270bp OligoMPRA，MPRAnalyze <code>getAlpha</code>）上，
纯序列固定模型能否比同等规模的组成模型更好地预测<em>留出组</em>？——是 assay 特异的有监督学习问题，
<u>不是</u>自主启动子 / 本地 293FT D28 / 上下文因果性。</p>
<p><b>结论：<span class="neg">门控通过并执行了 3 次固定拟合，但结果为阴性/描述性试点（verdict LIMITED；
不晋升任何模型，无 operational PASS）。</span></b>
按协议，分组证据<em>并集</em>（实际重叠 ∪ 作者父瓦片 family ∪ 精确/RC ∪ 无标签近同源）把 7537 生物行
合并为 <b>113</b> 组；单次确定性 SHA 划分得 TRAIN 7219 行/92 组、TEST 318 行/<b>21</b> 组
（父级 pre-fit 修正后的可行性门控 10/5 组、1000/300 行 = <b>通过</b>）。因 TEST 组数 21 &lt; 30，
条件 bootstrap/置信区间不可用，故只能给出<em>无显著性</em>的描述性结果：R1（H1 序列核）留出
Spearman ρ = %.3f，反而低于 R0 组成模型 ρ = %.3f，组平衡 MAE 也未改善 → 未达预注册的正面阈值。</p>

<h2>一、输入与来源校验（Phase A，无标签）</h2>
<ul>
<li>冻结表 SHA <code>%s</code> 校验 <b>通过</b>；使用全部 <b>%d</b> 生物行，<b>%d</b> 对照
独立保留、不进训练/测试指标；alpha 全部有限（%d/%d），未做 p 值/活性/GC 选择。</li>
<li>300bp oligo 上 270bp 插入 DNA 身份：raw[15:285] 与序列逐行一致 <b>%d/%d</b>（大小写软屏蔽后相等），
未做参考 DNA 补充/裁剪/按标签定向。</li>
<li>分组来源：batch41 区间重叠组件 %d；作者 ROI 父瓦片 family %d；精确/RC 重复 0；近同源合并 0 →
<b>并集 %d 组</b>。</li>
</ul>

<h2>二、近同源扫描（无标签，先于任何标签分布打分）</h2>
<p>以 canonical(fwd/RC) 15-mer 重叠仅生成候选（<u>绝不作为对齐证据</u>），对 %d 个跨组候选对做
双向 Myers 位向量编辑距离（全长 270bp）：<b>最小编辑距离 = %d bp</b>；满足“≥90%% 一致且 ≥90%% 双向覆盖”
(ed≤27、零 gap) 的跨组对 = <b>0</b> → 无同源合并边。残留非穷尽（共享 &lt;8 个 15-mer 的对未遍历）已如实披露，
未作任何“所有 family 均孤立”的假声明。</p>

<h2>三、单次划分与门控</h2>
<p>冻结规则：canonical groupID = 组成员中最小的 <code>sequence_sha256</code>；
<code>h=SHA256('b42-group-v1|'+groupID)</code>，<code>int(h,16)%%100&lt;80</code>→TRAIN 否则 TEST。
跨划分精确/RC 重复 = %s。门控（父级 pre-fit 修正）：train≥1000 行/≥10 组，
test≥300 行/≥5 组，非单一组，train 目标方差非零 → <b>PASS=%s</b>。
（初始协议副本保留，修正已披露：≥30 组仅用于正式条件 bootstrap。）</p>

<h2>四、三次固定拟合（Phase B，执行账本）</h2>
<p>经审查的 batch27 <code>exec_gate.py</code>（原子锁+预注册）：预算 3，先哑写盘探测（不实例化拟合），
每个拟合前 <code>reserve</code> 后 <code>run</code>。<b>已消耗 3/3</b>，无 CV/调参/多随机种子/全量重拟合、
无隐藏重试。%s。样本权重 1/训练组大小（归一化均值 1）；特征均值/尺度仅用 TRAIN，
常数特征尺度取 1；截距 = 加权训练标签均值。</p>

<h2>五、留出组 TEST 评估（Phase C，一次）</h2>
<table><tr><th>模型</th><th>pooled ρ</th><th>R²</th><th>MAE</th><th>组平衡 MAE</th></tr>%s</table>
<p>零拟合常数基线（训练加权均值）MAE = %.4f，组平衡 MAE = %.4f。</p>
<p>Top10（k=32=ceil(0.1×318)，确定性 ID 并列规则）：</p>
<table><tr><th>模型</th><th>precision</th><th>enrichment</th></tr>%s</table>
<p>同一 TEST ID 上缓存的 batch41 冻结基线（仅比较 ρ 与 top10，不跨单位比 MAE）：</p>
<table><tr><th>基线</th><th>ρ</th><th>top10 precision</th></tr>%s</table>
<p>逐组宏 ρ（≥10 行且非恒定，使用 17/%d 组）：R1=%.3f，R0=%.3f，Rnull=%.3f。
固定分层（描述性，不用最佳子集挽救）：chr17 n=%d 时 R1 ρ=%.3f / R0 ρ=%.3f / cached PARM ρ=%.3f；
非 chr17 n=%d 时 R1 ρ=%.3f / R0 ρ=%.3f。</p>
<p><b>决策规则（阈值预注册）：</b>R1 ρ≥0.30 → %s；R1−R0 ρ≥0.05 → %s；条件 CI 下界&gt;0 → <b>不可用</b>（21&lt;30 组）；
组 MAE 至少低 10%% → %s；top10 不低于 R0 → %s。因 CI 无法计算，<b>不可能</b>给出 operational PASS。</p>

<h2>六、边界（如实保留）</h2>
<ul>
<li>回顾性复用：batch41 已暴露全部标签，本实验<u>不是</u>新的未触碰外部验证。</li>
<li>公开 HEK293FT ≠ 本地 293FT；保留既往 D28 校准 FAIL、PARM 软件 600/2000bp 不支持、v6.3/v6.4 指针不变。</li>
<li>Rnull 为单次打乱参照，非置换 p 值；无显著性结论。</li>
<li>阴性结果不得用于否定有监督学习的全部可能；仅说明<em>本</em>固定核/固定 alpha 设置下未优于组成模型。</li>
</ul>
<p>下一步：见 <code>next_batch_proposal.json</code>（一个具体有界方案）。</p>
</body></html>""" % (
    d["r1_rho"], d["r0_rho"], src["frozen_table_sha256"], src["bio_rows"], src["control_rows"],
    src["bio_alpha_finite"], src["bio_rows"], src["upper270_ok"], src["upper270_ok"] + src["upper270_bad"],
    frozen["union_sources"]["n_components_batch41"], frozen["union_sources"]["n_author_families"],
    frozen["counts"]["groups"], hom["n_candidate_pairs_scanned"], hom["min_edit_distance_found"],
    frozen["cross_split_exact_or_rc"], g["PASS"], fits_short, rows, base["mae"], base["group_balanced_mae"],
    trow, crow, met["per_group"]["R1"]["n_groups_used"], met["per_group"]["R1"]["macro_rho"],
    met["per_group"]["R0"]["macro_rho"], met["per_group"]["Rnull"]["macro_rho"],
    st["chr17"]["n"], st["chr17"]["R1_rho"], st["chr17"]["R0_rho"], st["chr17"]["cached_parm_rho"],
    st["non_chr17"]["n"], st["non_chr17"]["R1_rho"], st["non_chr17"]["R0_rho"],
    d["criteria"]["R1_test_rho_ge_0.30"], d["criteria"]["R1_minus_R0_rho_ge_0.05"],
    d["criteria"]["groupMAE_at_least_10pct_lower"], d["criteria"]["top10_not_lower"])
(REP / "report.html").write_text(html, encoding="utf-8")
print("report.html written", len(html))

# ---- SHA manifest (immutable; excludes itself + mutable status/review) ------
MANIFEST = REP / "sha_manifest.json"
excl_names = {"sha_manifest.json", "status.json", "worker_selfcheck.json",
              "remaining_work.json", "next_batch_proposal.json", "independent_review.json"}
paths = []
for base in (C.B42_DATA, C.B42_RUN, REP):
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in excl_names and ".tmp" not in p.name \
                and not p.name.endswith(".lock") and "mail" not in p.name.lower():
            paths.append(p)
entries = []
for p in sorted(set(paths)):
    entries.append({"path": str(p.relative_to(C.ROOT)).replace("\\", "/"),
                    "sha256": C.sha256_file(p), "bytes": p.stat().st_size})
manifest = {"batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
            "excludes": sorted(excl_names) + ["manifest itself", "mutable status/verify/mail"],
            "n_files": len(entries), "files": entries}
C.atomic_write_json(MANIFEST, manifest)
print("manifest files", len(entries))

# ---- version registry + changelog (atomic append, old history unchanged) ---
reg_path = C.ROOT / "reports" / "breakthrough_20261002" / "version_registry.json"
reg = json.load(open(reg_path, encoding="utf-8"))
entry = {
    "record_type": "supervised_pilot_descriptive",
    "batch": "batch42", "version": C.VERSION,
    "kind": "assay-specific supervised fixed-kernel pilot on public HEK293FT 270bp OligoMPRA alpha; union-group split; 3 fixed fits; DESCRIPTIVE (no CI)",
    "appended_by": "batch42 worker (DeepSeek Flash implementer; parent-authorized append-only)",
    "appended_at": C.now_local(),
    "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
    "gate": {"union_groups": frozen["counts"]["groups"], "train_rows": g["train_rows"],
             "train_groups": g["train_groups"], "test_rows": g["test_rows"],
             "test_groups": g["test_groups"], "PASS": g["PASS"],
             "amendment": "pre-fit feasibility 10/5 groups; formal bootstrap >=30"},
    "fits": bf["attempts"], "fits_consumed": bf["status_after"]["consumed"],
    "test_metrics": {"R1_pooled_spearman": met["R1"]["pooled_spearman"],
                     "R0_pooled_spearman": met["R0"]["pooled_spearman"],
                     "delta_rho": d["delta_rho"], "groupMAE_improvement_frac": d["groupMAE_improvement_frac"]},
    "verdict": d["verdict"], "operational_PASS": False,
    "scope": "regional/group-held-out retrospective public assay; NOT new unseen family / local 293FT D28 / autonomous promoter",
    "preserved_prior": ["local 293FT D28 FAIL", "PARM software 600 / 2000bp unsupported", "v6.3/v6.4 pointers unchanged"],
}
reg.setdefault("batch_appends", []).append(entry)
reg["last_appended_at"] = C.now_local()
reg["last_appended_by"] = "batch42 worker"
C.atomic_write_json(reg_path, reg)

chg = C.ROOT / "reports" / "breakthrough_20261002" / "CHANGELOG.txt"
with open(chg, "a", encoding="utf-8") as fh:
    fh.write("%s batch42 %s: union-group split (113 groups; TRAIN 7219/92, TEST 318/21) gate PASS after pre-fit 10/5 amendment; 3 fixed fits (R0 comp342 a100, R1 H1-1280 a1, Rnull shuffled seed20261043); TEST R1 rho %.3f vs R0 %.3f -> DESCRIPTIVE LIMITED (21<30 groups, no CI, no operational PASS); no promotion.\n" % (
        C.now_local(), C.VERSION, met["R1"]["pooled_spearman"], met["R0"]["pooled_spearman"]))
print("registry+changelog appended")
