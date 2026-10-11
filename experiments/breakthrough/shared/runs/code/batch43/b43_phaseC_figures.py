# -*- coding: utf-8 -*-
"""batch43 Phase C: <=3 scientifically useful figures + decision.json.

Figures use ONLY the cached/derived tables from Phase B.  No decorative HTML.
"""
from __future__ import annotations
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import b43_common as C

T0 = time.time()
R = C.B43_REPORT
FIG = C.B43_FIG
FIG.mkdir(parents=True, exist_ok=True)

ed = C.jload(R / "error_decomposition.json")
rk = C.jload(R / "rank_diagnostics.json")
db = C.jload(R / "distribution_table.json")
m42 = C.jload(C.B42_METRICS)

LEARNED = ["R0", "R1", "Rnull"]
COLORS = {"R0": "#1f77b4", "R1": "#d62728", "Rnull": "#7f7f7f", "const": "#2ca02c"}

# ---- fig1: original held-group comparison + same-unit constant baseline -----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
names = ["R0", "R1", "Rnull", "parm(rank)", "h1(rank)", "gc(rank)"]
vals, cols = [], []
for n in names:
    key = n.split("(")[0]
    d = rk["TEST"][key]
    vals.append(np.nan if d["group_mean_spearman"] is None else d["group_mean_spearman"])
    cols.append(COLORS.get(key, "#9467bd"))
ax[0].bar(range(len(names)), vals, color=cols)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(range(len(names)))
ax[0].set_xticklabels(names, rotation=20, ha="right")
ax[0].set_title("TEST: pooled Spearman (rank metric)")
ax[0].set_ylabel("Spearman ρ")
ax[0].set_ylim(-0.3, 0.3)
for i, v in enumerate(vals):
    ax[0].text(i, (v if np.isfinite(v) else 0), "%.3f" % v, ha="center",
               va="bottom" if v >= 0 else "top", fontsize=8)

mae_names = ["R0", "R1", "Rnull"]
row_mae = [ed["TEST"][m]["mae"] for m in mae_names]
grp_mae = [ed["TEST"][m]["group_balanced_mae"] for m in mae_names]
x = np.arange(len(mae_names))
ax[1].bar(x - 0.2, row_mae, 0.4, label="row MAE", color=["#1f77b4", "#d62728", "#7f7f7f"])
ax[1].bar(x + 0.2, grp_mae, 0.4, label="group-balanced MAE", color=["#aec7e8", "#ff9896", "#c7c7c7"])
const_mae = m42["zero_fit_train_weighted_mean_baseline"]["mae"]
const_gmae = m42["zero_fit_train_weighted_mean_baseline"]["group_balanced_mae"]
ax[1].axhline(const_mae, color="green", ls="--", lw=1.2, label="zero-fit const MAE %.4f" % const_mae)
ax[1].axhline(const_gmae, color="green", ls=":", lw=1.2, label="const group-MAE %.4f" % const_gmae)
ax[1].set_xticks(x); ax[1].set_xticklabels(mae_names)
ax[1].set_title("TEST: same-unit MAE (alpha units)")
ax[1].set_ylabel("MAE (alpha units)")
ax[1].legend(fontsize=7)
fig.suptitle("batch43 fig1 — original batch42 held-group result (unchanged) + zero-fit constant baseline", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / "fig1_heldgroup_comparison.png", dpi=130)
plt.close(fig)

# ---- fig2: MSE decomposition (between-group bias vs within-group) ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
groups = ["R0", "R1", "Rnull", "const"]
for j, (part, key) in enumerate((("row-weighted (1/N)", "row_weighted"), ("group-equal (1/G)", "group_equal"))):
    bet = [ed["TEST"][m][key]["between_b2"] for m in groups]
    wit = [ed["TEST"][m][key]["within"] for m in groups]
    x = np.arange(len(groups))
    ax[j].bar(x, bet, 0.6, label="between-group bias  mean(b_g^2)", color="#ff7f0e")
    ax[j].bar(x, wit, 0.6, bottom=bet, label="within-group error", color="#4c72b0")
    ax[j].set_xticks(x); ax[j].set_xticklabels(groups)
    ax[j].set_title("TEST MSE decomposition — %s" % part)
    ax[j].set_ylabel("MSE (alpha^2 units)")
    for i, m in enumerate(groups):
        wf = ed["TEST"][m][key]["within_frac"]
        ax[j].text(i, bet[i] + wit[i] + 0.0008, "within %.0f%%" % (100 * wf), ha="center", fontsize=8)
    if j == 0:
        ax[j].legend(fontsize=7)
fig.suptitle("batch43 fig2 — group-mean-bias vs within-group MSE (MSE_g=b_g^2+within_g; NOT a correction, oracle group means)",
             fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig2_error_decomposition.png", dpi=130)
plt.close(fig)

# ---- fig3: fixed distribution / evidence scope -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
rows = C.B42.load_frozen_rows(with_labels=True)
gmap = C.load_groups()
bio = [r for r in rows if r["control"] == 0]
ctl = [r for r in rows if r["control"] == 1]
trv = [r["alpha_primary"] for r in bio if gmap[r["label"]]["split"] == "TRAIN"]
tev = [r["alpha_primary"] for r in bio if gmap[r["label"]]["split"] == "TEST"]
cv = [r["alpha_primary"] for r in ctl]
ax[0].boxplot([trv, tev, cv], tick_labels=["TRAIN\nn=7219", "TEST\nn=318", "controls\nn=85"], showfliers=False)
ax[0].set_title("raw alpha distribution (same units)")
ax[0].set_ylabel("alpha")
sizes = db["TEST"]["overall"]["group_sizes_sorted_desc"]
ax[1].bar(range(len(sizes)), sizes, color="#4c72b0")
ax[1].axhline(10, color="r", ls="--", lw=1, label="within-rho min rows = 10")
ax[1].set_title("TEST group sizes (21 groups)")
ax[1].set_xlabel("group rank"); ax[1].set_ylabel("rows")
ax[1].legend(fontsize=7)
ax[2].axis("off")
txt = ("Evidence / target scope (fixed)\n"
       "• target: local 293FT novel-promoter/truncation D28 — NOT tested here\n"
       "• here: PUBLIC HEK293FT OligoMPRA getAlpha, 270bp, retrospective reuse\n"
       "• TEST 318 rows / 21 groups (<30) → descriptive, no CI/significance\n"
       "• source parents: TEST %d families; TRAIN %d families\n"
       "• chr17 TEST n=%d, non-chr17 n=%d\n"
       "• controls(scrambled context) descriptive only, no binary truth\n"
       "• preserved: D28 FAIL, 2000bp unsupported, UNVALIDATED_FOR_293FT\n"
       % (db["TEST"]["overall"]["source_parent_families"],
          db["TRAIN"]["overall"]["source_parent_families"],
          db["TEST"]["chr17"]["rows"], db["TEST"]["non_chr17"]["rows"]))
ax[2].text(0.0, 1.0, txt, va="top", ha="left", fontsize=8, family="monospace")
ax[2].set_title("fixed distribution / scope")
fig.suptitle("batch43 fig3 — fixed cohort distribution + evidence scope (no new strata/resampling)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig3_distribution_scope.png", dpi=130)
plt.close(fig)
print("figures done")

# ---- decision.json + Chinese conclusion ------------------------------------
tw = {m: ed["TEST"][m]["row_weighted"] for m in groups}
gw = {m: ed["TEST"][m]["group_equal"] for m in groups}
decision = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
    "record_type": "decision",
    "posthoc_diagnostic_reuse_of_exposed_test": True,
    "original_failure_persists": {
        "value": True,
        "R1_minus_R0_pooled_rho": rk["TEST"]["R1"]["n_groups_total"] and (ed and (-0.06735343227629197)),
        "note": "R1 did NOT beat R0; all pre-registered positive criteria failed; verdict LIMITED unchanged",
        "preserved_metrics": ed["original_metrics_unchanged"],
    },
    "question": "are errors predominantly group-level bias or poor within-group ranking?",
    "decomposition_answer": {
        "TEST_row_weighted_within_frac": {m: tw[m]["within_frac"] for m in groups},
        "TEST_group_equal_within_frac": {m: gw[m]["within_frac"] for m in groups},
        "dominant_term": "WITHIN-group residual (per-construct) — roughly 71–88% of TEST MSE across learned models",
        "group_level_bias_share": "≈12–29% of TEST MSE; largest for R1 (its larger total MSE also carries more group-level bias)",
        "caveat": "group means use TEST labels = ORACLE diagnostic quantities, NOT a deployable correction or improved prediction",
    },
    "what_it_resolves": [
        "Locates the residual mass: within-group (ordering/heterogeneity), not mainly group-level offset.",
        "Shows the learned models' TEST error is not a single group shift that a group-mean correction could remove.",
        "Separates group-mean-level rank skill (near zero, sometimes negative) from within-group rank skill (near zero).",
    ],
    "what_it_does_not_resolve": [
        "Does not make the negative result positive and adds no operational PASS.",
        "Does not test local 293FT / D28 / novel-family or 2000bp; public HEK293FT OligoMPRA only.",
        "Cannot establish assay repeatability or an upper bound on attainable rho from replicate-aggregated labels.",
        "Does not prove low capacity or noisy labels from low TRAIN performance, nor impossibility from a negative TEST.",
        "Cannot establish cell/vector causal mechanisms from group-level shifts.",
    ],
    "new_learning_question_justified": {
        "value": True,
        "bounded": ("a NEW, pre-registered, group-heldout learning question is justified only if it keeps the frozen "
                    "union grouping, does NOT tune against this exposed test, and targets WITHIN-group ordering / richer "
                    "sequence representation; the earlier overlap-only resplit and alpha tuning remain rejected"),
        "not_authorized_here": True,
    },
    "exact_missing_evidence": [
        "matched-cell local 293FT assay with sequence/barcode identity — none",
        "independent per-group replicate measurements for a noise ceiling — none",
        "prospective (untouched) confirmation split — none (batch41 exposure + batch42 reuse)",
        ">=30 held-out groups for conditional CI — unavailable (21)",
    ],
    "no_new_operational_PASS_criterion": True,
    "preserved_defaults": {"local_293FT_D28": "FAIL", "A20_G20_2000bp": "UNSUPPORTED",
                           "293FT_status": "UNVALIDATED_FOR_293FT",
                           "production_pointer": "v6.3 main / v6.4 cellline pilot (unchanged)"},
}
C.atomic_write_json(R / "decision.json", decision)

conclusion_zh = (
    "batch43 v6.30.0-group-error-diagnostic.rc1 结论（事后诊断，非新验证）\n"
    "1) 原负结果保持：在已冻结的 batch42 分组留出试点上，R1 未胜 R0（pooled ρ 0.091 vs 0.158；"
    "group-balanced MAE 0.175 vs 0.107），全部预定阳性判据未达，verdict 仍为 LIMITED，无任何晋升。\n"
    "2) 误差分解：对学习型同单位模型 e=p-y，按组计算 b_g、MSE_g 与 within_g，逐组验证 MSE_g=b_g^2+within_g"
    "（最大残差≈2.8e-17）。TEST 上 within（组内）项占 MSE 约 71–88%，between（组均值偏置）占约 12–29%。"
    "故误差以组内排序/异质性为主，而非单一组水平偏移。\n"
    "3) 秩诊断：TEST 组均预测 vs 组均 alpha 的 Spearman——R0 0.086、R1 −0.179、Rnull −0.244、"
    "GC 0.216（仅秩比较），组内 macro rho 均接近 0（R0 0.118、R1 0.086、GC 0.163）。"
    "常数基线 rho 记为空并注明原因（非常数零）。\n"
    "4) 边界：组均值来自 TEST 标签，是诊断用的 oracle，不构成可部署校正或更好预测；"
    "21 个 TEST 组<30，无 CI/显著性；不能据此断言能力/噪声/泛化不可能，也不能推断细胞/载体因果。"
    "PUBLIC HEK293FT OligoMPRA 与本地 293FT/D28 不是同一实验，D28 FAIL、2000bp 不支持、"
    "UNVALIDATED_FOR_293FT 全部保留。\n"
    "5) 纠正：batch42 metrics.json 中两处 cached_parm_mae（chr17 0.4664 / non_chr17 0.3536）为跨单位无效指标，"
    "标注 INVALID；驳回 I15 与整体合规 PASS；旧 canonical 评审与旧结果原字节不改。\n"
)
(R / "conclusion_zh.txt").write_text(conclusion_zh, encoding="utf-8")

C.atomic_write_json(C.B43_RUN / "phaseC_figures.json",
                    {"batch": C.BATCH, "phase": "C_figures", "generated_at": C.now_local(),
                     "elapsed_s": round(time.time() - T0, 2), "figures": ["fig1_heldgroup_comparison.png",
                     "fig2_error_decomposition.png", "fig3_distribution_scope.png"],
                     "script_sha256": C.sha256_file(__file__)})
print("PHASE C done %.2fs" % (time.time() - T0))
