# -*- coding: utf-8 -*-
"""BATCH15 metrics_v2 figures (minimal layout repairs over frozen batch14 b14_plot_fixed.py).

Read-only over batch14 sources; writes ONLY reports/breakthrough_20261002/batch15/metrics_v2.
Changes vs frozen b14_plot_fixed.py (kept minimal, no numerical/method edits):
  1) fig01 footer: "误差条" -> "注释区间" (only CI text is shown, no error bars).
  2) fig02 legend moved to reserved whitespace ABOVE the top axes (no longer covers fold1
     full point); pooled-delta CI annotation moved ABOVE the upper CI cap with padding.
  3) fig05 shared legend moved to a dedicated reserved strip between subtitle and axes
     (no longer overlaps the main title).
All metric values and source paths are byte-identical to batch14.
"""
import os, json, csv, textwrap, warnings, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from scipy import stats
from PIL import Image

import b15_common as C

ROOT = "E:/promoter_atlas"
OUT = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch15", "metrics_v2")
os.makedirs(OUT, exist_ok=True)

SRC = {
    "batch09_metrics": os.path.join(ROOT, "reports", "breakthrough_20261002", "batch09", "metrics.json"),
    "batch10_metrics": os.path.join(ROOT, "reports", "breakthrough_20261002", "batch10", "metrics.json"),
    "batch13_metrics": os.path.join(ROOT, "reports", "breakthrough_20261002", "batch13", "metrics.json"),
    "batch13_oof":    os.path.join(ROOT, "data", "breakthrough_20261002", "batch13", "oof_predictions.tsv"),
    "batch06_review": os.path.join(ROOT, "reports", "breakthrough_20261002", "batch06", "independent_review.json"),
    "cellline_metrics": os.path.join(ROOT, "reports", "cellline_v2_20261001", "model_metrics.tsv"),
}

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_fonts():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "fonts", "msyh_regular.ttf"),
              os.path.join(here, "fonts", "msyh_bold.ttf")):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            pass
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"

def add_subtitle(fig, text, y=0.965, fontsize=12.5, color="#005A50", weight="bold"):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=fontsize, color=color,
             fontweight=weight, linespacing=1.4)

def add_footer(fig, text, width=95, y=0.012, fontsize=9.3, color=C.DARK):
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))
    fig.text(0.5, y, wrapped, ha="center", va="bottom", fontsize=fontsize, color=color, linespacing=1.45)

def save(fig, stem, dpi=180):
    png = os.path.join(OUT, stem + ".png")
    svg = os.path.join(OUT, stem + ".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg

def bar_labels(ax, bars, vals, fmt="%.4f", color=C.DARK, fs=10.5, dy=2):
    for r, v in zip(bars, vals):
        ax.annotate(fmt % v, (r.get_x()+r.get_width()/2, v), ha="center", va="bottom",
                    fontsize=fs, color=color, xytext=(0, dy), textcoords="offset points")

# ============================================================================
def fig01():
    b9 = load_json(SRC["batch09_metrics"]); b10 = load_json(SRC["batch10_metrics"])
    gap = b10["primary_feature_gap"]; gr = gap["ridge"]; gh = gap["hgb"]
    data = {
        "Spearman 排序相关（越高越好）": {
            "Ridge": (b9["arms"]["ridge_raw"]["spearman_vs_raw_truth"], b10["augmented_arms"]["ridge_aug"]["spearman_vs_raw_truth"]),
            "HGB":   (b9["arms"]["hgb_raw"]["spearman_vs_raw_truth"],   b10["augmented_arms"]["hgb_aug"]["spearman_vs_raw_truth"]),
        },
        "rawS R²（越高越好）": {
            "Ridge": (b9["arms"]["ridge_raw"]["r2"], b10["augmented_arms"]["ridge_aug"]["r2"]),
            "HGB":   (b9["arms"]["hgb_raw"]["r2"],   b10["augmented_arms"]["hgb_aug"]["r2"]),
        },
        "rawS MAE（越低越好）": {
            "Ridge": (b9["arms"]["ridge_raw"]["mae"], b10["augmented_arms"]["ridge_aug"]["mae"]),
            "HGB":   (b9["arms"]["hgb_raw"]["mae"],   b10["augmented_arms"]["hgb_aug"]["mae"]),
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    learners = ["Ridge", "HGB"]; x = np.arange(len(learners)); w = 0.36
    for ax, (title, vals) in zip(axes, data.items()):
        base = [vals[l][0] for l in learners]; full = [vals[l][1] for l in learners]
        b1 = ax.bar(x - w/2, base, w, color=C.GRAY, label="base（45 特征）")
        b2 = ax.bar(x + w/2, full, w, color=C.TEAL, label="full（1069 特征）")
        bar_labels(ax, b1, base); bar_labels(ax, b2, full, color="#005A50")
        ax.set_xticks(x); ax.set_xticklabels(learners, fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylim(0, max(base + full) * 1.20)
        ax.grid(axis="y", linestyle="--", alpha=0.35); ax.set_axisbelow(True)
        ax.tick_params(labelsize=10.5)
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.95)
    axes[1].set_xlabel("学习者", fontsize=11.5); axes[2].set_xlabel("学习者", fontsize=11.5)

    fig.suptitle("01 · S 特征增益：base45 → full1069（同一固定测试集）", fontsize=15.5, y=0.985)
    add_subtitle(fig,
        "Δ Spearman（97.5%% 配对组件 bootstrap CI）：Ridge +%.4f [%.4f, %.4f]；HGB +%.4f [%.4f, %.4f]"
        % (gr["point_estimate"], gr["lo"], gr["hi"], gh["point_estimate"], gh["lo"], gh["hi"]), y=0.935)
    add_footer(fig, "范围：跨组织基因级 S（rawS 单位，非时间稳定性/启动子活性）｜n_test=1867 基因，n_eligible=18769，同一 corrected batch05 分割｜注释区间为 batch10 既有 97.5% 区间（非本批次新造）")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16, wspace=0.28)
    fig._meta = {"kind": "fig01"}
    return fig

# ============================================================================
def fig02():
    m = load_json(SRC["batch13_metrics"])
    folds = m["fold_metrics"]; fk = ["fold0","fold1","fold2","fold3","fold4"]
    fb = [folds[k]["base"]["spearman"] for k in fk]
    ff = [folds[k]["full"]["spearman"] for k in fk]
    fd = [folds[k]["spearman_delta_full_minus_base"] for k in fk]
    pooled = m["pooled_oof"]; ci = m["primary_contrast"]["bootstrap"]
    comps = [3674, 3681, 3681, 3682, 3682]; genes = m["fold_gene_counts"]
    mae_b = pooled["base"]["mae"]; mae_f = pooled["full"]["mae"]; rel = (mae_b - mae_f) / mae_b
    delta = pooled["full_minus_base"]["spearman_delta"]; lo, hi = ci["lo"], ci["hi"]

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], top=0.80, bottom=0.14,
                          left=0.07, right=0.98, hspace=0.46, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, :])
    xs = np.arange(5)
    ax0.plot(xs, fb, "o-", color=C.GRAY, lw=2, ms=9, label="base（45 特征）")
    ax0.plot(xs, ff, "s-", color=C.TEAL, lw=2, ms=9, label="full（1069 特征）")
    for i in range(5):
        ax0.plot([xs[i], xs[i]], [fb[i], ff[i]], color="#B0BEC5", lw=1.4, zorder=1)
        ax0.annotate("Δ+%.4f" % fd[i], (xs[i], max(fb[i], ff[i])), ha="center", va="bottom",
                     fontsize=9, color=C.DARK, xytext=(0, 4), textcoords="offset points")
    ax0.set_xticks(xs); ax0.set_xticklabels(["折1","折2","折3","折4","折5"], fontsize=12)
    ax0.set_ylabel("Spearman（对 rawS）", fontsize=12)
    ax0.set_title("五折成对：base vs full Spearman（5/5 折为正）", fontsize=13.5)
    ax0.grid(axis="y", linestyle="--", alpha=0.35); ax0.set_axisbelow(True)
    ax0.set_ylim(min(fb)*0.985, max(ff)*1.03)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar([0], [delta], 0.5, color=C.TEAL, yerr=[[delta-lo],[hi-delta]], capsize=6,
            error_kw=dict(lw=1.6, color=C.DARK))
    ax1.set_xlim(-0.7, 0.7); ax1.set_xticks([]); ax1.set_ylim(0, hi*1.70)
    ann = ax1.annotate("Δ Spearman = +%.4f\n95%% CI [%.4f, %.4f]" % (delta, lo, hi),
                 (0, hi), ha="center", va="bottom", fontsize=10.5, color="#005A50",
                 xytext=(0, 10), textcoords="offset points", fontweight="bold")
    ax1.set_title("pooled OOF Spearman 增益\n（base %.4f → full %.4f）" % (pooled["base"]["spearman"], pooled["full"]["spearman"]), fontsize=11.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35); ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[1, 1])
    r2b = pooled["base"]["r2"]; r2f = pooled["full"]["r2"]
    bars = ax2.bar(["base", "full"], [r2b, r2f], 0.5, color=[C.GRAY, C.TEAL])
    bar_labels(ax2, bars, [r2b, r2f], fs=11)
    ax2.set_ylim(0, r2f*1.22); ax2.set_title("pooled OOF R²（rawS）", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.35); ax2.set_axisbelow(True); ax2.tick_params(labelsize=11)

    ax3 = fig.add_subplot(gs[1, 2])
    bars = ax3.bar(["base", "full"], [mae_b, mae_f], 0.5, color=[C.GRAY, C.TEAL])
    bar_labels(ax3, bars, [mae_b, mae_f], fs=11)
    ax3.set_ylim(0, mae_b*1.22)
    ax3.set_title("pooled OOF MAE（rawS）\n相对下降 %.2f%%" % (rel*100), fontsize=12)
    ax3.grid(axis="y", linestyle="--", alpha=0.35); ax3.set_axisbelow(True); ax3.tick_params(labelsize=11)

    fig.suptitle("02 · S 五折交叉验证稳健性（batch13）", fontsize=15.5, y=0.985)
    add_subtitle(fig, "pooled OOF：Spearman 0.4032 → 0.4571（Δ +0.0539，95%% CI [%.4f, %.4f]，既有 bootstrap）" % (lo, hi), y=0.945)
    handles0, labels0 = ax0.get_legend_handles_labels()
    leg = fig.legend(handles0, labels0, loc="upper center", bbox_to_anchor=(0.5, 0.922),
                     ncol=2, fontsize=10, frameon=False)
    add_footer(fig, "组件分块｜n_eligible=18769｜每折基因数 %s，组件数 %s｜pooled OOF 来自 5 个分别训练模型（非单一可部署模型）｜回顾性复用数据，无外部/前瞻声称｜折间训练重叠，非独立重复" % (genes, comps))
    fig._meta = {"kind": "fig02", "legend": leg, "ci_annotation": ann, "ax0": ax0, "ax1": ax1, "hi": hi}
    return fig

# ============================================================================
def fig03():
    rows = C.read_tsv(SRC["batch13_oof"])
    obs  = np.array([float(r["S_raw_full"]) for r in rows])
    base = np.array([float(r["base_pred"]) for r in rows])
    full = np.array([float(r["full_pred"]) for r in rows])
    assert obs.size == 18769
    mb = dict(spearman=C.spearman(obs, base), r2=C.r2_score(obs, base), mae=C.mae_score(obs, base))
    mf = dict(spearman=C.spearman(obs, full), r2=C.r2_score(obs, full), mae=C.mae_score(obs, full))
    pm = load_json(SRC["batch13_metrics"])["pooled_oof"]
    assert abs(mb["spearman"]-pm["base"]["spearman"])<1e-6 and abs(mf["spearman"]-pm["full"]["spearman"])<1e-6

    lo = min(obs.min(), base.min(), full.min()); hi = max(obs.max(), base.max(), full.max())
    pad = (hi - lo) * 0.04; lim = (lo - pad, hi + pad)
    grid = 52
    h1, _, _ = np.histogram2d(base, obs, bins=grid, range=[lim, lim])
    h2, _, _ = np.histogram2d(full, obs, bins=grid, range=[lim, lim])
    vmax = max(h1.max(), h2.max()); norm = LogNorm(vmin=1, vmax=vmax)

    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], top=0.82, bottom=0.16,
                          left=0.08, right=0.98, wspace=0.16)
    axb = fig.add_subplot(gs[0, 0]); axf = fig.add_subplot(gs[0, 1], sharex=axb, sharey=axb)
    cax = fig.add_subplot(gs[0, 2])

    for ax, pred, met, lab, col in [(axb, base, mb, "base（45 特征）", C.GRAY),
                                    (axf, full, mf, "full（1069 特征）", C.TEAL)]:
        hb = ax.hexbin(pred, obs, gridsize=grid, extent=[lim[0], lim[1], lim[0], lim[1]],
                       cmap="viridis", norm=norm, mincnt=1, linewidths=0)
        ax.plot(lim, lim, "--", color="#37474F", lw=1.4, label="y = x")
        ax.set_title(lab, fontsize=13, pad=8)
        ax.text(0.03, 0.97, "Spearman = %.4f\nR² = %.4f\nMAE = %.4f" % (met["spearman"], met["r2"], met["mae"]),
                transform=ax.transAxes, ha="left", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.3))
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
        ax.grid(linestyle="--", alpha=0.2); ax.tick_params(labelsize=10)
    axb.set_ylabel("观测 rawS（负 MAD 单位）", fontsize=12)
    for ax in (axb, axf):
        ax.set_xlabel("预测 rawS（负 MAD 单位）", fontsize=12)
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("计数（对数色标）", fontsize=11); cb.ax.tick_params(labelsize=9)

    fig.suptitle("03 · S 预测分布：观测值 vs 预测值（batch13 OOF，全部 18769 样本）", fontsize=15, y=0.985)
    add_subtitle(fig, "左：base（45 特征）；右：full（1069 特征）；虚线 y=x；共享坐标轴与对数计数色标", y=0.945, color=C.DARK, weight="normal", fontsize=11.5)
    add_footer(fig, "rawS 为负 MAD 单位（非 0..1 排序）｜全部 18769 个 OOF 样本（无随机抽样）｜可视化回归向均值与剩余误差，无虚构置信区间")
    fig._meta = {"kind": "fig03"}
    return fig

# ============================================================================
def fig04():
    rv = load_json(SRC["batch06_review"]); rm = rv["recomputed_primary_metrics"]["test_full"]
    ridge = rm["finalridge"]; hgb = rm["hgb"]; combo = rm["combined_fixedmean"]
    models = ["Ridge45", "HGB45\n（非线性）", "残差 CNN\n+ Ridge"]
    colors = [C.GRAY, C.TEAL, C.ORANGE]
    spe = [ridge["spearman"], hgb["spearman"], combo["spearman"]]
    r2  = [ridge["r2"], hgb["r2"], combo["r2"]]
    mae = [ridge["mae"], hgb["mae"], combo["mae"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    x = np.arange(len(models)); w = 0.6
    for ax, (title, vals) in zip(axes, [("Spearman（越高越好）", spe), ("R²（越高越好）", r2), ("MAE（越低越好）", mae)]):
        bars = ax.bar(x, vals, w, color=colors)
        bar_labels(ax, bars, vals, fs=10)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10.5)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylim(0, max(vals)*1.20)
        ax.grid(axis="y", linestyle="--", alpha=0.35); ax.set_axisbelow(True); ax.tick_params(labelsize=10.5)

    fig.suptitle("04 · E 模型比较（batch06，Score_E 排序目标，测试 n=5782 基因）", fontsize=15, y=0.985)
    add_subtitle(fig, "HGB45 最优（Spearman 0.748）；残差 CNN+Ridge（0.740）未超过 HGB45（负/混合结果如实保留）", y=0.935, color="#BF360C")
    add_footer(fig, "目标 E = 基因表达 Score_E 排序（0..1 排序量纲，非 v6.3/rawS 量纲）｜Ridge/HGB 使用 45 特征；残差 CNN 额外学习原始序列残差再加 Ridge｜残差组合为固定均值（2 种子）组合")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.17, wspace=0.30)
    fig._meta = {"kind": "fig04"}
    return fig

# ============================================================================

# ---- rendered text bounding-box validation ---------------------------------
def _disp(artist, fig):
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer)

def boxes_overlap(a, b):
    if a is None or b is None:
        return False
    return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)

def layout_checks(stem, fig):
    fig.canvas.draw()
    meta = getattr(fig, "_meta", {})
    checks = {}
    if stem == "01_s_feature_gain":
        foot = " ".join([t.get_text() for t in fig.texts])
        checks["footer_says_annotated_intervals"] = ("注释区间" in foot and "误差条" not in foot)
    if stem == "02_s_cv_robustness":
        leg = meta.get("legend"); ax0 = meta.get("ax0"); ax1 = meta.get("ax1")
        ann = meta.get("ci_annotation"); hi = meta.get("hi")
        st = fig._suptitle
        ren = fig.canvas.get_renderer()
        leg_b = _disp(leg, fig); st_b = _disp(st, fig); ax0_b = ax0.get_window_extent(ren)
        checks["legend_vs_suptitle_overlap"] = boxes_overlap(leg_b, st_b)
        checks["legend_above_axes"] = bool(leg_b.ymin >= ax0_b.ymax - 2)
        checks["legend_ymin"] = round(float(leg_b.ymin), 1)
        checks["axes_top_ymax"] = round(float(ax0_b.ymax), 1)
        if ann is not None:
            ann_b = _disp(ann, fig)
            cap_pt = ax1.transData.transform((0, hi))
            checks["ci_annotation_bottom_px"] = round(float(ann_b.ymin), 1)
            checks["ci_cap_top_px"] = round(float(cap_pt[1]), 1)
            checks["ci_annotation_above_cap"] = bool(ann_b.ymin >= cap_pt[1])
            checks["ci_annotation_inside_axes"] = bool(ax1.get_window_extent(ren).contains(ann_b.x0, ann_b.y0))
    if stem == "05_cellline_generalization":
        leg = meta.get("legend"); axes = meta.get("axes")
        st = fig._suptitle
        ren = fig.canvas.get_renderer()
        leg_b = _disp(leg, fig); st_b = _disp(st, fig)
        checks["legend_vs_suptitle_overlap"] = boxes_overlap(leg_b, st_b)
        ax0_b = axes[0].get_window_extent(ren)
        checks["legend_between_subtitle_and_axes"] = bool(leg_b.ymax <= st_b.ymin and leg_b.ymin >= ax0_b.ymax - 2)
        checks["legend_ymax"] = round(float(leg_b.ymax), 1)
        checks["suptitle_ymin"] = round(float(st_b.ymin), 1)
        checks["axes_top_ymax"] = round(float(ax0_b.ymax), 1)
    checks["n_legends"] = len(fig.legends)
    checks["n_axes"] = len(fig.axes)
    return checks

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)

def main():
    setup_fonts()
    figs = [("01_s_feature_gain", fig01), ("02_s_cv_robustness", fig02),
            ("03_s_prediction_distribution", fig03), ("04_e_model_comparison", fig04)]
    manifest = {"report_version": "metrics_v2-20261003-r1",
                "parent_batch14": "metrics-snapshot-20261003-r1",
                "figures": []}
    layout = {}
    for stem, fn in figs:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fn()
            layout[stem] = layout_checks(stem, fig)
            png, svg = save(fig, stem)
            glyph_warns = [str(x.message) for x in w if "missing from font" in str(x.message) or "Glyph" in str(x.message)]
        with Image.open(png) as im:
            w_px, h_px = im.size
        manifest["figures"].append({
            "stem": stem,
            "png": png.replace("\\", "/"),
            "svg": svg.replace("\\", "/"),
            "png_sha256": sha256_file(png),
            "svg_sha256": sha256_file(svg),
            "png_width_px": int(w_px),
            "png_height_px": int(h_px),
            "glyph_warnings": glyph_warns if glyph_warns else [],
        })
        print("saved", stem, "layout:", layout[stem], flush=True)

    atomic_write_json(os.path.join(OUT, "layout_validation.json"), {
        "report_version": "metrics_v2-20261003-r1",
        "checks": layout,
        "summary": {
            "fig02_legend_vs_suptitle_overlap": layout["02_s_cv_robustness"].get("legend_vs_suptitle_overlap"),
            "fig02_legend_above_axes": layout["02_s_cv_robustness"].get("legend_above_axes"),
            "fig02_ci_above_cap": layout["02_s_cv_robustness"].get("ci_annotation_above_cap"),
            "fig01_footer_says_annotated_intervals": layout["01_s_feature_gain"].get("footer_says_annotated_intervals"),
        },
    })
    atomic_write_json(os.path.join(OUT, "figure_manifest.json"), manifest)
    print("WROTE layout_validation.json and figure_manifest.json", flush=True)

if __name__ == "__main__":
    main()
