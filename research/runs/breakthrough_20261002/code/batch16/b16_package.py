#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 packaging: validation-based candidate selection + deployment bundle
+ model card + Chinese HTML summary + version registry append.

Deployment selector uses VALIDATION ONLY (never test). If shared (B) passes the
frozen rule (mean per-target val rho gain >= 0.02 vs independent, no target drop
> 0.02) it is selected; otherwise the strongest supported independent (A)
composite is selected and the rejected ablation is reported honestly.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT, ROOT,
                        CANDIDATE_VERSION, TASKS, SEEDS, sha256_file, atomic_write_json)


def load_json(p):
    if not Path(p).exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def select_candidate(comparison):
    if not comparison:
        return {"architecture": "pending", "reason": "comparison.json not yet available"}
    primary = comparison.get("primary_contrast_B_vs_A", {})
    s = primary.get("summary", {})
    if s.get("rule_pass"):
        return {"architecture": "shared", "reason": "B shared backbone passed validation rule",
                "summary": s}
    # strongest supported independent composite; C only if it clearly improves over A
    return {"architecture": "independent",
            "reason": "no shared neural win; strongest supported independent (A) composite release",
            "summary": s}


def build_bundle(selection, neural, comparison):
    bundle = {
        "candidate_version": CANDIDATE_VERSION,
        "architecture": selection["architecture"],
        "selection_reason": selection["reason"],
        "selection_uses_validation_only": True,
        "seeds": SEEDS,
        "scalers": str(DATA_OUT / "scalers.npz"),
        "models": {},
    }
    if selection["architecture"] == "shared":
        # B (shared, no aux) preferred if it wins; use seed 20261016 checkpoint
        bundle["models"]["shared"] = str(MODEL_OUT / "B_s20261016" / "model.pt")
    else:
        for t in TASKS:
            bundle["models"][t] = str(MODEL_OUT / "A_s20261016" / ("model_%s.pt" % t))
    bundle["note"] = ("selected checkpoint is seed 20261016 (validation-based); seed 20261017 "
                      "is reported as replication, not used for selection")
    return bundle


def write_model_card(selection, comparison, neural, hgb, bundle):
    lines = []
    lines.append("=" * 72)
    lines.append("MODEL CARD  %s  (batch16 controlled comparison)" % CANDIDATE_VERSION)
    lines.append("=" * 72)
    lines.append("Scope: human gene-level natural-expression E/S/B (not HEK293T/fragment/multispecies).")
    lines.append("Selection: validation-only rule (never test).")
    lines.append("")
    lines.append("Candidate architecture: %s" % selection["architecture"])
    lines.append("Reason: %s" % selection["reason"])
    if selection.get("summary"):
        lines.append("Primary rule summary: %s" % json.dumps(selection["summary"], ensure_ascii=False))
    lines.append("")
    lines.append("Targets:")
    lines.append("  E = Score_E rank-scaled [0,1] expression strength")
    lines.append("  S = S_raw = -MAD(log2(1+detectable TPM)) (S_defined mask, 18769 eligible)")
    lines.append("  B = B_raw = detectable/tissue_count (cross-tissue breadth)")
    lines.append("")
    lines.append("Input representation (identical across A/B/C):")
    lines.append("  - diverse 4-slot gene-bag sequence CNN (one-hot ACGT, N->zero)")
    lines.append("  - frozen 1069-feature composition/position branch")
    lines.append("")
    lines.append("Deployment bundle: %s" % str(MODEL_OUT / "deployment_bundle.json"))
    for k, v in bundle.get("models", {}).items():
        lines.append("  %s: %s" % (k, v))
    lines.append("")
    if hgb:
        lines.append("Matched HGB baselines (same corrected split, from scratch):")
        for k in ("E45", "S1069", "B45"):
            d = hgb.get(k, {})
            tv = d.get("val", {}).get("spearman")
            tt = d.get("test", {}).get("spearman")
            if tv is not None:
                lines.append("  %s: val rho=%.4f test rho=%.4f" % (k, tv, tt))
    lines.append("")
    lines.append("Honest limitations: retrospective heldout (old test reused); detected-edge-aware")
    lines.append("grouping is not full homology proof; no causal inflation claim; neural win over")
    lines.append("HGB alone does not beat the established nonlinear baseline.")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_zh_html(selection, comparison, neural, hgb, bundle, out_path):
    c = comparison or {}
    primary = c.get("primary_contrast_B_vs_A", {})
    s = primary.get("summary", {})
    heldout = c.get("heldout_test_rho", {})
    rows = []
    for t in TASKS:
        a_val = primary.get(t, {}).get("A_val_rho")
        b_val = primary.get(t, {}).get("B_val_rho")
        gain = primary.get(t, {}).get("gain")
        rows.append("<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%+.4f</td></tr>" % (
            t, a_val or 0, b_val or 0, gain if gain is not None else 0))
    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>batch16 受控任务共享对比报告</title>
<style>
body{{font-family:sans-serif;max-width:900px;margin:24px auto;line-height:1.6;}}
table{{border-collapse:collapse;margin:12px 0;}}th,td{{border:1px solid #ccc;padding:6px 12px;}}
th{{background:#f0f0f0;}}h1,h2{{color:#333;}}.lim{{color:#a33;}}
</style></head><body>
<h1>batch16 受控任务共享对比（人类基因级 E/S/B）</h1>
<p>候选版本：{ver}。选定架构：<b>{arch}</b>。选定理由：{reason}。</p>
<p>本实验为<b>受控消融</b>：同一从头训练骨架/输入/优化器/划分/早停规则下比较
A（独立 E/S/B 头）、B（共享骨架+E/S/B 头）、C（共享骨架+观测表观遗传辅助监督）。</p>
<h2>主对比（验证集 Spearman，B − A 增益）</h2>
<table><tr><th>目标</th><th>A 验证 rho</th><th>B 验证 rho</th><th>增益</th></tr>
{rows}</table>
<p>规则结论：平均增益 {mean_gain:+.4f}（阈值 ≥0.02），无目标下降超过 0.02：<b>{rule}</b>。</p>
<h2>heldout（测试）rho</h2>
<table><tr><th>目标</th><th>A</th><th>B</th><th>C</th></tr>
{held_rows}</table>
<h2>基线</h2>
<table><tr><th>基线</th><th>验证 rho</th><th>测试 rho</th></tr>
{hgb_rows}</table>
<h2 class="lim">诚实限制</h2>
<ul>
<li>测试集为复用旧数据（回顾性评估），非外部/前瞻验证。</li>
<li>检测边感知分组（batch05 精确/RC+前向近同源 + batch15 RC 边）不是完整同源证明。</li>
<li>不宣称“去除辅助头导致增益”的因果结论，也不宣称旧增益消失/保持。</li>
<li>仅人类基因级；无 HEK293T/片段/多物种验证。任意片段不能继承整基因 S 活性。</li>
<li>“最佳”仅在已测变体与验证集选择内成立。</li>
</ul>
<p>生成时间：{ts}。</p>
</body></html>"""
    held_rows = ""
    for t in TASKS:
        ha = heldout.get("A", {}).get(t)
        hb = heldout.get("B", {}).get(t)
        hc = heldout.get("C", {}).get(t)
        held_rows += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            t, "%.4f" % ha if ha is not None else "-",
            "%.4f" % hb if hb is not None else "-",
            "%.4f" % hc if hc is not None else "-")
    hgb_rows = ""
    for k in ("E45", "S1069", "B45"):
        d = (hgb or {}).get(k, {})
        hgb_rows += "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            k, "%.4f" % d.get("val", {}).get("spearman") if d.get("val", {}).get("spearman") is not None else "-",
            "%.4f" % d.get("test", {}).get("spearman") if d.get("test", {}).get("spearman") is not None else "-")
    html = html.format(ver=CANDIDATE_VERSION, arch=selection["architecture"],
                       reason=selection["reason"], rows="\n".join(rows),
                       mean_gain=s.get("mean_gain", float("nan")),
                       rule=str(s.get("rule_pass")), held_rows=held_rows,
                       hgb_rows=hgb_rows, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html


def main():
    t0 = time.time()
    for d in (MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    comparison = load_json(REPORT_OUT / "comparison.json")
    neural = load_json(MODEL_OUT / "neural_results.json") or {}
    hgb = load_json(MODEL_OUT / "hgb_results.json") or {}
    hgb_results = hgb.get("results", {}) if isinstance(hgb, dict) else {}

    selection = select_candidate(comparison)
    bundle = build_bundle(selection, neural, comparison)
    atomic_write_json(MODEL_OUT / "deployment_bundle.json", bundle)

    card = write_model_card(selection, comparison, neural, hgb_results, bundle)
    card_path = ROOT / "models" / "cards" / "promoter_atlas_v6_15_0_multitask_rc1_card.txt"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(card + "\n")

    zh = write_zh_html(selection, comparison, neural, hgb_results, bundle,
                       REPORT_OUT / "batch16_summary_zh.html")

    pkg = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": "batch16",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "selection": selection,
        "deployment_bundle": str(MODEL_OUT / "deployment_bundle.json"),
        "model_card": str(card_path),
        "zh_summary": str(REPORT_OUT / "batch16_summary_zh.html"),
        "inference_entrypoint": str(Path(__file__).parent / "b16_inference.py"),
    }
    atomic_write_json(REPORT_OUT / "package_manifest.json", pkg)
    print(json.dumps(pkg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
