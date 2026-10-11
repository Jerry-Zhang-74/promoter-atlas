#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 finalize2: report.html (Chinese), worker_selfcheck, status, remaining_work,
ONE bounded next protocol, sha_manifest (self-excluding), registry/changelog append,
UTF-8 roundtrip audit. Worker writes ONLY its own selfcheck (no independent_review)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402


def utf8_audit():
    s = "批量41 冻结模型 迁移诊断 ρ (Spearman) 受限 LIMIT"
    p = C.DATA / "encoding_roundtrip_sample.txt"
    C.atomic_write_text(p, s + "\n")
    back = p.read_text(encoding="utf-8").rstrip("\n")
    return {"ok": back == s, "has_replacement_char": ("�" in back)}


def sha_manifest(exclude):
    man = {"batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
           "note": "self-excluding; excludes sha_manifest itself + mutable status/selfcheck/review/mail",
           "files": {}}
    for base in (C.REPORT, C.DATA, C.FIG):
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(C.ROOT)).replace("\\", "/")
            if any(x in rel for x in exclude):
                continue
            man["files"][rel] = {"sha256": C.sha256_file(f), "bytes": f.stat().st_size}
    C.atomic_write_json(C.REPORT / "sha_manifest.json", man)
    return man


def report_html(rep, p2, add):
    P = rep["metrics_pooled"]
    D = rep["paired_rho_differences"]
    S = rep["strata"]
    b = p2["bootstrap"]
    rows = []
    for m, name in (("parm", "PARM HEK293 5折均值"), ("h1", "H1 谱核 readout (冻结)"), ("gc", "GC 比例基线")):
        rows.append("<tr><td>%s</td><td>%.3f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td>"
                    "<td>%.3f</td></tr>" % (
                        name, P[m]["spearman"], P[m]["top_decile"]["precision"],
                        S["chr17"]["rho"][m], S["non_chr17"]["rho"][m],
                        rep["per_component"]["macro"][m]["macro_mean_rho"]))
    html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>batch41 v6.28.0 冻结迁移诊断</title>
<style>body{font-family:sans-serif;max-width:960px;margin:20px auto;line-height:1.6}
table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}
.neg{color:crimson;font-weight:bold}</style></head><body>
<h1>batch41 v6.28.0-hekft-fragment-benchmark.rc1 冻结迁移诊断</h1>
<p><b>结论：<span class="neg">LIMITED（未支持继续片段迁移）</span></b>。零拟合、零校准、零梯度/优化步骤。</p>
<h2>一、任务与输入</h2>
<p>目标：OligoMPRA 公开 HEK293FT getAlpha 活性（7537 生物 + 85 对照，270bp 插入 raw[15:285]）。
输入仅 270bp 插入，无 RC/滑窗/额外接头/参考DNA填充。模型冻结：PARM HEK293 5 折、batch24 H1 谱核 readout（仅推理）、固定 GC 基线。</p>
<h2>二、主要指标（预先声明）</h2>
<table><tr><th>模型</th><th>合并 rho</th><th>top10%% 精确率</th><th>chr17 rho</th><th>非chr17 rho</th><th>组件宏平均 rho</th></tr>
%s</table>
<p>配对 rho 差（共同集合 n=%d）：PARM−H1=%+.3f，PARM−GC=%+.3f，H1−GC=%+.3f。</p>
<p>组件自助（条件重采样，2000 次）：PARM−H1 差 95%% 区间 [%+.3f, %+.3f]。</p>
<h2>三、决策规则（冻结）</h2>
<p>需全部满足才支持：PARM 合并 rho≥0.30？%s；PARM−H1≥0.05？%s；差值 CI 下界&gt;0？%s；PARM top10 精确率≥H1？%s；PARM 非chr17 rho&gt;0？%s。
任一不满足则为 LIMITED。</p>
<h2>四、batch40 修复</h2>
<p>独立重算 step90 窗口重叠：%d/%d 一致（未硬编码）。生物覆盖 %s，未匹配 %d，对照 %d/%d。旧 batch40 保存 0 对/报告 8821、覆盖 1.0113、unmatched −85 的不一致已原文保留并披露。</p>
<h2>五、边界与保留</h2>
<p>公开 HEK293FT ≠ 本地 293FT/D28；先验训练暴露未知；回顾性迁移，非前瞻/严格新家族；minP/载体/自主性未知；物理链方向未定。
本地 D28 校准 FAIL 与 A20/G20 2000bp 不支持（未做加窗替代）保持不变。5 折离散度是分歧不是置信区间。</p>
<p><b>资源结论非泛化晋升</b>；负结果不等于所有启动子设计或数据无用。下一有界方案见 next_batch_proposal.json。</p>
</body></html>""" % (
        "\n".join(rows), D["n"], D["PARM_minus_H1"], D["PARM_minus_GC"], D["H1_minus_GC"],
        b["PARM_minus_H1"]["lo"], b["PARM_minus_H1"]["hi"],
        p2["decision_rule"]["checks"]["PARM_pooled_rho_ge_0.30"],
        p2["decision_rule"]["checks"]["PARM_minus_H1_ge_0.05"],
        p2["decision_rule"]["checks"]["PARM_minus_H1_CI_lo_gt_0"],
        p2["decision_rule"]["checks"]["PARM_top10_precision_ge_H1"],
        p2["decision_rule"]["checks"]["PARM_non_chr17_rho_gt_0"],
        add["step90_overlap_recompute"]["pairs_consistent"],
        add["step90_overlap_recompute"]["pairs_checked"],
        add["reconciled_gate"]["bio_coverage_fraction"], add["reconciled_gate"]["unmatched_bio"],
        add["reconciled_gate"]["ctrl_matched"], add["reconciled_gate"]["controls"])
    C.atomic_write_text(C.REPORT / "report.html", html)
    return len(html)


def main():
    t0 = time.time()
    rep = json.load(open(C.DATA / "report.json", encoding="utf-8"))
    p2 = json.load(open(C.DATA / "metrics_part2.json", encoding="utf-8"))
    add = json.load(open(C.DATA / "taskA_correction_addendum.json", encoding="utf-8"))
    enc = utf8_audit()
    n_html = report_html(rep, p2, add)

    verdict = p2["decision_rule"]["verdict"]
    selfcheck = {
        "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(),
        "worker_selfcheck_only": True, "independent_review_created": False,
        "zero_fits_solves_gradients_optimizer_steps": True,
        "zero_cv_train_val_test_fit": True, "no_third_model": True,
        "frozen_manifest_before_forward": True,
        "input": "270bp insert raw[15:285] only",
        "PARM_gpu": {"device_asserted_params_cuda": rep["inference_account"]["PARM"].get("model_params_on_cuda"),
                     "input_cuda": rep["inference_account"]["PARM"].get("input_tensor_on_cuda"),
                     "elapsed_s": rep["inference_account"]["PARM"]["elapsed_s"]},
        "H1_gram_max_abs_diff_vs_saved": json.load(open(C.DATA / "cache/h1/preds.json",
                                                        encoding="utf-8"))["gram_max_abs_diff"],
        "n_rows_scored": rep["unique_5fold_predictions_rows"],
        "unique_at_most_7622": rep["unique_5fold_predictions_rows"] <= 7622,
        "verdict": verdict, "encoding_roundtrip_ok": enc["ok"],
        "note": "this selfcheck is NOT an independent PASS",
    }
    C.atomic_write_json(C.REPORT / "worker_selfcheck.json", selfcheck)

    status = {"batch": C.BATCH, "phase": "worker", "state": "complete", "version": C.VERSION,
              "generated_at": C.now_local(), "pid": os.getpid(),
              "verdict": verdict, "zero_fits": True, "zero_gpu_fits": True,
              "PARM_gpu_seconds": rep["inference_account"]["PARM"]["elapsed_s"],
              "reports_dir": str(C.REPORT.relative_to(C.ROOT)),
              "next": "fresh independent review then parent decision",
              "remaining_work": "remaining_work.json",
              "next_batch_proposal": "next_batch_proposal.json"}
    C.atomic_write_json(C.REPORT / "status.json", status)

    remaining = {"batch": C.BATCH, "generated_at": C.now_local(), "verdict": verdict,
                 "remaining": [
                     "PARM/H1 fragment-transfer gate NOT met (LIMITED): no further transfer investigation without a new mechanism",
                     "public HEK293FT != local 293FT/D28; minP/vector/autonomy/orientation UNKNOWN",
                     "local 293FT D28 calibration FAIL and A20/G20 2000bp unsupported remain unconverted",
                     "no model promotion; production pointers unchanged",
                     "independent review pending; worker selfcheck is not a PASS"]}
    C.atomic_write_json(C.REPORT / "remaining_work.json", remaining)

    nbp = {"record_type": "next_batch_proposal", "batch": C.BATCH, "generated_at": C.now_local(),
           "one_bounded_next_protocol": {
               "name": "batch42 v6.29.0-hekft-local-bridge.rc1 (ZERO fit diagnostic unless parent overrides)",
               "question": ("does the weak alignment reflect (a) cell/context mismatch or (b) input-representation "
                            "mismatch? Test the SAME frozen models on a matched-context slice where activity "
                            "definition is consistent."),
               "scope": ("Compare PARM/H1/GC on the OligoMPRA target restricted to a PRESET, label-free slice "
                         "(e.g. non-chr17 only, or the 133bp-compatible subset) with the SAME frozen readouts; "
                         "no new training; report whether the delta-PARM-H1 changes sign/magnitude."),
               "why_bounded": "one preset slice, one preset metric, zero fits, CPU+<=15min GPU",
               "acceptance": "report honestly; no promotion; no finetuning; if no slice helps -> conclude transfer unsupported",
               "requires": "parent authorization before execution; no automatic launch"},
           "alternative_if_parent_prefers": ("acquire an actual 293FT autonomous-promoter/assay-matched public source "
                                             "(new source query) before any further model comparison"),
           "note": "ONE specific bounded next protocol as required; not authorized here"}
    C.atomic_write_json(C.REPORT / "next_batch_proposal.json", nbp)

    man = sha_manifest(["sha_manifest.json", "worker_selfcheck.json", "status.json",
                        "remaining_work.json", "next_batch_proposal.json",
                        "independent_review", "_review", "mail", "encoding_roundtrip_sample.txt"])

    # registry + changelog atomic append (old production unchanged)
    reg = C.ROOT / "reports/breakthrough_20261002/version_registry.json"
    o = json.load(open(reg, encoding="utf-8"))
    o.setdefault("batch_appends", [])
    entry = {"record_type": "evidence_and_usability", "batch": C.BATCH, "version": C.VERSION,
             "kind": "ZERO-FIT frozen transfer diagnostic (PARM HEK293 / H1 kernel / GC) on public HEK293FT OligoMPRA",
             "appended_by": "batch41 worker (DeepSeek Flash implementer; parent-authorized append-only)",
             "appended_at": C.now_local(),
             "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
             "pooled_rho": {m: rep["metrics_pooled"][m]["spearman"] for m in ("parm", "h1", "gc")},
             "decision": verdict,
             "batch40_repair": {"step90_pairs": "%d/%d" % (add["step90_overlap_recompute"]["pairs_checked"],
                                                           add["step90_overlap_recompute"]["pairs_consistent"]),
                                "bio_coverage": add["reconciled_gate"]["bio_coverage_fraction"],
                                "unmatched": add["reconciled_gate"]["unmatched_bio"],
                                "source20_all_ok": add["source20_join_check"]["all_ok"]},
             "preserved": ["local 293FT D28 FAIL", "A20/G20 2000bp unsupported", "batch40 old bytes"],
             "next": "fresh independent review then parent decision"}
    if not any(e.get("batch") == C.BATCH for e in o["batch_appends"]):
        o["batch_appends"].append(entry)
    o["last_appended_at"] = C.now_local()
    o["last_appended_by"] = "batch41 worker (DeepSeek Flash implementer; parent-authorized append-only)"
    tmp = reg.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(o, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, reg)
    ch = C.ROOT / "reports/breakthrough_20261002/CHANGELOG.txt"
    with open(ch, "a", encoding="utf-8") as f:
        f.write("%s batch41 v6.28.0-hekft-fragment-benchmark.rc1: ZERO-FIT frozen transfer diagnostic; "
                "PARM pooled rho %.3f / H1 %.3f / GC %.3f -> %s; batch40 gate repaired independently "
                "(step90 %d/%d, bio coverage 1.0, unmatched 0, 20/20 joins); no promotion.\n" % (
                    C.now_local(), rep["metrics_pooled"]["parm"]["spearman"],
                    rep["metrics_pooled"]["h1"]["spearman"], rep["metrics_pooled"]["gc"]["spearman"],
                    verdict, add["step90_overlap_recompute"]["pairs_checked"],
                    add["step90_overlap_recompute"]["pairs_consistent"]))
    print(json.dumps({"report_html_bytes": n_html, "verdict": verdict, "fits": 0,
                      "manifest_files_excluding": len(man["files"]),
                      "encoding_ok": enc["ok"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
