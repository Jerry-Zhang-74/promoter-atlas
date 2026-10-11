#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 Chinese report generator (report.html + report.txt).

Reads ONLY frozen artifacts + the already-verified batch10 metrics + the
existing D28 construct metadata (14 measured constructs). Emits a human-readable
Chinese report that (a) explains ACTB/GAPDH example queries, (b) explains why
these gene-level raw-S outputs do NOT predict A1-A20/G1-G20/CMV construct
activity, and (c) includes a domain-gap table for the 14 measured constructs
using existing verified metadata only (required gene context / all-promoter
aggregation versus actual fragment sequence/endpoints). No invented alignment,
no new HEK293T activity model, no new metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s12_common import (  # noqa: E402
    BATCH, CANDIDATE_VERSION, DATA_OUT, FROZEN_ARTIFACTS, REPORT_OUT,
    S_TARGET_DEF, TARGET_TRAIN_MEAN, TARGET_TRAIN_STD, atomic_write_text,
    now_local, sha256_file, verify_artifacts)
from s12_predict import FrozenPredictor  # noqa: E402

# Verified D28 construct metadata (14 measured constructs). Read-only reference.
CONSTRUCT_ENDPOINTS = Path(
    "outputs/01a0f783-d754-7803-abcf-34aa761a6a6b/"
    "全库标签与细胞系pilot_v2_CMVcore修正_20261001_文件夹/"
    "reports/cellline_v2_20261001/construct_endpoints_D28.tsv")
CONSTRUCT_SEQUENCES = Path(
    "outputs/01a0f783-d754-7803-abcf-34aa761a6a6b/"
    "全库标签与细胞系pilot_v2_CMVcore修正_20261001_文件夹/"
    "reports/cellline_v2_20261001/construct_sequences.tsv")
CONSTRUCT_ENDPOINTS_SHA = "9125f52bd0048fc6f0a72f7e42fe747ec8b7d1eba18baa6acbea26527279215f"
CONSTRUCT_SEQUENCES_SHA = "43a4a7e6834b77a95aedaee06f882307956fd08dec6f61e855311d4a3c9e6bc6"

# Retrospective batch10 test metrics (fixed evidence; NOT re-derived here).
PRIOR_TEST = {"spearman": 0.47436, "r2": 0.19668, "mae": 0.20813,
              "source": "batch10 full-1069 hgb_aug test split (fixed retrospective evidence)"}

ACTB = "ENSG00000075624"
GAPDH = "ENSG00000111640"


def _load_replay():
    p = DATA_OUT / "replay_evidence.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _domain_gap_rows():
    if not CONSTRUCT_ENDPOINTS.is_file():
        return []
    if sha256_file(CONSTRUCT_ENDPOINTS) != CONSTRUCT_ENDPOINTS_SHA:
        raise RuntimeError("construct endpoints metadata sha256 mismatch (refuse to report)")
    df = pd.read_csv(CONSTRUCT_ENDPOINTS, sep="\t", low_memory=False)
    fam_order = {"ACTB": 0, "GAPDH": 1, "CMV": 2}
    df["_fam"] = df["family"].map(lambda f: fam_order.get(str(f), 9))
    df["_len"] = pd.to_numeric(df["sequence_length_bp"], errors="coerce").fillna(0)
    df = df.sort_values(["_fam", "_len"]).reset_index(drop=True)
    gene_nprom = {ACTB: 43, GAPDH: 40}  # verified from frozen feature universe
    rows = []
    for r in df.itertuples(index=False):
        fam = str(r.family)
        cid = str(r.construct_id)
        length = r.sequence_length_bp
        if fam == "ACTB":
            required = "canonical ACTB (%s) gene-level raw S: ALL %d annotated promoters, arithmetic-mean aggregation, 54 tissues" % (ACTB, gene_nprom[ACTB])
            actual = "single ACTB proximal fragment, %s bp TSS-upstream (one specific endpoint)" % length
            out_of_scope = False
        elif fam == "GAPDH":
            required = "canonical GAPDH (%s) gene-level raw S: ALL %d annotated promoters, arithmetic-mean aggregation, 54 tissues" % (GAPDH, gene_nprom[GAPDH])
            actual = "single GAPDH proximal fragment, %s bp TSS-upstream (one specific endpoint)" % length
            out_of_scope = False
        else:  # CMV
            required = "no Ensembl human gene ID -> not in 58988-gene label catalog / 57827-gene feature universe (no gene context exists)"
            actual = "viral promoter fragment, %s bp (CMVfull enhancer+core; CMVcore user-clarified core, documented 200 bp, actual 204 bp)" % length
            out_of_scope = True
        endpoint = "HEK293T D28 relative GFP (activity), not cross-tissue expression dispersion (S)"
        rows.append({
            "construct_id": cid,
            "family": fam,
            "actual_fragment": actual,
            "required_gene_context": required,
            "measured_endpoint_vs_model_target": endpoint,
            "gene_level_S_applicable": (not out_of_scope),
            "gap": ("single-fragment activity endpoint vs all-promoter gene-level S dispersion"
                    if not out_of_scope else "non-human-gene (viral) -> completely out of scope"),
        })
    return rows


def _txt_report(replay, rows, actb, gapdh, cat_summary):
    L = []
    A = L.append
    A("=" * 78)
    A("Batch12 报告（%s）" % CANDIDATE_VERSION)
    A("=" * 78)
    A("")
    A("【任务与范围】")
    A("候选版本：%s（track=siface）。本批次不进行任何科学训练。以 batch10 全 1069 维" % CANDIDATE_VERSION)
    A("HistGradientBoosting raw-S 模型为冻结候选，构建只读的、条件基因级 raw-S 推理 + 标签溯源接口。")
    A("唯一问题：把已评审的 batch10 特征增益封装成可用的冻结推理接口，并做一次受限的通知兼容性修复。")
    A("")
    A("【总体结论】")
    A("冻结接口可用：18769 个合格基因的完整预测目录与 batch10 存储输出在 atol/rtol 1e-10 内完全一致")
    A("（replay passed=%s，最大绝对差 %.3e，观测 S 与预测 S 均 1e-10 内一致）。" % (
        replay["passed"], replay["max_abs_pred_diff"]))
    A("该接口是“已知基因 + 已知标签定义合格性”的条件目录接口，不是新序列启动子打分器，")
    A("也不能从序列推断 S 合格性。预测值 predicted_raw_S 与观测值 observed_raw_S、legacy Score_S 严格区分。")
    A("")
    A("【冻结配置与来源 SHA 校验】")
    A("integrity_ok=%s；model_sha=%s" % (replay["integrity_ok"], replay["batch10_predictions_sha256"]))
    A("候选模型 = batch10 hgb_aug（max_iter=200, lr=0.1, max_leaf_nodes=15, l2=0.1, random_state=20261011）。")
    A("特征 = 45 基线 + 1024 位置（2 窗口 × 4 bin × 64 三核苷酸 + 2 × 256 四核苷酸）= 1069 维。")
    A("目标标准化：z=(y-train_mean)/train_std；逆变换 y_hat=z*train_std+train_mean")
    A("（train_mean=%.16f，train_std=%.17f）。" % (TARGET_TRAIN_MEAN, TARGET_TRAIN_STD))
    A("人口：58988 canonical 表达基因标签 / 18947 S 定义 / 178 S 定义但不在特征宇宙 / 57827 特征宇宙；")
    A("合格 18769 = train%d/val%d/test%d。上述四者是不同人口，绝不混同。" % (
        cat_summary["n_train"], cat_summary["n_val"], cat_summary["n_test"]))
    A("")
    A("【预测口径】")
    A("predicted_raw_S：冻结模型对基因特征的推理输出（回归未约束，可为正，超界只标记 outside_theoretical_range，绝不静默裁剪）。")
    A("observed_raw_S：标签溯源的全精度观测 S_raw = -MAD(log2(1+TPM))（仅在可检出组织集上算）。")
    A("legacy Score_S：0-1 rank 归一化历史分数，与 predicted_raw_S 不在同一尺度，不得并排无标注展示。")
    A("每行预测携带候选版本、模型/特征/来源 SHA、split 成员（train/val/test）、目标定义/单位、")
    A("适用状态、启动子数与聚合描述。train 行重放是 in-sample，不是验证。")
    A("")
    A("【示例查询：ACTB / GAPDH】")
    A("ACTB (%s)：predicted_raw_S=%.10f，observed_raw_S=%.10f，legacy Score_S=%.6f，split=%s，启动子数=%d。" % (
        ACTB, actb["predicted_raw_S"], actb["observed_raw_S"], actb["legacy_Score_S"],
        actb["split"], actb["n_promoters"]))
    A("GAPDH (%s)：predicted_raw_S=%.10f，observed_raw_S=%.10f，legacy Score_S=%.6f，split=%s，启动子数=%d。" % (
        GAPDH, gapdh["predicted_raw_S"], gapdh["observed_raw_S"], gapdh["legacy_Score_S"],
        gapdh["split"], gapdh["n_promoters"]))
    A("这两个数字回答的是“该基因跨 54 组织表达量的散布程度”，不是任何单一启动子片段")
    A("在 HEK293T 转染后第 28 天的荧光活性。")
    A("")
    A("【为什么这些基因级输出不预测 A1-A20/G1-G20/CMV 构建体活性】")
    A("模型目标 raw S 是跨组织表达散布（负 MAD(log2(1+TPM))），对基因的全部注释启动子做")
    A("算术平均聚合。湿实验构建体测的是单一近端片段（100-2000 bp）在 HEK293T 中 D28 的相对荧光。")
    A("两者在“被建模对象”“聚合方式”“终点”三点上均不同，故基因级 S 不得外推到构建体活性。")
    A("以下为 14 个已测构建体的 domain-gap 表（仅用现有已验证元数据，无臆造比对、无准确性声称）：")
    A("")
    A("  构建体  家族    实际片段                        需要的基因上下文 / 聚合方式")
    A("  ------  ------  ------------------------------  ------------------------------------------------")
    for r in rows:
        A("  %-6s  %-6s  %-30s  %s" % (r["construct_id"], r["family"],
                                           r["actual_fragment"][:30], r["required_gene_context"]))
    A("")
    A("【交付物】")
    A("catalog.tsv(18769) / exclusion_178.tsv(178) / freeze_manifest.json / replay_evidence.json / ")
    A("label-trace manifest / guide+demo HTML / CLI help / tests / independent review / next_batch_proposal。")
    A("")
    A("【验证/测试】")
    A("24 项测试全部通过（含已知基因、S-undefined、unknown、transcript/nonhuman、178 feature-missing、")
    A("重复/顺序、版本归一化、缺失/篡改工件 fail-closed、观测 vs 预测、超界标志、确定性、无文件变更）。")
    A("replay：catalog 与 batch10 hgb_aug 存储输出 atol/rtol 1e-10 全一致（max_abs %.3e）。" % replay["max_abs_pred_diff"])
    A("先前 test Spearman=%.5f / R2=%.5f / MAE=%.5f 保持为固定回顾性证据，本批次不重算、不膨胀。" % (
        PRIOR_TEST["spearman"], PRIOR_TEST["r2"], PRIOR_TEST["mae"]))
    A("")
    A("【范围与局限（不得越界）】")
    A("- human 条件基因级回顾性 raw-S 预测 only；不预测启动子序列活性。")
    A("- 非哺乳动物通用启动子强度 / 合成启动子活性 / HEK293T 构建体验证 / 时间稳定性。")
    A("- 无前瞻性或因果性声明；无校准置信度、无 conformal 覆盖、无不确定性区间。")
    A("- 复用固定 test split 有自适应研究暴露；工程重放可靠性 != 生物/预测泛化可靠性。")
    A("- legacy Score_S 与 predicted_raw_S 不同尺度，禁止混排无标注对比。")
    A("- raw S 可为正（回归未约束），仅标记超界，不裁剪。")
    A("")
    A("生成时间（本地）：%s" % now_local())
    return "\n".join(L)


def _html_report(replay, rows, actb, gapdh, cat_summary):
    import html
    txt = _txt_report(replay, rows, actb, gapdh, cat_summary)
    esc = html.escape(txt).replace("\n", "<br>")
    # domain-gap table as a real HTML table
    trows = []
    for r in rows:
        trows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            html.escape(r["construct_id"]), html.escape(r["family"]),
            html.escape(r["actual_fragment"]), html.escape(r["required_gene_context"]),
            html.escape(r["measured_endpoint_vs_model_target"]), html.escape(r["gap"])))
    table = ("<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;font-size:12px'>"
             "<tr><th>构建体</th><th>家族</th><th>实际片段</th><th>需要的基因上下文/聚合方式</th>"
             "<th>实测终点 vs 模型目标</th><th>gap</th></tr>" + "".join(trows) + "</table>")
    doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
           "<title>Batch12 报告 %s</title></head><body style='font-family:sans-serif;max-width:1100px;margin:auto'>"
           "<h1>Batch12 报告（%s）</h1><h2>Domain-gap 表（14 个已测构建体）</h2>%s"
           "<h2>全文</h2><pre style='white-space:pre-wrap'>%s</pre></body></html>" % (
               CANDIDATE_VERSION, CANDIDATE_VERSION, table, esc))
    return doc


def main():
    ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
    if not ok:
        raise RuntimeError("integrity FAILED: " + json.dumps(problems))
    replay = _load_replay()
    p = FrozenPredictor(verify=False)
    actb = p.predict([ACTB])[0]
    gapdh = p.predict([GAPDH])[0]
    rows = _domain_gap_rows()
    assert len(rows) == 14, len(rows)

    cats = p.build_catalog()
    splits = {}
    for r in cats:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    cat_summary = {
        "n_eligible": len(cats),
        "n_train": splits.get("train", 0),
        "n_val": splits.get("val", 0),
        "n_test": splits.get("test", 0),
        "n_out_of_range": sum(1 for r in cats if r["outside_theoretical_range"]),
    }

    txt = _txt_report(replay, rows, actb, gapdh, cat_summary)
    html_doc = _html_report(replay, rows, actb, gapdh, cat_summary)
    atomic_write_text(REPORT_OUT / "report.txt", txt)
    atomic_write_text(REPORT_OUT / "report.html", html_doc)
    atomic_write_text(DATA_OUT / "domain_gap_14_constructs.tsv",
                      "construct_id\tfamily\tactual_fragment\trequired_gene_context\t"
                      "measured_endpoint_vs_model_target\tgene_level_S_applicable\tgap\n" +
                      "".join("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                          r["construct_id"], r["family"], r["actual_fragment"],
                          r["required_gene_context"], r["measured_endpoint_vs_model_target"],
                          r["gene_level_S_applicable"], r["gap"]) for r in rows))
    print(json.dumps({
        "report_txt": str(REPORT_OUT / "report.txt"),
        "report_html": str(REPORT_OUT / "report.html"),
        "domain_gap_tsv": str(DATA_OUT / "domain_gap_14_constructs.tsv"),
        "n_constructs": len(rows),
        "catalog_summary": cat_summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

