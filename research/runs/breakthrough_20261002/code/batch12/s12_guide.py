#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 standalone guide/demo HTML + CLI help.

Writes reports/breakthrough_20261002/batch12/guide_demo.html and cli_help.txt.
The guide is standalone and readable; it shows real example ACTB/GAPDH queries,
the CLI help text, and the applicability boundary (why gene-level S does not
predict construct activity). No hosted service; read-only on frozen inputs.
"""
from __future__ import annotations

import html
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s12_common import (  # noqa: E402
    CANDIDATE_VERSION, REPORT_OUT, atomic_write_text, now_local, verify_artifacts,
    FROZEN_ARTIFACTS)
from s12_predict import FrozenPredictor, build_parser  # noqa: E402

ACTB = "ENSG00000075624"
GAPDH = "ENSG00000111640"


def _cli_help():
    buf = io.StringIO()
    with redirect_stdout(buf):
        build_parser().print_help()
    return buf.getvalue()


def _example():
    p = FrozenPredictor(verify=False)
    a = p.predict([ACTB])[0]
    g = p.predict([GAPDH])[0]
    return a, g


def main():
    ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
    if not ok:
        raise RuntimeError("integrity FAILED: " + json.dumps(problems))
    actb, gapdh = _example()
    help_text = _cli_help()

    lines = []
    A = lines.append
    A("=" * 78)
    A("Batch12 冻结基因级 raw-S 接口 —— 指南 / 演示（%s）" % CANDIDATE_VERSION)
    A("=" * 78)
    A("")
    A("这是什么：只读的、条件基因级 raw-S 推理 + 标签溯源接口。冻结 batch10 全 1069 维")
    A("HistGradientBoosting 模型，不做任何训练。输入显式 canonical Ensembl 人类基因 ID，")
    A("输出该基因跨 54 组织表达散布（raw S）的预测值与溯源信息。")
    A("")
    A("这不是什么：不是新序列启动子打分器；不能从序列推断 S 合格性；不预测 A1-A20/G1-G20/CMV")
    A("构建体在 HEK293T 的活性。legacy Score_S（0-1 排序）与 predicted_raw_S 不在同一尺度。")
    A("")
    A("【示例】")
    A("ACTB %s: predicted_raw_S=%.10f, observed_raw_S=%.10f, legacy Score_S=%.6f, split=%s, 启动子数=%d" % (
        ACTB, actb["predicted_raw_S"], actb["observed_raw_S"], actb["legacy_Score_S"],
        actb["split"], actb["n_promoters"]))
    A("GAPDH %s: predicted_raw_S=%.10f, observed_raw_S=%.10f, legacy Score_S=%.6f, split=%s, 启动子数=%d" % (
        GAPDH, gapdh["predicted_raw_S"], gapdh["observed_raw_S"], gapdh["legacy_Score_S"],
        gapdh["split"], gapdh["n_promoters"]))
    A("")
    A("【子命令】")
    A("predict  条件 raw-S 预测（显式基因 ID，args 或 --file 批量）")
    A("trace    标签溯源查询（观测 E/S/B + 分数 + 合格/缺失理由 + 来源文件/哈希）")
    A("catalog  生成完整 18769 基因预测目录 + 178 排除清单")
    A("version  打印候选版本号")
    A("")
    A("【CLI 帮助】")
    A(help_text.rstrip())
    A("")
    A("【示例命令】")
    A('  python s12_predict.py predict --genes ENSG00000075624 --format tsv')
    A('  python s12_predict.py trace --genes ENSG00000111640 --format json')
    A('  python s12_predict.py catalog')
    A("")
    A("【适用边界（为什么基因级 S 不预测构建体活性）】")
    A("模型目标 raw S = -MAD(log2(1+TPM))（跨组织散布），对基因全部注释启动子做算术平均聚合。")
    A("湿实验 14 个构建体测的是单一近端片段（100-2000 bp）在 HEK293T 中 D28 的相对荧光。")
    A("两者在被建模对象、聚合方式、终点三点均不同。CMV 无 Ensembl 基因 ID，完全越界。")
    A("")
    A("生成时间（本地）：%s" % now_local())
    txt = "\n".join(lines)

    atomic_write_text(REPORT_OUT / "cli_help.txt", help_text)

    esc = html.escape(txt).replace("\n", "<br>")
    html_doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
                "<title>Batch12 指南/演示 %s</title></head>"
                "<body style='font-family:sans-serif;max-width:1000px;margin:auto'>"
                "<h1>Batch12 冻结基因级 raw-S 接口 —— 指南/演示（%s）</h1>"
                "<pre style='white-space:pre-wrap;font-size:13px'>%s</pre>"
                "</body></html>" % (CANDIDATE_VERSION, CANDIDATE_VERSION, esc))
    atomic_write_text(REPORT_OUT / "guide_demo.html", html_doc)

    print(json.dumps({"guide_demo_html": str(REPORT_OUT / "guide_demo.html"),
                      "cli_help_txt": str(REPORT_OUT / "cli_help.txt")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

