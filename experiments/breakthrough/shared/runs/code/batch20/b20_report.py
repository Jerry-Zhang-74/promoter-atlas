#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained local HTML report for batch20 raw-input predictions."""
from __future__ import annotations

import html
import json
from pathlib import Path

from b20_common import atomic_write_json

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1b1b1b}
h1{font-size:20px} h2{font-size:16px;margin-top:24px;border-bottom:1px solid #ddd}
table{border-collapse:collapse;margin:8px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:3px 7px;text-align:left}
th{background:#f2f2f2}
.warn{background:#fff3cd;border:1px solid #ffe08a;padding:8px;margin:8px 0}
.ok{background:#e7f6e7;border:1px solid #b7e0b7;padding:8px;margin:8px 0}
.bad{background:#fde8e8;border:1px solid #f5b5b5;padding:8px;margin:8px 0}
code{background:#f5f5f5;padding:1px 4px}
.small{color:#555;font-size:12px}
"""


def _tbl(rows, cols):
    out = ["<table><tr>" + "".join("<th>%s</th>" % html.escape(str(c)) for c in cols) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % html.escape(str(r.get(c, ""))) for c in cols) + "</tr>")
    out.append("</table>")
    return "".join(out)


def write_html(path, result: dict) -> None:
    path = Path(path)
    p = []
    p.append("<!doctype html><html><head><meta charset='utf-8'><title>Promoter Atlas "
             "batch20 raw-sequence report</title><style>%s</style></head><body>" % CSS)
    p.append("<h1>Promoter Atlas v6.16.0-sequence.rc1 &mdash; raw-sequence prediction report</h1>")
    p.append("<p class='small'>generated_local %s &middot; mode <code>%s</code> &middot; "
             "input <code>%s</code></p>" % (html.escape(result.get("generated_at_local", "")),
                                            html.escape(result.get("mode", "")),
                                            html.escape(str(result.get("input_name", "")))))
    p.append("<div class='warn'><b>Scope:</b> %s</div>" % html.escape(result.get("scope_note", "")))
    for warn in result.get("warnings", []):
        p.append("<div class='%s'>%s</div>" % (warn.get("level", "warn"), html.escape(warn.get("text", ""))))

    if result.get("mode") == "gene":
        p.append("<h2>Gene database mode &mdash; OBSERVED catalog labels vs MODEL predictions</h2>")
        p.append("<div class='warn'>OBSERVED = measured/catalog values from the traceable source label table. "
                 "PREDICTED = model output. They are different quantities and are never merged.</div>")
        rows = result.get("rows", [])
        p.append(_tbl(rows, ["query", "gene_id", "symbol", "n_promoters", "obs_E", "obs_S", "obs_B",
                             "obs_S_defined", "pred_E", "pred_S", "pred_B", "split", "note"]))
        p.append("<h2>Definitions</h2>")
        p.append(_tbl([{"k": k, "v": v} for k, v in result.get("definitions", {}).items()], ["k", "v"]))
        if result.get("errors"):
            p.append("<h2>Rejected queries (honest errors, no fake values)</h2>")
            p.append(_tbl(result["errors"], ["query", "error"]))
    else:
        cols = result.get("columns", [])
        rows = result.get("rows", [])
        p.append("<h2>Per-sequence predictions</h2>")
        p.append(_tbl(rows, cols))

    p.append("<h2>Provenance</h2>")
    prov = result.get("provenance", {})
    p.append("<pre class='small'>%s</pre>" % html.escape(json.dumps(prov, ensure_ascii=False, indent=2)))
    p.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(p), encoding="utf-8")


def write_json(path, result: dict) -> None:
    atomic_write_json(path, result)
