#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""联合模型预测、筛选及文件处理。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


COL_ALIASES = {"_E": ("Score_E_pred", "Score_E"), "_S": ("Score_S_pred", "Score_S"),
               "_B": ("Score_B_pred", "Score_B")}


def read_predictions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError("prediction TSV is empty")
    header = rows[0]
    for key, names in COL_ALIASES.items():
        if not any(n in header for n in names):
            raise ValueError("prediction TSV missing one of %s" % (names,))
    for r in rows:
        for key, names in COL_ALIASES.items():
            col = next(n for n in names if n in header)
            r[key] = float(r[col])
    return rows


def rank(rows, b_gate=0.5, b_strata=(0.0, 0.35, 0.5, 0.65, 1.0)):
    e_order = sorted(rows, key=lambda d: (-d["_E"], d["candidate_id"]))
    s_order = sorted(rows, key=lambda d: (-d["_S"], d["candidate_id"]))
    for i, r in enumerate(e_order, 1):
        r["rank_E"] = i
    for i, r in enumerate(s_order, 1):
        r["rank_S"] = i
    for r in rows:
        r["passes_B_gate"] = int(r["_B"] >= b_gate)
        r["B_gate_threshold"] = b_gate
        bucket = "B<%.2f" % b_strata[1]
        for lo, hi in zip(b_strata[:-1], b_strata[1:]):
            if lo <= r["_B"] < hi or (hi == b_strata[-1] and r["_B"] >= hi):
                bucket = "%.2f<=B<%.2f" % (lo, hi) if hi < b_strata[-1] else "B>=%.2f" % lo
                break
        r["B_gate_stratum"] = bucket
    rows.sort(key=lambda d: (-d["passes_B_gate"], -d["_E"], d["candidate_id"]))
    return rows


def run(input_path, out="ranked_candidates.tsv", b_gate=0.5):
    rows = read_predictions(input_path)
    rows = rank(rows, b_gate=b_gate)
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    base_fields = [k for k in rows[0] if not k.startswith("_")]
    extra = [k for k in ("passes_B_gate", "B_gate_threshold", "B_gate_stratum", "rank_E", "rank_S")
             if k not in base_fields]
    fields = base_fields + extra
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote %d ranked candidates to %s" % (len(rows), out_path))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="output TSV from predict.py")
    ap.add_argument("--out", default="ranked_candidates.tsv")
    ap.add_argument("--b-gate", type=float, default=0.5, help="Score_B gate threshold")
    args = ap.parse_args(argv)
    run(args.input, args.out, args.b_gate)


if __name__ == "__main__":
    main()
