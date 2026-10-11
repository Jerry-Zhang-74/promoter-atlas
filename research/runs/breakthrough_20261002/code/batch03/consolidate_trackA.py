#!/usr/bin/env python3
"""Batch03 Track A: consolidate pilot + baseline + reference metrics.

Reads the independent scratch-model metric JSONs and writes a single trackA_summary.json and a
compact comparison TSV. Read-only with respect to prior artifacts.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch03"
PRED = REPORT / "trackA_predictions"

REFERENCES = {}


def load_metrics(tag):
    p = PRED / f"metrics_{tag}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def row(metrics, split_name, subset):
    d = metrics.get(split_name, {}).get(subset)
    if not d:
        return None
    return {"n": d["n"], "spearman": d["spearman"], "r2": d["r2"], "mae": d["mae"], "mse": d["mse"]}


def main():
    tags = [
        ("v6.6.0-scratchgene.rc1_s20261011", "single (scratch CNN, gene-mean)"),
        ("v6.6.0-scratchgene.rc2_s20261011", "bag (scratch CNN, mean+max)"),
        ("v6.6.0-seqfeatridge.rc1_s20261011", "seqfeat ridge (composition)"),
    ]
    summary = {"references": REFERENCES, "models": {}}
    rows = []
    for tag, label in tags:
        m = load_metrics(tag)
        if m is None:
            print("missing", tag, file=sys.stderr)
            continue
        entry = {"tag": tag, "label": label,
                 "candidate": m.get("candidate"), "mode": m.get("mode"),
                 "selected_alpha": m.get("selected_alpha")}
        for split_name in ["val", "test"]:
            entry[f"{split_name}_full"] = row(m, split_name, "full")
            entry[f"{split_name}_unexposed"] = row(m, split_name, "unexposed")
        summary["models"][tag] = entry
        rows.append({"model": tag, "label": label,
                     "val_full_sp": entry["val_full"]["spearman"],
                     "val_unexp_sp": entry["val_unexposed"]["spearman"],
                     "test_full_sp": entry["test_full"]["spearman"],
                     "test_unexp_sp": entry["test_unexposed"]["spearman"]})

    (REPORT / "trackA_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(REPORT / "trackA_comparison.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
