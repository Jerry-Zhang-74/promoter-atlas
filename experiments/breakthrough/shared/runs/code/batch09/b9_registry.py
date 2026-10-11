#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final batch09 SHA registry (run after every artifact is on disk)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ, FEATURES_NPZ,
                       DATA_OUT, MODEL_OUT, REPORT_OUT, CODE_OUT, sha256_file)


def main():
    source_files = [EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ, FEATURES_NPZ]
    code_files = sorted(CODE_OUT.glob("b9_*.py"))
    artifact_files = sorted([
        *DATA_OUT.glob("*.tsv"),
        MODEL_OUT / "config.json",
        MODEL_OUT / "models.joblib",
        MODEL_OUT / "X_scaler.joblib",
        MODEL_OUT / "rank_reference.npz",
        MODEL_OUT / "transforms.json",
        REPORT_OUT / "metrics.json",
        REPORT_OUT / "preflight_summary.json",
        REPORT_OUT / "tests_summary.json",
        REPORT_OUT / "status.json",
        REPORT_OUT / "report.txt",
        REPORT_OUT / "report.html",
        REPORT_OUT / "next_batch_proposal.json",
        REPORT_OUT / "BATCH10_DRAFT.txt",
        REPORT_OUT / "sha_registry.tsv",
    ])
    lines = ["category\tpath\tsha256\tbytes"]
    for p in source_files:
        lines.append("source\t%s\t%s\t%d" % (p, sha256_file(p), p.stat().st_size))
    for p in code_files:
        lines.append("code\t%s\t%s\t%d" % (p, sha256_file(p), p.stat().st_size))
    for p in artifact_files:
        if p.exists():
            lines.append("artifact\t%s\t%s\t%d" % (p, sha256_file(p), p.stat().st_size))
    (REPORT_OUT / "sha_registry.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", REPORT_OUT / "sha_registry.tsv", "with", len(lines) - 1, "rows")


if __name__ == "__main__":
    main()
