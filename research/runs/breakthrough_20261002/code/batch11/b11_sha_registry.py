#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 SHA registry writer (sha_registry.json).

Computes SHA256 over every frozen batch11 artifact that must be reproducible
and records relative paths + byte sizes. Read-only: never refits or rewrites
any artifact except the registry itself.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (ROOT, DATA_OUT, MODEL_OUT, REPORT_OUT, TRACKA_NPZ, SPLIT_NPZ,
                        FEATURES_NPZ, MEM400, MEM2000, EXPR, B10_FEATURES1024,
                        CANDIDATE_VERSION, BATCH, sha256_file, atomic_write_json)  # noqa: E402


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def main():
    code_dir = Path(__file__).resolve().parent
    files = [
        TRACKA_NPZ, SPLIT_NPZ, FEATURES_NPZ, MEM400, MEM2000, EXPR, B10_FEATURES1024,
        DATA_OUT / "eligible_genes.tsv",
        DATA_OUT / "predictions.tsv",
        DATA_OUT / "replay.json",
        DATA_OUT / "attempts.jsonl",
        MODEL_OUT / "config.json",
        MODEL_OUT / "attempt_001_fit" / "models.joblib",
        MODEL_OUT / "attempt_001_fit" / "X_scaler_A.joblib",
        MODEL_OUT / "attempt_001_fit" / "X_scaler_B.joblib",
        MODEL_OUT / "attempt_001_fit" / "transforms.json",
        MODEL_OUT / "attempt_001_fit" / "attempt_meta.json",
        REPORT_OUT / "preflight_summary.json",
        REPORT_OUT / "metrics.json",
        REPORT_OUT / "tests_summary.json",
        REPORT_OUT / "status.json",
        REPORT_OUT / "worker_self_check.json",
        REPORT_OUT / "report.txt",
        REPORT_OUT / "report.html",
        REPORT_OUT / "next_batch_proposal.json",
        REPORT_OUT / "BATCH12_DRAFT.txt",
    ]
    code_files = sorted(code_dir.glob("b11_*.py"))
    entries = {}
    for p in files:
        if p.exists():
            entries[rel(p)] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        else:
            entries[rel(p)] = {"bytes": None, "sha256": None, "missing": True}
    for p in code_files:
        entries[rel(p)] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    registry = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "algorithm": "sha256",
        "n_files": len(entries),
        "n_missing": int(sum(1 for v in entries.values() if v.get("missing"))),
        "files": dict(sorted(entries.items())),
    }
    atomic_write_json(REPORT_OUT / "sha_registry.json", registry)
    print(json.dumps({k: registry[k] for k in ("written_at", "n_files", "n_missing")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

