#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHA manifest for batch23 immutable scientific outputs.

Excludes the manifest itself and MUTABLE/process files (status.json, selfcheck.json,
independent_review.json, *.tmp). Revisioned scientific outputs are listed immutably.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (ROOT, DATA_OUT, REPORT_OUT, RUN_OUT, CODE_OUT,  # noqa: E402
                        now_local, sha256_file, atomic_write_json)

MUTABLE = {"status.json", "selfcheck.json", "independent_review.json", "manifest.json"}
CARDS = ROOT / "models" / "cards"


def collect():
    items = {}
    for base in (CODE_OUT, RUN_OUT, DATA_OUT, REPORT_OUT, ROOT / "figures" / "batch23"):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix in (".tmp", ".pyc") or "__pycache__" in p.parts:
                continue
            if p.parent.name == "batch23" or "batch23" in p.parts:
                if p.name in MUTABLE:
                    continue
            items[str(p.relative_to(ROOT))] = sha256_file(p)
    card = CARDS / "promoter_atlas_v6_17_1_sequence_evidence_rc1_card.txt"
    if card.exists():
        items[str(card.relative_to(ROOT))] = sha256_file(card)
    return items


def main():
    items = collect()
    out = {"generated_at_local": now_local(), "batch": "batch23",
           "version": "v6.17.1-sequence-evidence.rc1",
           "excludes": sorted(MUTABLE), "policy":
           "manifest excludes itself and mutable status/selfcheck/review files",
           "n_files": len(items), "files": items}
    atomic_write_json(REPORT_OUT / "manifest.json", out)
    print("[b23_manifest] %d files" % len(items))
    for k in items:
        print("  ", k)


if __name__ == "__main__":
    main()
