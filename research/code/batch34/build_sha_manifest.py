# -*- coding: utf-8 -*-
"""Batch34 SHA manifest (excludes itself, mutable status.json, and any mail ledger); then verify all."""
import hashlib, json, os, time
from pathlib import Path
REPO = Path(r"E:/promoter_atlas")
DIRS = [REPO / "code/batch34",
        REPO / "data/breakthrough_20261002/batch34",
        REPO / "reports/breakthrough_20261002/batch34"]
EXCLUDE_NAMES = {"sha_manifest.json", "status.json", "mail_ledger.json", "stage_email_status.json"}
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


entries = {}
for d in DIRS:
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        entries[rel] = {"sha256": sha(p), "bytes": p.stat().st_size}

man = {"batch": "batch34", "version": "v6.23.0-scoped-inference.rc1", "generated_at": NOW,
       "excludes": sorted(EXCLUDE_NAMES) + ["this manifest itself", "shared mutable CHANGELOG/version_registry"],
       "n_entries": len(entries), "entries": entries}
mp = REPO / "reports/breakthrough_20261002/batch34/sha_manifest.json"
mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

# verify
bad = []
for rel, e in entries.items():
    p = REPO / rel
    if not p.exists() or sha(p) != e["sha256"]:
        bad.append(rel)
res = {"batch": "batch34", "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "n_entries": len(entries), "n_ok": len(entries) - len(bad), "all_ok": not bad, "mismatches": bad}
(REPO / "reports/breakthrough_20261002/batch34/sha_verify_result.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"manifest_entries": len(entries), "verify_all_ok": res["all_ok"], "bad": bad}, ensure_ascii=False))
