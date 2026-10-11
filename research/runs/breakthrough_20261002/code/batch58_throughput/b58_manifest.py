# -*- coding: utf-8 -*-
"""batch58 immutable manifest builder.

Hashes the IMMUTABLE batch58 code + model artifacts. MUTABLE progress/review/mail files
(status json, worker/heartbeat, current_training_status.json, logs, gpu_*.json, THROUGHPUT_*)
are EXCLUDED on purpose so the manifest is stable across re-runs. Existing batch52/53/57 code
and all pre-existing models are never included.
"""
from __future__ import annotations
import hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EXCLUDE_SUBSTR = ("status", "worker", "heartbeat", "review", "mail", "latest", "current_training",
                  "gpu_", "log_", "process", "resume_evidence", "coverage_proof", ".lock", "_result")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build():
    files = {}
    for d, exts in ((HERE, (".py", ".md", ".json")),
                    (os.path.join(ROOT, "data", "breakthrough_20261002", "batch58"),
                     (".ckpt", ".json"))):
        for dp, _dn, fn in os.walk(d):
            for f in fn:
                if not f.endswith(exts):
                    continue
                if any(s in f for s in EXCLUDE_SUBSTR):
                    continue
                p = os.path.join(dp, f)
                files[os.path.relpath(p, ROOT)] = sha256(p)
    return {"batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_files": len(files),
            "excluded": list(EXCLUDE_SUBSTR), "files_sha256": dict(sorted(files.items()))}


if __name__ == "__main__":
    out = build()
    dest = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch58", "immutable_manifest.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"dest": dest, "n_files": out["n_files"]}))
