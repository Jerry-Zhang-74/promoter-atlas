"""Optional schema-only compatibility fix for the batch35 canonical review.

Adds a top-level ``verdict`` DICTIONARY {integrity, compliance, scientific_gain}
mirroring the existing original top-level strings (unchanged), so the live
stage-email watcher can classify the artifact.  A byte-exact backup of the
original is written first.  This is NOT a new independent review: the original
author / session / clock / verdicts / caveats are preserved verbatim; only one
additive key plus an explicit addendum (with the actual edit time) is added.
"""
from __future__ import annotations
import hashlib, json, os, datetime, sys

ROOT = "E:/promoter_atlas"
P = os.path.join(ROOT, "reports/breakthrough_20261002/batch35/independent_review.json")
sys.path.insert(0, os.path.join(ROOT, "runs/breakthrough_20261002/code/batch19/stage_email_rc3"))


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


orig = open(P, "rb").read()
sha = sha_b(orig)
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bak = P.replace(".json", ".orig_backup_" + ts + ".json")
open(bak, "wb").write(orig)
print("backup", os.path.relpath(bak, ROOT), "sha", sha)

d = json.loads(orig.decode("utf-8"))
keys_before = set(d.keys())
newd = dict(d)
if "verdict" not in newd:
    newd["verdict"] = {"integrity": d.get("integrity"),
                       "compliance": d.get("compliance"),
                       "scientific_gain": d.get("scientific_gain")}
else:
    # already compatible: do not duplicate
    print("verdict already present; no change")
newd["schema_addendum"] = {
    "edited_at": now_iso(),
    "editor": "batch36 worker (schema-only watcher-compatibility)",
    "not_a_new_independent_review": True,
    "original_sha256": sha,
    "byte_exact_backup": os.path.relpath(bak, ROOT).replace("\\", "/"),
    "note": "Added top-level verdict dictionary mirroring the original integrity/compliance strings; "
            "all original author/session/clock/verdict/caveat fields preserved verbatim.",
}
open(P, "w", encoding="utf-8").write(json.dumps(newd, indent=2, ensure_ascii=False))
print("keys preserved:", keys_before.issubset(set(newd.keys())))

# verify the live watcher classifier now accepts it
try:
    import stage_watcher as W
    cls = W.classify_review_verdict(newd)
    print("classifier:", json.dumps(cls))
except Exception as e:
    print("classifier import failed:", e)

