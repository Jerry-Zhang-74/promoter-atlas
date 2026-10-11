# -*- coding: utf-8 -*-
"""Batch35 SHA manifest + read-only verification.

FIX (defect#2). In batch34 the verification result lived INSIDE the hashed tree and was
overwritten by the verifier, so it hashed itself stale (57/58). Here:
  * the manifest EXCLUDES itself, mutable verification outputs, status, later review and mail ledger;
  * verification outputs are written OUTSIDE the hashed set (reports/batch35/verification/);
  * verification is run READ-ONLY TWICE without rebuilding and the two runs must agree exactly.
"""
import hashlib, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HASH_DIRS = [REPO / "code/batch35",
             REPO / "data/breakthrough_20261002/batch35",
             REPO / "reports/breakthrough_20261002/batch35"]
VERIFY_DIR = REPO / "reports/breakthrough_20261002/batch35/verification"   # OUTSIDE hashed set
EXCLUDE_NAMES = {"sha_manifest.json", "status.json", "worker_selfcheck.json",
                 "mail_ledger.json", "stage_email_status.json", "independent_review.json",
                 "remaining_work.json", "test_evidence.json"}
MANIFEST = REPO / "reports/breakthrough_20261002/batch35/sha_manifest.json"
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hashed_paths():
    out = {}
    for d in HASH_DIRS:
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            if VERIFY_DIR in p.parents:      # verification outputs are outside the hashed set
                continue
            if p.name in EXCLUDE_NAMES:
                continue
            out[str(p.relative_to(REPO)).replace("\\", "/")] = p
    return out


# --- build (once) ---
paths = hashed_paths()
entries = {rel: {"sha256": sha(p), "bytes": p.stat().st_size} for rel, p in paths.items()}
man = {"batch": "batch35", "version": "v6.24.0-293ft-cohort-readiness.rc1", "generated_at": NOW,
       "hashed_dirs": [str(d.relative_to(REPO)).replace("\\", "/") for d in HASH_DIRS],
       "excludes": sorted(EXCLUDE_NAMES) + ["this manifest itself",
                                            "reports/batch35/verification/ (verification outputs, outside hashed set)",
                                            "shared mutable CHANGELOG/version_registry"],
       "verification_output_location": "reports/breakthrough_20261002/batch35/verification/ (outside hashed set)",
       "n_entries": len(entries), "entries": entries}
MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_readonly():
    """Re-read the FROZEN manifest and recompute hashes. No rebuild."""
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad, present = [], []
    for rel, e in frozen["entries"].items():
        p = REPO / rel
        if not p.exists() or sha(p) != e["sha256"]:
            bad.append(rel)
        present.append(rel)
    leaked = [r for r in present if r.endswith("sha_verify_result.json")]
    return {"verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_entries": len(present),
            "n_ok": len(present) - len(bad), "all_ok": not bad, "mismatches": bad,
            "verification_output_leaked_into_manifest": leaked,
            "manifest_self_sha256": sha(MANIFEST)}


VERIFY_DIR.mkdir(parents=True, exist_ok=True)
run1 = verify_readonly()
run2 = verify_readonly()
consistent = (run1["n_entries"] == run2["n_entries"] and run1["n_ok"] == run2["n_ok"]
              and run1["all_ok"] == run2["all_ok"] and run1["mismatches"] == run2["mismatches"])
result = {"batch": "batch35", "readonly_verification_runs": 2,
          "consistent_across_runs": consistent, "run1": run1, "run2": run2,
          "all_ok": run1["all_ok"] and run2["all_ok"] and consistent
                    and not run1["verification_output_leaked_into_manifest"]}
(VERIFY_DIR / "sha_verify_result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"manifest_entries": len(entries), "run1_ok": run1["all_ok"],
                  "run2_ok": run2["all_ok"], "consistent": consistent,
                  "leaked": run1["verification_output_leaked_into_manifest"],
                  "bad": run1["mismatches"], "all_ok": result["all_ok"]}, ensure_ascii=False))
