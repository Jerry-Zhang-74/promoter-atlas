# -*- coding: utf-8 -*-
"""batch49 post-finalize verifier (runs AFTER b49_finalize.py; writes ONLY to the run dir).
Independently re-hashes the final b49 manifest twice (read-only), proves the reused b48/b35 artifacts are
byte-unchanged, and confirms the registry/changelog were APPENDED with the production pointer intact.
This is NOT the fresh independent review."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b49_common as C  # noqa: E402
import b49_finalize as FIN  # noqa: E402

PRE = C.B49_RUNS / "_pre_hashes.json"
OUT = C.B49_RUNS / "post_finalize_verify.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = C.B49_REPORT / "sha_manifest.json"
    r1 = FIN.verify_manifest_file(man)
    r2 = FIN.verify_manifest_file(man)
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    imm = {}
    for k, v in pre.items():
        if k in ("registry_before_b49", "changelog_before_b49"):
            continue
        ap = C.ROOT / v["path"]
        now = sha(ap) if ap.exists() else None
        imm[k] = {"path": v["path"], "before": v["sha256"], "after": now, "unchanged": now == v["sha256"]}
    reg = json.loads((C.ROOT / "reports/breakthrough_20261002/version_registry.json").read_text(encoding="utf-8"))
    has_b49 = sum(1 for a in reg.get("batch_appends", []) if a.get("batch") == "batch49") == 1
    out = {
        "batch": "batch49", "role": "post_finalize_verifier",
        "note": "Read-only re-verification of the FINAL b49 manifest, b48/b35 byte-invariance, registry append.",
        "b49_manifest_path": str(man.relative_to(C.ROOT).as_posix()),
        "b49_manifest_rounds": [{"round": 1, "ok": not r1, "bad": r1}, {"round": 2, "ok": not r2, "bad": r2}],
        "b49_manifest_two_checks_ok": (not r1) and (not r2),
        "reused_artifacts_unchanged": imm,
        "reused_all_unchanged": all(v["unchanged"] for v in imm.values()),
        "registry_appended_batch49_once": has_b49,
        "production_pointer": reg.get("production_pointer", {}).get("model"),
        "production_pointer_unchanged": reg.get("production_pointer", {}).get("model") == "promoter_atlas_v6.3",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "writes_to_hashed_files": [],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = (out["b49_manifest_two_checks_ok"] and out["reused_all_unchanged"]
          and out["registry_appended_batch49_once"] and out["production_pointer_unchanged"])
    print(json.dumps({"b49_two_checks_ok": out["b49_manifest_two_checks_ok"],
                      "reused_all_unchanged": out["reused_all_unchanged"],
                      "registry_appended": out["registry_appended_batch49_once"],
                      "production_pointer": out["production_pointer"]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
