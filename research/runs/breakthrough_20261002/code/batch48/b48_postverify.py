# -*- coding: utf-8 -*-
"""batch48 post-finalize verifier. Runs AFTER b48_finalize.py. Writes ONLY to the run dir (never a
hashed file). Independently re-hashes the final b48 manifest twice (read-only) and proves that the b47
canonical defect bytes are still unchanged. This is NOT the fresh independent review."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b48_common as C  # noqa: E402
import b48_finalize as FIN  # noqa: E402

PRE = C.B48_RUNS / "_pre_hashes.json"
OUT = C.B48_RUNS / "post_finalize_verify.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = C.B48_REPORT / "sha_manifest.json"
    r1 = FIN.verify_manifest_file(man)
    r2 = FIN.verify_manifest_file(man)
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    b47 = {k: {"path": v["path"], "before": v["sha256"],
               "after": sha(C.ROOT / v["path"]) if (C.ROOT / v["path"]).exists() else None}
           for k, v in pre.items() if k.startswith("b47_")}
    for v in b47.values():
        v["unchanged"] = v["after"] == v["before"]
    out = {
        "batch": "batch48", "role": "post_finalize_verifier",
        "note": ("Read-only re-verification of the FINAL b48 manifest and b47 byte-invariance. Does not "
                 "constitute the fresh independent review."),
        "b48_manifest_path": str(man.relative_to(C.ROOT).as_posix()),
        "b48_manifest_rounds": [{"round": 1, "ok": not r1, "bad": r1},
                                {"round": 2, "ok": not r2, "bad": r2}],
        "b48_manifest_two_checks_ok": (not r1) and (not r2),
        "b47_files_unchanged": b47,
        "b47_all_unchanged": all(v["unchanged"] for v in b47.values()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "writes_to_hashed_files": [],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"b48_two_checks_ok": out["b48_manifest_two_checks_ok"],
                      "b47_all_unchanged": out["b47_all_unchanged"]}, ensure_ascii=False))
    return 0 if (out["b48_manifest_two_checks_ok"] and out["b47_all_unchanged"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
