# -*- coding: utf-8 -*-
"""batch46: immutable-output SHA256 manifest + TWO read-only verifications.
Excludes self (sha_manifest.json) and mutable verification/status/review/mail files."""
import json, hashlib, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C

EXCLUDE = {"sha_manifest.json", "status.json", "worker_selfcheck.json",
           "independent_review.json", "verification.json", "manifest_verification.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = [p for p in sorted(C.OUT.rglob("*")) if p.is_file() and p.name not in EXCLUDE]
    man = {"batch": C.BATCH, "version": C.VERSION,
           "generated_at": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
           "excludes": sorted(EXCLUDE), "n_files": len(files),
           "files": {str(p.relative_to(C.OUT)).replace("\\", "/"):
                     {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}}
    (C.OUT / "sha_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding="utf-8")
    rounds = []
    for rnd in (1, 2):
        ok, bad = True, []
        for rel, rec in man["files"].items():
            p = C.OUT / rel
            if not p.exists() or sha(p) != rec["sha256"] or p.stat().st_size != rec["bytes"]:
                ok, bad = False, bad + [rel]
        rounds.append({"round": rnd, "ok": ok, "bad": bad})
        print("verify round", rnd, "OK" if ok else ("MISMATCH " + str(bad)))
    C.write_json(C.OUT / "manifest_verification.json", {
        "batch": C.BATCH, "version": C.VERSION, "excludes": sorted(EXCLUDE),
        "n_files": man["n_files"], "rounds": rounds,
        "read_only": True, "self_and_mutable_excluded": True})
    print("manifest files:", man["n_files"])


if __name__ == "__main__":
    main()
