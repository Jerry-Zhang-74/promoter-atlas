# -*- coding: utf-8 -*-
"""batch59 worker: independent SHA verification of the two staged releases.

Read-only. ZERO fit/solve/backward/GPU/network. Verifies:
  * source checkpoint sha256 vs the two protocol-fixed expected SHAs,
  * every file listed in each release SHA256SUMS.txt (two independent read-only passes),
  * the pinned_core PINNED_CORE_SHA256.json manifest,
  * the two staged checkpoints inside each release.
Writes reports/breakthrough_20261002/batch59/verification.json.
"""
from __future__ import annotations
import hashlib, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")

EXPECTED = {
    "Nabs": "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
    "Njoint": "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee",
}
RELEASES = {
    "Nabs": os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-first"),
    "Njoint": os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-second"),
}
CKPT = {
    "Nabs": os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models", "Nabs", "Nabs_selected.ckpt"),
    "Njoint": os.path.join(ROOT, "data", "breakthrough_20261002", "batch52", "corrected", "models", "Njoint", "Njoint_selected.ckpt"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_dir(rel_dir):
    """Two independent read-only passes over SHA256SUMS.txt."""
    rows = []
    sums = os.path.join(rel_dir, "SHA256SUMS.txt")
    with open(sums, "r", encoding="utf-8") as f:
        entries = [ln.rstrip("\n") for ln in f if ln.strip()]
    ok = 0
    for _pass in (1, 2):
        pass_ok = 0
        for ln in entries:
            want, rel = ln.split("  ", 1)
            full = os.path.join(rel_dir, rel.replace("/", os.sep))
            got = sha256_file(full) if os.path.exists(full) else None
            if got == want:
                pass_ok += 1
            elif _pass == 1:
                rows.append({"file": rel, "expected": want, "observed": got})
        if _pass == 1:
            ok = pass_ok
        else:
            if pass_ok != ok:
                rows.append({"inconsistency": "pass2 != pass1", "pass1": ok, "pass2": pass_ok})
    return {"entries": len(entries), "matched": ok, "mismatches": rows}


def main():
    out = {"kind": "batch59_release_verification", "readonly": True,
           "zero_fit": True, "zero_gpu": True, "zero_network": True}
    rel = {}
    for arm in ("Nabs", "Njoint"):
        expected = EXPECTED[arm]
        src = sha256_file(CKPT[arm])
        bundle_ck = sha256_file(os.path.join(RELEASES[arm], "model", os.path.basename(CKPT[arm])))
        d = check_dir(RELEASES[arm])
        # pinned core manifest self-check
        pcm = os.path.join(RELEASES[arm], "code", "pinned_core", "PINNED_CORE_SHA256.json")
        pcm_ok = {}
        if os.path.exists(pcm):
            man = json.load(open(pcm, encoding="utf-8"))["files_sha256"]
            for fn, want in man.items():
                got = sha256_file(os.path.join(RELEASES[arm], "code", "pinned_core", fn))
                pcm_ok[fn] = (got == want)
        rel[arm] = {
            "release_dir": os.path.relpath(RELEASES[arm], ROOT).replace(os.sep, "/"),
            "expected_weight_sha256": expected,
            "source_checkpoint_sha256": src,
            "source_checkpoint_match": src == expected,
            "staged_bundle_checkpoint_sha256": bundle_ck,
            "staged_bundle_checkpoint_match": bundle_ck == expected,
            "manifest_files_sha256sums": d,
            "pinned_core_all_match": all(pcm_ok.values()) if pcm_ok else None,
            "pinned_core_detail": pcm_ok,
        }
    out["releases"] = rel
    out["all_weights_match"] = all(rel[a]["source_checkpoint_match"] and rel[a]["staged_bundle_checkpoint_match"]
                                   for a in rel)
    out["all_manifest_files_match"] = all(rel[a]["manifest_files_sha256sums"]["mismatches"] == [] for a in rel)
    os.makedirs(REP, exist_ok=True)
    with open(os.path.join(REP, "verification.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"all_weights_match": out["all_weights_match"],
                      "all_manifest_files_match": out["all_manifest_files_match"],
                      "nabs_files": rel["Nabs"]["manifest_files_sha256sums"]["matched"],
                      "njoint_files": rel["Njoint"]["manifest_files_sha256sums"]["matched"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
