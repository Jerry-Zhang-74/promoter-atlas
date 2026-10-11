#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch18 deliverable 2: extract the rc2 zip into a NEW owned folder and execute
the documented example + all head/validity tests from a FOREIGN working directory.

Writes reports/breakthrough_20261002/batch18/release_extract_test.json.
No network, no credentials, no training assets; writes only under batch18 dirs.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REL = ROOT / "models" / "breakthrough_20261002" / "batch18"
ZIP = REL / "promoter_atlas_v6.15.1-release.rc2.zip"
EXTRACT = REL / "extract_check"
FOREIGN = ROOT / "data" / "breakthrough_20261002" / "batch18" / "foreign_cwd"
OUT = ROOT / "reports" / "breakthrough_20261002" / "batch18" / "release_extract_test.json"
PY = sys.executable
PKGNAME = "release_v6.15.1-release.rc2"


def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args, cwd):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", env=env)
    return {"cmd": " ".join(str(a) for a in args), "cwd": str(cwd),
            "returncode": p.returncode, "stdout_tail": (p.stdout or "")[-1500:],
            "stderr_tail": (p.stderr or "")[-600:]}


def main():
    if EXTRACT.exists():
        shutil.rmtree(EXTRACT)
    EXTRACT.mkdir(parents=True, exist_ok=True)
    FOREIGN.mkdir(parents=True, exist_ok=True)
    zip_sha = sha_file(ZIP)
    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        zf.extractall(EXTRACT)
    ext = EXTRACT / PKGNAME

    vm = json.loads((ext / "VERSION_MANIFEST.json").read_text(encoding="utf-8"))
    mismatches = []
    for rel, h in vm["package_files_sha256"].items():
        f = ext / rel
        if not f.exists() or sha_file(f) != h:
            mismatches.append(rel)

    code = ext / "code"
    checks = [
        run([PY, str(code / "run_example.py")], FOREIGN),
        run([PY, str(code / "test_input_validation.py")], FOREIGN),
        run([PY, str(code / "promoter_atlas_predict.py"),
             "--input", str(code / "example_input.npz"),
             "--out", str(FOREIGN / "preds_foreign.json")], FOREIGN),
    ]
    preds_ok = False
    pj = FOREIGN / "preds_foreign.json"
    if pj.exists():
        d = json.loads(pj.read_text(encoding="utf-8"))
        preds_ok = d.get("candidate_version") == "v6.15.1-release.rc2" and d.get("n", 0) >= 1

    report = {
        "candidate_version": "v6.15.1-release.rc2",
        "zip": str(ZIP.relative_to(ROOT)).replace("\\", "/"),
        "zip_sha256": zip_sha,
        "zip_bytes": ZIP.stat().st_size,
        "n_zip_entries": len(names),
        "extract_dir": str(EXTRACT.relative_to(ROOT)).replace("\\", "/"),
        "foreign_cwd": str(FOREIGN.relative_to(ROOT)).replace("\\", "/"),
        "all_entries": sorted(names),
        "manifest_index_files": len(vm["package_files_sha256"]),
        "manifest_hash_mismatches": mismatches,
        "manifest_index_ok": len(mismatches) == 0,
        "checks": checks,
        "predictions_json_written_and_valid": preds_ok,
        "verdict": ("PASS" if (len(mismatches) == 0 and all(c["returncode"] == 0 for c in checks)
                               and preds_ok) else "FAIL"),
        "notes": ["zip built AFTER all rc2 edits; extracted into a fresh owned folder",
                  "every documented command executed from a foreign working directory",
                  "validity checks include the F17-3 EMPTY batch (N=0) typed-error case"],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("verdict", "manifest_index_ok", "predictions_json_written_and_valid",
                       "zip_sha256", "n_zip_entries")}, ensure_ascii=False, indent=2))
    for c in checks:
        print(" check rc=%d %s" % (c["returncode"], c["cmd"][:80]))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

