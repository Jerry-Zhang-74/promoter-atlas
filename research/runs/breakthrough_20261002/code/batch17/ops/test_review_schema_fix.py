#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 ops (deliverable 5b tests): stage_watcher review-verdict schema fix.

Applies stage_watcher_review_schema.patch to a COPY of the live watcher (the
live watcher is never modified or restarted), imports the patched module and
checks (1) the REAL batch16 independent_review.json is now classifiable and
would emit batch16:review_final, (2) all previously supported schemas are
unchanged, (3) unknown shapes stay fail-closed.
Run: python -X utf8 test_review_schema_fix.py
"""
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SRC_DIR = ROOT / "runs" / "breakthrough_20261002" / "code" / "stage_email_v3"
PATCH = HERE / "stage_watcher_review_schema.patch"
STAGE = HERE / "_staging" / "watchertest" / "runs" / "breakthrough_20261002" / "code" / "stage_email_v3"
REVIEW = ROOT / "reports" / "breakthrough_20261002" / "batch16" / "independent_review.json"
LIVE = SRC_DIR / "stage_watcher.py"

results = []


def check(name, cond, detail=""):
    results.append({"name": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


def main():
    if (HERE / "_staging" / "watchertest").exists():
        shutil.rmtree(HERE / "_staging" / "watchertest")
    STAGE.mkdir(parents=True)
    for f in SRC_DIR.glob("*.py"):
        shutil.copy2(f, STAGE / f.name)
    r = subprocess.run(["patch", "-p1", "--forward", "-i", str(PATCH)],
                       cwd=str(HERE / "_staging" / "watchertest"),
                       capture_output=True, text=True)
    check("review-schema patch applies to watcher copy", r.returncode == 0,
          "rc=%d %s" % (r.returncode, r.stdout.strip()))

    sys.path.insert(0, str(STAGE))
    sys.path.insert(0, str(ROOT / "runs" / "breakthrough_20261002" / "code"))
    sys.modules.pop("stage_watcher", None)
    sw = importlib.import_module("stage_watcher")
    check("patched watcher importable", hasattr(sw, "_classify_scoped_top_verdict"))

    # live watcher (read-only) is unmodified
    rc = subprocess.run(["git", "diff", "--quiet", "--", str(LIVE.relative_to(ROOT))],
                        cwd=str(ROOT), capture_output=True)
    check("live stage_watcher.py untouched by this work", rc.returncode == 0,
          "git diff rc=%d" % rc.returncode)

    # (1) REAL batch16 artifact
    rj = json.loads(REVIEW.read_text(encoding="utf-8"))
    cls = sw.classify_review_verdict(rj)
    check("batch16 verdict dict no longer blocked", cls["blocked"] is False, str(cls.get("block_reason")))
    check("batch16 integrity machine verdict = PASS", cls["verdict"] == "PASS", str(cls["verdict"]))
    check("batch16 scientific text preserved verbatim",
          cls["scientific_original"] == "NONE_CLAIMED (honest negative result; no shared-backbone win; independent composite release)",
          str(cls["scientific_original"]))
    check("batch16 scientific machine verdict NOT guessed", cls["scientific_verdict"] is None,
          str(cls["scientific_verdict"]))

    # (2) previously supported schemas unchanged
    top = sw.classify_review_verdict({"verdict": "PASS"})
    check("top-level PASS still PASS", top["verdict"] == "PASS" and not top["blocked"], str(top))
    nested = sw.classify_review_verdict({"protocol_compliance": {"verdict": "PASS"},
                                         "scientific_acceptance": {"verdict": "FAIL"}})
    check("nested schema still supported", nested["kind"] == "nested" and nested["verdict"] == "PASS"
          and nested["scientific_verdict"] == "FAIL", str(nested))
    empty = sw.classify_review_verdict({})
    check("empty artifact -> kind none", empty["kind"] == "none" and not empty["blocked"], str(empty))
    conflict = sw.classify_review_verdict({"verdict": "PASS",
                                           "protocol_compliance": {"verdict": "PASS"},
                                           "scientific_acceptance": {"verdict": "PASS"}})
    check("conflicting schema stays blocked", conflict["blocked"] is True, str(conflict["block_reason"]))

    # (3) fail-closed on unknown shapes
    bad = sw.classify_review_verdict({"verdict": {"foo": "bar"}})
    check("unknown verdict dict blocked", bad["blocked"] is True, str(bad["block_reason"]))
    amb = sw.classify_review_verdict({"verdict": {"integrity": "maybe"}})
    check("non-PASS/FAIL integrity blocked", amb["blocked"] is True, str(amb["block_reason"]))
    scoped_sci = sw.classify_review_verdict({"verdict": {"integrity": "PASS", "scientific": "PASS"}})
    check("scoped integrity+scientific both carried",
          scoped_sci["verdict"] == "PASS" and scoped_sci["scientific_verdict"] == "PASS", str(scoped_sci))

    npass = sum(1 for r in results if r["pass"])
    print("\n%d/%d checks passed" % (npass, len(results)))
    (HERE / "test_review_schema_fix.results.json").write_text(
        json.dumps({"module": "stage_watcher_review_schema.patch", "n_checks": len(results),
                    "n_passed": npass, "all_pass": npass == len(results),
                    "checks": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
