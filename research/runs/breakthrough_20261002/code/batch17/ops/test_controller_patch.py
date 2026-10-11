#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 ops (deliverable 5a tests, controller level).

Applies budget_controller_carryforward.patch to a COPY of the live controller
(the live one is never modified), then instantiates the patched class and
exercises _init_status and _run_fresh against the REAL control_23 manifest to
prove a fresh/adopt launch now carries original 142.67 + topup 100.00 and keeps
cumulative spend continuous, instead of resetting top-ups to zero.
Run: python -X utf8 test_controller_patch.py
"""
import argparse
import importlib
import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SRC_CTRL = ROOT / "runs" / "breakthrough_20261002" / "code" / "budget_control"
PATCH = HERE / "budget_controller_carryforward.patch"
STAGE = HERE / "_staging" / "patchtest"
MANIFEST = ROOT / "runs" / "breakthrough_20261002" / "control_23" / "status.json"
REAL_FB = ROOT / "runs" / "breakthrough_20261002" / "control_24" / "funding_basis.json"

results = []


def check(name, cond, detail=""):
    results.append({"name": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


def prepare():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "runs" / "breakthrough_20261002" / "code" / "budget_control").mkdir(parents=True)
    base = STAGE / "runs" / "breakthrough_20261002" / "code" / "budget_control"
    for f in ("budget_common.py", "low_balance_mailer.py", "__init__.py"):
        shutil.copy2(SRC_CTRL / f, base / f)
    shutil.copy2(SRC_CTRL / "budget_controller.py", base / "budget_controller.py")
    shutil.copy2(HERE / "funding_carryforward.py", base / "funding_carryforward.py")
    r = subprocess.run(["patch", "-p1", "--forward", "-i", str(PATCH)], cwd=str(STAGE),
                       capture_output=True, text=True, shell=False)
    print("patch rc=%d out=%s err=%s" % (r.returncode, r.stdout.strip(), r.stderr.strip()))
    return base, r.returncode


def make_controller(base, out_dir):
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(ROOT / "runs" / "overnight_20261002"))
    for m in list(sys.modules):
        if m in ("budget_controller", "budget_common", "funding_carryforward"):
            del sys.modules[m]
    bc = importlib.import_module("budget_controller")
    args = argparse.Namespace(
        mode="fresh", baseline_cny="142.67", prior_spent_cny="128.63",
        manifest=str(MANIFEST), out_dir=str(out_dir), protocol_file="PROTOCOL.txt",
        science_status_file="status.json", model="deepseek-flash",
        worker_deadline="2026-10-03T15:00:00+08:00",
        review_deadline="2026-10-03T17:00:00+08:00")
    return bc, bc.BudgetController(args)


def main():
    base, rc = prepare()
    check("patch applies to controller copy", rc == 0, "rc=%d" % rc)
    out = STAGE / "out"
    out.mkdir(parents=True, exist_ok=True)
    bc, c = make_controller(base, out)

    # (A) adopt path
    c._init_status(Decimal("114.04"), None, None)
    st = c.status
    check("adopt baseline_cny = effective 242.67", st["baseline_cny"] == "242.67", st["baseline_cny"])
    check("adopt baseline_original_cny = 142.67", st.get("baseline_original_cny") == "142.67", str(st.get("baseline_original_cny")))
    check("adopt carried_topup_cny = 100.00", st.get("carried_topup_cny") == "100.00", str(st.get("carried_topup_cny")))
    check("adopt cumulative_topup_cny = 100.00 (not 0)", st["cumulative_topup_cny"] == "100.00", st["cumulative_topup_cny"])
    check("adopt spent continuous = 128.63", st["spent_cny"] == "128.63", st["spent_cny"])

    # (B) fresh path (no worker/reviewer launch; stub the phase runners)
    calls = {}
    c._write_status = lambda: None
    c._run_llm_phase = lambda *a, **k: (calls.setdefault("phase", a[0]), (0, None))[1]
    c._handoff = lambda why: 0
    c._run_review = lambda x: (calls.setdefault("review", True), 0)[1]
    c._fresh_worker_prompt = lambda: "stub prompt"
    rc2 = c._run_fresh(Decimal("114.04"))
    st2 = c.status
    check("fresh baseline_cny = effective 242.67", st2["baseline_cny"] == "242.67", st2["baseline_cny"])
    check("fresh baseline_original_cny = 142.67", st2.get("baseline_original_cny") == "142.67", str(st2.get("baseline_original_cny")))
    check("fresh cumulative_topup_cny = 100.00 (not 0)", st2["cumulative_topup_cny"] == "100.00", st2["cumulative_topup_cny"])
    check("fresh spent continuous = 128.63", st2["spent_cny"] == "128.63", st2["spent_cny"])

    # balance.jsonl written with carried topup
    lines = [json.loads(x) for x in (out / "balance.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    check("balance.jsonl topup_cny carries 100.00", lines and lines[-1]["topup_cny"] == "100.00",
          str(lines[-1] if lines else None))

    # (C) regression: legacy default when no manifest present
    c.manifest_path = None
    fb = c._funding_basis()
    check("no-manifest default baseline 142.67", fb["original_baseline_cny"] == Decimal("142.67"), str(fb["original_baseline_cny"]))
    check("no-manifest zero topup", fb["carried_topup_cny"] == Decimal("0"), str(fb["carried_topup_cny"]))

    npass = sum(1 for r in results if r["pass"])
    print("\n%d/%d checks passed" % (npass, len(results)))
    outp = HERE / "test_controller_patch.results.json"
    outp.write_text(json.dumps({"module": "budget_controller_carryforward.patch", "n_checks": len(results),
                                "n_passed": npass, "all_pass": npass == len(results),
                                "checks": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
