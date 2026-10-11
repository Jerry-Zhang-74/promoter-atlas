# -*- coding: utf-8 -*-
"""Run the artificial no-fit software self-checks and write ONE evidence JSON.

Usage: python selfcheck.py --out <path>   (default: runs/breakthrough_20261002/batch53/code_selfcheck.json)
No training, no gradients, no GPU, no network. This is the batch53 worker selfcheck.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script):
    p = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "tests", script)],
                       capture_output=True, text=True)
    name = "no_fit_evidence.json" if "nofit" in script else "pipeline_test_evidence.json"
    ev = os.path.join(HERE, "tests", name)
    data = json.load(open(ev, encoding="utf-8")) if os.path.exists(ev) else {}
    return {"script": script, "returncode": p.returncode, "evidence": data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "batch53", "code_selfcheck.json"))
    a = ap.parse_args()
    res = [_run("test_nofit.py"), _run("test_pipeline.py")]
    ok = all(r["returncode"] == 0 for r in res)
    payload = {"kind": "batch53_worker_selfcheck", "all_passed": ok, "suites": res}
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"all_passed": ok, "out": out}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
