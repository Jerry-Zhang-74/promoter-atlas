# -*- coding: utf-8 -*-
"""Run the batch57 repaired artificial no-fit software tests and write ONE evidence JSON.

Usage: python selfcheck_rc2.py [--out PATH]
No training, no solve, no gradients, no GPU, no network.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "reports",
                                           "breakthrough_20261002", "batch57", "code_selfcheck.json"))


def _run(script, ev_name):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""      # hard CPU-only
    env["PYTHONUTF8"] = "1"
    p = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "tests", script)],
                       capture_output=True, text=True, env=env)
    ev = os.path.join(HERE, "tests", ev_name)
    data = json.load(open(ev, encoding="utf-8")) if os.path.exists(ev) else {}
    return {"script": script, "returncode": p.returncode, "evidence": data,
            "stderr_tail": p.stderr[-800:] if p.returncode != 0 else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    res = [_run("test_nofit_rc2.py", "no_fit_evidence_rc2.json"),
           _run("test_pipeline_rc2.py", "pipeline_test_evidence_rc2.json")]
    ok = all(r["returncode"] == 0 for r in res)
    payload = {"kind": "batch57_worker_selfcheck", "all_passed": ok, "suites": res}
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    print(json.dumps({"all_passed": ok, "out": out,
                      "passed": [s["evidence"].get("passed") for s in res]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
