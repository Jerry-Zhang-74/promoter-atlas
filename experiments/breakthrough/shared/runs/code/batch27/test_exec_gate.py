# -*- coding: utf-8 -*-
"""BATCH27 C: mock unit tests for exec_gate. ZERO estimator fits (fake callbacks only)."""
import os, sys, json, threading, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exec_gate import ExecGate, BudgetExceeded
RES = {}
def check(name, cond, detail=""):
    RES[name] = {"pass": bool(cond), "detail": detail}
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + detail if detail else ""))

tmp = tempfile.mkdtemp(prefix="gate_")
led = os.path.join(tmp, "led.jsonl")
CALLS = {"n": 0}
def fake_ok():
    CALLS["n"] += 1; return "ok"
def fake_raise():
    CALLS["n"] += 1; raise RuntimeError("boom")

# 1) one failed callback then retry consumes 2
g = ExecGate(led, budget=4)
try:
    g.run("fit", fake_raise)
except RuntimeError: pass
g.run("fit", fake_ok)
st = g.status()
check("failed_then_retry_consumes_2", st["consumed"] == 2, "consumed=%d" % st["consumed"])

# 2) fifth call of max4 rejected BEFORE callback
g2 = ExecGate(os.path.join(tmp, "l2.jsonl"), budget=4)
c2 = {"n": 0}
def ok(): c2["n"] += 1
for _ in range(4): g2.run("fit", ok)
rejected = False
before = c2["n"]
try:
    g2.reserve("fit")
except BudgetExceeded:
    rejected = True
check("fifth_rejected_before_callback", rejected and c2["n"] == before == 4, "calls=%d" % c2["n"])

# 3) concurrent registration serialized (threads)
g3 = ExecGate(os.path.join(tmp, "l3.jsonl"), budget=1000)
N = 40
def worker():
    try: g3.run("fit", lambda: None)
    except Exception: pass
ts = [threading.Thread(target=worker) for _ in range(N)]
for t in ts: t.start()
for t in ts: t.join()
s3 = g3.status()
check("concurrent_registration_serialized", s3["consumed"] == N, "consumed=%d/%d" % (s3["consumed"], N))
lines = sum(1 for _ in open(os.path.join(tmp, "l3.jsonl"), encoding="utf-8"))
check("no_lost_updates_ledger_lines", lines >= 2 * N, "lines=%d" % lines)

# 4) unknown outcomes not silently reset: reserved w/o terminal stays consumed after reload
g4 = ExecGate(os.path.join(tmp, "l4.jsonl"), budget=2)
tok = g4.reserve("fit")           # reserved, no complete/fail (simulated crash)
g4b = ExecGate(os.path.join(tmp, "l4.jsonl"), budget=2)
s4 = g4b.status()
check("crash_reserved_stays_consumed", s4["consumed"] == 1, "consumed=%d" % s4["consumed"])
g4b.run("fit", lambda: None)      # second slot
try:
    g4b.reserve("fit"); rej = False
except BudgetExceeded: rej = True
check("crash_then_next_rejected", rej, "status=%s" % json.dumps(g4b.status()))

# 5) valid resume references existing fit (no new slot); bad identity = new execution
g5 = ExecGate(os.path.join(tmp, "l5.jsonl"), budget=2)
ident = {"model_sha": "abc", "opt_sha": "def", "rng": "ghi", "epoch": 7}
t5 = g5.reserve("fit"); g5.complete(t5, identity=ident)
r = g5.reserve("resume", resume_of=t5["exec_id"], resume_identity=ident)
check("valid_resume_no_consume", g5.status()["consumed"] == 1 and r["consumes"] is False,
      "consumed=%d" % g5.status()["consumed"])
bad = False
try:
    g5.reserve("resume", resume_of=t5["exec_id"], resume_identity={"model_sha": "zzz"})
except ValueError: bad = True
check("bad_resume_identity_rejected", bad)
check("unknown_outcome_not_reset_after_reload",
      ExecGate(os.path.join(tmp, "l4.jsonl"), 2).status()["consumed"] == 2)

n_pass = sum(1 for v in RES.values() if v["pass"])
out = {"batch": "batch27", "kind": "exec_gate_mock_tests", "n_tests": len(RES), "n_pass": n_pass,
       "zero_estimator_fits": True, "results": RES}
json.dump(out, open("reports/breakthrough_20261002/batch27/exec_gate_tests.json", "w", encoding="utf-8"), indent=2)
print("RESULT %d/%d passed" % (n_pass, len(RES)))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if n_pass == len(RES) else 1)
