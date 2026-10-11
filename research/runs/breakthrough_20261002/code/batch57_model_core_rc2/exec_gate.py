# -*- coding: utf-8 -*-
"""batch57 wrapper around the ALREADY-REVIEWED estimator-execution gate.

Owner spec (batch52/PROTOCOL.txt): "Use already reviewed exec_gate implementation; every
solve / first gradient and every restarted training execution must atomically consume a slot
BEFORE execution. Gate verifies this new ledger max4 and config SHA."

The reviewed implementation is vendored byte-for-byte as _exec_gate_vendored.py
(sha256 e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571, identical to
runs/breakthrough_20261002/code/batch27/exec_gate.py). We refuse to run if the vendored
bytes ever diverge, so the gate cannot silently change under the ledger.

BATCH57 REPAIRS:
  D2  resume is UNSUPPORTED and fail-closed. The vendored gate has a resume path that can
      reserve a NON-CONSUMING slot; batch53's Trainer advertised that path but never restored
      checkpoint state (a fit-budget integrity hole). Here `RESUME_SUPPORTED=False`,
      `ResumeUnsupported` is raised on any resume request, and `reserve_fit()` never passes
      resume arguments -- so EVERY attempted execution consumes a slot.
  D6  the batch53 handoff claimed "3 planned + 1 reserve = MAX 4" while the real batch52
      ledger already held consumed slots. `actual_status()` reads the real ledger, and the
      PLAN comment no longer invents a reserve.

NO numerical estimator logic here. Import has no side effects beyond hashing one file.
"""
from __future__ import annotations
import hashlib, json, os

_REVIEWED_SHA256 = "e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571"
_VENDORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_exec_gate_vendored.py")


class ResumeUnsupported(RuntimeError):
    """Raised when checkpoint resume is requested. Resume is fail-closed in batch57."""


def _verify_vendored() -> str:
    with open(_VENDORED, "rb") as f:
        b = f.read()
    sha = hashlib.sha256(b).hexdigest()
    if sha != _REVIEWED_SHA256:
        raise RuntimeError(
            "vendored exec_gate bytes changed: %s != reviewed %s" % (sha, _REVIEWED_SHA256))
    return sha


VENDORED_SHA256 = _verify_vendored()

from _exec_gate_vendored import ExecGate, BudgetExceeded  # noqa: E402

# Fit budget fixed by batch52 PROTOCOL: MAX 4 estimator executions TOTAL (historical ungated
# wrong-target R0 counts). There is NO reserve: after the planned R0/Nabs/Njoint slots are
# spent, only the actual remaining count from the real ledger applies.
PLANNED_PURPOSES = ("R0", "Nabs", "Njoint")
MAX_ATTEMPTS = 4
RESUME_SUPPORTED = False

# batch52 RUN_POLICY.json (read dynamically at launch) fixes this budget target.
_ACCEPTED_LEDGER_BUDGET = MAX_ATTEMPTS


def open_ledger(ledger_path: str, budget: int = MAX_ATTEMPTS):
    """Return an ExecGate bound to `ledger_path`. Does NOT reserve anything."""
    if int(budget) != MAX_ATTEMPTS:
        raise ValueError("batch52 fixes the estimator-attempt budget at %d" % MAX_ATTEMPTS)
    return ExecGate(ledger_path, budget)


def reserve_fit(gate, purpose: str, resume_from=None, resume_identity=None):
    """Reserve a CONSUMING slot. Resume is unsupported -> fail closed BEFORE any reservation.

    This guarantees D2: no non-consuming slot can ever be granted, and a restarted execution
    consumes a new slot (the vendored gate marks every non-resume reservation consumes=True)."""
    if resume_from is not None or resume_identity is not None:
        raise ResumeUnsupported(
            "checkpoint resume is UNSUPPORTED in batch57 (fail-closed); re-run from scratch so "
            "the restart consumes a fresh slot. Checkpoints remain READABLE for inspection/eval.")
    return gate.reserve(purpose)


def actual_status(ledger_path: str, budget: int = MAX_ATTEMPTS) -> dict:
    """D6: ACTUAL remaining attempts from the REAL ledger (no fictitious reserve)."""
    recs = []
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    recs.append(json.loads(ln))
    consuming = [r for r in recs if r.get("kind") == "reserved" and r.get("consumes")]
    consumed = len(consuming)
    return {"budget": budget, "consumed": consumed, "remaining": max(0, budget - consumed),
            "reserve": 0, "resume_supported": RESUME_SUPPORTED,
            "purposes": [r.get("purpose") for r in consuming]}


__all__ = ["ExecGate", "BudgetExceeded", "ResumeUnsupported", "open_ledger", "reserve_fit",
           "actual_status", "MAX_ATTEMPTS", "PLANNED_PURPOSES", "RESUME_SUPPORTED",
           "VENDORED_SHA256"]
