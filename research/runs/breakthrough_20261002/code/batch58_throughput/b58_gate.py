# -*- coding: utf-8 -*-
"""batch58 throughput-variant execution gate.

Separate NEW ledger with cap THREE (parent-authorized b58-only allowance):
  (1) profile gradients execution
  (2) fresh Nabs resource-variant training
  (3) genuine continuation from a newly saved b58 checkpoint (controlled graceful stop)
This supersedes the old batch32 ceiling ONLY for b58; the original batch52 ledger (MAX 4)
is UNTOUCHED and remains the frozen scientific comparison budget.

Every reservation is CONSUMING (consumes=True): there is NO non-consuming resume loophole.
The reviewed vendored exec_gate (_exec_gate_vendored.py, sha256 e1ff80a6...) is reused
byte-for-byte and re-verified here; only the budget/profile is new.
"""
from __future__ import annotations
import hashlib, json, os

_REVIEWED_SHA256 = "e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571"
_VENDORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_exec_gate_vendored.py")

MAX_ATTEMPTS = 3
PLANNED_PURPOSES = ("b58-profile", "b58-Nabs-throughput-variant", "b58-resume-continuation")


class GateBytesChanged(RuntimeError):
    pass


def _verify_vendored() -> str:
    with open(_VENDORED, "rb") as f:
        b = f.read()
    sha = hashlib.sha256(b).hexdigest()
    if sha != _REVIEWED_SHA256:
        raise GateBytesChanged(
            "vendored exec_gate bytes changed: %s != reviewed %s" % (sha, _REVIEWED_SHA256))
    return sha


VENDORED_SHA256 = _verify_vendored()

from _exec_gate_vendored import ExecGate, BudgetExceeded  # noqa: E402


def open_ledger(ledger_path: str, budget: int = MAX_ATTEMPTS) -> ExecGate:
    if int(budget) != MAX_ATTEMPTS:
        raise ValueError("batch58 fixes the b58 estimator-attempt budget at %d" % MAX_ATTEMPTS)
    return ExecGate(ledger_path, budget)


def reserve(gate: ExecGate, purpose: str):
    """Reserve a CONSUMING slot. Never passes resume_of/resume_identity (no free resume)."""
    return gate.reserve(purpose)


def actual_status(ledger_path: str, budget: int = MAX_ATTEMPTS) -> dict:
    recs = []
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    recs.append(json.loads(ln))
    consuming = [r for r in recs if r.get("kind") == "reserved" and r.get("consumes")]
    return {"ledger": ledger_path, "budget": budget, "consumed": len(consuming),
            "remaining": max(0, budget - len(consuming)), "reserve": 0,
            "purposes": [r.get("purpose") for r in consuming],
            "vendored_sha256": VENDORED_SHA256, "planned": list(PLANNED_PURPOSES)}


__all__ = ["ExecGate", "BudgetExceeded", "open_ledger", "reserve", "actual_status",
           "MAX_ATTEMPTS", "PLANNED_PURPOSES", "VENDORED_SHA256", "GateBytesChanged"]
