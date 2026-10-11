# -*- coding: utf-8 -*-
"""batch53 wrapper around the ALREADY-REVIEWED estimator-execution gate.

Owner spec (batch52/PROTOCOL.txt): "Use already reviewed exec_gate implementation;
every solve / first gradient and every restarted training execution must atomically
consume a slot BEFORE execution. Gate verifies this new ledger max4 and config SHA."

The reviewed implementation is vendored byte-for-byte as _exec_gate_vendored.py
(sha256 e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571, identical to
runs/breakthrough_20261002/code/batch27/exec_gate.py). We refuse to run if the vendored
bytes ever diverge, so the gate cannot silently change under the ledger.

NO numerical estimator logic here. Import has no side effects beyond hashing one file.
"""
from __future__ import annotations
import hashlib, os

_REVIEWED_SHA256 = "e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571"
_VENDORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_exec_gate_vendored.py")


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

# Fit budget fixed by batch52 PROTOCOL: 3 planned (R0, Nabs, Njoint) + 1 reserve = MAX 4.
PLANNED_PURPOSES = ("R0", "Nabs", "Njoint")
MAX_ATTEMPTS = 4


def open_ledger(ledger_path: str, budget: int = MAX_ATTEMPTS):
    """Return an ExecGate bound to `ledger_path`. Callers pass the batch52 ledger path.

    This function does NOT reserve anything; it only constructs the gate. Reservation
    happens inside the training entrypoints immediately before the first gradient/solve.
    """
    if int(budget) != MAX_ATTEMPTS:
        raise ValueError("batch52 fixes the estimator-attempt budget at %d" % MAX_ATTEMPTS)
    return ExecGate(ledger_path, budget)


__all__ = ["ExecGate", "BudgetExceeded", "open_ledger", "MAX_ATTEMPTS",
           "PLANNED_PURPOSES", "VENDORED_SHA256"]
