# -*- coding: utf-8 -*-
"""batch30 estimator-EXECUTION gate (append-only, atomic, cross-process lock).
Provenance: logic copied VERBATIM from runs/breakthrough_20261002/code/batch27/exec_gate.py
  sha256 e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571 (READ-ONLY original).
NO numerical estimator logic here.
"""
from __future__ import annotations
import json, os, time, uuid, errno


class BudgetExceeded(Exception):
    pass


class ExecGate:
    def __init__(self, ledger_path, budget):
        self.ledger = str(ledger_path); self.budget = int(budget)
        d = os.path.dirname(self.ledger)
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = self.ledger + ".lock"

    def _acquire(self, timeout=30.0):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd); return
            except OSError as e:
                if e.errno not in (errno.EEXIST, errno.EACCES, errno.EPERM):
                    raise
                if time.time() - t0 > timeout:
                    raise TimeoutError("gate lock timeout")
                time.sleep(0.005)

    def _release(self):
        for _ in range(50):
            try:
                os.remove(self._lock); return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.005)

    def _read(self):
        recs = []
        if os.path.exists(self.ledger):
            with open(self.ledger, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        return recs

    def _append(self, rec):
        with open(self.ledger, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

    @staticmethod
    def _consuming(recs):
        return [r for r in recs if r.get("kind") == "reserved" and r.get("consumes")]

    def status(self):
        self._acquire()
        try:
            recs = self._read(); n = len(self._consuming(recs))
            return {"budget": self.budget, "consumed": n, "remaining": self.budget - n, "records": len(recs)}
        finally:
            self._release()

    def reserve(self, purpose, resume_of=None, resume_identity=None):
        self._acquire()
        try:
            recs = self._read(); consumes = True
            if resume_of is not None:
                base = [r for r in recs if r.get("kind") == "completed" and r.get("exec_id") == resume_of]
                if not base:
                    raise ValueError("resume_of references no completed execution: %s" % resume_of)
                stored = base[-1].get("identity")
                if resume_identity and stored and stored == resume_identity:
                    consumes = False
                else:
                    raise ValueError("resume identity mismatch; a repeated solve is a NEW execution")
            n = len(self._consuming(recs))
            if consumes and n >= self.budget:
                raise BudgetExceeded("fit budget exhausted: %d/%d consumed" % (n, self.budget))
            token = {"exec_id": uuid.uuid4().hex, "purpose": purpose, "resume_of": resume_of,
                     "consumes": consumes, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            self._append({"kind": "reserved", **token}); return token
        finally:
            self._release()

    def complete(self, token, identity=None):
        self._acquire()
        try:
            self._append({"kind": "completed", "exec_id": token["exec_id"], "purpose": token["purpose"],
                          "identity": identity, "consumes": token.get("consumes", True),
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        finally:
            self._release()

    def fail(self, token, error=""):
        self._acquire()
        try:
            self._append({"kind": "failed", "exec_id": token["exec_id"], "purpose": token["purpose"],
                          "error": str(error), "consumes": token.get("consumes", True),
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        finally:
            self._release()

    def run(self, purpose, fn, resume_of=None, resume_identity=None, identity=None):
        token = self.reserve(purpose, resume_of=resume_of, resume_identity=resume_identity)
        try:
            out = fn()
        except BaseException as e:
            self.fail(token, e); raise
        self.complete(token, identity=identity); return out
