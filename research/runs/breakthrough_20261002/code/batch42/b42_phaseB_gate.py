# -*- coding: utf-8 -*-
"""batch42 Phase B (gate decision): load reviewed ExecGate, verify ledger
writeability with a DUMMY record (no estimator instantiated), and record the
freeze-gate decision.  Reservations R0/R1/Rnull are performed immediately before
each solve in b42_phaseB_fits.py (reserve-then-run), never here.
"""
from __future__ import annotations
import importlib.util
import json
import time

import b42_common as C

T0 = time.time()
ledger = C.B42_RUN / "exec_gate_ledger.jsonl"
spec = importlib.util.spec_from_file_location("exec_gate", C.EXEC_GATE)
eg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg_mod)

frozen = json.load(open(C.B42_DATA / "frozen_split.json", encoding="utf-8"))
gate = frozen["gate"]
gate_obj = eg_mod.ExecGate(ledger, C.MAX_FITS)

probe_path = C.B42_RUN / "write_probe.json"
C.atomic_write_json(probe_path, {"dummy": True, "ts": C.now_local(),
                                 "note": "writeability probe, no estimator, no solve"})
probe_ok = probe_path.exists()
status_before = gate_obj.status()
decision = "PROCEED_FITS_AUTHORIZED" if gate["PASS"] else "ZERO_FIT_GATE_FAILED"

out = {
    "batch": C.BATCH, "version": C.VERSION, "phase": "B_gate_decision",
    "generated_at": C.now_local(), "elapsed_s": round(time.time() - T0, 2),
    "exec_gate_path": str(C.EXEC_GATE), "exec_gate_used": True,
    "budget": C.MAX_FITS, "ledger": str(ledger),
    "writeability_probe": probe_ok, "dummy_no_solve": True,
    "freeze_gate": gate, "decision": decision,
    "reservation_performed_here": False,
    "reservation_note": "R0/R1/Rnull reserved via ExecGate.run immediately before each solve in b42_phaseB_fits.py",
    "estimator_solves_performed_here": 0,
    "status_before": status_before,
    "gate_amendment": "PARENT PRE-FIT GROUP-COUNT CORRECTION 2026-10-04 02:42: feasibility train>=10 / test>=5 groups (rows 1000/300 unchanged); formal bootstrap still >=30 test groups",
    "script_sha256": C.sha256_file(__file__),
}
C.atomic_write_json(C.B42_RUN / "phaseB_gate.json", out)
print(json.dumps({k: out[k] for k in ("decision", "freeze_gate", "writeability_probe")}, ensure_ascii=False))
