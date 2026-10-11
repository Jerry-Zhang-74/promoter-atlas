# -*- coding: utf-8 -*-
"""batch30 atomic-append registry/changelog entry.  Preserves ALL existing entries and pointers.
Adds a batch30-readiness record to reports/breakthrough_20261002/{version_registry.json,CHANGELOG.txt}.
Idempotent on (batch, version): re-running does not duplicate."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C

REG = C.REPORTS_BT / "version_registry.json"
CHG = C.REPORTS_BT / "CHANGELOG.txt"
ENTRY = {"record_type": "readiness", "batch": "batch30", "version": "v6.20.2-sure-native-readiness.rc1",
         "kind": "code/data readiness; NOT a trained model; ZERO estimator fits; ZERO optimizer steps",
         "appended_by": "batch30 worker (DeepSeek Flash equivalent implementer; parent-authorized append-only)",
         "appended_at": C.now_local(),
         "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
         "repairs": ["native-length input (one common builder, no padding at network, batch<=32 incl. eval)",
                     "durable latest.pt/best.pt + explicit verified --resume (structural)",
                     "zero-gradient preflight (hooks/widths/padding-invariance/state-identity/exec-gate)",
                     "mandatory 100-row independent source audit",
                     "bounded 90/90 alignment homology audit"],
         "evidence": {"audit100": "reports/breakthrough_20261002/batch30/audit100.json",
                      "preflight": "reports/breakthrough_20261002/batch30/tests.json",
                      "homology": "reports/breakthrough_20261002/batch30/homology_alignment_audit.json",
                      "split_freeze": "reports/breakthrough_20261002/batch30/split_freeze.json",
                      "prefit_gate": "reports/breakthrough_20261002/batch30/prefit_gate.json"},
         "next": "batch31 PROPOSED (4 executions) - NOT authorized by batch30; parent to decide after fresh review"}


def main():
    C.atomic_write_json(C.REP30 / "version_registry_entry.json", ENTRY)
    reg = C.read_json(REG) or {}
    appends = reg.get("batch_appends", [])
    if not any(a.get("batch") == "batch30" and a.get("version") == ENTRY["version"] for a in appends):
        appends.append(ENTRY)
        reg["batch_appends"] = appends
        reg["last_appended_at"] = ENTRY["appended_at"]
        reg["last_appended_by"] = ENTRY["appended_by"]
        tmp = str(REG) + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, REG)
    line = ("[%s] batch30 %s -> CODE/DATA READINESS (NOT a trained model, NO promotion, ZERO fits). "
            "Repaired native-length input (one common builder, no padding at network, batch<=32 incl eval); "
            "durable latest.pt/best.pt + verified --resume (structural only); zero-gradient preflight 5/5; "
            "mandatory 100-row independent source audit 100/100 exact (max|dy|=0.0); bounded 90/90 alignment "
            "homology audit. prefit_gate: see reports/breakthrough_20261002/batch30/prefit_gate.json. "
            "Production pointer unchanged (v6.3 main / v6.4 cellline pilot). No performance improvement claimed.\n" % (ENTRY["appended_at"], ENTRY["version"]))
    existing = CHG.read_text(encoding="utf-8") if CHG.exists() else ""
    if "batch30 v6.20.2-sure-native-readiness.rc1" not in existing:
        with open(CHG, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n" + line); f.flush(); os.fsync(f.fileno())
    print("registry append ok")


if __name__ == "__main__":
    main()
