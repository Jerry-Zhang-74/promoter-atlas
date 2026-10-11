# -*- coding: utf-8 -*-
"""batch30 prefit_gate.json builder (ZERO fits).  ready=true ONLY if EVERY mandatory item passes.
No placeholder fields, no canned PASS: every item records the file + observed value it read."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C


def main():
    items = {}

    a = C.read_json(C.REP30 / "audit100.json")
    items["audit100_present"] = {"ok": a is not None, "file": "audit100.json",
                                 "sha256": C.sha256_file(C.REP30 / "audit100.json") if a else None}
    items["audit100_100_exact"] = {"ok": bool(a and a.get("all_100_present_and_exact") and a.get("n_selected") == 100),
                                   "observed": None if not a else {"n_present": a["n_present"], "max_abs_dy": a["max_abs_dy"]}}

    t = C.read_json(C.REP30 / "tests.json")
    items["zerograd_preflight_pass"] = {"ok": bool(t and t["n_fail"] == 0 and t["n_pass"] >= 5
                                                 and t["gradients_taken"] is False and t["optimizer_steps"] == 0
                                                 and t["estimator_fits"] == 0),
                                        "observed": None if not t else {"n_pass": t["n_pass"], "n_fail": t["n_fail"],
                                                                        "gradients_taken": t["gradients_taken"],
                                                                        "optimizer_steps": t["optimizer_steps"]}}

    h = C.read_json(C.REP30 / "homology_alignment_audit.json")
    items["homology_audit_complete"] = {"ok": bool(h and not h.get("capped", True) and int(h.get("n_candidates_unexamined", 1)) == 0
                                                 and int(h.get("n_alignments_performed", 0)) > 0),
                                        "observed": None if not h else {"n_alignments": h["n_alignments_performed"],
                                                                        "n_unexamined": h["n_candidates_unexamined"],
                                                                        "capped": h["capped"],
                                                                        "n_confirmed_90_90": h["n_confirmed_90_90"],
                                                                        "n_confirmed_cross_block": h["n_confirmed_cross_block"]}}

    s = C.read_json(C.REP30 / "split_freeze.json")
    items["split_freeze_gates"] = {"ok": bool(s and s.get("gate_all") and not s.get("audit_capped_or_unexamined")),
                                   "observed": None if not s else {"counts": s["counts"], "gate_all": s["gate_all"],
                                                                   "hash_equivalent_to_b29": s["hash_equivalent_to_b29_split"]}}

    # ZERO fits: no estimator ledger reservation, no checkpoint artifacts, no test opened
    led = C.RUN30 / "exec_gate" / "ledger.jsonl"
    n_res = 0
    if led.exists():
        n_res = sum(1 for line in led.read_text(encoding="utf-8").splitlines() if line.strip())
    att = C.RUN30 / "attempts"
    n_ckpt = len(list(att.rglob("*.pt"))) if att.exists() else 0
    test_open = (C.REP30 / "selection_manifest.json").exists() or (C.REP30 / "test_metrics.json").exists()
    items["zero_fits_and_no_trainer"] = {"ok": (n_res == 0 and n_ckpt == 0 and not test_open),
                                         "observed": {"ledger_records": n_res, "checkpoint_files": n_ckpt,
                                                      "test_opened": bool(test_open)}}

    items["frozen_input_hashes_recorded"] = {"ok": bool(s and s.get("rows_tsv_sha256") and s.get("encoded_npz_sha256")
                                                        and s.get("split_tsv_sha256")),
                                             "observed": None if not s else {"rows": s["rows_tsv_sha256"],
                                                                             "encoded": s["encoded_npz_sha256"],
                                                                             "split": s["split_tsv_sha256"]}}

    allok = all(v["ok"] for v in items.values())
    out = {"batch": "batch30", "version": "v6.20.2-sure-native-readiness.rc1",
           "kind": "prefit_gate", "ready_for_training": bool(allok),
           "n_items": len(items), "n_pass": sum(1 for v in items.values() if v["ok"]),
           "items": items,
           "note": ("ready=true means the MANDATORY readiness evidence for a NEW fit is present and passing. "
                    "It does NOT certify empirical training-continuation equivalence (structural only), "
                    "does NOT authorize batch31 by itself, and is NOT a performance improvement."),
           "written_at": C.now_local()}
    C.atomic_write_json(C.REP30 / "prefit_gate.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
