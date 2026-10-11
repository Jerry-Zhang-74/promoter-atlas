# -*- coding: utf-8 -*-
"""batch47 Protocol D: atomic append of the new batch47 entry to version_registry.json + CHANGELOG.txt.
Preserves the production pointer and EVERY existing entry; never rewrites old reports."""
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b47_common as C  # noqa: E402

REG = C.ROOT / "reports/breakthrough_20261002/version_registry.json"
CHG = C.ROOT / "reports/breakthrough_20261002/CHANGELOG.txt"
NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def atomic_write_text(path, text):
    tmp = Path(str(path) + ".b47tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert "production_pointer" in reg and reg["production_pointer"]["model"] == "promoter_atlas_v6.3"
    before_keys = set(reg.keys())
    before_n = len(reg.get("batch_appends", []))
    existing_ids = {a.get("batch") for a in reg.get("batch_appends", [])}
    assert "batch47" not in existing_ids
    st = json.loads((C.B47_REPORT / "status.json").read_text(encoding="utf-8"))
    ev = json.loads((C.B47_REPORT / "test_evidence.json").read_text(encoding="utf-8"))
    entry = {
        "record_type": "scoped_input_routing_workflow_integration",
        "batch": "batch47", "version": C.VERSION,
        "kind": ("NEW version of the b20 command routing: explicit `fragment` (raw FASTA -> frozen b35 PARM "
                 "HEK293 engine, read-only reuse) and explicit `gene-observed` (catalog label lookup, no model/GPU). "
                 "ZERO fit/solve/calibration/optimizer/gradient/new-weights. Neural forward = frozen PARM only."),
        "appended_by": "batch47 worker (DeepSeek Flash implementer; parent-authorized append-only)",
        "appended_at": NOW,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "cli": {"path": "runs/breakthrough_20261002/code/batch47/b47_cli.py",
                "supported": ["fragment", "gene-observed"],
                "unsupported_explicit": ["mpra-fasta", "gene", "gene-bag-fasta", "constructs"]},
        "scope_correction": ("b20 raw-FASTA routed to the old 164bp K562 model and its constructs output still said "
                             "HEK293T; b47 routes raw FASTA to the frozen PARM HEK293 engine and forbids the legacy "
                             "constructs wording. Caller target cell (default 293FT) is METADATA only and never selects weights."),
        "score_label": C.SCORE_LABEL,
        "model_source_cell": C.MODEL_SOURCE_CELL,
        "target_cell_default": C.DEFAULT_TARGET_CELL,
        "biological_validation_status": "UNVALIDATED_FOR_293FT",
        "length_support": "1..600 inclusive; >600 (incl. 2000) -> UNSUPPORTED_LENGTH, score null; no truncation/window/RC/DNA change",
        "test_evidence": {"path": "reports/breakthrough_20261002/batch47/test_evidence.json",
                          "status": ev["status"], "all_ok": ev["all_ok"],
                          "gpu_seconds_used": ev.get("gpu_seconds_used"),
                          "one_worker_true_cuda_invocation": True, "no_performance_claim": True},
        "b46_addendum": {"path": "reports/breakthrough_20261002/batch47/b46_addendum_nonfinite_correction.json",
                         "fix": "dmn_K562_extended_real +Infinity -> finite 0.3809115737640978 (max - min; only the MEDIAN "
                                "candidate dropped out); dmm_K562_extended_real stays +Infinity; per-RNA-rep max-min "
                                "completed for all 244 triplets. b46 canonical bytes NOT rewritten; old compliance FAIL preserved."},
        "preserved": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)",
                      "2000bp unsupported", "batch44 source branch CLOSED", "b46 old compliance FAIL"],
        "decision": "WORKFLOW_INTEGRATION_PASS (not a scientific performance gain, not 293FT validation)",
        "next": "fresh independent review then parent decision",
    }
    reg["batch_appends"] = reg.get("batch_appends", []) + [entry]
    reg["last_appended_at"] = NOW
    reg["last_appended_by"] = "batch47 worker"
    assert set(reg.keys()) >= before_keys
    assert len(reg["batch_appends"]) == before_n + 1
    proto = json.loads(REG.read_text(encoding="utf-8"))["production_pointer"]
    assert reg["production_pointer"] == proto
    atomic_write_text(REG, json.dumps(reg, ensure_ascii=False, indent=2))

    line = ("[%s] batch47 %s -> NEW scoped input routing (workflow integration, ZERO fit/GPU-except-frozen-forward): "
            "explicit `fragment` (raw FASTA -> frozen read-only b35 PARM HEK293 engine; L_max=600 center-pad, no RC/"
            "window/DNA change; >600 incl 2000 -> null) and explicit `gene-observed` (catalog label lookup, no model/GPU, "
            "missing S -> null). Caller target cell (default 293FT) is METADATA only; model cell fixed HEK293; "
            "UNVALIDATED_FOR_293FT. Legacy 164bp K562 route / HEK293T constructs wording / gene-bag NOT exposed. "
            "Integration tests PASS (one true-CUDA control parity 3.1036716; two read-only manifest checks). "
            "b46 addendum: dmn_K562_extended_real corrected to finite 0.3809115737640978 (median-only dropout); b46 bytes "
            "preserved, old compliance FAIL retained. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). "
            "NOT a performance or 293FT-validation claim. All prior entries preserved." % (NOW, C.VERSION))
    old = CHG.read_text(encoding="utf-8")
    if not old.endswith("\n"):
        old += "\n"
    atomic_write_text(CHG, old + "\n" + line + "\n")
    print("registry appends now", len(reg["batch_appends"]), "production", reg["production_pointer"]["model"])
    print("changelog appended 1 line; batch47 present:", "batch47" in {a.get("batch") for a in reg["batch_appends"]})


if __name__ == "__main__":
    main()
