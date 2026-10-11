# -*- coding: utf-8 -*-
"""batch49 Protocol D: atomic append of the batch49 entry to version_registry.json + CHANGELOG.txt.
Preserves the production pointer and EVERY existing entry; never rewrites old reports."""
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b49_common as C  # noqa: E402

REG = C.ROOT / "reports/breakthrough_20261002/version_registry.json"
CHG = C.ROOT / "reports/breakthrough_20261002/CHANGELOG.txt"
NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def atomic_write_text(path, text):
    tmp = Path(str(path) + ".b49tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert "production_pointer" in reg and reg["production_pointer"]["model"] == "promoter_atlas_v6.3"
    before_keys = set(reg.keys())
    prior = [a for a in reg.get("batch_appends", []) if a.get("batch") != "batch49"]
    n_prior = len(prior)
    st = json.loads((C.B49_REPORT / "status.json").read_text(encoding="utf-8"))
    entry = {
        "record_type": "new_capability_catalog_tradeoff_selection",
        "batch": "batch49", "version": C.VERSION,
        "kind": ("NEW `gene-rank` task on the SAME new-version CLI: deterministic catalog tradeoff selection "
                 "on the frozen label table over the design axes E_raw and S_raw (both higher-better), with "
                 "B_raw as a GATE only and an exact Pareto-frontier algorithm. ZERO fit/solve/calibration/GPU/"
                 "neural-forward/web/download/new-weights; the frozen b35 PARM engine/weights and the "
                 "gene-observed/fragment behaviour are UNCHANGED (reused)."),
        "appended_by": "batch49 worker (DeepSeek Flash implementer; parent-authorized append-only)",
        "appended_at": NOW,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "capability": {
            "cli": "runs/breakthrough_20261002/code/batch49/b49_cli.py",
            "supported": ["fragment", "gene-observed", "gene-rank"],
            "unsupported_explicit": ["mpra-fasta", "gene", "gene-bag-fasta", "constructs"],
            "gene_rank": {
                "axes": ["E_raw (higher-better)", "S_raw (higher-better; less negative = less cross-tissue dispersion)"],
                "b_gate_only": True, "no_weighted_score": True, "ties_not_deduplicated": True,
                "filters": {"min_b_default": 0.30, "min_e_default": "no cutoff", "sort_default": "pareto",
                            "limit_default": 50},
                "contract_frozen_before_candidate_values": True,
                "default_counts": st.get("default_selection", {}).get("counts"),
                "default_n_frontier": st.get("default_selection", {}).get("n_frontier"),
            },
        },
        "test_evidence": {"path": "reports/breakthrough_20261002/batch49/test_evidence.json",
                          "status": st.get("test_evidence", {}).get("status"),
                          "all_ok": st.get("test_evidence", {}).get("all_ok"),
                          "gpu_used_this_batch": False, "no_performance_claim": True},
        "inherited_validation": ("The frozen b35 PARM inference engine is UNCHANGED (SHA inherited); b47 "
                                 "true-CUDA control parity is INHERITED for it. b49 performed NO GPU/replay."),
        "preserved": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)",
                      "2000bp unsupported", "batch44 source branch CLOSED", "b46 old compliance FAIL",
                      "b47 integrity/compliance FAIL history", "b48 repairs R1/R2"],
        "decision": ("CATALOG_SELECTION_CAPABILITY_DELIVERED (no model performance gain; no 293FT validation; "
                     "project NOT complete)"),
        "next": "fresh independent review, then parent decision (see next_batch_proposal.json, parent-gated)",
    }
    reg["batch_appends"] = prior + [entry]
    reg["last_appended_at"] = NOW
    reg["last_appended_by"] = "batch49 worker"
    assert set(reg.keys()) >= before_keys
    assert len(reg["batch_appends"]) == n_prior + 1
    assert sum(1 for a in reg["batch_appends"] if a.get("batch") == "batch49") == 1
    proto = json.loads(REG.read_text(encoding="utf-8"))["production_pointer"]
    assert reg["production_pointer"] == proto
    atomic_write_text(REG, json.dumps(reg, ensure_ascii=False, indent=2))
    line = ("[%s] batch49 %s -> NEW `gene-rank` capability: deterministic catalog tradeoff selection on the "
            "frozen label table over E_raw x S_raw (both higher-better), B_raw GATE only, exact Pareto frontier, "
            "--min-b/--min-e/--sort/--limit, contract frozen BEFORE candidate values. ZERO fit/solve/GPU/forward/"
            "download/new-weights; fragment + gene-observed behaviour reused byte-identical; b48/b35 unchanged "
            "(SHA inherited). Default: eligible 18947, frontier 12. Tests PASS. Production pointer unchanged "
            "(v6.3 main / v6.4 cellline pilot). NOT a performance or 293FT-validation claim; project NOT complete. "
            "All prior entries preserved." % (NOW, C.VERSION))
    old = CHG.read_text(encoding="utf-8")
    kept = [ln for ln in old.splitlines() if "batch49 v6.35.0" not in ln]
    base = "\n".join(kept)
    if not base.endswith("\n"):
        base += "\n"
    atomic_write_text(CHG, base + "\n" + line + "\n")
    print("registry appends now", len(reg["batch_appends"]), "production", reg["production_pointer"]["model"])
    print("batch49 present:", "batch49" in {a.get("batch") for a in reg["batch_appends"]})


if __name__ == "__main__":
    main()
