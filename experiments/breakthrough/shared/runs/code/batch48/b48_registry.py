# -*- coding: utf-8 -*-
"""batch48 Protocol D: atomic append of the batch48 entry to version_registry.json + CHANGELOG.txt.
Preserves the production pointer and EVERY existing entry; never rewrites old reports."""
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b48_common as C  # noqa: E402

REG = C.ROOT / "reports/breakthrough_20261002/version_registry.json"
CHG = C.ROOT / "reports/breakthrough_20261002/CHANGELOG.txt"
NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def atomic_write_text(path, text):
    tmp = Path(str(path) + ".b48tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert "production_pointer" in reg and reg["production_pointer"]["model"] == "promoter_atlas_v6.3"
    before_keys = set(reg.keys())
    # idempotent append: drop any prior batch48 entry (clean re-run) and keep EVERY other entry
    prior = [a for a in reg.get("batch_appends", []) if a.get("batch") != "batch48"]
    n_prior = len(prior)
    st = json.loads((C.B48_REPORT / "status.json").read_text(encoding="utf-8"))
    entry = {
        "record_type": "correctness_repair_plus_usage",
        "batch": "batch48", "version": C.VERSION,
        "kind": ("Minimal scoped repair of TWO b47 correctness defects + an accurate zh USAGE. Capability is "
                 "IDENTICAL to b47 (fragment + gene-observed); NO performance improvement is claimed or shown. "
                 "ZERO fit/solve/GPU/neural-forward/download; the frozen b35 PARM engine/weights are unchanged."),
        "appended_by": "batch48 worker (DeepSeek Flash implementer; parent-authorized append-only)",
        "appended_at": NOW,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "repairs": {
            "R1_finalization_ordering": ("b47 froze sha_manifest BEFORE writing remaining_work.json (not excluded, "
                                         "embeds generated_at) -> 1/29 stale sha and a false final-verification pass; "
                                         "b48 writes all immutable files first, then the manifest, then two READ-ONLY "
                                         "verifications, with mutable status excluded."),
            "R2_gene_observed_observedScore_S": ("b47 wrongly said observedScore_S ranks only defined-S genes; the "
                                                 "published catalog ranked the FULL 58988 rows with undefined S at the "
                                                 "bottom (build_candidate_e_labels.rank_scale none_value=-inf). b48 "
                                                 "states the full-catalog rank + denominator 58987 + defined mask."),
            "usage_zh": "PowerShell-first runnable examples; cmd optional and labelled; no internal parent-approval language.",
        },
        "cli": {"path": "runs/breakthrough_20261002/code/batch48/b48_cli.py",
                "supported": ["fragment", "gene-observed"],
                "unsupported_explicit": ["mpra-fasta", "gene", "gene-bag-fasta", "constructs"]},
        "score_label": C.SCORE_LABEL, "model_source_cell": C.MODEL_SOURCE_CELL,
        "target_cell_default": C.DEFAULT_TARGET_CELL, "biological_validation_status": "UNVALIDATED_FOR_293FT",
        "inherited_validation": ("b47 independent true-CUDA control parity 3.1036716 is INHERITED for the UNCHANGED "
                                 "inference code; b48 performed NO replay."),
        "test_evidence": {"path": "reports/breakthrough_20261002/batch48/test_evidence.json",
                          "status": st.get("test_evidence", {}).get("status"),
                          "all_ok": st.get("test_evidence", {}).get("all_ok"),
                          "gpu_used_this_batch": False, "no_performance_claim": True},
        "corrective_acceptance": {
            "path": "runs/breakthrough_20261002/batch48/CORRECTIVE_ACCEPTANCE.txt",
            "rejects": ["b47 false claim: local D28 FAIL means 'no traceable data' (prediction failure != measurement absence)",
                        "repeated inventory/UNKNOWN-only batches as a substitute for new evidence"]},
        "preserved": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)", "2000bp unsupported",
                      "batch44 source branch CLOSED", "b46 old compliance FAIL", "b47 integrity/compliance FAIL history"],
        "decision": "WORKFLOW_INTEGRATION_REPAIRS_PASS (no scientific performance gain; no 293FT validation; project NOT complete)",
        "next": "fresh independent review, then parent decision (see next_batch_proposal.json, parent-gated)",
    }
    reg["batch_appends"] = prior + [entry]
    reg["last_appended_at"] = NOW
    reg["last_appended_by"] = "batch48 worker"
    assert set(reg.keys()) >= before_keys
    assert len(reg["batch_appends"]) == n_prior + 1
    assert sum(1 for a in reg["batch_appends"] if a.get("batch") == "batch48") == 1
    proto = json.loads(REG.read_text(encoding="utf-8"))["production_pointer"]
    assert reg["production_pointer"] == proto
    atomic_write_text(REG, json.dumps(reg, ensure_ascii=False, indent=2))
    line = ("[%s] batch48 %s -> CORRECTNESS REPAIR of b47 + accurate zh USAGE (capability same as b47; ZERO fit/GPU/"
            "forward; b47 CUDA evidence INHERITED, not replayed). R1: fixed finalization ORDER (immutable files before "
            "manifest; two READ-ONLY verifications; mutable status excluded) -- b47 had 1/29 stale sha (remaining_work) "
            "and a false final-verification pass. R2: corrected gene-observed observedScore_S to the true FULL-58988 "
            "rank with undefined S at the bottom (denominator 58987, mask S_defined_main==1), sourced from "
            "build_candidate_e_labels.py. Corrective acceptance: rejects b47's false 'D28 FAIL = no traceable data' and "
            "repeated inventory. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). NOT a performance or "
            "293FT-validation claim; project NOT complete. All prior entries preserved." % (NOW, C.VERSION))
    old = CHG.read_text(encoding="utf-8")
    kept = [ln for ln in old.splitlines() if "batch48 v6.34.1" not in ln]
    base = "\n".join(kept)
    if not base.endswith("\n"):
        base += "\n"
    atomic_write_text(CHG, base + "\n" + line + "\n")
    print("registry appends now", len(reg["batch_appends"]), "production", reg["production_pointer"]["model"])
    print("batch48 present:", "batch48" in {a.get("batch") for a in reg["batch_appends"]})


if __name__ == "__main__":
    main()
