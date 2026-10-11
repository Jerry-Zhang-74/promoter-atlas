# -*- coding: utf-8 -*-
"""Atomically append the batch34 evidence version to version_registry.json + CHANGELOG.txt."""
import json, os, time
from pathlib import Path
REPO = Path(r"E:/promoter_atlas")
B = REPO / "reports/breakthrough_20261002/batch34"
REG = REPO / "reports/breakthrough_20261002/version_registry.json"
LOG = REPO / "reports/breakthrough_20261002/CHANGELOG.txt"
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")

entry = {
 "record_type": "evidence_and_usability",
 "batch": "batch34",
 "version": "v6.23.0-scoped-inference.rc1",
 "kind": "scoped FASTA->score inference interface + primary-methods closure; ZERO estimator/optimizer/gradient; official PARM HEK293 forward only (eval/no_grad)",
 "appended_by": "batch34 worker (DeepSeek Flash implementer; parent-authorized append-only)",
 "appended_at": NOW,
 "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
 "sectionA": {"runnable_fasta_interface": True, "model": "frozen official PARM HEK293 5-fold equal mean (unmodified, SHA-verified)",
              "tests_all_ok": True, "test_files": 8, "control_reproduction_abs_diff": 0.0,
              "unsupported_gt600_null": True, "invalid_duplicate_empty_fail_before_inference": True,
              "worker_gpu_forward_s": 2.114, "gpu_cap_incl_review_s": 600,
              "biological_validation_status": "UNVALIDATED_FOR_293FT"},
 "sectionB": {"full_studies": 2, "usable_quant_cohort_found": False,
              "293FT_role": "actual promoter-activity assay host (TH GFP reporter 17195153; survivin pSur 23847683), not only packaging",
              "strongest_lead": "PMID 41959502 (HEK293FT MPRA, MAPT locus) metadata-only; bounded check proposed",
              "limitation": "bounded-search result, NOT proof of non-existence; 293FT performance UNRESOLVED"},
 "sectionC": {"corrected_tube_figure": "reports/breakthrough_20261002/batch34/figures/fig_corrected_tube_intervals.png",
              "readme": "reports/breakthrough_20261002/batch34/README.md",
              "h1_h2_nomenclature_fixed": True, "no_promotion": True},
 "next": "fresh separate review; parent decides bounded batch35 (293FT cohort-readiness check)",
}
(B / "version_registry_entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

# atomic append to registry
d = json.loads(REG.read_text(encoding="utf-8"))
d.setdefault("batch_appends", []).append(entry)
d["last_appended_at"] = NOW
d["last_appended_by"] = "batch34 worker (parent-authorized append-only)"
tmp = REG.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, REG)

line = ("[%s] batch34 v6.23.0-scoped-inference.rc1 -> EVIDENCE + USABILITY (NOT a trained model, NO promotion, ZERO fits). "
        "A: minimal direct user-FASTA -> frozen official PARM HEK293 score (5-fold equal mean, unmodified SHA-verified source; "
        "central zero-pad L_max=600); JSON/TSV per-record with input_validity/software_length_support/model_source_cell=HEK293/"
        "target_requested_cell=293FT/biological_validation_status=UNVALIDATED_FOR_293FT; >600 UNSUPPORTED null score no tiling; "
        "empty/duplicate fatal before inference; mixed records deterministic; frozen 8-file test manifest all-ok; 2 controls reproduce "
        "batch31 saved scores (abs_diff 0.0, tol 1e-4); worker GPU forward 2.114 s (cap 600 s incl review). "
        "B: 293FT verified as an ACTUAL promoter-activity ASSAY host in 2 primary studies (PMID 17195153 TH-GFP reporter; PMID 23847683 "
        "survivin pSur 161/272/990 bp) - not merely packaging - but NO accessible matched sequence+quant-label 293FT cohort within bound; "
        "strongest lead PMID 41959502 (HEK293FT MPRA, MAPT locus) metadata-only. 293FT performance UNRESOLVED. "
        "C: corrected technical-tube figure (individual points + min..max interval, NOT symmetric +/-range); H1 (spectrum kernel) vs H2 "
        "(positional) nomenclature fixed, no auto-promotion. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved.\n")
with open(LOG, "a", encoding="utf-8") as f:
    f.write("\n" + line)
    f.flush(); os.fsync(f.fileno())
print(json.dumps({"registry_appends": len(d["batch_appends"]), "changelog_appended": True}, ensure_ascii=False))
