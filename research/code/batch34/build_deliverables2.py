# -*- coding: utf-8 -*-
"""Batch34 C: evidence card, remaining work, next proposal, status, worker selfcheck."""
import json, os, time
from pathlib import Path

B = Path(r"E:/promoter_atlas/reports/breakthrough_20261002/batch34")
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")
VERSION = "v6.23.0-scoped-inference.rc1"

evidence_card = {
 "batch": "batch34", "version": VERSION, "generated_at": NOW,
 "nomenclature_fix": {
   "H1": "whole-fragment SPECTRUM kernel (k-mer spectrum) - the model with 133bp public HEK in-domain evaluation; remains EXPERIMENTAL",
   "H2": "POSITIONAL model - distinct; NOT to be confused with H1. No automatic promotion of either.",
   "note": "prior reports conflated H1/H2; this batch states the distinction and does NOT switch models"},
 "existing_public_HEK_performance": {
   "eval_set": "HEK673 (133 bp, official PARM evaluation space; cross-cell vs user 293FT)",
   "PARM_rho": 0.6906, "R1_rho": 0.4566, "H1_rho": 0.5264,
   "PARM_minus_R1_CI95": [0.179, 0.291],
   "source": "reports/breakthrough_20261002/batch31 (frozen saved scores; cited only, NOT re-inferred here)"},
 "failed_local_293FT_direction": {
   "endpoint": "D28 log10(GFP-positive gated median / same-batch WT-negative median)",
   "macro_direction_agreement_PARM": 0.40, "n_pairs": 21,
   "baseline_increase_fraction": 0.567, "baseline_decrease_fraction": 0.433,
   "verdict": "exploratory rule FAIL; NO model is validated for new 293FT families",
   "cell_context": "user chassis = 293FT (2026-10-03 correction); PARM HEK293 = CROSS-CELL"},
 "batch34_interface": {"what": "direct user FASTA -> frozen PARM HEK293 score (JSON/TSV)",
                        "status": "software input path completed; score biological_validation_status=UNVALIDATED_FOR_293FT",
                        "must_not_be_read_as": "293FT activity, D28 GFP, probability, or contest readiness"},
 "license": {"model": "PARM (vansteensellab) academic local evaluation; weights/source NOT redistributed",
             "deliverable": "wrapper code only; no licensed weights or third-party source copied into batch34 as ours"},
}
(B / "evidence_card.json").write_text(json.dumps(evidence_card, ensure_ascii=False, indent=2), encoding="utf-8")

remaining_work = {
 "batch": "batch34", "version": VERSION, "generated_at": NOW,
 "unresolved": [
   "NEW-SEQUENCE performance in the user's 293FT context is UNRESOLVED: no model validated on 293FT families.",
   "Batch34 A provides a usable FASTA->score path, but the score is a HEK293 cross-cell descriptor (UNVALIDATED_FOR_293FT).",
   "Batch34 B found 293FT used as an ACTIVITY-ASSAY host in 2 studies but NO accessible matched sequence+quantitative-label cohort within bound.",
   "Strongest discovered lead (HEK293FT MPRA, MAPT locus, PMID 41959502) is metadata-only; availability not yet verified.",
   "H1 (spectrum kernel) vs H2 (positional) distinction must persist; neither is promoted.",
],
 "next_actions": [
   "fresh reviewer: independently direct-call the official PARM routine on 2 valid records to crosscheck the wrapper (<=2 min GPU, within batch 10 min).",
   "fresh reviewer: verify unsupported>600 -> null score (no hidden windowing), duplicate/invalid/empty handling, and that software length support is NOT called 293FT applicability.",
   "parent: decide bounded next task (see next_batch_proposal.json).",
],
 "not_blocked": "293FT performance being unresolved does NOT block useful computation or the delivered interface.",
 "protected": "v6.3 main / v6.4 cellline pilot pointers unchanged; batch31/32/33 assets read-only.",
}
(B / "remaining_work.json").write_text(json.dumps(remaining_work, ensure_ascii=False, indent=2), encoding="utf-8")

proposal = {
 "batch": "batch34", "version": VERSION, "generated_at": NOW,
 "proposed_next": "batch35 v6.24.0-293ft-cohort-readiness.rc1",
 "title": "Bounded DATA-AVAILABILITY check for a usable 293FT-context promoter-fragment activity cohort",
 "why": "Primary objective is new-sequence activity prediction in the user's 293FT chassis. The MAPT-locus HEK293FT MPRA (PMID 41959502 / PMC13060879) is the strongest discovered lead with sequence libraries + activity readouts in HEK293FT.",
 "scope": {
   "fits": 0, "gpu_inference": 0, "training": 0,
   "retrieval_cap_mb": 50, "targeted_queries_cap": 4,
   "do": ["inspect the preprint data-availability statement + GEO/accession record",
          "determine whether per-sequence (promoter fragment) + quantitative activity label tables in HEK293FT are downloadable",
          "produce a frozen 'usable 293FT cohort' VERDICT + (if available) a draft data/split/config spec (NO training)"],
   "do_not": ["no genome/FASTQ-wide download", "no model fitting", "no download of unrelated data",
              "do NOT rename public HEK293 (PARM) or HEK293T (STR) as 293FT"]},
 "acceptance": ["cohort verdict is evidence-based with exact accession/URL and file sizes",
                "any proposed split freezes sequence-family/RC isolation before labels",
                "if no usable cohort: state so as a bounded result, keep 293FT performance unresolved"],
 "needs_parent_approval": True,
 "integration_rule": "verified new data may enter training only AFTER data/split/config are frozen and the parent approves; no tuning on the 14 local constructs or the 673 set.",
}
(B / "next_batch_proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
print("evidence_card + remaining_work + proposal written")
