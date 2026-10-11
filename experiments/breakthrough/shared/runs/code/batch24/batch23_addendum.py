#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24: batch23 closure errata/addendum (origins preserved read-only)."""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C

B23 = C.ROOT / "reports" / "breakthrough_20261002" / "batch23"

ADD = {
    "generated_at_local": C.now_local(),
    "batch": "batch24", "version": C.VERSION,
    "purpose": ("errata/addendum for batch23 closure; all batch23 originals remain unchanged "
                "(read-only) and are not rewritten, only corrected in this new artifact"),
    "corrections": [
        {"id": "E1", "target": "reports/breakthrough_20261002/batch23/public_source_inventory.json "
                              "(reused_retrospective_reference.note)",
         "issue": "arithmetic typo '674 of 676 members usable after the 3 SCP1 mutants'",
         "correct": "673 of 676 members usable after the 3 SCP1 mutants (676 - 3 = 673)",
         "status": "corrected here; original file preserved"},
        {"id": "E2", "target": "batch23 conclusion wording",
         "issue": "'no processed sequence-activity table exists'",
         "correct": ("'no processed sequence-activity table was LOCATED WITHIN THE BOUND' - an "
                     "unlocated table is not proof of non-existence"),
         "status": "corrected here; original file preserved"},
        {"id": "E3", "target": "batch23 GSE306816 rejection reason",
         "issue": "rejected categorically because it measures engineered STR variants",
         "correct": ("insufficient reason on its own: this project also targets engineered "
                     "promoter/fragment sequences. batch24/PROTOCOL qualifier applied below."),
         "status": "corrected here; see batch24/gse306816_followup.json"},
        {"id": "E4", "target": "batch23 near-homology audit wording",
         "issue": "non-overlap of 20-mers alone was described as a homology/source-family audit",
         "correct": ("20-mer non-overlap is ONLY a bounded exact-substring check. Exact substring "
                     "source-unseen is NOT proof of absence of all homology (no alignment-based "
                     "or remote-homology screening was performed)."),
         "status": "qualified here; original file preserved"},
        {"id": "E5", "target": "reports/breakthrough_20261002/batch23/local_diagnostic.json "
                              "(per_fold_descriptive, pooled_descriptive mae/r2)",
         "issue": ("per-fold and pooled MAE/R2 were computed across DIFFERENT endpoint scales "
                   "(model output units vs local log10(GFP)), which the batch21/23 protocol "
                   "prohibited"),
         "correct": ("ALL cross-endpoint error statistics in the batch23 local diagnostic "
                     "(pooled_descriptive.*.mae, pooled_descriptive.*.r2, and the per-fold mae/r2 "
                     "entries) are INVALIDATED and must be excluded from any derived report or "
                     "summary and must not be quoted as model quality. They were NOT recomputed "
                     "here (no refits, no re-quoting). The numerical RANK replay "
                     "(Spearman/rank and pairwise sign agreement) remains usable as descriptive "
                     "evidence only."),
         "status": "invalidated in derived use; originals preserved"},
        {"id": "E6", "target": "batch23 independent_review.json process identity",
         "issue": ("an intermediate 'resumed_process_pid': 50048 was a transient child writer "
                   "process, not the resumed CLI identity"),
         "correct": ("operator note: the parent-observed resumed Flash review CLI was PID 48780, "
                     "launched 2026-10-03T16:21:39.579749+08:00 and exited "
                     "2026-10-03T16:23:14.876952+08:00. The canonical review already records "
                     "48780 plus a child-writer note."),
         "status": "operator note recorded; review file preserved unmodified"},
    ],
    "preserved_originals": {
        "batch23_report_md": C.sha256_file(B23 / "report.md"),
        "batch23_public_source_inventory": C.sha256_file(B23 / "public_source_inventory.json"),
        "batch23_local_diagnostic": C.sha256_file(B23 / "local_diagnostic.json"),
        "batch23_independent_review": C.sha256_file(B23 / "independent_review.json"),
    },
}
C.atomic_write_json(C.REPORT_OUT / "batch23_closure_addendum.json", ADD)

lines = ["# batch24 - batch23 closure errata / addendum", "",
         "All batch23 originals remain read-only and unchanged; these are corrections recorded in a",
         "new batch24 artifact only.", ""]
for c in ADD["corrections"]:
    lines += ["## %s - %s" % (c["id"], c["target"]), "",
              "- issue: %s" % c["issue"], "- correct: %s" % c["correct"],
              "- status: %s" % c["status"], ""]
import pathlib
pathlib.Path(C.REPORT_OUT / "batch23_closure_addendum.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
print("wrote addendum with", len(ADD["corrections"]), "corrections")
