#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch18 deliverable 1: truthful-provenance copy of the batch17 resume diagnostic (F17-1).

F17-1 (batch17 independent_review.json): resume_diagnostic.json recorded
provenance.code_diag_sha256 = e7ba5709... for b17_resume_diag.py, but the
on-disk script was EDITED AFTER the run (mtime 11:24:43) and now hashes to
a3121f22....  The recorded hash therefore does not tie the on-disk script to the
numbers it produced.

Fix (this file): PRESERVE the old value as the RUN-TIME CLAIMED hash (a
self-report written by the run), ADD the current post-run on-disk hash, and
state the exact edit chronology.  Do NOT replace a run-time hash with the
current code and imply current code produced the old numbers.  If the original
run revision is unavailable (it is), say so explicitly: an independent
recomputation from the SAVED .npz arrays verifies the NUMBERS, not the exact
source-at-run.

Also: carry the original raw differences and the PARTIAL resume status verbatim;
document the ORIGINAL <=16 total production minibatch bound exceedance
(actual 20 = 12 GPU + 8 CPU) as a VIOLATION, not a retroactive re-interpretation.

Read-only w.r.t. batch17; writes only reports/breakthrough_20261002/batch18/.
"""
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas
B17 = ROOT / "reports" / "breakthrough_20261002" / "batch17"
OUT = ROOT / "reports" / "breakthrough_20261002" / "batch18"
DIAG = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch17" / "b17_resume_diag.py"
DIAG_JSON = B17 / "resume_diagnostic.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def iso_mtime(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).astimezone().isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(DIAG_JSON.read_text(encoding="utf-8"))

    run_time_claimed = d.get("provenance", {}).get("code_diag_sha256")
    current_hash = sha256(DIAG)
    diag_mtime = iso_mtime(DIAG)
    json_mtime = iso_mtime(DIAG_JSON)

    # ---- F17-1: truthful provenance -------------------------------------
    d["candidate_version"] = "v6.15.1-release.rc2"
    d["batch"] = "batch18"
    d["kind"] = ("bounded production-path checkpoint/resume diagnostic (carried forward "
                 "from batch17 with corrected provenance; no rerun, no weight change)")
    d["provenance"]["code_diag"] = "runs/breakthrough_20261002/code/batch17/b17_resume_diag.py"
    d["provenance"]["run_time_claimed_code_diag_sha256"] = run_time_claimed
    d["provenance"]["current_on_disk_code_diag_sha256"] = current_hash
    d["provenance"].pop("code_diag_sha256", None)
    d["provenance"]["old_field_name_replaced"] = ("provenance.code_diag_sha256 (value preserved "
                                                  "above as run_time_claimed_code_diag_sha256)")
    d["provenance"]["run_revision_recoverable"] = False
    d["provenance"]["run_revision_note"] = (
        "The exact source revision that produced these numbers is NOT recoverable: the "
        "diagnostic script was EDITED AFTER the run (see edit_chronology). The run-time "
        "claimed hash ({rt}) is a SELF-REPORT written by the run, not an independent digest of "
        "the on-disk file; the current on-disk file hashes to {cur}. Do NOT read the "
        "current-code hash as evidence that the current code produced the old numbers."
        .format(rt=run_time_claimed, cur=current_hash))
    d["provenance"]["independent_verification_scope"] = (
        "An independent recomputation from the SAVED .npz arrays (data/breakthrough_20261002/"
        "batch17/resume_diag/*.npz) verifies the reported DIFFERENCES. It does NOT reconstruct "
        "the exact source revision used at run time.")

    d["edit_chronology"] = [
        {"at_local": "2026-10-03T11:17:39+0800",
         "event": ("resume diagnostic run; resume_diagnostic.json first written; recorded "
                   "provenance.code_diag_sha256 = " + str(run_time_claimed) +
                   " (the script revision that was on disk at that moment)")},
        {"at_local": "2026-10-03T11:21:00+0800",
         "event": ("parent PARENT_REVIEW_NOTE_1121 requested stronger uncertainty language and "
                   "actual total-minibatch accounting")},
        {"at_local": json_mtime,
         "event": ("resume_diagnostic.json REVISED in place (uncertainty wording + minibatch "
                   "reconciliation). Its recorded_at_local (11:17:39) was NOT updated by the "
                   "worker; the file mtime is the authoritative edit time and is recorded here.")},
        {"at_local": diag_mtime,
         "event": ("b17_resume_diag.py EDITED AFTER the run (wording revision). Its hash changed "
                   "from " + str(run_time_claimed) + " to " + current_hash + ".")},
        {"at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "event": "batch18 truthful-provenance copy written (this file); no rerun performed."},
    ]

    # ---- F17-1 part 2: original bound exceedance as a documented violation ----
    d["bounds_violation"] = {
        "declared_bound": ("batch17 PROTOCOL item 3 declared '<=16 production minibatches' as a "
                           "TOTAL bound for the whole diagnostic"),
        "declared_total_production_minibatch_bound": 16,
        "actual_total_minibatches": 20,
        "actual_gpu_minibatches": 12,
        "actual_cpu_minibatches": 8,
        "processes_launched": 7,
        "status": "EXCEEDED (documented violation; NOT retroactively redefined)",
        "note": ("The original total bound (16) was exceeded by 4 minibatches (20 total). The "
                 "earlier 'bounds_respected' phrasing that scoped the <=16 bound to the single "
                 "CUDA job is a RETROACTIVE REINTERPRETATION and is preserved verbatim below as "
                 "superseded text, explicitly labelled. No scientific full fit occurred "
                 "(full_fits 0)."),
    }
    if "bounds_respected" in d:
        d["original_bounds_respected_text_SUPERSEDED"] = d.pop("bounds_respected")

    recon = d.get("minibatch_reconciliation", {})
    recon["status"] = "TOTAL BOUND VIOLATED: 20 > 16 (see bounds_violation)"
    recon["note"] = ("Total across all 7 processes: 12 GPU (CUDA) + 8 CPU = 20 minibatches, which "
                     "EXCEEDS the original TOTAL <=16 bound by 4. This is reported as a violation; "
                     "the bound is NOT redefined to cover only the single CUDA job. All diagnostic "
                     "runs are listed, none hidden; full_fits 0.")
    d["minibatch_reconciliation"] = recon

    d["scope_limits"] = [
        "single control pair on CUDA; no distribution or band is established",
        "no exclusive-cause attribution to convolution kernels is proven",
        "CPU equality is evidence for the tested CPU path only",
        "original TOTAL <=16 production-minibatch bound EXCEEDED (20 total: 12 GPU + 8 CPU); "
        "documented violation, not retroactively redefined",
        "run-time source revision not recoverable (script edited after the run); numbers are "
        "verified by saved-array recomputation, not by source-at-run reconstruction",
        "raw observed differences retained as recorded; no rounded-away or hidden runs",
    ]
    d["carried_forward_note"] = (
        "This is a provenance-corrected COPY of reports/breakthrough_20261002/batch17/"
        "resume_diagnostic.json (sha256 " + sha256(DIAG_JSON) + "). Raw observed differences, the "
        "PARTIALLY VERIFIED label and the original run provenance are carried forward unchanged; "
        "only the provenance fields, the bound-violation documentation and the candidate/batch "
        "labels were corrected. No diagnostic was rerun and no scientific weight was touched.")

    outp = OUT / "resume_diagnostic_rc2.json"
    outp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(outp), "run_time_claimed": run_time_claimed,
                      "current_on_disk": current_hash,
                      "bounds_violation": d["bounds_violation"]["status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

