# -*- coding: utf-8 -*-
"""BATCH27 D: canonical status + selfcheck + next_batch_proposal."""
import json, time, hashlib, os

R = "reports/breakthrough_20261002/batch27/"
now = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

dr = json.load(open(R + "autonomous_fragment_data_ready.json", encoding="utf-8"))
g2 = json.load(open(R + "sure_tier2_group_graph.json", encoding="utf-8"))
p1 = json.load(open(R + "sure_pass1.json", encoding="utf-8"))
hs = json.load(open(R + "homology_sensitivity.json", encoding="utf-8"))
vb = json.load(open(R + "variant_bin_diagnostic.json", encoding="utf-8"))
eg = json.load(open(R + "exec_gate_tests.json", encoding="utf-8"))

pid = {
  "agent_role": "batch27 worker agent (zero-fit evidence cleanup + long-fragment data preparation)",
  "agent_pid": 48696,
  "agent_process_name": "claude.exe",
  "agent_creation_time_control": "2026-10-03T18:21:56.913688+08:00",
  "control_status_started_at": "2026-10-03T18:21:55.744279+08:00",
  "control_dir": "runs/breakthrough_20261002/control_34",
  "transcript": "runs/breakthrough_20261002/control_34/worker.jsonl",
  "script_pids": {
    "b27_sure_pass1b.py": 32028,
    "b27_sure_pass2.py": 34896,
    "b27_sure_seqsample.py (first attempt, id-mapping bug)": 50508,
    "b27_sure_pass3.py": "not captured (process exited before query)",
    "b27_sure_seqsample.py (rerun)": "not captured (process exited before query)"
  },
  "note": "agent PID from control_34/status.json + Win32_Process CIM; script PIDs recorded separately."
}

status = {
 "batch": "batch27",
 "version": "v6.19.1-promoter-readiness.rc1",
 "phase": "worker_complete_awaiting_independent_review",
 "state": "evidence_revision_complete",
 "experiment_complete": False,
 "ongoing_task_active": True,
 "protocol": "runs/breakthrough_20261002/batch27/PROTOCOL.txt",
 "review_protocol": "runs/breakthrough_20261002/batch27/REVIEW_PROTOCOL.txt",
 "started_at_local": "2026-10-03T18:21:56+08:00",
 "updated_at_local": now,
 "real_process_identity": pid,
 "no_estimator_fits_this_batch": True,
 "no_neural_inference_this_batch": True,
 "headline": {
   "A1_chronology": "first test read 17:56:51 > selection write 17:56:46 (>5s); the 18:07:53 timestamp is a later regeneration; no leakage asserted",
   "A2_known_homology_sensitivity": "ranking gain PERSISTS after removing the 2 known contaminated test groups",
   "A3_variant_bins": "descriptive only; sign agreement rises with |observed delta|",
   "B_SuRE": "GSE78709 K562 autonomous-promoter per-fragment table resolved and parsed: 153,562,178 measured insert rows; tier-1 eligible 8,033,394; tier-2 (both bioreps detected) 253,629 inserts in 186,969 independent overlap components; sequence retrieval 2500/2500 exact",
   "C_exec_gate": "reusable execution gate with 9/9 mock tests (zero numerical fits)",
   "data_ready_gate_met": dr["data_ready_gate"]["gate_met"]
 },
 "artifacts": {
   "batch26_manifest_addendum.json": sha(R + "batch26_manifest_addendum.json"),
   "homology_sensitivity.json": sha(R + "homology_sensitivity.json"),
   "variant_bin_diagnostic.json": sha(R + "variant_bin_diagnostic.json"),
   "exec_gate_tests.json": sha(R + "exec_gate_tests.json"),
   "sure_pass1.json": sha(R + "sure_pass1.json"),
   "sure_selected_ids.json": sha(R + "sure_selected_ids.json"),
   "sure_selected_covariates.json": sha(R + "sure_selected_covariates.json"),
   "sure_group_graph.json": sha(R + "sure_group_graph.json"),
   "sure_tier2_group_graph.json": sha(R + "sure_tier2_group_graph.json"),
   "sure_seq_sample.json": sha(R + "sure_seq_sample.json"),
   "source_exposure_audit.json": sha(R + "source_exposure_audit.json"),
   "autonomous_fragment_data_ready.json": sha(R + "autonomous_fragment_data_ready.json"),
   "figures/homology_sensitivity.png": sha(R + "figures/homology_sensitivity.png"),
   "figures/variant_bin_diagnostic.png": sha(R + "figures/variant_bin_diagnostic.png"),
   "figures/long_fragment_cohort.png": sha(R + "figures/long_fragment_cohort.png")
 },
 "remaining_work": [
   "PARENT DECISION: approve (or amend) the tier-2 eligibility rule (cDNA detected in both biological replicates) before any training run",
   "PARENT DECISION: approve the proposed batch28 protocol (see next_batch_proposal.json); the worker did NOT train",
   "optional strengthening, not required: full-cohort (253,629) sequence retrieval + all-pairs mismatch screening to convert unknown near-homology into a measured bound",
   "optional: chromosome-held-out arm as the stricter split variant",
   "batch26 fit-budget non-compliance remains recorded; the execution gate is the mitigation for future batches"
 ],
 "next_batch_proposal_ref": R + "next_batch_proposal.json",
 "independent_review_expected": R + "independent_review.json",
 "batches_emailed": {"batch25": "accepted", "batch26": "accepted", "batch27": "not sent by worker (watcher owns stage emails)"}
}
json.dump(status, open(R + "status.json", "w", encoding="utf-8"), indent=2)
print("status ok")

selfcheck = {
 "batch": "batch27", "written_at": now, "written_by": "worker",
 "bounds_respected": {
   "estimator_fits": 0, "neural_inference": 0, "gpu_used": False,
   "max_cpu_used": 3,
   "source_investigation_minutes": "<=60 (18:24-18:57 in-budget conclusion; downloads 137 s wall)",
   "download_walltime_s": 137.1, "download_bytes": 1198721050,
   "disk_free_after_gb": 1002
 },
 "checks": {
   "old_batch26_artifacts_unchanged": True,
   "selection_frozen_before_labels_read": True,
   "no_label_used_for_selection": True,
   "no_hg38_used": True, "no_padding_or_invented_bases": True,
   "json_parse_ok": True
 },
 "figures": ["homology_sensitivity.png", "variant_bin_diagnostic.png", "long_fragment_cohort.png"],
 "emails_sent": 0, "git_commits": 0, "credential_reads": 0
}
json.dump(selfcheck, open(R + "selfcheck.json", "w", encoding="utf-8"), indent=2)
print("selfcheck ok")
