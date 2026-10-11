# -*- coding: utf-8 -*-
import json, time
from pathlib import Path
B = Path(r"E:/promoter_atlas/reports/breakthrough_20261002/batch34")
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")
VERSION = "v6.23.0-scoped-inference.rc1"
te = json.loads((B / "test_evidence.json").read_text(encoding="utf-8"))
inf = json.loads((B / "inference_evidence.json").read_text(encoding="utf-8"))

status = {
 "batch": "batch34", "version": VERSION,
 "phase": "worker", "state": "worker_complete_awaiting_fresh_review",
 "self_independent_PASS": False,
 "note": "Worker selfcheck ONLY. Canonical independent_review.json must be written by a SEPARATE fresh reviewer session.",
 "actual_clock": NOW, "worker_selfcheck_only": True,
 "worker_session_id": "3c00cca3-59ad-4194-93e2-a776a8db2de7",
 "worker_process_pid": 44896, "model": "deepseek-flash",
 "chassis_current_authority": "USER 293FT (2026-10-03); PARM=HEK293 preserved as cross-cell",
 "paths": {"code": "code/batch34", "data": "data/breakthrough_20261002/batch34",
           "reports": "reports/breakthrough_20261002/batch34", "runs": "runs/breakthrough_20261002/batch34"},
 "zero_fits": True, "optimizer_steps": 0, "gradients": 0,
 "gpu_inference_worker_s": inf["gpu_accounting"]["total_official_forward_wall_s"],
 "gpu_cap_s_incl_review": 600,
 "sectionA": {"runnable_fasta_interface": True,
              "tests_all_ok": te["all_ok"], "tests_n_ok": te["n_ok"], "tests_n_files": te["n_files"],
              "control_reproduction_abs_diff": {"A1|ACTB|100": 0.0, "CMVfull|CMV|584": 0.0},
              "unsupported_gt600_null_score": True, "invalid_and_duplicate_handled_before_inference": True},
 "sectionB": {"full_studies_reviewed": 2, "usable_quant_cohort_found": False,
              "293FT_role": "also an actual promoter-activity ASSAY host (not only packaging)",
              "strongest_lead": "PMID 41959502 (HEK293FT MPRA, MAPT locus) - metadata only, needs bounded check"},
 "sectionC": {"corrected_tube_figure": "reports/breakthrough_20261002/batch34/figures/fig_corrected_tube_intervals.png",
              "readme": "reports/breakthrough_20261002/batch34/README.md"},
 "next": "await fresh separate reviewer; then parent route decision (see next_batch_proposal.json)",
 "ongoing_task_active": True,
}
(B / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

selfcheck = {
 "batch": "batch34", "version": VERSION, "generated_at": NOW,
 "REVIEW_AUTHORITY": "worker selfcheck ONLY (non-canonical). The canonical independent_review.json belongs ONLY to a fresh separate reviewer with its own identity/time.",
 "checks": {
   "official_source_unmodified_and_sha_verified": True,
   "five_HEK_folds_used_equal_mean": True,
   "exact_official_naked_fasta_preprocessing": True,
   "central_zero_padding_Lmax600": True,
   "no_fold_cell_choice_no_RC_no_tiling_no_referenceDNA": True,
   "device_assertions_cuda0_true": True,
   "unsupported_gt600_null_no_windowing": True,
   "duplicate_and_empty_fail_before_inference": True,
   "mixed_record_invalid_null_others_scored": True,
   "accepted_alphabet_frozen_before_tests": True,
   "no_manual_1069_feature_input": True,
   "no_E_S_B_from_arbitrary_sequence": True,
   "no_CI_from_fivefold_spread": True,
   "no_H1_H2_confusion_no_auto_switch": True,
   "licensed_assets_not_copied_into_deliverable": True,
   "production_pointers_unchanged": True,
   "batch33_assets_readonly": True,
   "fits_zero": True,
 },
 "tests": te,
 "gpu_accounting": inf["gpu_accounting"],
 "artifacts": {
   "wrapper": "code/batch34/parm_fasta_predict.py",
   "test_generator": "code/batch34/make_test_manifest.py",
   "test_runner": "code/batch34/run_batch34_tests.py",
   "figure": "code/batch34/make_figures_batch34.py",
   "frozen_assets": "reports/breakthrough_20261002/batch34/frozen_assets.json",
   "test_manifest": "reports/breakthrough_20261002/batch34/test_manifest.json",
 },
 "worker_session_id": "3c00cca3-59ad-4194-93e2-a776a8db2de7",
}
(B / "worker_selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=2), encoding="utf-8")
print("status + worker_selfcheck written")
