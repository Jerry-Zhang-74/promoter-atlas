# -*- coding: utf-8 -*-
import json, hashlib, datetime
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
B = ROOT/"reports/breakthrough_20261002/batch29"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
INTEG = "FAIL (protocol: E2 padded to maxlen2000 not native-length; worker pre-fit 100-row target check missing; saved-artifact numerics reproduced exactly)"
SCI = ("NONE / NOT ACCEPTED. Batch29 produced NO valid scientific candidate. E2 (CNN) is not the approved native-length experiment: "
 "b29_target.py pads every encoded row to maxlen2000 with code 4 and train_cnn/predict_all pass full-width codes[ch] to LegNetInspired, which maps "
 "code 4 to an all-zero channel and pools unmasked mean/max over all 2000 positions, i.e. padded pooling, exactly what the protocol forbade. "
 "Only 2 of the 4 declared estimator executions were consumed (E1 ridge, E2 true-CNN seed 20261101); E3/E4 never launched. The worker terminated "
 "after the parent HOLD. TEST labels were never opened (no selection_manifest.json, no test_metrics.json/test_predictions.npz on disk), so no "
 "test-score selection occurred. Production pointer unchanged (v6.3 main / v6.4 cellline pilot). No promotion.")
COMP = ("FAIL WITH STRONG DISCLOSURE. Execution-gate discipline PASSED (ledger reserved-before-compute; 2/4 slots consumed, no 5th/failed/smoke arm; "
 "real cuda:0 GPU telemetry). Numeric/data provenance PASSED (source SHA 8868fa64...458b85 stable; independent full-table re-derivation reproduces "
 "every headline count EXACTLY: n_rows 153562178; n_eligible 37282554; N_D 203742844; N_R1 37770653; N_R2 74081034; N_iPCR 424419578; "
 "90/90 selected-row target values match with max abs dy 0.0). BUT the native-length contract FAILED (padding), the mandatory pre-fit 100-row "
 "target verification was NOT evidenced by the worker, the checkpoint is not a true resume path, and split near-homology is a k-mer/containment "
 "APPROXIMATION, not confirmed 90pct-coverage/90pct-identity alignments (honestly disclosed in split.json).")
BASIS = ("数值/数据完整性独立复现通过（源SHA稳定、全表计数与90条选中行目标独立重算完全一致，评测指标从保存预测零拟合复现）；"
 "但协议合规不通过：E2 实际对 maxlen=2000 做了 padding 池化而非获批的 native-length，且 worker 拟合前未提供强制的 100 行目标核验。")
FULLT = ("numeric/data integrity reproduced independently (stable source SHA, exact full-table recount and exact 90-row target recompute, metrics "
 "replayed from saved predictions with zero fits); protocol compliance FAILED (E2 used padded pooling to maxlen2000 instead of the approved "
 "native-length contract; mandatory pre-fit 100-row target verification not evidenced by the worker).")
review = {
 "batch": "batch29", "version": "v6.20.1-sureproxy.rc1",
 "task": ("per-insert K562 SuRE autonomous-promoter proxy (D_i=1+P_i, P=plasmid PCRrep1+2), chromosome-block held-out, ridge + compact LegNet CNN from scratch (native length 200-2000bp)"),
 "review_kind": "fresh independent DeepSeek Flash review (read-only verification; ZERO estimator fits, ZERO optimizer steps)",
 "reviewer": "fresh independent reviewer, distinct from the batch29 worker",
 "reviewed_at": now, "review_timestamp_source": "actual system clock at review completion",
 "review_protocol": "runs/breakthrough_20261002/batch29/REVIEW_PROTOCOL.txt",
 "worker_protocol": "runs/breakthrough_20261002/batch29/PROTOCOL.txt",
 "parent_gate_hold_read": True, "parent_ops_addendum_read": True,
 "zero_extra_fits": True, "fits_performed_by_reviewer": 0, "optimizer_steps_by_reviewer": 0,
 "neural_inference_by_reviewer": "CPU-only forward passes on the saved E2 checkpoint to quantify the padding effect (no gradient, no optimizer step)",
 "gpu_used_by_reviewer": False, "does_not_send_email": True, "does_not_read_credentials": True, "does_not_start_controller": True,
 "verdict": {"integrity": INTEG, "scientific_gain": SCI, "compliance": COMP},
 "verdict_detail": {"numeric_data_integrity": "PASS", "protocol_compliance": "FAIL",
   "scientific_acceptance": "NO_GAIN_NOT_ACCEPTED", "breakthrough_accepted": False,
   "domain": "K562 SuRE per-fragment RNA/plasmid PROXY (not published genome-coverage activity; NOT HEK293T; not GFP)"},
 "integrity_basis": BASIS, "integrity_full_text": FULLT,
 "execution_gate_audit": {"ledger": "runs/breakthrough_20261002/batch29/exec_gate/ledger.jsonl",
   "reservations": 2, "completions": 2, "slots_declared": 4, "slots_consumed": 2, "slots_remaining": 2,
   "slots": [
     {"exec_id": "122c581410514e94b63f508956d82983", "purpose": "E1_ridge", "reserved": "2026-10-03T20:03:18+08:00", "completed": "2026-10-03T20:03:24+08:00", "identity": "ridge342_alpha100"},
     {"exec_id": "f0d38512ebe84517a7488b9e5c1a6cd8", "purpose": "E2_cnn_true_seed20261101", "reserved": "2026-10-03T20:03:29+08:00", "completed": "2026-10-03T20:04:57+08:00", "identity": "legnet64_seed20261101_true"}],
   "reserved_before_compute": True, "no_hidden_fifth_arm": True, "no_smoke_gradient": True, "pretests_were_forward_only": True,
   "note": "Reservations precede each fit; no extra/failed/retry arm observed. E3/E4 not attempted."},
 "gpu_telemetry_audit": {"telemetry": "runs/breakthrough_20261002/batch29/gpu_telemetry.jsonl",
   "preflight_pid": 30212, "preflight_proc_start": "2026-10-03T20:03:10", "preflight_nvidia": "RTX 5060 Laptop GPU, 8151 MiB, 150 MiB, 0 pct", "n_params": 110145,
   "train_pid": 47368, "train_proc_start": "2026-10-03T20:03:27", "train_end_nvidia": "8151 MiB, 396 MiB, 20 pct", "train_elapsed_s": 86.7,
   "true_gpu": True, "device": "cuda:0", "worker_process_pid": 44444, "worker_started": "2026-10-03T19:46:52",
   "worker_state": "failed", "worker_returncode": 4294967295, "worker_terminated_by_parent_hold": True},
 "numeric_replay": {"artifact": "reports/breakthrough_20261002/batch29_review/replay.json",
   "method": "independent metric implementation recomputed from saved val_pred.npz with zero fits",
   "E1_ridge": {"n_val": 566, "spearman": 0.13859843031303667, "mae": 0.4349792691773507, "group_balanced_mae": 0.43543234101156, "abs_diff_vs_reported": 0.0},
   "E2_cnn_true_seed20261101": {"n_val": 566, "spearman": 0.0882537386666388, "mae": 0.31814060249797116, "group_balanced_mae": 0.31866056128402787, "abs_diff_vs_reported": 0.0},
   "idx_and_labels_consistent": True},
 "source_verification": {"artifact": "reports/breakthrough_20261002/batch29_review/source_verify.json",
   "method": "independent single streaming pass over the frozen SuRE gz (read-only, no fit)",
   "source_sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85", "source_sha_recheck": "stable across 3 consecutive reads",
   "reproduced_exactly": {"n_rows": 153562178, "n_eligible": 37282554, "n_bad_coord_or_strand": 0, "n_noncanonical_chrom": 553226,
     "n_length_out_of_range": 8724178, "n_below_depth_P_lt_1": 107002220, "N_D": 203742844, "N_R1": 37770653, "N_R2": 74081034,
     "N_iPCR": 424419578, "N_plasmid_P": 50180666},
   "row_recompute": {"n_checked": 90, "count_mismatches": 0, "max_abs_y_diff": 0.0},
   "pre_fit_gate_note": "This reviewer performed the protocol-required deterministic target recompute AFTER the fact; the worker did NOT evidence it before fitting (finding P2).",
   "author_code_recovered": {"normalize_counts.py": "151e5270932a25989810852c27891c66b5359448d17341dcd051de5d44d5a8e8",
     "get_onehot_from_normalized_data.py": "bbacdace1d3a2cb30b8dbec9fe6b4f547c44f256c829f854972bddc6dc7c5e84"},
   "iPCR_semantics": "author code uses the iPCR column as the per-fragment normalizer (NEW material finding); this does NOT change the approved D=1+P proxy and does NOT by itself prove GSE78709 column identity (name-level match only)."},
 "padding_violation_evidence": {"artifact": "reports/breakthrough_20261002/batch29_review/padding_demo.json",
   "code": "b29_target.py sets codes to full((n,maxlen),4); train_cnn/predict_all pass codes[ch] (full width 2000) to LegNetInspired, which does one_hot then slices to 4 channels so code 4 becomes an all-zero channel, then unmasked mean/max over 2000 positions.",
   "encoded_width": 2000, "native_len_max": 2000, "fraction_pad_cells": 0.5343,
   "cpu_forward_demo_120_val": {"mean_abs_diff_y": 0.0087, "max_abs_diff_y": 0.0169, "pearson_pad_vs_native": 0.9682},
   "conclusion": "CONFIRMED contract violation. The numeric effect on this model outputs is modest (about 0.01 y-units, rho about 0.97) because padded positions drive near-constant BN activations, BUT the experiment is still not the approved native-length run and its val metric cannot be certified as such."},
 "test_isolation": {"test_labels_opened": False, "selection_manifest_present": False, "test_metrics_present": False, "test_predictions_present": False,
   "note": "b29_eval.py requires selection_manifest.json (absent) and was never executed; no test artifacts on disk. No test-score selection."},
 "checkpoint_finding": {"path": "runs/breakthrough_20261002/batch29/attempts/E2_cnn_true_seed20261101/ckpt.pt",
   "sha256": "666ea03c1d3320ad800cb8783c9dac24b913fa932c93082b8beef988eab751f5",
   "defect": "epoch is 3 (best) but global_step is 14679 (END-of-fit counter) and RNG/optimizer states were captured at fit end; best-model weights are paired with post-fit RNG/optimizer state, so NOT a faithful resume point.",
   "saved_only_after_whole_fit": True},
 "split_finding": {"artifact": "reports/breakthrough_20261002/batch29/split.json",
   "method": "connected components of interval overlap + exact/RC duplicates + k-mer/containment near-homolog approximation (k=13, min_shared 10, containment 0.25)",
   "confirmed_alignment_90_90": False,
   "disclosure": "split.json explicitly states this is a k-mer/containment approximation with capped pair search, NOT a full all-pairs Smith-Waterman guarantee.",
   "multiblock_components_dropped_label_free": 15, "inserts_excluded_multiblock": 987},
 "findings": [
   {"id": "P1", "severity": "high", "status": "open", "title": "Native-length contract violated: E2 trained/inferred on maxlen2000 padded inputs with unmasked mean/max pooling. E2 is not the approved equal-exact-length no-padded-pooling experiment."},
   {"id": "P2", "severity": "high", "status": "open", "title": "Mandatory pre-fit deterministic 100-row raw-count/target recompute was NOT evidenced by the worker before fitting (this reviewer performed it after the fact: 90/90 exact, max abs dy 0.0)."},
   {"id": "P3", "severity": "medium", "status": "open", "title": "Checkpoint is not a true resume path: best-epoch weights paired with end-of-fit global_step and RNG/optimizer state; saved only after the whole fit."},
   {"id": "P4", "severity": "medium", "status": "open", "title": "Split near-homology uses a k-mer/containment approximation, not confirmed 90pct/90pct alignments; residual uncapped all-pairs homology not bounded."},
   {"id": "P5", "severity": "low", "status": "closed-ok", "title": "Execution gate, GPU telemetry, source SHA and library offsets are correct; independent recount reproduces every headline number exactly."}],
 "limitations_of_this_review": [
   "No test metrics exist to review (batch29 stopped at validation; test never opened).",
   "The padding demonstration used 120 val rows on CPU only; it bounds, not fully enumerates, the padding effect.",
   "The independent source recount validates data extraction and target math, not the scientific adequacy of the D=1+P proxy.",
   "This review is zero-fit and cannot certify a corrected native-length run; that requires the repaired bounded batch."],
 "repair_plan": {
   "native_length_fix": "Slice every batch to its exact length: for a length-L bucket pass codes[ch, :L]; or build per-length tensors. Assert tensor width equals L before forward; add a unit test that a padded vs sliced forward on a fixed row differ and that the training width equals the bucket length.",
   "true_resume_fix": "Snapshot model/optimizer/scheduler/RNG at the BEST epoch (copy state at improvement time), and store epoch with matching global_step; verify bitwise-identical continue-train on a bounded 2-epoch dry resume.",
   "pre_fit_gate_fix": "Implement the deterministic 100-row recompute as an explicit pre-fit gate artifact (precomputed_100rows.json) that must exist and pass BEFORE the gate ledger is opened for any estimator.",
   "homology_fix": "Either run confirmed 90pct-coverage/90pct-identity alignments for candidate near-homolog merges, or keep the approximation but DISCLOSE it as an approximation and flag multi-block components (already dropped label-free).",
   "budget": "E3/E4 (2 remaining declared slots) may be re-attempted ONLY under corrected code in a new batch; do not spend them under the old padded path."},
 "next_steps": "Parent to freeze a NEW bounded batch30 with corrected native-length code, true resume snapshot and the pre-fit 100-row gate, then re-run the <=4 arm comparison (ridge + 2 true CNN seeds + shuffle) under <=3 GPUh. See next_batch_proposal.json.",
 "ongoing_task_active": True, "experiment_complete": False, "batch29_artifacts_preserved": True, "does_not_resolve_proxy_validity": True,
}
(B/"independent_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
chk = json.loads((B/"independent_review.json").read_text(encoding="utf-8"))
assert chk["verdict"]["integrity"].startswith("FAIL")
print("ok integrity=", chk["verdict"]["integrity"][:45], "bytes=", (B/"independent_review.json").stat().st_size)
print("sha256:", sha(B/"independent_review.json"))
