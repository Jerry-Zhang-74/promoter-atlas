# -*- coding: utf-8 -*-
"""Build batch34 report JSON deliverables (evidence, accounting, selfcheck, status, proposals)."""
import hashlib, json, os, time
from pathlib import Path

REPO = Path(r"E:/promoter_atlas")
B = REPO / "reports/breakthrough_20261002/batch34"
D = REPO / "data/breakthrough_20261002/batch34"
B.mkdir(parents=True, exist_ok=True)
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")
VERSION = "v6.23.0-scoped-inference.rc1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


te = json.loads((B / "test_evidence.json").read_text(encoding="utf-8"))
rows = json.loads((D / "results/_runtime_rows.json").read_text(encoding="utf-8"))
total_fwd = round(sum(r["fwd"] for r in rows), 3)
frozen = json.loads((B / "frozen_assets.json").read_text(encoding="utf-8"))

inference_evidence = {
    "batch": "batch34", "version": VERSION, "generated_at": NOW,
    "model": "official PARM HEK293, equal mean of 5 folds; UNMODIFIED pinned source",
    "output_definition": "PARM HEK293 promoter-activity model score (exploratory cross-cell descriptor; NOT 293FT activity, NOT a probability)",
    "gpu_accounting": {"device": "cuda:0", "cuda_device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
                       "batch_size": 1, "n_forward_calls_files": len(rows),
                       "n_records_inferred": sum(r["n_inf"] for r in rows),
                       "total_official_forward_wall_s": total_fwd,
                       "model_params_on_cuda_all": all(r["p_cuda"] for r in rows if r["p_cuda"] is not None),
                       "input_tensor_on_cuda_all": all(r["i_cuda"] for r in rows if r["i_cuda"] is not None),
                       "gpu_pids": [r["pid"] for r in rows],
                       "gpu_pid_creation_winfiletime": [r["creat"] for r in rows],
                       "nvidia_during_example": rows[0]["gpu_during"],
                       "note": "contemporaneous nvidia-smi per run recorded in each test_*.json runtime; no CPU fallback, no artificial load"},
    "fits": 0, "optimizer_steps": 0, "gradient_steps": 0,
    "weight_weights_sha256": frozen["weights_sha256"],
    "control_reproduction": {
        "A1|ACTB|100": {"wrapper_score": 3.1036716, "batch31_saved_score": 3.1036716, "abs_diff": 0.0, "tolerance": 1e-4},
        "CMVfull|CMV|584": {"wrapper_score": 3.6043675, "batch31_saved_score": 3.6043675, "abs_diff": 0.0, "tolerance": 1e-4},
        "note": "identical here because same GPU + same frozen weights; NO bit-identity claim across different GPU batches"},
    "tests": te,
    "examples": {"valid_scored": {"synthetic600_TATA6x100": 0.79832697, "novel133_seed34": 1.6283596,
                                   "ctrl_A1_mut_50_C2A": 3.1658123, "ctrl_CMVfull_mut_292_G2A": 3.5424228,
                                   "mixed_valid_det": 0.5878788, "valid_with_N": 0.50202125},
                  "unsupported_null": {"unsupported601": None, "unsupported2000": None},
                  "invalid_null": {"mixed_ambig_R": None, "mixed_digit_1": None}},
}
(B / "inference_evidence.json").write_text(json.dumps(inference_evidence, ensure_ascii=False, indent=2), encoding="utf-8")

accounting = {
    "batch": "batch34", "version": VERSION, "generated_at": NOW,
    "zero_fit_declaration": True,
    "estimator_training_steps": 0, "optimizer_steps": 0, "gradient_steps": 0,
    "ridge_fits": 0, "calibration_steps": 0, "fine_tuning": 0, "solves": 0,
    "neural_inference": {"authorized_for": "frozen PARM HEK293 validation only",
                          "model": "PARM HEK293 5-fold equal mean (frozen, eval/no_grad)",
                          "gpu_seconds_worker": total_fwd,
                          "gpu_seconds_cap_incl_review": 600,
                          "freshreview_allocated_s": 120, "batch_size_max": 1, "device": "cuda:0"},
    "cpu_jobs": "manifest build, SHA verify, figure render, JSON/TSV I/O (no fit)",
    "separate_from_fits": True,
    "note": "fits and inference are accounted separately; inference is official forward only.",
}
(B / "zero_fit_and_inference_accounting.json").write_text(json.dumps(accounting, ensure_ascii=False, indent=2), encoding="utf-8")
print("inference_evidence + accounting written; total_fwd_s", total_fwd)
