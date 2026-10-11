# -*- coding: utf-8 -*-
"""batch58 finalizer: add the honest memory-target assessment to current_training_status.json,
write the throughput analysis addendum + TRAINING_DONE marker + worker status. No GPU, no fit.
Idempotent; all writes land in reports/breakthrough_20261002/batch58/.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch58")
RUN = os.path.join(ROOT, "runs", "breakthrough_20261002", "batch58")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "batch57_model_core_rc2")))
from io_utils import atomic_write_json, now_iso  # noqa: E402


def main():
    prof = json.load(open(os.path.join(REP, "throughput_profile.json"), encoding="utf-8"))
    stp = json.load(open(os.path.join(REP, "current_training_status.json"), encoding="utf-8"))
    rows = [{"batch": r.get("batch"), "ok": r.get("ok"), "oom": bool(r.get("oom")),
             "median_s": r.get("median_s"), "samples_per_s": r.get("samples_per_s"),
             "nvidia_used_gib": r.get("nvidia_used_gib"), "nvidia_free_gib": r.get("nvidia_free_gib"),
             "safe_ge_0.7gib_free": bool(r.get("nvidia_free_gib") is not None
                                         and r["nvidia_free_gib"] >= 0.7)}
            for r in prof["results"]]
    safe = [r for r in rows if r["ok"] and r["safe_ge_0.7gib_free"]]
    safe_rates = [r["samples_per_s"] for r in safe]
    spread = (max(safe_rates) - min(safe_rates)) / max(safe_rates) if safe_rates else None
    in_target = [r["batch"] for r in safe if r["nvidia_used_gib"] is not None
                 and 6.7 <= r["nvidia_used_gib"] <= 7.1]
    analysis = {
        "batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
        "selector": "PARENT1012 rule: max samples/second (batch/median_s) among memory-safe candidates",
        "nvidia_total_mib": prof["nvidia_total_mib"],
        "resident_cohort_gib": prof["resident_cohort_gib"],
        "candidates": rows,
        "selected_batch": prof["selected_batch"],
        "memory_target_gib": [6.7, 7.1], "min_free_gib_floor": 0.7,
        "candidates_inside_memory_target": in_target,
        "actual_training_peak_device_mib": stp.get("peak_actual_vram_mib"),
        "actual_training_peak_torch_gib": (stp.get("process2") or {}).get("peak_cuda_mb", 0) / 1024.0,
        "memory_target_reached": bool(in_target) and prof["selected_batch"] in in_target,
        "honest_memory_note": (
            "The requested ~6.7-7.1 GiB device target was NOT reached by the selected batch. "
            "Only batch 2048 touched it (nvidia_used 7.61 GiB) but left only 0.015 GiB free "
            "(< 0.7 GiB floor) and ran ~22x slower per step (586 vs 6402 samples/s at 1024), so "
            "it is UNSAFE and was rejected; batch 1024 used 5.62 GiB at 6402 samples/s. Per the "
            "PARENT1012 correction the 7 GiB target is SECONDARY to maximum safe throughput, so "
            "the fastest measured memory-safe batch was selected and usage was NOT padded."),
        "throughput_flatness_note": (
            "Measured throughput is essentially flat from batch 64..1024 (%s samples/s spread "
            "%.1f%%); values for batch<=64 are 3-repeat medians of ~8 ms steps and are within "
            "run-to-run noise, so 'fastest' here means 'largest measured samples/s among safe "
            "candidates', not a robust separation from batch 1024."
            % (("%d..%d" % (int(min(safe_rates)), int(max(safe_rates)))) if safe_rates else "n/a",
               100.0 * (spread or 0.0))),
        "no_tuning_from_scores": True,
        "resource_configuration_variant": True,
        "no_performance_superiority_claim": True,
        "analysis_at": now_iso(),
    }
    atomic_write_json(os.path.join(REP, "throughput_analysis.json"), analysis)

    stp["memory_target_assessment"] = {
        "requested_gib": [6.7, 7.1], "reached": analysis["memory_target_reached"],
        "actual_peak_device_mib": stp.get("peak_actual_vram_mib"),
        "min_free_giB_floor": 0.7,
        "reason": analysis["honest_memory_note"],
        "detail": "reports/breakthrough_20261002/batch58/throughput_analysis.json",
    }
    atomic_write_json(os.path.join(REP, "current_training_status.json"), stp)

    done = {
        "batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
        "status": "TRAINING_DONE", "variant": "throughput_variant",
        "target_column": "Log2RPM_HEK293",
        "epochs_completed": (stp.get("process2") or {}).get("epochs_completed"),
        "full_pass_coverage": stp["resume_evidence"]["full_pass_coverage"],
        "global_samples_final": stp["resume_evidence"]["global_samples_final"],
        "expected_samples_2_epochs": stp["resume_evidence"]["expected_samples_2_epochs"],
        "real_resume_verified": (stp["resume_evidence"]["carried_opt_steps_match"]
                                 and stp["resume_evidence"]["process2_resumed"]),
        "selected_effective_batch": stp.get("selected_batch_final"),
        "sample_throughput_samples_per_s": stp.get("sample_throughput_samples_per_s"),
        "peak_actual_vram_mib": stp.get("peak_actual_vram_mib"),
        "total_training_elapsed_s": stp.get("total_training_elapsed_s"),
        "gate_consumed": stp["gate_after"]["consumed"], "gate_budget": stp["gate_after"]["budget"],
        "final_checkpoint": stp["checkpoint_hashes"]["latest"],
        "no_test_evaluation": True, "no_new_fit_allowance_requested": True,
        "finished_at": now_iso(),
    }
    atomic_write_json(os.path.join(REP, "TRAINING_DONE.json"), done)

    ws = {"batch": "batch58", "version": "v6.39.0-throughput-checkpoint.rc1",
          "state": "TRAINING_DONE", "stage": "awaiting independent fresh review",
          "code_ready": os.path.join(HERE, "CODE_READY.json"),
          "selfcheck": {"passed": 9, "total": 9},
          "gate": stp["gate_after"],
          "artifacts": {
              "current_training_status": os.path.join(REP, "current_training_status.json"),
              "throughput_profile": os.path.join(REP, "throughput_profile.json"),
              "throughput_analysis": os.path.join(REP, "throughput_analysis.json"),
              "resume_evidence": os.path.join(REP, "resume_evidence.json"),
              "coverage_proof": os.path.join(REP, "coverage_proof.json"),
              "THROUGHPUT_READY": os.path.join(REP, "THROUGHPUT_READY.json"),
              "handoff": os.path.join(ROOT, "runs", "breakthrough_20261002", "batch52",
                                      "THROUGHPUT_READY.txt")},
          "updated_at": now_iso()}
    atomic_write_json(os.path.join(REP, "worker_status.json"), ws)
    print(json.dumps({"ok": True, "selected": prof["selected_batch"],
                      "memory_target_reached": analysis["memory_target_reached"],
                      "epochs": done["epochs_completed"],
                      "coverage": done["full_pass_coverage"],
                      "resume_verified": done["real_resume_verified"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
