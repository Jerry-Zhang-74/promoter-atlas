#!/usr/bin/env python3
"""Batch05 frozen split/config manifest — FROZEN BEFORE any new val/test eval.

Records source-data, split-artifact and code hashes, arms/candidates,
hyperparameters and run ids. Written BEFORE training/evaluation so the split,
thresholds, seeds and code are all hash-pinned prior to heldout score access.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import sha256_file, atomic_write_json  # noqa: E402

RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch05"
B3 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"


def sha(p):
    return sha256_file(Path(p))


def main():
    manifest = {
        "written_at": "frozen BEFORE training/evaluation (batch05)",
        "protocol_ref": "runs/breakthrough_20261002/batch05/PROTOCOL.txt (APPROVED v1)",
        "split": {
            "name": "exact/RC + detected-near-identity blocked (NOT homology-free)",
            "builder": "code/batch05/b5_split.py",
            "deterministic": True,
            "target_labels_not_used": True,
            "seed": 20261011,
        },
        "arms": {
            "B1_repeat_bag": {"candidate_version": "v6.6.0-scratchgene.rc6",
                              "arm_key": "repeat_bag",
                              "description": "anchor p0=(epoch+g)%n repeated in ALL 4 slots; concat(mean,max) pooling"},
            "B2_diverse_bag": {"candidate_version": "v6.6.0-scratchgene.rc7",
                               "arm_key": "diverse_bag",
                               "description": "anchor matches B1; slots=(p0+k)%n for k=0..3 (cycle when n<4); same pooling/head"},
        },
        "annotation_diagnostics": {
            "candidate_version": "v6.6.0-annridge.rc2",
            "feature_variants": ["log1p(promoter_count)", "44 composition (no count)", "44 composition + log1p(count)"],
            "alpha_grid": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            "inner_split": "group-preserving component fold, seed 20261011",
            "scaling": "train-only (inner-train for selection, full train for final fit)",
            "note": "descriptive baseline set, NOT new NN arms, NOT causal",
        },
        "hyperparameters": {
            "lr": 0.001, "optimizer": "AdamW weight_decay=1e-4", "epochs": 12,
            "batch_size": 32, "seed": 20261011, "loss": "MSE on raw gene Score_E",
            "epoch_rule": "final.pt = epoch 12 (predeclared max); no heldout epoch/model selection",
            "one_hot": "ACGT one-hot, N -> zero", "n_slots": 4,
            "bag_anchor": "p0=(epoch+g)%n identical for both arms",
            "n_lt4_policy": "deterministic cycling fills 4 slots (reported)",
        },
        "inference_policy": {
            "primary": "matched 4-slot, fixed epoch=0 rotation, mean+max pooling, SAME policy both arms",
            "frozen_before_eval": True,
        },
        "run_ids": [
            {"arm": "repeat_bag", "seed": 20261011, "attempt_dir": "repeat_bag_s20261011"},
            {"arm": "diverse_bag", "seed": 20261011, "attempt_dir": "diverse_bag_s20261011"},
        ],
        "source_hashes": {
            "data_npz": sha(DATA / "trackA_gene_promoters.npz"),
            "memmap_sym400": sha(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"),
            "memmap_up2000": sha(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"),
            "batch03_model_source": sha(B3 / "scratch_gene_model.py"),
            "batch03_baseline_source": sha(B3 / "baseline_seq_ridge.py"),
        },
        "split_artifact_hashes": {
            "split_plan.json": sha(DATA05 / "split_plan.json"),
            "split_arrays.npz": sha(DATA05 / "split_arrays.npz"),
            "split_assignment.tsv": sha(DATA05 / "split_assignment.tsv"),
            "components.tsv": sha(DATA05 / "components.tsv"),
            "near_identity_pairs.tsv": sha(DATA05 / "near_identity_pairs.tsv"),
            "exact_rc_summary.json": sha(DATA05 / "exact_rc_summary.json"),
        },
        "code_hashes": {
            "b5_common.py": sha(CODE / "b5_common.py"),
            "b5_split.py": sha(CODE / "b5_split.py"),
            "b5_data.py": sha(CODE / "b5_data.py"),
            "b5_model.py": sha(CODE / "b5_model.py"),
            "train_bag_v5.py": sha(CODE / "train_bag_v5.py"),
            "eval_bag_v5.py": sha(CODE / "eval_bag_v5.py"),
            "ann_diagnostics_v5.py": sha(CODE / "ann_diagnostics_v5.py"),
            "resume_check_v5.py": sha(CODE / "resume_check_v5.py"),
            "inference_entrypoint_v5.py": sha(CODE / "inference_entrypoint_v5.py"),
        },
        "limits": {
            "max_gpu_walltime_min": 60,
            "max_epochs_per_run": 12,
            "n_final_runs": 2,
            "resume_check_min": 3,
            "batch_max": 32,
            "cpu_threads": 4,
        },
    }
    atomic_write_json(RUNS / "frozen_manifest.json", manifest)
    manifest["manifest_sha256"] = sha(RUNS / "frozen_manifest.json")
    atomic_write_json(RUNS / "frozen_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
