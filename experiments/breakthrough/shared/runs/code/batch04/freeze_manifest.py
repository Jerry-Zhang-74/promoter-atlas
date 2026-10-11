#!/usr/bin/env python3
"""Batch04 selection/run manifest — FROZEN BEFORE any new val/test evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import sha256_file, atomic_write_json  # noqa: E402

RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch04"
B3 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch03"
CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch04"


def sha(p):
    return sha256_file(Path(p))


def main():
    manifest = {
        "written_at": "2026-10-02T23:47:00+08:00 (original FROZEN before any new val/test evaluation)",
        "code_fix": {
            "at": "2026-10-03T00:05:00+08:00",
            "reason": "deterministic pooling bugfix BEFORE any new val/test evaluation: the batch03 GeneBagModel used CUDA-atomic index_add_/scatter_reduce_ for mean/max pooling, which is non-bitwise-reproducible across runs. pool_meanmax_4slot() implements the mathematically identical concat(mean,max) over the fixed 4 slots via reshape+reduction (no atomics), restoring bitwise reproducibility (resume_check: weights_sha_match=true, max_weight_absdiff=0.0).",
            "scientific_design_unchanged": True,
            "candidate_versions_unchanged": ["v6.6.0-scratchgene.rc4", "v6.6.0-scratchgene.rc5"],
            "superseded_outputs_retained": "models/breakthrough_20261002/batch04/_superseded_nondeterministic_pooling/",
            "selection_unchanged": True,
        },
        "protocol_ref": "runs/breakthrough_20261002/batch04/PROTOCOL.txt (APPROVED v1)",
        "question": "gene-promoter diversity under matched input/head/update count (bounded controlled contrast, NOT a broad search to raise heldout scores)",
        "arms": {
            "B1_repeat_bag": {
                "candidate_version": "v6.6.0-scratchgene.rc4",
                "arm_key": "repeat_bag",
                "description": "one anchor promoter per gene per epoch (p0=(epoch+g)%n), repeated in ALL 4 slots; concat(mean,max) pooling -> Linear(256,1)",
            },
            "B2_diverse_bag": {
                "candidate_version": "v6.6.0-scratchgene.rc5",
                "arm_key": "diverse_bag",
                "description": "first anchor matches B1 (p0=(epoch+g)%n); slots=(p0+k)%n for k=0..3 (deterministic cycling when n<4); same pooling/head",
            },
        },
        "annotation_diagnostics": {
            "candidate_version": "v6.6.0-annridge.rc1",
            "feature_variants": ["log1p(promoter_count)", "44 composition (no count)", "44 composition + log1p(count)"],
            "alpha_grid": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            "inner_split": "batch03 inner 90/10 gene fold, seed 20261011",
            "note": "descriptive baseline set, NOT new NN arms, NOT causal",
        },
        "hyperparameters": {
            "lr": 0.001,
            "optimizer": "AdamW weight_decay=1e-4",
            "epochs": 12,
            "batch_size": 32,
            "seeds": [20261011, 20261012, 20261013],
            "loss": "MSE on raw gene Score_E",
            "epoch_rule": "final.pt = epoch 12 (predeclared max); no post-hoc epoch choice; no LR grid",
            "one_hot": "ACGT one-hot, N -> zero",
            "n_slots": 4,
            "bag_anchor": "p0=(epoch+g)%n identical for both arms",
            "n_lt4_policy": "deterministic cycling fills 4 slots; repeated promoters carry extra sampling weight in mean (reported)",
        },
        "inference_policy": {
            "primary": "matched 4-slot, fixed epoch=0 rotation, mean+max pooling, SAME policy both arms",
            "descriptive": "all-promoter mean+max (batch03 rc2 style; train/inference mismatch for B1 by design)",
            "frozen_before_eval": True,
        },
        "strata_thresholds": {
            "count_strata": [[1, 1], [2, 4], [5, 10 ** 9]],
            "gc_edges": [0.4525, 0.57],
            "derived_from": "TRAIN only (count distribution + GC tertiles)",
        },
        "limits": {
            "max_gpu_walltime_min": 90,
            "max_epochs_per_run": 12,
            "n_final_runs": 6,
            "resume_check_min": 5,
            "batch_max": 32,
            "cpu_threads": 4,
        },
        "source_hashes": {
            "data_npz": sha(DATA / "trackA_gene_promoters.npz"),
            "data_genes": sha(DATA / "trackA_genes.tsv"),
            "data_promoters": sha(DATA / "trackA_promoters.tsv"),
            "data_prep_meta": sha(DATA / "trackA_prep_meta.json"),
            "batch03_model_source": sha(B3 / "scratch_gene_model.py"),
            "batch03_train_source": sha(B3 / "train_scratch_gene.py"),
            "batch03_common_source": sha(B3 / "b3_common.py"),
        },
        "code_hashes": {
            "b4_common.py": sha(CODE / "b4_common.py"),
            "b4_model.py": sha(CODE / "b4_model.py"),
            "train_bag_v2.py": sha(CODE / "train_bag_v2.py"),
            "eval_bag_v2.py": sha(CODE / "eval_bag_v2.py"),
            "ann_diagnostics.py": sha(CODE / "ann_diagnostics.py"),
        },
        "run_ids": [
            {"arm": "repeat_bag", "seed": 20261011, "attempt_dir": "repeat_bag_s20261011"},
            {"arm": "repeat_bag", "seed": 20261012, "attempt_dir": "repeat_bag_s20261012"},
            {"arm": "repeat_bag", "seed": 20261013, "attempt_dir": "repeat_bag_s20261013"},
            {"arm": "diverse_bag", "seed": 20261011, "attempt_dir": "diverse_bag_s20261011"},
            {"arm": "diverse_bag", "seed": 20261012, "attempt_dir": "diverse_bag_s20261012"},
            {"arm": "diverse_bag", "seed": 20261013, "attempt_dir": "diverse_bag_s20261013"},
        ],
    }
    atomic_write_json(RUNS / "selection_run_manifest.json", manifest)
    manifest["manifest_sha256"] = sha(RUNS / "selection_run_manifest.json")
    atomic_write_json(RUNS / "selection_run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
