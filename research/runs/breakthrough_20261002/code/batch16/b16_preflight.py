#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 executable preflight: freeze architecture/config/data/code SHA BEFORE fit.

Records all source/data/code SHAs, the frozen neural/HGB config, split, targets,
and architecture summary into model_16/config.json + code_hashes.json. This is the
immutable freeze point for the batch16 controlled comparison (A/B/C + 3 HGB).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (ROOT, CODE_OUT, DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT,
                        TRACKA_NPZ, FEATURES45_NPZ, FEATURES1024_NPZ, SPLIT_NPZ,
                        MEM400, EXPR, CANON, OMICS_TSV, CANDIDATE_VERSION, TRACK,
                        BATCH, SEEDS, HGB, NEURAL, TASKS, AUX_FIELDS,
                        N_FEATURE_UNIVERSE, N_SLOTS, L400, sha256_file, atomic_write_json)
from b16_model import SingleTaskModel, MultiTaskModel, n_params

import torch  # noqa: E402


def main():
    t0 = time.time()
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # ---- parameter counts for the frozen architecture ----
    a_models = {t: SingleTaskModel(t) for t in TASKS}
    b_model = MultiTaskModel(n_aux=0)
    c_model = MultiTaskModel(n_aux=len(AUX_FIELDS))
    param_counts = {
        "A_E": n_params(a_models["E"]), "A_S": n_params(a_models["S"]), "A_B": n_params(a_models["B"]),
        "A_total": sum(n_params(a_models[t]) for t in TASKS),
        "B": n_params(b_model), "C": n_params(c_model),
    }

    config = {
        "candidate_version": CANDIDATE_VERSION,
        "track": TRACK,
        "batch": BATCH,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze_note": "config.json written BEFORE any batch16 fit; immutable freeze point",
        "scope": "human gene-level natural-expression E/S/B; controlled neural task-sharing comparison",
        "seeds": SEEDS,
        "seeds_are_run_ids": True,
        "max_neural_fits": 10,
        "max_hgb_baseline_fits": 3,
        "split": {
            "source": str(SPLIT_NPZ),
            "name": "corrected component-union (batch05 + batch15 RC edges), SHA256 fold allocation",
            "sha256": sha256_file(SPLIT_NPZ),
            "manifest": str(DATA_OUT / "split" / "split_manifest.json"),
            "n_train": 46367, "n_val": 5609, "n_test": 5851,
            "label_independent": True,
        },
        "targets": {
            "E": "Score_E rank-scaled [0,1]",
            "S": "S_raw = -MAD(log2(1+detectable TPM)), S_defined mask (18769 eligible)",
            "B": "B_raw = detectable/tissue_count",
        },
        "neural_config": NEURAL,
        "neural_architecture": {
            "sequence_branch": "Trunk Conv1d(4,64,k7)+BN+ReLU -> Conv1d(64,64,k5)+BN+ReLU -> Conv1d(64,128,k5)+BN+ReLU; mean-over-positions -> 128/slot",
            "bag": "diverse 4-slot; concat(mean,max) over slots -> 256",
            "feature_branch": "Linear(1069,128)+ReLU+Linear(128,128) -> 128",
            "combined_rep": "concat(256,128)=384",
            "heads": "Linear(384,1) per task; C aux head Linear(128,7) on per-slot embedding BEFORE pooling",
            "identical_inputs_across_arms": True,
            "param_counts": param_counts,
            "note": "independent arm A owns 3 backbones (more total params); B/C share one backbone",
        },
        "aux": {
            "fields": AUX_FIELDS,
            "n_fields": len(AUX_FIELDS),
            "excluded": ["cage_tss", "procap_tss"],
            "excluded_reason": "TSS-activity proxies of E",
            "level": "promoter (before gene pooling), same selected slots",
            "loss_coef": NEURAL["aux_loss_coef"],
        },
        "hgb_config": HGB,
        "hgb_baselines": {
            "E45": "HGB on E, 45 features",
            "S1069": "HGB on S, 1069 features",
            "B45": "HGB on B, 45 features",
        },
        "evaluation": {
            "primary": "per-target validation Spearman gain of shared vs independent (mean), no target drop > 0.02",
            "threshold": {"rho_gain_min": 0.02, "no_drop_more_than": 0.02},
            "secondary": "R2/MAE per valid target common rows; heldout paired component bootstrap 95% delta",
            "selection": "validation-only checkpoint choice (patience 3); no test selection; no best-seed choice",
            "heldout": "evaluated once after candidate freeze; never used to redraw split",
        },
        "source_hashes": {
            "trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
            "features45.npz": sha256_file(FEATURES45_NPZ),
            "features1024.npz": sha256_file(FEATURES1024_NPZ),
            "split_arrays.npz": sha256_file(SPLIT_NPZ),
            "promoter_sym400_uint8.memmap": sha256_file(MEM400),
            "gene_expression_table.tsv": sha256_file(EXPR),
            "gene_expression_label_table.tsv": sha256_file(CANON),
            "promoter_omics_targets_v1.tsv": sha256_file(OMICS_TSV),
        },
        "data_hashes": {
            "labels.npz": sha256_file(DATA_OUT / "labels.npz"),
            "data.npz": sha256_file(DATA_OUT / "data.npz"),
            "scalers.npz": sha256_file(DATA_OUT / "scalers.npz"),
        },
    }
    atomic_write_json(MODEL_OUT / "config.json", config)

    code_hashes = {}
    for f in sorted(CODE_OUT.glob("b16_*.py")):
        code_hashes[f.name] = sha256_file(f)
    atomic_write_json(MODEL_OUT / "code_hashes.json", code_hashes)

    print(json.dumps({
        "candidate_version": CANDIDATE_VERSION,
        "param_counts": param_counts,
        "seeds": SEEDS,
        "config": str(MODEL_OUT / "config.json"),
        "n_code_files": len(code_hashes),
        "elapsed_s": round(time.time() - t0, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
