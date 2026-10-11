#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 deliverable 2: build the portable, runnable release directory + zip.

Copies the fixed independent-arm (A_s20261016) E/S/B weights, train-fitted
transforms and self-contained inference code into a package, builds a small
reproducible example (gene bag + features -> expected predictions), cross-checks
the packaged predictions against the ACTUAL saved batch16 predictions, writes a
source/code/split/model SHA manifest and the negative ablation results, then zips.
Read-only w.r.t. batch16; writes only under models/breakthrough_20261002/batch17/.
"""
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[4]
CODE17 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch17"
B16 = ROOT / "models" / "breakthrough_20261002" / "batch16"
DATA16 = ROOT / "data" / "breakthrough_20261002" / "batch16"
REL_ROOT = ROOT / "models" / "breakthrough_20261002" / "batch17"
PKG = REL_ROOT / "release_v6.15.1-release.rc1"
MEM400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
N_HUMAN = 507365
L400 = 400
VERSION = "v6.15.1-release.rc1"


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def build_setup():
    for sub in ("models", "transforms", "code", "labels"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    model_sha = {}
    for t in ("E", "S", "B"):
        src = B16 / "A_s20261016" / ("model_%s.pt" % t)
        dst = PKG / "models" / ("model_%s.pt" % t)
        shutil.copy2(src, dst)
        model_sha[t] = {"source": str(src.relative_to(ROOT)), "sha256": sha256_file(dst)}
    shutil.copy2(DATA16 / "scalers.npz", PKG / "transforms" / "scalers.npz")
    shutil.copy2(CODE17 / "pkg_promoter_atlas_predict.py", PKG / "code" / "promoter_atlas_predict.py")
    shutil.copy2(CODE17 / "pkg_run_example.py", PKG / "code" / "run_example.py")
    docs = CODE17 / "pkg_docs"
    shutil.copy2(docs / "README.md", PKG / "README.md")
    shutil.copy2(docs / "MODEL_CARD.md", PKG / "MODEL_CARD.md")
    shutil.copy2(docs / "requirements.txt", PKG / "requirements.txt")
    shutil.copy2(docs / "labels_README.md", PKG / "labels" / "README.md")
    print("copied weights + transforms + code + docs")
    return model_sha


def build_example():
    data = np.load(DATA16 / "data.npz", allow_pickle=True)
    gene_split = data["gene_split"].astype(str)
    gene_gid = data["gene_gid"].astype(str)
    mask = {t: data["%s_mask" % t].astype(bool) for t in ("E", "S", "B")}
    slot_r400 = data["slot_r400"].astype(np.int64)
    X = data["X"].astype(np.float64)
    mm = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    allheads = mask["E"] & mask["S"] & mask["B"]
    test_idx = list(np.where((gene_split == "test") & allheads)[0][:3])
    train_idx = list(np.where((gene_split == "train") & allheads)[0][:1])
    n_containing = None
    for g in np.where(allheads)[0][:20000]:
        if (np.asarray(mm[slot_r400[g]], dtype=np.uint8) == 4).any():
            n_containing = int(g)
            break
    picks = test_idx + train_idx
    if n_containing is not None and n_containing not in picks:
        picks.append(n_containing)
    seqs = np.stack([np.asarray(mm[slot_r400[g]], dtype=np.uint8) for g in picks]).astype(np.uint8)
    feats = X[picks].astype(np.float64)
    gids = np.array([gene_gid[g] for g in picks], dtype="U32")
    splits = np.array([gene_split[g] for g in picks], dtype="U8")
    np.savez(PKG / "code" / "example_input.npz",
             seqs=seqs, features=feats, gene_ids=gids, split=splits,
             y_E=data["E"][picks], y_S=data["S"][picks], y_B=data["B"][picks],
             mask_E=mask["E"][picks], mask_S=mask["S"][picks], mask_B=mask["B"][picks])
    print("example genes:", [(str(g), str(s)) for g, s in zip(gids, splits)])
    return data, gene_split, gene_gid, mask, slot_r400, X, picks, seqs, feats, gids


def build_predictions_and_crosscheck(data, gene_split, gene_gid, mask, picks, seqs, feats, gids):
    sc = np.load(DATA16 / "scalers.npz", allow_pickle=True)
    sys.path.insert(0, str(PKG / "code"))
    import promoter_atlas_predict as pap
    predictor = pap.PromoterAtlasGenePredictor(package_root=PKG)
    res = predictor.predict_batch(seqs, feats)
    predicted = {str(g): {t: float(res[t][i]) for t in ("E", "S", "B")} for i, g in enumerate(gids)}

    cross = []
    maxdiff = 0.0
    for task in ("E", "S", "B"):
        for split in ("test", "train", "val"):
            f = B16 / "A_s20261016" / ("pred_%s_%s.npy" % (task, split))
            if not f.exists():
                continue
            idx = np.where((gene_split == split) & mask[task])[0]
            saved = np.load(f)
            saved_raw = saved[0].astype(np.float64) * float(sc["%s_std" % task]) + float(sc["%s_mean" % task])
            pos = {int(g): j for j, g in enumerate(idx)}
            for g in picks:
                if int(g) in pos and gene_split[g] == split and mask[task][g]:
                    mine = predicted[str(gene_gid[g])][task]
                    theirs = float(saved_raw[pos[int(g)]])
                    cross.append({"gene": str(gene_gid[g]), "task": task, "split": split,
                                  "packaged": mine, "saved": theirs, "absdiff": abs(mine - theirs)})
    if cross:
        maxdiff = max(c["absdiff"] for c in cross)
    print("packaged-vs-saved cross-check: n=%d max_absdiff=%.3e" % (len(cross), maxdiff))
    expected = {
        "candidate_version": VERSION,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tolerance": 1e-5,
        "note": "raw-unit predictions from the shipped weights; reproduce with code/run_example.py",
        "predictions": predicted,
    }
    (PKG / "code" / "example_expected_output.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cross, maxdiff, predicted


def write_manifests(model_sha, cross, maxdiff):
    metrics = json.load(open(REPORT17JS, encoding="utf-8"))
    neg = {
        "candidate_version": VERSION,
        "source_candidate": "v6.15.0-multitask.rc1",
        "kind": "negative / rejected ablation results (shared-network selection)",
        "frozen_validation_rule": "mean per-target validation rho gain >= 0.02 AND no target drop > 0.02 (mean-of-seed)",
        "result": {arm: {"mean_gain": metrics["validation_rule_applied"][arm]["mean_gain"],
                         "no_drop_gt_0.02": metrics["validation_rule_applied"][arm]["no_drop_gt_0.02"],
                         "rule_pass": metrics["validation_rule_applied"][arm]["rule_pass"],
                         "per_target": metrics["validation_rule_applied"][arm]["per_target"]}
                   for arm in ("B", "C")},
        "conclusion": "shared backbone (B) and shared+aux (C) were REJECTED on validation; independent composite ships",
        "retrospective_heldout_ensemble": {arm: {t: metrics["mean_of_seed_and_ensemble"][arm][t]["test"]["ensemble"]["spearman"]
                                                 for t in ("E", "S", "B")} for arm in ("A", "B", "C")},
        "paired_component_bootstrap": {arm: {t: {k: metrics["paired_component_bootstrap"][arm][t][k]
                                                 for k in ("base_point_estimate", "ci95_low", "ci95_high", "n_components")}
                                             for t in ("E", "S", "B")} for arm in ("B", "C")},
        "caveats": metrics["known_confounds_prominent"],
    }
    (PKG / "negative_ablation_results.json").write_text(json.dumps(neg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = {
        "candidate_version": VERSION,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": _torch_version(),
        "cuda_available_at_package_time": _cuda_available(),
        "device_note": "inference runs on CPU; no GPU required",
    }
    (PKG / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    schema = {
        "candidate_version": VERSION,
        "input": {
            "seqs": {"shape": "(N,4,400) or (4,400)", "dtype": "uint8",
                     "alphabet": "A,C,G,T,N -> 0,1,2,3,4 (N -> encoded 4, one-hot all-zero)",
                     "order": "4 fixed diverse promoter slots per gene, each 400 bp"},
            "features": {"shape": "(N,1069) or (1069,)", "dtype": "float (finite)",
                         "scaling": "RAW (unstandardized); the shipped transforms apply (x-mean)/std",
                         "order": "44 region composition + 1 log1p_count + 1024 kmer-bin features (fixed)"},
            "batch_rule": "len(seqs) must equal len(features)",
        },
        "output": {
            "predictions.E": "Score_E rank-scaled expression strength [0,1]",
            "predictions.S": "S_raw = -MAD(log2(1+detectable TPM)); defined for B_raw>=0.30 AND detectable>=3",
            "predictions.B": "B_raw = detectable_tissue_count / tissue_count [0,1]",
            "is_model_prediction": True, "is_catalog_observed": False,
        },
        "errors": {"class": "PromoterAtlasInputError (ValueError)",
                   "raised_for": ["wrong seqs shape", "wrong seqs dtype", "seqs value outside 0..4",
                                  "wrong features length", "non-finite features", "seqs/features length mismatch"]},
        "unsupported": ["arbitrary promoter-fragment activity", "HEK293T/synthetic-construct activity",
                        "multi-species", "temporal-stability reading of S"],
    }
    (PKG / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    files = {}
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files[str(p.relative_to(PKG)).replace("\\", "/")] = sha256_file(p)
    split_manifest = json.load(open(DATA16 / "split" / "split_manifest.json", encoding="utf-8"))
    vm = {
        "candidate_version": VERSION,
        "source_candidate": "v6.15.0-multitask.rc1 (batch16 weights; packaging/provenance repair only)",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architectures_shipped": {"selected": "independent (arm A)", "reference_seed": 20261016,
                                  "rejected": ["B shared", "C shared+aux"]},
        "model_files": model_sha,
        "package_files_sha256": files,
        "source_references": {
            "batch16_deployment_bundle": "models/breakthrough_20261002/batch16/deployment_bundle.json",
            "batch16_neural_results": "models/breakthrough_20261002/batch16/neural_results.json",
            "batch16_comparison": "reports/breakthrough_20261002/batch16/comparison.json",
            "batch16_independent_review": "reports/breakthrough_20261002/batch16/independent_review.json",
            "batch17_corrected_metrics": "reports/breakthrough_20261002/batch17/corrected_release_metrics.json",
            "batch17_numeric_source_table": "reports/breakthrough_20261002/batch17/corrected_release_metrics.tsv",
            "resume_diagnostic": "reports/breakthrough_20261002/batch17/resume_diagnostic.json",
        },
        "split": {"name": split_manifest.get("split_name", "component-union corrected split"),
                  "sha256": split_manifest.get("split_sha256", split_manifest.get("output_hashes", {}).get("split_arrays.npz", ""))},
        "packaged_vs_saved_prediction_crosscheck": {"n_comparisons": len(cross), "max_absdiff": maxdiff,
                                                    "tolerance": 1e-5, "verdict": "PASS" if maxdiff <= 1e-5 else "FAIL"},
        "notes": ["weights are unchanged batch16 fits; version bump is packaging only",
                  "no training assets required to run the example"],
    }
    (PKG / "VERSION_MANIFEST.json").write_text(json.dumps(vm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return vm, neg


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def make_zip():
    zpath = REL_ROOT / "promoter_atlas_v6.15.1-release.rc1.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, arcname=str(Path(PKG.name) / p.relative_to(PKG)).replace("\\", "/"))
    print("zip:", zpath, "bytes:", zpath.stat().st_size)
    return zpath


if __name__ == "__main__":
    REPORT17JS = ROOT / "reports" / "breakthrough_20261002" / "batch17" / "corrected_release_metrics.json"
    t0 = time.time()
    model_sha = build_setup()
    data, gene_split, gene_gid, mask, slot_r400, X, picks, seqs, feats, gids = build_example()
    cross, maxdiff, predicted = build_predictions_and_crosscheck(data, gene_split, gene_gid, mask, picks, seqs, feats, gids)
    vm, neg = write_manifests(model_sha, cross, maxdiff)
    zp = make_zip()
    print(json.dumps({"version": VERSION, "n_files": len(vm["package_files_sha256"]),
                      "crosscheck": vm["packaged_vs_saved_prediction_crosscheck"],
                      "zip": str(zp), "elapsed_s": round(time.time() - t0, 2)}, ensure_ascii=False, indent=2))
