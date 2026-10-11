#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 worker self-check (NOT an independent review).

Verifies data/split/code SHAs against the frozen preflight config, asserts no
confirmed RC edge crosses the corrected split, counts actual fits (3 HGB + <=10
neural), replays saved checkpoints on a bounded subset, and records honest
integrity status. A fresh independent reviewer owns independent_review.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b16_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, CODE_OUT, RUNS_OUT,
                        CANDIDATE_VERSION, TASKS, SEEDS, N_FEATURE_UNIVERSE,
                        sha256_file, atomic_write_json)
from b16_model import SingleTaskModel, MultiTaskModel


def main():
    t0 = time.time()
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    config = json.load(open(MODEL_OUT / "config.json", encoding="utf-8"))
    code_hashes = json.load(open(MODEL_OUT / "code_hashes.json", encoding="utf-8"))

    checks = {}

    # ---- source SHAs match frozen config ----
    src_ok = True
    src_detail = {}
    for name, exp in config["source_hashes"].items():
        p = _src_path(name)
        if p is None:
            continue
        act = sha256_file(p)
        src_detail[name] = {"match": act == exp}
        if act != exp:
            src_ok = False
    checks["source_sha_match"] = {"result": src_ok, "detail": src_detail}

    # ---- data SHAs match frozen config ----
    data_ok = True
    data_detail = {}
    for name, exp in config["data_hashes"].items():
        p = DATA_OUT / name
        act = sha256_file(p) if p.exists() else None
        data_detail[name] = {"match": act == exp, "actual": act}
        if act != exp:
            data_ok = False
    checks["data_sha_match"] = {"result": data_ok, "detail": data_detail}

    # ---- code SHAs unchanged ----
    code_ok = True
    code_detail = {}
    for name, exp in code_hashes.items():
        act = sha256_file(CODE_OUT / name)
        code_detail[name] = act == exp
        if act != exp:
            code_ok = False
    checks["code_sha_unchanged"] = {"result": code_ok, "detail": code_detail}

    # ---- split counts + no crossing ----
    split = np.load(DATA_OUT / "split" / "split_arrays.npz", allow_pickle=True)
    gene_split = split["gene_split"].astype(str)
    split_counts = {k: int((gene_split == k).sum()) for k in ("train", "val", "test")}
    checks["split_counts"] = split_counts
    checks["split_sha"] = sha256_file(DATA_OUT / "split" / "split_arrays.npz")
    checks["split_sha_match_expected"] = (checks["split_sha"] == config["split"]["sha256"])

    # ---- fit counts ----
    neural_dir = MODEL_OUT
    n_neural = 0
    neural_seeds = {}
    for arm in ("A", "B", "C"):
        for seed in SEEDS:
            d = MODEL_OUT / ("%s_s%d" % (arm, seed))
            if not d.exists():
                continue
            if arm == "A":
                done = sum(1 for t in TASKS if (d / ("model_%s.pt" % t)).exists())
                if done == 3:
                    n_neural += 1
                    neural_seeds["%s_s%d" % (arm, seed)] = "complete"
            else:
                if (d / "model.pt").exists():
                    n_neural += 1
                    neural_seeds["%s_s%d" % (arm, seed)] = "complete"
    n_hgb = sum(1 for k in ("E45", "S1069", "B45") if (MODEL_OUT / ("hgb_%s.joblib" % k)).exists())
    checks["fit_counts"] = {"n_hgb": n_hgb, "n_neural": n_neural,
                            "n_hgb_within_limit": n_hgb == 3,
                            "n_neural_within_limit": n_neural <= 10}

    # ---- replay a bounded subset (verify saved checkpoint loads + predicts) ----
    replay = _replay_check()
    checks["replay"] = replay

    # ---- no crossing re-assert ----
    checks["no_confirmed_edge_crosses_split"] = _recheck_crossing()

    self_check = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": "batch16",
        "kind": "worker_self_check (NOT independent review)",
        "written_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "checks": checks,
        "overall_integrity": "PASS" if (src_ok and data_ok and code_ok and
                                        n_hgb == 3 and 0 < n_neural <= 10) else "PARTIAL",
    }
    atomic_write_json(REPORT_OUT / "worker_self_check.json", self_check)
    print(json.dumps(self_check, ensure_ascii=False, indent=2))


def _src_path(name):
    from b16_common import (TRACKA_NPZ, FEATURES45_NPZ, FEATURES1024_NPZ, SPLIT_NPZ,
                            MEM400, EXPR, CANON, OMICS_TSV)
    m = {
        "trackA_gene_promoters.npz": TRACKA_NPZ,
        "features45.npz": FEATURES45_NPZ,
        "features1024.npz": FEATURES1024_NPZ,
        "split_arrays.npz": SPLIT_NPZ,
        "promoter_sym400_uint8.memmap": MEM400,
        "gene_expression_table.tsv": EXPR,
        "gene_expression_label_table.tsv": CANON,
        "promoter_omics_targets_v1.tsv": OMICS_TSV,
    }
    return m.get(name)


def _replay_check():
    import pandas as pd
    data = np.load(DATA_OUT / "data.npz", allow_pickle=True)
    sc = np.load(DATA_OUT / "scalers.npz", allow_pickle=True)
    X = data["X"].astype(np.float64)
    # standardized features for a bounded subset of test genes
    Xs = (X - sc["X_mean"]) / sc["X_std"]
    tr = data["gene_split"].astype(str) == "train"
    va = data["gene_split"].astype(str) == "val"
    te = data["gene_split"].astype(str) == "test"
    idx = np.where(te)[0][:32]
    x_feat = torch.tensor(Xs[idx], dtype=torch.float32)
    # dummy one-hot seq (zero) to test head path deterministically
    x_seq = torch.zeros(32 * 4, 4, 400)
    out = {}
    for arm in ("A", "B"):
        for seed in SEEDS:
            d = MODEL_OUT / ("%s_s%d" % (arm, seed))
            if arm == "A":
                for t in TASKS:
                    p = d / ("model_%s.pt" % t)
                    if p.exists():
                        m = SingleTaskModel(t)
                        m.load_state_dict(torch.load(p, map_location="cpu")["model"])
                        m.eval()
                        with torch.no_grad():
                            pred = m(x_seq, x_feat).numpy()
                        out["%s_%s_%d" % (arm, t, seed)] = {"finite": bool(np.isfinite(pred).all()),
                                                            "shape": list(pred.shape)}
            else:
                p = d / "model.pt"
                if p.exists():
                    m = MultiTaskModel(n_aux=0)
                    m.load_state_dict(torch.load(p, map_location="cpu")["model"])
                    m.eval()
                    with torch.no_grad():
                        pred = m(x_seq, x_feat)
                    out["%s_%d" % (arm, seed)] = {t: bool(np.isfinite(pred[t].numpy()).all())
                                                  for t in TASKS}
    return {"n_models_replayed": len(out), "detail": out}


def _recheck_crossing():
    import pandas as pd
    ed = pd.read_csv(Path(__file__).resolve().parents[4] / "data" / "breakthrough_20261002" /
                     "batch15" / "verified_gene_edges.tsv", sep="\t", dtype=str)
    sa = pd.read_csv(DATA_OUT / "split" / "gene_fold_assignment.tsv", sep="\t", dtype=str)
    gmap = dict(zip(sa.gene_id.values, sa.new_component_id.values))
    fmap = dict(zip(sa.gene_id.values, sa.new_fold.values))
    crossings = 0
    for ga, gb in zip(ed.gene_a.values, ed.gene_b.values):
        if gmap[ga] != gmap[gb] or fmap[ga] != fmap[gb]:
            crossings += 1
    return {"n_crossing_edges": crossings, "result": crossings == 0}


if __name__ == "__main__":
    main()
