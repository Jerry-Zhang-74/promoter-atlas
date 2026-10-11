#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch20 preliminary metrics + latency.

A. K562 MPRA: ensemble prediction on the held-out test split vs observed.
B. Gene E/S/B: independent replay of the rc2 held-out test split through the
   batch20 inference path (must reproduce the shipped retrospective Spearman).
C. Latency: per-call wall time and device for both models.

Writes reports/breakthrough_20261002/batch20/preliminary_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b20_common as C
import b20_gene as G
import b20_gene_model as GM
import b20_mpra as M


def _avg_rank(x):
    """Average (tie-corrected) ranks; required because B_raw is ratio-valued with
    heavy ties, where argsort-based ranks give a biased Spearman."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a, b):
    ra = _avg_rank(a); rb = _avg_rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def mpra_test_metrics(pred_obj):
    import pandas as pd
    df = pd.read_csv(C.MPRA_SUPERVISED, sep="\t", low_memory=False)
    te = df[df["split"] == "test"].reset_index(drop=True)
    seqs = [str(s) for s in te["sequence"].tolist()]
    t0 = time.time()
    pred = M.predict_legnet(seqs)
    dt = time.time() - t0
    y = te["activity"].to_numpy(float)
    p = np.asarray(pred["ensemble_mean"], float)
    # Family-leakage diagnosis: the mpra_supervised.tsv split column is NOT the
    # batch02 grouped-by-family split, so test/train family overlap inflates this.
    tr_fam = set(df[df["split"] == "train"]["family_id"].astype(str))
    te_fam = te["family_id"].astype(str)
    leak = float(te_fam.isin(tr_fam).mean())
    return {
        "n": int(len(te)),
        "spearman": round(spearman(p, y), 5),
        "mae": round(mae(p, y), 5),
        "latency_s": round(dt, 2),
        "latency_per_seq_ms": round(1000 * dt / max(1, len(te)), 3),
        "family_leakage_fraction_of_test_rows": round(leak, 4),
        "honest_grouped_cv_reference": {"source": "batch02 grouped-by-family OOF (legnet)",
                                        "spearman": 0.79577, "mae": 0.16872},
        "note": ("RANDOM split, NOT grouped by family: %.1f%% of test rows share a family_id with train, so this "
                 "value is LEAKAGE-INFLATED and must NOT be quoted as generalization. The honest estimate is the "
                 "batch02 grouped-by-family OOF Spearman 0.79577." % (100 * leak)),
    }


def gene_test_metrics(cat, predictor):
    # Use the FROZEN batch16 targets (identical artifacts to the rc2 replay). The
    # catalog label table stores rounded columns unsuitable for metric replay.
    z = np.load(C.B16_LABELS_NPZ, allow_pickle=True)
    split = z["gene_split"].astype(str)
    assert len(split) == cat["X"].shape[0]
    te = np.where(split == "test")[0]
    seqs = np.zeros((len(te), C.N_SLOTS, C.L400), dtype=np.uint8)
    for k, gi in enumerate(te):
        seqs[k] = G.slot_sequences(int(gi))
    feats = cat["X"][te].astype(np.float64)
    t0 = time.time()
    out = predictor.predict(seqs, feats, batch=64)
    dt = time.time() - t0
    res = {}
    for t in ("E", "S", "B"):
        y = z[t].astype(float)[te]
        m = z[t + "_mask"].astype(bool)[te]
        yy, pp = y[m], out[t][m]
        res[t] = {"n": int(m.sum()), "spearman": round(spearman(pp, yy), 5), "mae": round(mae(pp, yy), 5),
                  "unit": ("Score_E" if t == "E" else ("S_raw" if t == "S" else "B_raw"))}
    return {"n_test_genes": int(len(te)), "latency_s": round(dt, 2),
            "latency_per_gene_ms": round(1000 * dt / max(1, len(te)), 3), "per_task": res,
            "expected_from_rc2": {"E": 0.7427954, "S": 0.4341868, "B": 0.7261787},
            "labels_source": str(C.B16_LABELS_NPZ)}


def latency_single(predictor, gene_idx):
    seqs = G.slot_sequences(gene_idx)[None, ...]
    fv = np.load(C.B16_DATA_NPZ, allow_pickle=True)["X"][gene_idx][None, :].astype(np.float64)
    predictor.predict(seqs, fv, batch=1)  # warm-up (excluded)
    t0 = time.time()
    for _ in range(5):
        predictor.predict(seqs, fv, batch=1)
    return round(1000 * (time.time() - t0) / 5, 2)


def main():
    C.set_thread_limits()
    device = "cuda" if _cuda() else "cpu"
    cat = G.load_catalog()
    predictor = GM.GeneESBPredictor()
    out = {
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device,
        "python": sys.version.split()[0],
        "candidate_version": GM.CANDIDATE_VERSION,
        "model_sha": predictor.model_sha,
        "source_sha": M.source_sha(),
        "mpra_test": mpra_test_metrics(predictor),
        "gene_test": gene_test_metrics(cat, predictor),
    }
    out["latency_gene_single_ms"] = latency_single(predictor, int(np.where(cat["gene_split"] == "test")[0][0]))
    p = C.REPORT_OUT / "preliminary_metrics.json"
    C.atomic_write_json(p, out)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str)[:1600])
    print("wrote", p)


def _cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
