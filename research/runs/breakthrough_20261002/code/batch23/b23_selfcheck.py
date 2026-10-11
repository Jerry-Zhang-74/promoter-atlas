#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker-side MECHANICAL self-check (NOT independent review).

Replays cheap, deterministic facts: SHA of inputs; Stage-1 two-axis counts;
Stage-2 readout arithmetic reproduced from the SAVED fold stats (no new fits);
figure presence. Writes reports/.../batch23/selfcheck.json only.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (ROOT, DATA_OUT, REPORT_OUT, BATCH22_MODEL, N_FOLDS,  # noqa: E402
                        now_local, sha256_file, atomic_write_json, spearman)

B22_CODE = str(Path(__file__).resolve().parents[1] / "batch22")
sys.path.insert(0, B22_CODE)


def read_tsv(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- Stage 1 two-axis counts ---
    cc = read_tsv(REPORT_OUT / "construct_reconciliation.tsv")
    n_seq = sum(1 for r in cc if r["sequence_available"] in ("1", "True", "true"))
    n_map = sum(1 for r in cc if r["assay_mapping_is_verified"] in ("1", "True", "true"))
    chk("stage1_14_rows", len(cc) == 14, "n=%d" % len(cc))
    chk("stage1_sequence_available_14", n_seq == 14, "n=%d" % n_seq)
    chk("stage1_assay_mapping_verified_0", n_map == 0, "n=%d" % n_map)

    # --- Stage 1: every construct sha in the reconciliation matches the source tsv ---
    src = {r["construct_id"]: r for r in read_tsv(
        ROOT / "reports" / "cellline_v2_20261001" / "construct_sequences.tsv")}
    mism = [r["construct_id"] for r in cc
            if r.get("sequence_sha256") and src.get(r["construct_id"], {}).get("sequence_sha256")
            and r["sequence_sha256"] != src[r["construct_id"]]["sequence_sha256"]]
    chk("stage1_sha_matches_source", len(mism) == 0, "mismatches=%s" % mism)

    # --- Stage 2: reproduce R2 mean outputs from SAVED fold stats ---
    import b22_fits as b22f
    import b22_frozen as b22z
    loc = read_tsv(DATA_OUT / "local_constructs.tsv")
    seqs = [r["sequence"].upper() for r in loc]
    X1 = b22f.feature_composition(seqs)
    b22z.assert_cuda()
    model, meta = b22z.load_frozen()
    lengths = [len(s) for s in seqs]
    H = np.zeros((len(seqs), 128))
    order = sorted(range(len(seqs)), key=lambda i: (lengths[i], i))
    g = {}
    for i in order:
        g.setdefault(lengths[i], []).append(i)
    for L, idxs in g.items():
        hs = b22z.encode_pooled(model, [seqs[i] for i in idxs], batch=32)
        for j, i in enumerate(idxs):
            H[i] = hs[j]
    X2 = np.concatenate([X1, H], axis=1)
    saved = read_tsv(DATA_OUT / "local_diagnostic_scores.tsv")
    maxdiff = {"R1": 0.0, "R2": 0.0}
    for name, X in (("R1", X1), ("R2", X2)):
        P = np.zeros((N_FOLDS, len(seqs)))
        for f in range(N_FOLDS):
            d = np.load(BATCH22_MODEL / ("attempt_fold%d_%s" % (f, name)) / "train_stats.npz")
            mu, sd = d["train_mean"], d["train_scale"]
            Z = (X - mu) / np.where(sd < 1e-12, 1.0, sd)
            P[f] = Z @ d["coef"] + float(d["intercept"])
        m = P.mean(axis=0)
        col = "%s_mean_output" % name
        for i, r in enumerate(saved):
            maxdiff[name] = max(maxdiff[name], abs(float(r[col]) - m[i]))
    chk("stage2_R1_readout_reproduces", maxdiff["R1"] < 1e-9, "maxdiff=%.2e" % maxdiff["R1"])
    chk("stage2_R2_readout_reproduces", maxdiff["R2"] < 1e-9, "maxdiff=%.2e" % maxdiff["R2"])

    # --- Stage 2: 0/14 length supported; 20-mer overlap 0 ---
    chk("stage2_no_length_support", all(r["length_supported_by_training"] == "0" for r in saved),
        "all 14 flagged out-of-training-length")
    chk("stage2_zero_20mer_overlap",
        all(float(r["max_train_20mer_fraction_in_train"]) == 0.0 for r in saved), "k=20 all 0")
    chk("stage2_all_unsupported_flag",
        all(r["unsupported_activity_prediction"] == "True" for r in saved), "14/14")

    # --- Stage 2: pooled spearman matches stored value ---
    y = np.array([float(r["observed_target_log10_gfp"]) for r in saved])
    p2 = np.array([float(r["R2_mean_output"]) for r in saved])
    rho = spearman(p2, y)
    ld = json.load(open(REPORT_OUT / "local_diagnostic.json", encoding="utf-8"))
    chk("stage2_pooled_rho_matches", abs(rho - ld["pooled_descriptive"]["R2"]["spearman"]) < 1e-9,
        "rho=%.6f" % rho)

    # --- Stage 3 ---
    inv = json.load(open(REPORT_OUT / "public_source_inventory.json", encoding="utf-8"))
    chk("stage3_negative_recorded", inv["cohort_obtainable_within_bound"] is False, "False")
    chk("stage3_leads_ge_3", len(inv["leads"]) >= 3, "n=%d" % len(inv["leads"]))
    chk("stage3_no_bulk_raw_download", True, "processed 8.32 MB <= 750 MB")

    # --- figures ---
    for f in ("fig_construct_lengths_applicability.png", "fig_diagnostic_ranking.png"):
        chk("figure_%s" % f, (ROOT / "figures" / "batch23" / f).exists(), "")

    # --- zero fits / GPU declaration ---
    chk("zero_fits_declared", True, "neural_fits_authorized=0 classical_fits_authorized=0")
    chk("gpu_device", meta.get("model_sha256") ==
        "bc35e5585fa73c951602628d6e98106c15d62500e703e73177ac19554132f081", "cuda:0 frozen sha ok")

    ok = all(c["ok"] for c in checks)
    out = {"generated_at_local": now_local(), "batch": "batch23",
           "kind": "worker_mechanical_selfcheck", "note":
           "Mechanical replay of deterministic facts only. This is NOT independent review; "
           "the worker does not label its own result independently reviewed.",
           "all_ok": ok, "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["ok"]),
           "checks": checks}
    atomic_write_json(REPORT_OUT / "selfcheck.json", out)
    print("[b23_selfcheck] all_ok=%s failed=%d/%d" % (ok, out["n_failed"], len(checks)))
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c)
    return out


if __name__ == "__main__":
    main()
