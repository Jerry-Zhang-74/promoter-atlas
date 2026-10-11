#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: UNTUNED LOCAL DIAGNOSTIC (batch23).

Applies the FIVE SAVED batch22 fold readouts (R1 composition ridge, R2 composition +
frozen 128-d ridge) to each of the 14 ACTUAL native full inserts, using the EXACT
saved batch22 preprocessing. No truncation, no padding, no reference restoration, no
repeated bags, ZERO fitting. Outputs are exploratory_model_score with
unsupported_activity_prediction=true; they are NOT physical/validated HEK293T activity.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
B22_CODE = str(Path(__file__).resolve().parents[1] / "batch22")
sys.path.insert(0, B22_CODE)

from b23_common import (  # noqa: E402
    ROOT, DATA_OUT, REPORT_OUT, MODEL_OUT, BATCH22_MODEL, BATCH22_SEQ_TABLE,
    TRAIN_LEN, N_FOLDS, now_local, sha256_file, atomic_write_json, set_thread_limits,
    spearman, pearson, mae, r2, rc)


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def kmer_set(seqs, k=20):
    s = set()
    for x in seqs:
        x = x.upper()
        for i in range(len(x) - k + 1):
            s.add(x[i:i + k])
    return s


def max_train_similarity(local_seq, train_kmers, k=20):
    """Fraction of the local sequence's k-mers contained anywhere in the training
    sequence set (both strands). 0..1; a descriptive composition-overlap diagnostic."""
    s = local_seq.upper()
    total = max(0, len(s) - k + 1)
    if total == 0:
        return {"k": k, "n_kmers": 0, "n_in_train": 0, "frac_in_train": 0.0}
    hit = 0
    for i in range(total):
        if s[i:i + k] in train_kmers:
            hit += 1
    return {"k": k, "n_kmers": total, "n_in_train": hit, "frac_in_train": hit / total}


def main():
    set_thread_limits(4)
    t0 = now_local()
    import b22_fits as b22f          # frozen composition features (READ-ONLY reuse)
    import b22_frozen as b22z        # frozen encoder (READ-ONLY reuse)

    # ---- local constructs (native full inserts) ----
    loc = read_tsv(DATA_OUT / "local_constructs.tsv")
    cids = [r["construct_id"] for r in loc]
    fams = [r["family"] for r in loc]
    seqs = [r["sequence"].upper() for r in loc]
    lengths = [len(s) for s in seqs]
    y_obs = np.array([float(r["endpoint_target_value_log10"]) for r in loc], dtype=np.float64)

    # ---- batch22 training sequences (target-domain reference) ----
    train_rows = read_tsv(BATCH22_SEQ_TABLE)
    train_seqs = [r["sequence"].upper() for r in train_rows]
    train_rc = [rc(s) for s in train_seqs]
    train_k = kmer_set(train_seqs + train_rc, k=20)
    train_k_by_k = {k: kmer_set(train_seqs + train_rc, k=k) for k in (20, 15, 12, 8)}
    train_len_set = sorted({len(s) for s in train_seqs})

    # ---- R1 composition features: EXACT batch22 function ----
    X1 = b22f.feature_composition(seqs)          # (14, 86)
    assert X1.shape[1] == 86, X1.shape

    # ---- frozen 128-d repr via cuda:0 (strict load + assertions) ----
    b22z.assert_cuda()
    model, meta = b22z.load_frozen()
    # group equal lengths (encoder contract) and encode
    H = np.zeros((len(seqs), 128), dtype=np.float64)
    order = sorted(range(len(seqs)), key=lambda i: (lengths[i], i))
    groups = {}
    for i in order:
        groups.setdefault(lengths[i], []).append(i)
    for L, idxs in groups.items():
        hs = b22z.encode_pooled(model, [seqs[i] for i in idxs], batch=32)
        for j, i in enumerate(idxs):
            H[i] = hs[j]
    assert H.shape == (len(seqs), 128)
    X2 = np.concatenate([X1, H], axis=1)          # (14, 214)
    assert X2.shape[1] == 214, X2.shape

    # ---- the FIVE saved fold readouts (R1, R2) ----
    fold_pred = {"R1": np.zeros((N_FOLDS, len(seqs))), "R2": np.zeros((N_FOLDS, len(seqs)))}
    fold_meta = []
    for f in range(N_FOLDS):
        for name, X in (("R1", X1), ("R2", X2)):
            d = np.load(BATCH22_MODEL / ("attempt_fold%d_%s" % (f, name)) / "train_stats.npz")
            mu = d["train_mean"]; sd = d["train_scale"]; coef = d["coef"]; b = float(d["intercept"])
            Z = (X - mu) / np.where(sd < 1e-12, 1.0, sd)
            fold_pred[name][f] = Z @ coef + b
            fold_meta.append({"fold": f, "recipe": name, "dim": int(X.shape[1]),
                              "n_train_stats": int(len(coef))})
    mean_pred = {n: fold_pred[n].mean(axis=0) for n in ("R1", "R2")}
    spread = {n: fold_pred[n].std(axis=0, ddof=0) for n in ("R1", "R2")}

    # ---- composition distance (3-mer cosine vs mean training composition) ----
    def tri_vec(s):
        from b23_common import BASE
        v = np.zeros(64)
        s = s.upper()
        for i in range(len(s) - 2):
            a = BASE.get(s[i]); b2 = BASE.get(s[i + 1]); c = BASE.get(s[i + 2])
            if a is None or b2 is None or c is None:
                continue
            v[(a * 4 + b2) * 4 + c] += 1
        n = v.sum()
        return v / n if n else v

    train_tri = np.array([tri_vec(s) for s in train_seqs])
    train_tri_mean = train_tri.mean(axis=0)

    def comp_dist(s):
        v = tri_vec(s)
        d = np.linalg.norm(v) * np.linalg.norm(train_tri_mean)
        return float(1.0 - (v @ train_tri_mean) / d) if d > 0 else float("nan")

    # ---- per-construct records ----
    recs = []
    for i, cid in enumerate(cids):
        sim = max_train_similarity(seqs[i], train_k, k=20)
        sim_by_k = {("k%d" % k): max_train_similarity(seqs[i], train_k_by_k[k], k=k)["frac_in_train"]
                    for k in (20, 15, 12, 8)}
        rec = {
            "construct_id": cid, "family": fams[i],
            "sequence_length_bp": lengths[i],
            "training_length_bp": TRAIN_LEN,
            "length_supported_by_training": int(lengths[i] == TRAIN_LEN),
            "R1_fold_outputs": [float(v) for v in fold_pred["R1"][:, i]],
            "R2_fold_outputs": [float(v) for v in fold_pred["R2"][:, i]],
            "R1_mean_output": float(mean_pred["R1"][i]),
            "R2_mean_output": float(mean_pred["R2"][i]),
            "R1_fold_spread_sd": float(spread["R1"][i]),
            "R2_fold_spread_sd": float(spread["R2"][i]),
            "observed_target_log10_gfp": float(y_obs[i]),
            "exploratory_model_score_units": "batch22 hek_norm (log2 HEK293 exp/input) scale",
            "unsupported_activity_prediction": True,
            "output_unavailable": False,
            "max_train_20mer_fraction_in_train": sim["frac_in_train"],
            "max_train_20mer_n_in_train": sim["n_in_train"],
            "n_20mers": sim["n_kmers"],
            "train_kmer_fraction_by_k": sim_by_k,
            "composition_distance_3mer_cosine": comp_dist(seqs[i]),
            "assay_mapping_certainty": loc[i]["assay_mapping_certainty"],
            "assay_to_plasmid_barcode_verified": loc[i]["assay_to_plasmid_barcode_verified"],
            "cell_assay_endpoint_mismatch": "trained HEK293 bulk log2(RNA/input); local HEK293T "
                                            "transient plasmid GFP D28 second-sort log10(relGFP)",
            "flags": [],
        }
        if not rec["length_supported_by_training"]:
            rec["flags"].append("out_of_training_length")
        rec["flags"].append("cell_context_mismatch")
        rec["flags"].append("assay_mapping_unverified_barcode")
        if sim["frac_in_train"] > 0.5:
            rec["flags"].append("high_train_20mer_overlap")
        if lengths[i] > 600:
            rec["flags"].append("length_far_above_training_domain")
        recs.append(rec)

    # ---- descriptive pooled metrics (strong OOD qualification; NO calibration) ----
    pooled = {}
    for n in ("R1", "R2"):
        p = mean_pred[n]
        pooled[n] = {"spearman": spearman(p, y_obs), "pearson": pearson(p, y_obs),
                     "mae": mae(p, y_obs), "r2": r2(p, y_obs), "n": len(p)}
    # also per individual fold (transparency; no fold selection)
    per_fold = {"R1": [], "R2": []}
    for n in ("R1", "R2"):
        for f in range(N_FOLDS):
            per_fold[n].append({"fold": f, "spearman": spearman(fold_pred[n][f], y_obs),
                                "mae": mae(fold_pred[n][f], y_obs)})
    fold_spearman_range = {n: [min(x["spearman"] for x in per_fold[n]),
                               max(x["spearman"] for x in per_fold[n])] for n in ("R1", "R2")}

    # ---- within-family ordering (descriptive; >=3 distinct constructs only) ----
    fams_u = sorted(set(fams))
    within_family = {}
    for fam in fams_u:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if len(idx) < 3:
            within_family[fam] = {"n": len(idx), "computed": False,
                                  "reason": "fewer than 3 distinct constructs in this family"}
            continue
        p = mean_pred["R2"][idx]; o = y_obs[idx]
        within_family[fam] = {"n": len(idx), "computed": True,
                              "spearman_R2": spearman(p, o),
                              "spearman_R1": spearman(mean_pred["R1"][idx], o)}

    # ---- pairwise sign agreement (distinct constructs) ----
    def sign_agreement(p, o):
        n = len(p); match = 0; denom = 0; ties = 0
        for i in range(n):
            for j in range(i + 1, n):
                dp = p[i] - p[j]; do = o[i] - o[j]
                if dp == 0 or do == 0:
                    ties += 1
                    continue
                denom += 1
                if (dp > 0) == (do > 0):
                    match += 1
        return {"n_pairs": n * (n - 1) // 2, "n_comparable": denom, "n_ties": ties,
                "n_agree": match, "agreement": (match / denom) if denom else None}
    pair_agree = {"R1": sign_agreement(mean_pred["R1"], y_obs),
                  "R2": sign_agreement(mean_pred["R2"], y_obs)}

    # ---- write outputs ----
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out_tsv = DATA_OUT / "local_diagnostic_scores.tsv"
    cols = ["construct_id", "family", "sequence_length_bp", "training_length_bp",
            "length_supported_by_training", "max_train_20mer_fraction_in_train",
            "max_train_20mer_n_in_train", "n_20mers", "composition_distance_3mer_cosine",
            "train_kmer_frac_k20", "train_kmer_frac_k15", "train_kmer_frac_k12", "train_kmer_frac_k8",
            "assay_mapping_certainty", "assay_to_plasmid_barcode_verified",
            "R1_mean_output", "R2_mean_output", "R1_fold_spread_sd", "R2_fold_spread_sd",
            "observed_target_log10_gfp", "exploratory_model_score_units",
            "unsupported_activity_prediction", "flags",
            "R1_f0", "R1_f1", "R1_f2", "R1_f3", "R1_f4",
            "R2_f0", "R2_f1", "R2_f2", "R2_f3", "R2_f4"]
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in recs:
            w.writerow([r["construct_id"], r["family"], r["sequence_length_bp"], r["training_length_bp"],
                        r["length_supported_by_training"], r["max_train_20mer_fraction_in_train"],
                        r["max_train_20mer_n_in_train"], r["n_20mers"], r["composition_distance_3mer_cosine"],
                        r["train_kmer_fraction_by_k"]["k20"], r["train_kmer_fraction_by_k"]["k15"],
                        r["train_kmer_fraction_by_k"]["k12"], r["train_kmer_fraction_by_k"]["k8"],
                        r["assay_mapping_certainty"], r["assay_to_plasmid_barcode_verified"],
                        r["R1_mean_output"], r["R2_mean_output"], r["R1_fold_spread_sd"], r["R2_fold_spread_sd"],
                        r["observed_target_log10_gfp"], r["exploratory_model_score_units"],
                        r["unsupported_activity_prediction"], ";".join(r["flags"])]
                       + r["R1_fold_outputs"] + r["R2_fold_outputs"])

    result = {
        "generated_at_local": now_local(),
        "started_local": t0,
        "batch": "batch23",
        "n_constructs": len(recs),
        "n_outputs": int(sum(0 if r["output_unavailable"] else 1 for r in recs)),
        "n_folds": N_FOLDS,
        "training_length_bp": TRAIN_LEN,
        "training_rows": len(train_seqs),
        "training_length_set": train_len_set,
        "frozen_model": meta,
        "zero_fits_this_stage": True,
        "ensemble_definition": "unweighted mean of the FIVE saved fold readouts; NOT a newly "
                               "trained estimator; fold spread is NOT a confidence interval",
        "pooled_descriptive": pooled,
        "per_fold_descriptive": per_fold,
        "fold_spearman_range": fold_spearman_range,
        "within_family_descriptive": within_family,
        "pairwise_sign_agreement": pair_agree,
        "records": recs,
        "caveats": [
            "exploratory_model_score only: batch22 hek_norm (HEK293 bulk log2 RNA/DNA) scale",
            "unsupported_activity_prediction=true for every construct",
            "different cell (HEK293T vs HEK293), different assay (transient plasmid GFP D28 vs bulk MPRA)",
            "different scale (log10 relGFP vs log2 exp/input): no absolute MAE/R2 comparison across scales",
            "no calibration and no target-driven weight/selection",
            "length support is NOT asserted merely because the encoder accepts the input dimension",
        ],
    }
    atomic_write_json(REPORT_OUT / "local_diagnostic.json", result)
    atomic_write_json(REPORT_OUT / "stage2_manifest.json", {
        "generated_at_local": now_local(),
        "outputs": {
            str(out_tsv.relative_to(ROOT)): sha256_file(out_tsv),
            "reports/breakthrough_20261002/batch23/local_diagnostic.json":
                sha256_file(REPORT_OUT / "local_diagnostic.json"),
        },
        "gpu": {"device": "cuda:0", "model_sha256": meta.get("model_sha256"),
                "n_seqs_encoded": len(seqs), "batch": 32},
    })
    print("[b23_diagnostic] n=%d R2_pooled_rho=%.4f R1_pooled_rho=%.4f max20mer=%s"
          % (len(recs), pooled["R2"]["spearman"], pooled["R1"]["spearman"],
             [round(r["max_train_20mer_fraction_in_train"], 2) for r in recs]))
    return result


if __name__ == "__main__":
    main()
