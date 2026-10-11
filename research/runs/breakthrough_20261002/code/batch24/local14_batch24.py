#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 local-14 exploratory diagnostic. ZERO fits, ZERO tuning.

H1 full-insert global spectrum only. H2 is UNSUPPORTED on non-133 bp inserts
(its positional contract is validated for exactly 133 bp) -> emit unsupported,
never a last-133/164 truncation. The H1 score is an exploratory numerical output
of 5 saved fold models averaged equally - NOT a validated activity prediction and
NOT calibrated to the local log10(GFP) endpoint.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C
import batch24_kernels as KK

C.set_thread_limits(4)
TRAIN_LEN = 133


def load_fold_models():
    models = []
    for f in range(C.N_FOLDS):
        d = C.MODEL_OUT / ("attempt_fold%d_H1" % f)
        z = np.load(d / "weights.npz", allow_pickle=True)
        models.append({"train_id": list(z["train_id"]), "w": z["w"],
                       "train_mean": float(z["train_mean"])})
    return models


def main():
    order, seqs, y, fold, grp, seen, k562 = C.load_rows()
    idx_of = {p: i for i, p in enumerate(order)}
    models = load_fold_models()
    rows = list(csv.DictReader(open(C.LOCAL_CONSTRUCTS, encoding="utf-8"), delimiter="\t"))
    train_seqs = [seqs[i] for i in range(len(order))]
    T4 = np.vstack([KK.feature_vector(t, 4) for t in train_seqs])
    T5 = np.vstack([KK.feature_vector(t, 5) for t in train_seqs])

    recs = []
    for r in rows:
        s = r["sequence"].strip().upper()
        L = len(s)
        c4 = KK.feature_vector(s, 4)
        c5 = KK.feature_vector(s, 5)
        kc = 0.5 * (T4 @ c4) + 0.5 * (T5 @ c5)
        scores = []
        for m in models:
            cols = np.array([idx_of[p] for p in m["train_id"]], dtype=np.int64)
            scores.append(float(kc[cols] @ m["w"] + m["train_mean"]))
        scores = np.array(scores, dtype=np.float64)
        length_ok = (L == TRAIN_LEN)
        if length_ok:
            reason = ("different cell/assay endpoint (local GFP) than the HEK293 bulk "
                      "log2(RNA/input) training target")
        else:
            reason = ("length %d bp != validated training length 133 bp; different "
                      "cell/assay endpoint; exploratory numerical score only" % L)
        recs.append({
            "construct_id": r["construct_id"], "family": r["family"],
            "documented_length_bp": int(r["documented_length_bp"]),
            "sequence_length_bp": L,
            "sequence_sha256": r["sequence_sha256"],
            "mapping_evidence": r["mapping_evidence"],
            "assay_mapping_certainty": r["assay_mapping_certainty"],
            "assay_to_plasmid_barcode_verified": int(r["assay_to_plasmid_barcode_verified"]),
            "endpoint": "log10 GFP (local retrospective construct)",
            "endpoint_target_value_log10": float(r["endpoint_target_value_log10"]),
            "H1_exploratory_model_score_folds": [float(v) for v in scores],
            "H1_exploratory_model_score_mean": float(scores.mean()),
            "H1_exploratory_model_score_min": float(scores.min()),
            "H1_exploratory_model_score_max": float(scores.max()),
            "H1_fold_spread": float(scores.max() - scores.min()),
            "H1_fold_spread_is_ci": False,
            "H1_activity_prediction_unsupported": True,
            "H1_unsupported_reason": reason,
            "H2_status": "unsupported",
            "H2_unsupported_reason": ("H2 positional contract validated only for exactly "
                                      "133 bp inserts; this construct is %d bp. No "
                                      "last-133/164 truncation applied." % L),
            "assay_mapping_barcode_unverified": int(not int(r["assay_to_plasmid_barcode_verified"])),
        })

    def sp(a, b):
        return C.spearman(np.asarray(a, float), np.asarray(b, float))

    pooled = {"n": len(recs),
              "spearman_exploratory_vs_measured": sp(
                  [x["H1_exploratory_model_score_mean"] for x in recs],
                  [x["endpoint_target_value_log10"] for x in recs])}
    fams = {}
    for fam in sorted(set(x["family"] for x in recs)):
        sub = [x for x in recs if x["family"] == fam]
        fams[fam] = {"n": len(sub)}
        if len(sub) >= 3:
            fams[fam]["spearman_exploratory_vs_measured"] = sp(
                [x["H1_exploratory_model_score_mean"] for x in sub],
                [x["endpoint_target_value_log10"] for x in sub])
        else:
            fams[fam]["computed"] = False
            fams[fam]["reason"] = "fewer than 3 distinct constructs in this family"
    n_pairs = n_tie_score = n_tie_meas = n_agree = 0
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            ds = (recs[i]["H1_exploratory_model_score_mean"]
                  - recs[j]["H1_exploratory_model_score_mean"])
            dm = (recs[i]["endpoint_target_value_log10"]
                  - recs[j]["endpoint_target_value_log10"])
            n_pairs += 1
            if ds == 0:
                n_tie_score += 1
            if dm == 0:
                n_tie_meas += 1
            if ds * dm > 0:
                n_agree += 1
    signblk = {"n_pairs": n_pairs, "n_tie_score": n_tie_score, "n_tie_meas": n_tie_meas,
               "n_agree": n_agree,
               "agreement": n_agree / n_pairs if n_pairs else float("nan"),
               "n_comparable": n_pairs - n_tie_score - n_tie_meas}

    out = {
        "generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
        "zero_fits_this_stage": True, "zero_tuning": True,
        "training_length_bp": TRAIN_LEN, "n_constructs": len(recs),
        "ensemble_definition": ("equal average of the FIVE saved H1 fold models; NOT a newly "
                                "trained estimator; fold spread is NOT a confidence interval"),
        "endpoint_scope": ("local retrospective log10(GFP) endpoint; NOT the HEK293 bulk "
                           "log2(RNA/input) training target; NO cross-scale MAE/R2 and NO "
                           "local calibration computed"),
        "H2_reason": "positional kernel validated only for exactly 133 bp; all 14 differ",
        "pooled_descriptive": pooled, "within_family_descriptive": fams,
        "pairwise_sign_agreement": signblk,
        "classification": {"H1": "exploratory_model_score",
                           "H1_unsupported_activity_prediction": True,
                           "H2": "unsupported"},
        "caveats": [
            "no validated activity claim; exploratory numerical score only",
            "assay-to-plasmid barcode mapping is UNVERIFIED for all 14 constructs",
            "CMVcore actual feature length 204 bp vs historical 200 bp label; CMVfull (584 bp) "
            "is a distinct enhancer-containing construct",
            "A6 and the 81 bp untransfected sample are excluded (not in this table)",
            "no best-fold selection: all five fold outputs are retained",
            "published HEK293 (training) is not HEK293T and not the local GFP assay",
        ],
        "records": recs,
    }
    C.atomic_write_json(C.REPORT_OUT / "local14_exploratory.json", out)
    cols = ["construct_id", "family", "documented_length_bp", "sequence_length_bp",
            "H1_exploratory_model_score_mean", "H1_fold_spread",
            "endpoint_target_value_log10", "H1_activity_prediction_unsupported",
            "H2_status", "assay_to_plasmid_barcode_verified", "assay_mapping_certainty"]
    with open(C.DATA_OUT / "local14_exploratory_scores.tsv", "w", encoding="utf-8",
              newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for x in recs:
            fh.write("\t".join(str(x[c]) for c in cols) + "\n")
    print(json.dumps({"pooled": pooled, "families": fams, "sign": signblk}, indent=1))


if __name__ == "__main__":
    main()
