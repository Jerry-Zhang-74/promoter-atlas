#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-input entrypoint for NEW promoter sequences (K562 164 bp legnet branch).

Declared input-quality rules (batch21 protocol):
  * length must equal 164 bp for this branch; other lengths are UNSUPPORTED and
    refused with a clear message (never padded by duplicating sequence).
  * IUPAC ambiguity codes (R Y S W K M B D H V) are refused; only ACGT and N are
    accepted, and N-fraction must be <= 5 percent else refused.
  * empty records and header-less FASTA are explicit errors.
  * every emitted row carries source_similarity and domain labels.
  * source-catalog rows are marked in_source; a fold-aware OOF value is emitted
    ONLY for rows inside the prior batch02 CV cohort. Any row outside that cohort
    gets no OOF value (never fabricated).
"""
from __future__ import annotations

import numpy as np

from b21_common import (B21InputError, MPRA_LEN, N_MAX_FRACTION, sha256_text)
import b21_mpra

ACGT = set("ACGT")
IUPAC = set("RYSWKMBDHVN")


def validate_sequence(s, name=""):
    """Return a quality dict or raise B21InputError. Never silently coerces."""
    raw = (s or "").strip()
    if raw == "":
        raise B21InputError("empty sequence%s" % (" for " + name if name else ""))
    up = raw.upper()
    bad = sorted({c for c in up if c not in ACGT and c not in IUPAC})
    if bad:
        raise B21InputError("invalid base(s) %s%s; only ACGT/N accepted"
                            % (",".join(bad), " for " + name if name else ""))
    iupac_used = sorted({c for c in up if c in IUPAC and c != "N"})
    if iupac_used:
        raise B21InputError("IUPAC ambiguity code(s) %s unsupported%s"
                            % (",".join(iupac_used), " for " + name if name else ""))
    if len(up) != MPRA_LEN:
        raise B21InputError("length %d bp unsupported for the %d bp K562 branch%s "
                            "(no sequence duplication/padding is performed)"
                            % (len(up), MPRA_LEN, " (" + name + ")" if name else ""))
    n_frac = up.count("N") / len(up)
    if n_frac > N_MAX_FRACTION:
        raise B21InputError("N-fraction %.3f exceeds declared limit %.3f%s; row unsupported"
                            % (n_frac, N_MAX_FRACTION, " for " + name if name else ""))
    return {"sequence": up, "length": len(up), "n_count": up.count("N"),
            "n_fraction": n_frac, "sequence_sha256": sha256_text(up)}


def _onehot(seqs):
    L = MPRA_LEN
    B = {"A": 0, "C": 1, "G": 2, "T": 3}
    O = np.zeros((len(seqs), L * 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        codes = np.array([B.get(c, 0) for c in s], dtype=np.int64)
        for j, c in enumerate(codes):
            O[i, j * 4 + c] = 1.0
    return O


def _nearest_catalog(seqs):
    """Bounded exact min-Hamming (fwd + RC) to the catalog, for n<=256 inputs."""
    cat = b21_mpra._catalog_sequences()
    cat_s, oligo = cat["seqs"], cat["oligo"]
    L = MPRA_LEN
    rc_cat = [s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1] for s in cat_s]
    Ocat = _onehot(cat_s); Ocat_rc = _onehot(rc_cat)
    out = []
    for s in seqs:
        Ox = _onehot([s])[0]
        df = L - Ocat @ Ox
        dr = L - Ocat_rc @ Ox
        jf = int(np.argmin(df)); jr = int(np.argmin(dr))
        if df[jf] <= dr[jr]:
            out.append({"min_hamming": int(round(float(df[jf]))),
                        "nearest_oligo": oligo[jf], "orientation": "forward"})
        else:
            out.append({"min_hamming": int(round(float(dr[jr]))),
                        "nearest_oligo": oligo[jr], "orientation": "reverse_complement"})
    return out


def predict_records(records, compute_nn=True):
    """records: [(name, sequence)]. Returns (rows, refused)."""
    cohort = b21_mpra._cohort_oof()
    cat = b21_mpra._catalog_sequences()
    cat_sha = {sha256_text(s) for s in cat["seqs"]}
    rows, refused, valid = [], [], []
    for name, seq in records:
        try:
            q = validate_sequence(seq, name)
        except B21InputError as e:
            refused.append({"name": name, "reason": str(e)})
            continue
        q["name"] = name
        valid.append(q)
    nn = [None] * len(valid)
    if compute_nn and 0 < len(valid) <= 256:
        try:
            nn = _nearest_catalog([q["sequence"] for q in valid])
        except Exception as e:
            nn = [{"min_hamming": None, "nearest_oligo": None, "orientation": None,
                   "note": "nn_error:%s" % repr(e)[:80]}] * len(valid)
    new_seqs, new_pos = [], []
    for i, q in enumerate(valid):
        if q["sequence"] not in cohort:
            new_seqs.append(q["sequence"]); new_pos.append(i)
    ens = b21_mpra.predict_ensemble(new_seqs) if new_seqs else None
    new_map = {p: k for k, p in enumerate(new_pos)}
    for i, q in enumerate(valid):
        s = q["sequence"]
        in_cohort = s in cohort
        in_catalog = q["sequence_sha256"] in cat_sha
        row = {"name": q["name"], "sequence_sha256": q["sequence_sha256"],
               "length": q["length"], "n_fraction": round(q["n_fraction"], 4),
               "in_source_catalog": bool(in_catalog), "in_prior_cv_cohort": bool(in_cohort)}
        if in_cohort:
            c = cohort[s]
            row.update({"prediction": c["oof_legnet"], "prediction_type": "fold_aware_oof",
                        "oof_available": True, "prior_cv_fold": c["fold"],
                        "prior_component_id": c["component_id"],
                        "source_observed_activity": c["activity"],
                        "source_similarity": "exact_in_source_catalog",
                        "domain": "K562_164bp_MPRA", "robustness": "UNCONFIRMED"})
        elif in_catalog:
            row.update({"prediction": None, "prediction_type": "in_source_no_oof",
                        "oof_available": False,
                        "source_similarity": "exact_in_source_catalog_outside_cv_cohort",
                        "domain": "K562_164bp_MPRA", "robustness": "UNCONFIRMED",
                        "note": "outside prior CV cohort: no honest OOF exists; all-15 "
                                "in-sample agreement is NOT reported as new-sequence accuracy"})
        else:
            k = new_map[i]
            row.update({"prediction": float(ens["ensemble_mean"][k]),
                        "prediction_type": "ensemble15_new_sequence",
                        "prediction_std_over_models": float(ens["ensemble_std_over_models"][k]),
                        "oof_available": False, "source_similarity": "out_of_source_catalog",
                        "domain": "K562_164bp_MPRA", "robustness": "UNCONFIRMED"})
        if nn[i] is not None:
            mh = nn[i].get("min_hamming")
            row["min_hamming_to_catalog"] = mh
            row["nearest_catalog_oligo"] = nn[i].get("nearest_oligo")
            row["nearest_orientation"] = nn[i].get("orientation")
            row["source_similarity_tier"] = ("exact_catalog_match" if mh == 0 else
                                             "not_computed" if mh is None else
                                             "near_duplicate_to_catalog_le16" if mh <= 16 else
                                             "novel_relative_to_catalog")
        else:
            row["source_similarity_tier"] = "not_computed"
        rows.append(row)
    return rows, refused
