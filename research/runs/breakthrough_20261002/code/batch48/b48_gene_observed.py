# -*- coding: utf-8 -*-
"""batch48 gene-observed: actual natural-gene CATALOG lookup (b20 sources authoritative, no network).

MINIMAL scoped copy of batch47 b47_gene_observed.py. The ONE substantive change (repair R2) is the
CORRECTED FIELD_DEFINITIONS.observedScore_S wording, derived from the ACTUAL label-generation source
(runs/scripts/build_candidate_e_labels.py) rather than from rounded output correlation:

  assign_scores() calls rank_scale(gene_rows, "S_raw", "Score_S", none_value=float("-inf")) over ALL
  catalog rows (n_total = 58988), then resets rows with S_raw is None to Score_S = None.
  rank_scale() sorts ASCENDING by sort_value (None -> -inf, i.e. undefined at the BOTTOM) and assigns
  rank / (n_total - 1). => undefined S occupies the BOTTOM band and is then stored as null; a DEFINED
  gene gets (n_undefined + rank_among_defined_ascending) / (n_total - 1), i.e. only the TOP band.
  So observedScore_S is NOT a rank restricted to defined-S genes.

Verified (read-only) against the frozen table labels/build/gene_expression_label_table.tsv:
  n_total = 58988, n_undefined(S_defined_main==0) = 40041, n_defined = 18947;
  min defined Score_S = 0.678811 == n_undefined/(n_total-1) = 40041/58987; max = 1.0.
This is a derived catalog-label lookup, NOT an individual promoter measurement, NOT 293FT / GFP
activity, and NOT a model prediction. It needs NO CUDA, torch, model weights, features or memmap.
"""
from __future__ import annotations

import csv

from b48_common import (ANALYSIS_SET, B48InputError, LABEL_TABLE, REQUIRED_LABEL_COLUMNS,
                        looks_like_dna, sha256_file)

SOURCE_EVIDENCE = {
    "generation_code": "runs/scripts/build_candidate_e_labels.py",
    "functions": {
        "rank_scale": "rank_scale(rows, key, out_key, none_value=None): sort ASCENDING by value "
                      "(None -> none_value), assign rank/(len(rows)-1).",
        "assign_scores": "assign_scores(gene_rows): rank_scale(E_raw->Score_E); rank_scale(B_raw->Score_B); "
                         "rank_scale(S_raw->Score_S, none_value=float('-inf')); then rows with S_raw None get "
                         "Score_S=None.",
    },
    "key_constants": {"DETECTABLE_TPM_THRESHOLD": 1.0, "B_MIN_FOR_MAIN": 0.30,
                      "MIN_DETECTABLE_TISSUES_FOR_S": 3},
    "verified_counts_from_frozen_table": {"n_total_rows": 58988, "n_S_defined": 18947, "n_S_undefined": 40041,
                                          "score_s_denominator": 58987,
                                          "min_defined_Score_S": 0.67881058538322,
                                          "max_Score_S": 1.0},
}

FIELD_DEFINITIONS = {
    "observedScore_E": {
        "source_column": "Score_E",
        "rank_scaled": True,
        "over": "ALL catalog gene rows (n_total = 58988)",
        "rank_denominator": "n_total - 1 = 58987",
        "definition": "rank_scale(E_raw) over the FULL catalog: 0-based ASCENDING rank / (n_total-1), in [0,1]",
        "E_raw": "mean over all collapsed tissues of log2(1+mean_tpm)",
        "defined_for": "all catalog gene rows (E_raw is present for every row)",
    },
    "S_raw": {
        "source_column": "S_raw",
        "definition": "-MAD(log2(1+mean_tpm)) over DETECTABLE tissues only (mean_tpm >= 1.0)",
        "units": "cross-tissue dispersion of expression; S is CROSS-TISSUE stability, NOT time",
        "defined_when": "is_in_main_subset==1 AND detectable_tissue_count>=3 (mask S_defined_main==1)",
        "missing_policy": "null with reason when undefined; NEVER filled with 0 or a prediction",
    },
    "B_raw": {
        "source_column": "B_raw",
        "definition": "detectable_tissue_count / tissue_count; detectable = mean_tpm >= 1.0",
        "units": "expression breadth fraction in [0,1]",
        "present_for": "all catalog gene rows",
    },
    "observedScore_S": {
        "source_column": "Score_S",
        "rank_scaled": True,
        "over": "ALL catalog gene rows (n_total = 58988), NOT only defined-S genes",
        "rank_denominator": "n_total - 1 = 58987",
        "definition": ("rank_scale(S_raw) over the FULL catalog. Rows whose S_raw is UNDEFINED (null) are "
                       "ordered at the BOTTOM (sort key = -inf) and then stored as null/empty. Therefore a "
                       "DEFINED gene gets observedScore_S = (n_undefined + rank_among_defined_ascending) / "
                       "(n_total - 1); defined genes occupy only the TOP band "
                       "[n_undefined/(n_total-1), 1] = [0.6788, 1.0]. It is NOT a rank restricted to "
                       "defined-S genes, and it is informational only (NOT the same units as S_raw; NOT time)."),
        "defined_mask": "S_defined_main == 1",
        "undefined_stored_as": "null (never 0, never a prediction)",
    },
    "observedScore_B": {
        "source_column": "Score_B",
        "rank_scaled": True,
        "over": "ALL catalog gene rows (n_total = 58988)",
        "rank_denominator": "n_total - 1 = 58987",
        "definition": "rank_scale(B_raw) over the FULL catalog: 0-based ASCENDING rank / (n_total-1); informational; NOT the same units as B_raw",
        "defined_for": "all catalog gene rows",
    },
    "is_in_main_subset": {"source_column": "is_in_main_subset", "definition": "1 iff B_raw >= 0.30 (B gate)"},
    "S_defined_main": {"source_column": "S_defined_main",
                       "definition": "1 iff S_raw is defined (is_in_main_subset==1 AND detectable_tissue_count>=3)"},
}


def _base(eid):
    return (eid or "").split(".")[0]


def _f(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def load_contract():
    """Load the label contract + symbol map. Fails with a SPECIFIC reason if absent/incomplete."""
    if not LABEL_TABLE.exists():
        raise B48InputError("LABEL_CONTRACT_ABSENT: label table not found at %s" % LABEL_TABLE)
    if not ANALYSIS_SET.exists():
        raise B48InputError("LABEL_CONTRACT_ABSENT: symbol map not found at %s" % ANALYSIS_SET)
    labels = {}
    with open(LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames or []
        missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in cols]
        if missing:
            raise B48InputError("LABEL_CONTRACT_ABSENT: label table missing required columns %s" % missing)
        for row in rdr:
            gid = (row.get("gene_id") or "").strip()
            if not gid:
                continue
            labels[gid] = {
                "Score_E": _f(row.get("Score_E")),
                "S_raw": _f(row.get("S_raw")),
                "B_raw": _f(row.get("B_raw")),
                "Score_S": _f(row.get("Score_S")),
                "Score_B": _f(row.get("Score_B")),
                "S_defined_main": _i(row.get("S_defined_main")),
                "is_in_main_subset": _i(row.get("is_in_main_subset")),
            }
    if not labels:
        raise B48InputError("LABEL_CONTRACT_ABSENT: label table has zero usable rows")

    id2sym = {}
    sym2ids = {}
    with open(ANALYSIS_SET, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames or []
        if "gene_id" not in cols or "gene_name" not in cols:
            raise B48InputError("LABEL_CONTRACT_ABSENT: symbol map lacks gene_id/gene_name columns")
        for row in rdr:
            base = _base((row.get("gene_id") or "").strip())
            sym = (row.get("gene_name") or "").strip()
            if not base or not sym:
                continue
            id2sym.setdefault(base, sym)
            sym2ids.setdefault(sym.upper(), set()).add(base)
    return {"labels": labels, "id2sym": id2sym,
            "sym2ids": {k: sorted(v) for k, v in sym2ids.items()}}


def resolve(contract, query):
    """Resolve a symbol/ID. status in OK / ABSENT / AMBIGUOUS / INVALID_QUERY."""
    q = (query or "").strip()
    if not q:
        return "INVALID_QUERY", "EMPTY_QUERY", None, None, []
    if looks_like_dna(q):
        msg = ("QUERY_LOOKS_LIKE_DNA_SEQUENCE: gene-observed expects a gene symbol or Ensembl ID; "
               "use the fragment command for raw DNA or FASTA input")
        return "INVALID_QUERY", msg, None, None, []
    labels = contract["labels"]
    if q.upper().startswith("ENSG"):
        base = _base(q).upper()
        if base in labels:
            return "OK", "", base, contract["id2sym"].get(base), []
        return "ABSENT", "ENSEMBL_ID_NOT_IN_CATALOG", None, None, []
    hits = contract["sym2ids"].get(q.upper(), [])
    if len(hits) == 0:
        return "ABSENT", "SYMBOL_NOT_IN_CATALOG", None, None, []
    if len(hits) > 1:
        cands = [{"gene_id": h, "symbol": contract["id2sym"].get(h)} for h in hits]
        return "AMBIGUOUS", "AMBIGUOUS_SYMBOL:%d" % len(hits), None, None, cands
    base = hits[0]
    if base not in labels:
        return "ABSENT", "SYMBOL_MAPS_TO_GENE_WITHOUT_LABEL_ROW", None, None, []
    return "OK", "", base, contract["id2sym"].get(base, q.upper()), []


def observed_record(contract, index, query, status, reason, base, symbol, candidates):
    rec = {
        "index": index, "query": query, "status": status, "reason": reason,
        "canonical_gene_id": base, "symbol": symbol,
        "observedScore_E": None, "S_raw": None, "S_raw_defined": 0, "B_raw": None,
        "observedScore_S": None, "observedScore_B": None,
        "is_in_main_subset": None, "S_defined_mask": 0,
        "candidates": candidates or None,
        "note": "",
    }
    if status != "OK":
        if status == "AMBIGUOUS":
            rec["note"] = "ambiguous symbol: %d catalog genes match; resolve before lookup" % len(candidates)
        else:
            rec["note"] = reason
        return rec
    row = contract["labels"][base]
    rec["observedScore_E"] = row["Score_E"]
    rec["B_raw"] = row["B_raw"]
    rec["observedScore_B"] = row["Score_B"]
    rec["is_in_main_subset"] = row["is_in_main_subset"]
    rec["S_defined_mask"] = row["S_defined_main"]
    if row["S_defined_main"] == 1:
        rec["S_raw"] = row["S_raw"]
        rec["S_raw_defined"] = 1
        rec["observedScore_S"] = row["Score_S"]
        rec["note"] = ("catalog-derived label; S defined (cross-tissue stability); observedScore_S is the "
                       "FULL-catalog rank (undefined S at bottom), informational only")
    else:
        rec["S_raw"] = None
        rec["S_raw_defined"] = 0
        rec["observedScore_S"] = None
        rec["note"] = ("catalog-derived label; S undefined for this gene (S_defined_main=0); "
                       "S NOT filled with 0 or a prediction")
    return rec


def source_sha():
    return {"label_table": {"path": str(LABEL_TABLE),
                            "sha256": sha256_file(LABEL_TABLE) if LABEL_TABLE.exists() else None},
            "symbol_map": {"path": str(ANALYSIS_SET),
                           "sha256": sha256_file(ANALYSIS_SET) if ANALYSIS_SET.exists() else None}}
