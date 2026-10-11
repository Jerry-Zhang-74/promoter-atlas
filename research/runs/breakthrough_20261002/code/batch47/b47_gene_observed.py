# -*- coding: utf-8 -*-
"""batch47 gene-observed: actual natural-gene CATALOG lookup (b20 sources authoritative, no network).

This is a derived catalog-label lookup, NOT an individual promoter measurement, NOT a 293FT /
GFP activity, and NOT a model prediction. It reads ONLY the traceable label table + symbol map and
therefore needs NO CUDA, NO torch, NO model weights, NO features and NO memmap.
"""
from __future__ import annotations

import csv

from b47_common import (ANALYSIS_SET, B47InputError, LABEL_TABLE, REQUIRED_LABEL_COLUMNS,
                        looks_like_dna, sha256_file)

FIELD_DEFINITIONS = {
    "observedScore_E": {
        "source_column": "Score_E",
        "definition": "rank_scale(E_raw) over all catalog gene rows: rank/(n-1) ascending, in [0,1]",
        "E_raw": "mean over all collapsed tissues of log2(1+mean_tpm)",
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
        "definition": "rank_scale(S_raw) over genes with defined S; informational; NOT the same units as S_raw",
    },
    "observedScore_B": {
        "source_column": "Score_B",
        "rank_scaled": True,
        "definition": "rank_scale(B_raw) over all catalog genes; informational; NOT the same units as B_raw",
    },
    "is_in_main_subset": {"source_column": "is_in_main_subset", "definition": "1 iff B_raw >= 0.30 (B gate)"},
    "S_defined_main": {"source_column": "S_defined_main", "definition": "1 iff S_raw is defined"},
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
        raise B47InputError("LABEL_CONTRACT_ABSENT: label table not found at %s" % LABEL_TABLE)
    if not ANALYSIS_SET.exists():
        raise B47InputError("LABEL_CONTRACT_ABSENT: symbol map not found at %s" % ANALYSIS_SET)
    labels = {}
    with open(LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames or []
        missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in cols]
        if missing:
            raise B47InputError("LABEL_CONTRACT_ABSENT: label table missing required columns %s" % missing)
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
        raise B47InputError("LABEL_CONTRACT_ABSENT: label table has zero usable rows")

    id2sym = {}
    sym2ids = {}
    with open(ANALYSIS_SET, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames or []
        if "gene_id" not in cols or "gene_name" not in cols:
            raise B47InputError("LABEL_CONTRACT_ABSENT: symbol map lacks gene_id/gene_name columns")
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
        "is_in_main_subset": None,
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
    if row["S_defined_main"] == 1:
        rec["S_raw"] = row["S_raw"]
        rec["S_raw_defined"] = 1
        rec["observedScore_S"] = row["Score_S"]
        rec["note"] = "catalog-derived label; S defined (cross-tissue stability)"
    else:
        rec["S_raw"] = None
        rec["S_raw_defined"] = 0
        rec["observedScore_S"] = None
        rec["note"] = ("catalog-derived label; S undefined for this gene (S mask=0); "
                       "S NOT filled with 0 or a prediction")
    rec["is_in_main_subset"] = row["is_in_main_subset"]
    return rec


def source_sha():
    return {"label_table": {"path": str(LABEL_TABLE),
                            "sha256": sha256_file(LABEL_TABLE) if LABEL_TABLE.exists() else None},
            "symbol_map": {"path": str(ANALYSIS_SET),
                           "sha256": sha256_file(ANALYSIS_SET) if ANALYSIS_SET.exists() else None}}
