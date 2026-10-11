# -*- coding: utf-8 -*-
"""batch49 gene-rank: deterministic CATALOG TRADEOFF SELECTION on the frozen label table (no model, no GPU).

The NEW batch49 capability. It ranks natural genes from the existing FROZEN catalog label table
(labels/build/gene_expression_label_table.tsv) using the two DESIGN AXES E_raw and S_raw, with B_raw as a
GATE only. It reuses the batch49 catalog loader (b49_gene_observed.load_contract) so the values/masks are
byte-identical to `gene-observed`.

Contract (FROZEN before any candidate value is printed; see freeze_contract()):
  * axes     : E_raw (mean cross-tissue log2(1+mean_tpm)) and S_raw (-MAD cross-tissue dispersion of
               log2(1+mean_tpm) over detectable tissues). BOTH are HIGHER-BETTER. S_raw is CROSS-TISSUE
               stability, NOT time; a LESS NEGATIVE S_raw = LESS cross-tissue dispersion.
  * B gate   : B_raw >= min_b (inclusive). B_raw is a GATE only -- it is NEVER a multiplicative third score.
  * eligible : finite E_raw, finite S_raw, finite B_raw AND S_defined_main == 1. Undefined S is NEVER
               filled with 0 and NEVER recalculated/overridden from B.
  * dominance: p dominates q iff E_raw(p) >= E_raw(q) and S_raw(p) >= S_raw(q) with at least one STRICT.
  * ties     : exact equal (E_raw, S_raw) pairs are ALL nondominated together (no dedup, no gene collapse).
  * frontier : eligible genes with NO dominator among ALL ELIGIBLE genes.
  * no score : no arbitrary weighted E+S and no product; E and S are axes, "E x S" is a design idea, not a
               numeric multiplication.
  * sorts    : expression -> E_raw desc, S_raw desc, gene_id asc; stability -> S_raw desc, E_raw desc,
               gene_id asc; pareto -> the frontier, E_raw desc, S_raw desc, gene_id asc.
  * limit    : DISPLAY only. The FULL frontier is computed FIRST; limit truncates the returned view.

These are catalog-derived OBSERVED labels -- NOT an individual promoter measurement, NOT a model prediction,
NOT a 293FT/GFP activity and NOT a wet-lab recommendation or an autonomous-promoter proof. No CUDA/torch/
numpy/weights/features; runs CUDA-hidden.
"""
from __future__ import annotations

import math

from b49_common import (B49InputError, DEFAULT_LIMIT, DEFAULT_MIN_B, DEFAULT_MIN_E, DEFAULT_SORT,
                        MAX_LIMIT, SORT_MODES)

# ---- FROZEN rule text (single source of truth; copied verbatim into freeze_contract) ----
RULE_ELIGIBILITY = ("finite E_raw AND finite S_raw AND finite B_raw AND S_defined_main == 1")
RULE_DOMINANCE = ("gene p DOMINATES gene q iff E_raw(p) >= E_raw(q) AND S_raw(p) >= S_raw(q) AND at least "
                  "one of the two is STRICT (>).")
RULE_TIE = ("exact equal (E_raw, S_raw) pairs are ALL nondominated together; genes are NEVER deduplicated "
            "or collapsed on ties.")
RULE_FRONTIER = "an eligible gene is on the Pareto frontier iff NO other ELIGIBLE gene dominates it."
RULE_B_GATE = ("B_raw is a GATE only (B_raw >= min_b inclusive); it is NEVER a multiplicative third score "
               "and never combined into a single number.")
RULE_NO_WEIGHTED_SCORE = ("no arbitrary weighted E+S and no product score; E_raw and S_raw are two design "
                          "axes ('E x S' is a design idea, not a numeric multiplication).")
RULE_AXES = {
    "E_raw": "mean over all collapsed tissues of log2(1+mean_tpm); HIGHER is better",
    "S_raw": "minus MAD of log2(1+mean_tpm) over DETECTABLE tissues; CROSS-TISSUE stability, NOT time; "
             "HIGHER (less negative) is better = LESS cross-tissue dispersion",
}
RULE_UNDEFINED_S = "undefined S_raw is NEVER filled with 0 and NEVER recalculated/overridden from B."
EXCLUSION_PRIORITY = ["INVALID", "UNDEFINED_S", "BELOW_B", "BELOW_E"]
EXCLUSION_PRIORITY_NOTE = ("each source row is counted in EXACTLY ONE bucket by this fixed priority; a row "
                           "that is BOTH undefined-S and below a filter is counted at its highest priority.")


def _is_finite_number(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)))


def validate_params(min_b, min_e, sort, limit):
    """Domain/finiteness checks for every numeric flag. No implicit/data-derived thresholds."""
    if not _is_finite_number(min_b):
        raise B49InputError("INVALID_MIN_B_NOT_FINITE: %r" % (min_b,))
    min_b = float(min_b)
    if not (0.0 <= min_b <= 1.0):
        raise B49InputError("INVALID_MIN_B_OUT_OF_RANGE_0_1: %r" % min_b)
    if min_e is not None:
        if not _is_finite_number(min_e):
            raise B49InputError("INVALID_MIN_E_NOT_FINITE: %r" % (min_e,))
        min_e = float(min_e)
    if sort not in SORT_MODES:
        raise B49InputError("INVALID_SORT_MODE: %r not in %s" % (sort, list(SORT_MODES)))
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise B49InputError("INVALID_LIMIT_NOT_INTEGER: %r" % (limit,))
    if not (1 <= limit <= MAX_LIMIT):
        raise B49InputError("INVALID_LIMIT_OUT_OF_RANGE_1_%d: %r" % (MAX_LIMIT, limit))
    return min_b, min_e, sort, limit


def freeze_contract(contract, min_b, min_e, sort, limit):
    """The FROZEN pre-registration: exact input SHA + schema + filter + dominance + tie + sort + limit.
    Computed and written BEFORE any candidate value. Deterministic (no data-derived thresholds)."""
    import b49_gene_observed as GO
    n_labels = len(contract["labels"])
    return {
        "batch": __import__("b49_common").BATCH,
        "version": __import__("b49_common").VERSION,
        "phase": "A_FREEZE_BEFORE_CANDIDATE_VALUES",
        "frozen_before_candidate_values": True,
        "inputs": GO.source_sha(),
        "schema": {
            "label_table_columns_used": ["gene_id", "E_raw", "S_raw", "B_raw", "Score_E", "Score_S",
                                         "Score_B", "S_defined_main", "is_in_main_subset"],
            "symbol_map_columns_used": ["gene_id", "gene_name", "gene_type"],
            "n_label_rows": n_labels,
        },
        "filter": {
            "min_b": {"value": min_b, "units": "B_raw fraction in [0,1]", "inclusive": True,
                      "role": "B GATE only", "default": DEFAULT_MIN_B},
            "min_e": {"value": min_e, "units": "E_raw (log2(1+mean_tpm))", "inclusive": True,
                      "role": "optional E_raw cutoff; None = NO cutoff", "default": DEFAULT_MIN_E},
        },
        "eligibility": RULE_ELIGIBILITY,
        "eligibility_requires_S_defined_main": 1,
        "dominance": RULE_DOMINANCE,
        "tie_rule": RULE_TIE,
        "frontier_definition": RULE_FRONTIER,
        "b_gate_rule": RULE_B_GATE,
        "no_weighted_score_rule": RULE_NO_WEIGHTED_SCORE,
        "axes": RULE_AXES,
        "undefined_S_rule": RULE_UNDEFINED_S,
        "exclusion_priority": EXCLUSION_PRIORITY,
        "exclusion_priority_note": EXCLUSION_PRIORITY_NOTE,
        "sort_modes": {
            "expression": "E_raw desc, S_raw desc, canonical gene_id asc",
            "stability": "S_raw desc, E_raw desc, canonical gene_id asc",
            "pareto": "the Pareto frontier, E_raw desc, S_raw desc, canonical gene_id asc",
        },
        "sort_selected": sort,
        "limit": {"value": limit, "affects": "DISPLAY only; the FULL frontier is computed first",
                  "default": DEFAULT_LIMIT, "bounds": [1, MAX_LIMIT]},
        "no_implicit_thresholds_from_candidate_quality": True,
        "catalog_is_cross_tissue_not_293FT": True,
    }


def classify(labels, min_b, min_e):
    """Disjoint classification with the FIXED exclusion priority INVALID > UNDEFINED_S > BELOW_B > BELOW_E.
    Returns (counts, eligible_points, invalid_examples). Never fills/recalcs S."""
    counts = {"full_source": 0, "invalid": 0, "undefined_S": 0, "below_B": 0, "below_E": 0, "eligible": 0}
    eligible = []
    invalid_examples = []
    for gid in sorted(labels):
        r = labels[gid]
        counts["full_source"] += 1
        E, S, B = r["E_raw"], r["S_raw"], r["B_raw"]
        sd = r["S_defined_main"]
        reason = None
        if E is None or not math.isfinite(E):
            reason = "E_RAW_MISSING_OR_NONFINITE"
        elif B is None or not math.isfinite(B):
            reason = "B_RAW_MISSING_OR_NONFINITE"
        elif sd not in (0, 1):
            reason = "S_DEFINED_FLAG_NOT_0_OR_1"
        elif sd == 1 and (S is None or not math.isfinite(S)):
            reason = "S_DEFINED_BUT_S_RAW_MISSING_OR_NONFINITE"
        if reason is not None:
            counts["invalid"] += 1
            if len(invalid_examples) < 50:
                invalid_examples.append({"gene_id": gid, "reason": reason})
            continue
        if sd == 0:
            counts["undefined_S"] += 1
            continue
        if B < min_b:
            counts["below_B"] += 1
            continue
        if min_e is not None and E < min_e:
            counts["below_E"] += 1
            continue
        counts["eligible"] += 1
        eligible.append({"gene_id": gid, "E": float(E), "S": float(S), "B": float(B)})
    return counts, eligible, invalid_examples


def pareto_frontier(points):
    """EXACT Pareto frontier (maximize E and S) in O(n log n) via an E-sorted grouped scan.

    Handles equal E as groups and equal (E,S) as full ties. Dominance: E>=,S>= with >=1 strict.
    Returns frontier rows already ordered E_raw desc, S_raw desc, gene_id asc.
    """
    pts = sorted(points, key=lambda p: (-p["E"], -p["S"], p["gene_id"]))
    frontier = []
    best_s = None
    i, n = 0, len(pts)
    while i < n:
        e_cur = pts[i]["E"]
        j = i
        while j < n and pts[j]["E"] == e_cur:
            j += 1
        group = pts[i:j]
        gmax = max(g["S"] for g in group)
        if best_s is None or gmax > best_s:
            for g in group:
                if g["S"] == gmax:
                    frontier.append(g)
            best_s = gmax if best_s is None else max(best_s, gmax)
        i = j
    return frontier


def _ordered_pool(eligible, frontier, sort):
    if sort == "pareto":
        return list(frontier)
    if sort == "expression":
        return sorted(eligible, key=lambda g: (-g["E"], -g["S"], g["gene_id"]))
    return sorted(eligible, key=lambda g: (-g["S"], -g["E"], g["gene_id"]))


def _row(contract, point, rank, is_frontier):
    gid = point["gene_id"]
    r = contract["labels"][gid]
    sym = contract["id2sym"].get(gid)
    biotype = contract["id2biotype"].get(gid) or "unknown"
    return {
        "rank": rank, "gene_id": gid, "symbol": sym, "biotype": biotype,
        "E_raw": r["E_raw"], "observedScore_E": r["Score_E"],
        "S_raw": r["S_raw"], "observedScore_S": r["Score_S"],
        "B_raw": r["B_raw"], "observedScore_B": r["Score_B"],
        "is_in_main_subset": r["is_in_main_subset"], "S_defined_main": r["S_defined_main"],
        "is_frontier": bool(is_frontier),
        "status": "ELIGIBLE_SELECTED",
        "note": ("catalog-derived observed label; not a model prediction, not 293FT/GFP activity; "
                 "S_raw is cross-tissue stability, not time"),
    }


def _mapping_warnings(contract, gene_ids):
    """Ambiguous symbols are KEPT (canonical gene_id output) and reported as mapping warnings."""
    w = {"n_missing_symbol": 0, "n_ambiguous_symbol": 0, "ambiguous_examples": []}
    for gid in gene_ids:
        sym = contract["id2sym"].get(gid)
        if not sym:
            w["n_missing_symbol"] += 1
            continue
        hits = contract["sym2ids"].get(sym.upper(), [])
        if len(hits) > 1:
            w["n_ambiguous_symbol"] += 1
            if len(w["ambiguous_examples"]) < 10:
                w["ambiguous_examples"].append({"gene_id": gid, "symbol": sym, "n_gene_ids": len(hits)})
    return w


def rank_catalog(contract, min_b=DEFAULT_MIN_B, min_e=DEFAULT_MIN_E, sort=DEFAULT_SORT,
                 limit=DEFAULT_LIMIT):
    """Full deterministic selection + counts. Returns the result dict (no file writes)."""
    contract_block = freeze_contract(contract, min_b, min_e, sort, limit)
    counts, eligible, invalid_examples = classify(contract["labels"], min_b, min_e)
    frontier = pareto_frontier(eligible)
    frontier_ids = set(g["gene_id"] for g in frontier)
    pool = _ordered_pool(eligible, frontier, sort)
    selected = pool[:limit]
    n_truncated = len(pool) - len(selected)

    selected_rows = [_row(contract, p, i + 1, p["gene_id"] in frontier_ids)
                     for i, p in enumerate(selected)]
    all_eligible_rows = []
    for i, p in enumerate(sorted(eligible, key=lambda g: (-g["E"], -g["S"], g["gene_id"]))):
        all_eligible_rows.append(_row(contract, p, i + 1, p["gene_id"] in frontier_ids))

    warnings = [{"level": "warn", "text":
                 "Catalog-derived OBSERVED labels from the frozen cross-tissue table: NOT an individual "
                 "promoter measurement, NOT a model prediction, NOT 293FT/GFP activity, NOT a wet-lab "
                 "recommendation and NOT an autonomous-promoter proof."},
                {"level": "warn", "text":
                 "E_raw and S_raw are the two DESIGN AXES (both higher-better); B_raw is a GATE only. No "
                 "weighted E+S and no product score is computed. Ties are NOT deduplicated."}]
    if min_b < 0.30:
        warnings.append({"level": "info", "text":
                         "min_b < 0.30 cannot expand the eligible universe: eligibility requires "
                         "S_defined_main == 1, and in this catalog S_defined_main == 1 implies "
                         "is_in_main_subset == 1 (B_raw >= 0.30). The S definition still limits eligibility."})
    wmap = _mapping_warnings(contract, [r["gene_id"] for r in selected_rows])
    if wmap["n_ambiguous_symbol"] or wmap["n_missing_symbol"]:
        warnings.append({"level": "warn", "text":
                         "symbol mapping: %d selected row(s) have an AMBIGUOUS symbol (multiple gene_ids) "
                         "and %d have no symbol; canonical gene_ids are kept (no candidate dropped)."
                         % (wmap["n_ambiguous_symbol"], wmap["n_missing_symbol"])})

    return {
        "status": "OK",
        "batch": __import__("b49_common").BATCH,
        "version": __import__("b49_common").VERSION,
        "command": "gene-rank",
        "task": "catalog_tradeoff_selection",
        "note": ("Deterministic catalog tradeoff selection on the frozen label table using the E_raw and "
                 "S_raw design axes; B_raw is a gate only. Catalog-derived observed labels (cross-tissue), "
                 "NOT 293FT activity and NOT a model prediction."),
        "frozen_contract": contract_block,
        "params": {"min_b": min_b, "min_e": min_e, "sort": sort, "limit": limit,
                   "executed_defaults": {"min_b": DEFAULT_MIN_B, "min_e": DEFAULT_MIN_E,
                                         "sort": DEFAULT_SORT, "limit": DEFAULT_LIMIT}},
        "counts": {
            "full_source": counts["full_source"],
            "invalid": counts["invalid"],
            "undefined_S": counts["undefined_S"],
            "below_B": counts["below_B"],
            "below_E": counts["below_E"],
            "eligible": counts["eligible"],
            "selected": len(selected_rows),
            "disjoint_priority": EXCLUSION_PRIORITY,
            "counts_sum_equals_full_source":
                (counts["invalid"] + counts["undefined_S"] + counts["below_B"] + counts["below_E"]
                 + counts["eligible"]) == counts["full_source"],
        },
        "returns": {"n_eligible": len(eligible), "n_frontier": len(frontier),
                    "n_pool": len(pool), "n_selected": len(selected_rows), "n_truncated": n_truncated,
                    "limit_affects_display_only": True,
                    "frontier_computed_before_limit": True,
                    "pool_definition": ("pareto_frontier" if sort == "pareto" else "all_eligible")},
        "selected": selected_rows,
        "mapping_warnings": wmap,
        "invalid_examples": invalid_examples,
        "warnings": warnings,
        "definitions": {
            "axes": RULE_AXES, "eligibility": RULE_ELIGIBILITY, "dominance": RULE_DOMINANCE,
            "tie_rule": RULE_TIE, "frontier": RULE_FRONTIER, "b_gate": RULE_B_GATE,
            "no_weighted_score": RULE_NO_WEIGHTED_SCORE,
            "observedScore_E": "FULL-catalog rank of E_raw (informational; NOT raw units)",
            "observedScore_S": "FULL-catalog rank of S_raw with undefined S at the bottom (informational)",
            "observedScore_B": "FULL-catalog rank of B_raw (informational)",
        },
        "claims": {"is_model_prediction": False, "is_293FT_activity": False, "is_gfp_measurement": False,
                   "has_confidence_interval": False, "is_wetlab_recommendation": False,
                   "is_autonomous_promoter_proof": False},
        "all_eligible_rows": all_eligible_rows,
    }
