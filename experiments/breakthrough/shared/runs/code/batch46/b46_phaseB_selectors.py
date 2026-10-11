# -*- coding: utf-8 -*-
"""batch46 Phase B selectors: fixed-selection utility within matched triplets.
Deterministic tie rule: lexicographically smallest info id. No pvalue/CI/bootstrap."""
import sys, json, math, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T
from b46_phaseB_analysis import per_triplet_records, LINEAGES

CELLS = ["K562", "HepG2"]


def pick(cands, keysel):
    """keysel(label)->value ; returns chosen label or None. tie -> smallest info id."""
    best, bestv = None, None
    for s in C.SELS:
        v = keysel(s)
        if v is None or not math.isfinite(v):
            continue
        if best is None or v > bestv or (v == bestv and cands[s]["info"] < cands[best]["info"]):
            best, bestv = s, v
    return best


def eval_selector(triplets, chosen_label_fn, cell):
    rows = []
    for t in triplets:
        c = t["cands"]
        chosen = chosen_label_fn(c)
        meas = {s: c[s][cell + "_meas"] for s in C.SELS}
        vals = [meas[s] for s in C.SELS]
        if chosen is None or any(v is None or not math.isfinite(v) for v in vals):
            rows.append({"triplet_id": t["triplet_id"], "usable": False,
                         "reason": "nonfinite_or_missing_measured"})
            continue
        others = [meas[s] for s in C.SELS if s != chosen]
        best_other = max(others)
        oracle = max(vals)
        uniform = sum(vals) / 3.0
        rows.append({"triplet_id": t["triplet_id"], "library": t["library"],
                     "lineage": t["lineage"], "generation": t["generation"], "usable": True,
                     "chosen_label": chosen, "chosen_info": c[chosen]["info"],
                     "chosen_meas": meas[chosen], "uniform_mean": uniform, "oracle_best": oracle,
                     "utility": meas[chosen] - uniform, "regret": meas[chosen] - oracle,
                     "hit_strict": int(meas[chosen] > best_other),
                     "hit_tie_inclusive": int(meas[chosen] >= best_other),
                     "n_tied_at_top": sum(1 for v in vals if v == oracle)})
    return rows


def agg(rows):
    usable = [r for r in rows if r["usable"]]
    def mean(v):
        return (sum(v) / len(v)) if v else None
    return {"n_triplets": len(rows), "n_usable": len(usable),
            "n_skipped": len(rows) - len(usable),
            "mean_utility_vs_uniform": mean([r["utility"] for r in usable]),
            "mean_regret_vs_oracle": mean([r["regret"] for r in usable]),
            "hit_rate_strict": mean([r["hit_strict"] for r in usable]),
            "hit_rate_tie_inclusive": mean([r["hit_tie_inclusive"] for r in usable]),
            "utility_sign_counts": {"positive": sum(1 for r in usable if r["utility"] > 0),
                                    "zero": sum(1 for r in usable if r["utility"] == 0),
                                    "negative": sum(1 for r in usable if r["utility"] < 0)},
            "regret_zero_count": sum(1 for r in usable if r["regret"] == 0)}


def main():
    frags = T.build_fragments()
    freeze = T.load_freezer()
    triplets = T.build_triplets(frags, freeze)

    def score_K562(c):
        return pick(c, lambda s: c[s]["pred_K562"])

    def score_HepG2(c):
        return pick(c, lambda s: c[s]["pred_HepG2"])

    def gc_high(c):
        return pick(c, lambda s: c[s]["GC"])

    selectors = {"K562_score": score_K562, "GC_high": gc_high, "HepG2_score": score_HepG2}
    out = {"batch": C.BATCH, "version": C.VERSION, "phase": "B_selector_utility",
           "definitions": {
               "chosen": "argmax of the named selector value among the 3 frozen candidates; "
                         "tie -> lexicographically smallest info id (frozen).",
               "uniform_mean": "mean of the three candidates' measured summaries = expected utility "
                               "of a UNIFORM RANDOM CHOICE AMONG THESE AUTHOR-SELECTED three (NOT arbitrary random sequences).",
               "utility": "chosen measured summary - uniform_mean.",
               "regret": "chosen measured summary - oracle_best (hindsight only; NOT a deployable upper guarantee).",
               "hit_strict": "chosen measured strictly greater than both others.",
               "hit_tie_inclusive": "chosen measured equals the maximum of the three (ties counted as hits).",
               "tie_rule": "lexicographically smallest info id on selector ties; measured ties handled explicitly above.",
               "no_inference": "about 10 related search lineages; no pvalue/CI/bootstrap computed."},
           "primary_K562": {}, "secondary_HepG2": {}}
    for lib in ("short", "long"):
        tl = [t for t in triplets if t["library"] == lib]
        for selname, fn in selectors.items():
            rows = eval_selector(tl, fn, "K562")
            out["primary_K562"].setdefault(selname, {})[lib] = agg(rows)
        # HepG2 secondary: SAME K562-selected candidate vs same uniform/GC choices
        rows_k = eval_selector(tl, score_K562, "HepG2")
        out["secondary_HepG2"].setdefault("K562_score_choice_eval_on_HepG2", {})[lib] = agg(rows_k)
        rows_g = eval_selector(tl, gc_high, "HepG2")
        out["secondary_HepG2"].setdefault("GC_high_choice_eval_on_HepG2", {})[lib] = agg(rows_g)
        rows_h = eval_selector(tl, score_HepG2, "HepG2")
        out["secondary_HepG2"].setdefault("HepG2_score_choice_eval_on_HepG2", {})[lib] = agg(rows_h)
    # selector-level detail tables (all rows)
    detail = {}
    for lib in ("short", "long"):
        tl = [t for t in triplets if t["library"] == lib]
        detail[lib] = {
            "K562_score_on_K562": eval_selector(tl, score_K562, "K562"),
            "GC_high_on_K562": eval_selector(tl, gc_high, "K562"),
            "K562_score_on_HepG2": eval_selector(tl, score_K562, "HepG2"),
            "GC_high_on_HepG2": eval_selector(tl, gc_high, "HepG2"),
            "HepG2_score_on_HepG2": eval_selector(tl, score_HepG2, "HepG2")}
    C.write_json(C.OUT / "selector_utility.json", out)
    C.write_json(C.OUT / "selector_detail.json", {"batch": C.BATCH, "version": C.VERSION, "rows": detail})
    for lib in ("short", "long"):
        p = out["primary_K562"]["K562_score"][lib]
        print(lib, "K562score on K562: n", p["n_usable"], "util", p["mean_utility_vs_uniform"],
              "hit_tie", p["hit_rate_tie_inclusive"], "hit_strict", p["hit_rate_strict"],
              "signs", p["utility_sign_counts"])
        g = out["primary_K562"]["GC_high"][lib]
        print(lib, "  GC_high on K562:", "util", g["mean_utility_vs_uniform"], "hit_tie", g["hit_rate_tie_inclusive"])
        s = out["secondary_HepG2"]["K562_score_choice_eval_on_HepG2"][lib]
        print(lib, "  K562score on HepG2:", "util", s["mean_utility_vs_uniform"], "hit_tie", s["hit_rate_tie_inclusive"])


if __name__ == "__main__":
    main()
