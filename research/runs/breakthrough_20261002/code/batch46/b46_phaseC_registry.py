# -*- coding: utf-8 -*-
"""batch46 Phase C: atomic append of (a) missing b45 entry marked priorcomplianceFAIL and
(b) new b46 entry, to version_registry.json + CHANGELOG.txt. Preserves production pointer
and EVERY existing entry; never rewrites old reports."""
import sys, json, os, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C

REG = C.ROOT / "reports/breakthrough_20261002/version_registry.json"
CHG = C.ROOT / "reports/breakthrough_20261002/CHANGELOG.txt"
NOW = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def atomic_write_text(path, text):
    tmp = Path(str(path) + ".b46tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert "production_pointer" in reg and reg["production_pointer"]["model"] == "promoter_atlas_v6.3"
    before_keys = set(reg.keys())
    before_n_appends = len(reg.get("batch_appends", []))
    existing_ids = {a.get("batch") for a in reg.get("batch_appends", [])}
    assert "batch45" not in existing_ids and "batch46" not in existing_ids
    prim = json.loads((C.OUT / "primary_contrasts.json").read_text(encoding="utf-8"))
    sel = json.loads((C.OUT / "selector_utility.json").read_text(encoding="utf-8"))
    b45_entry = {
        "record_type": "retroactive_registry_entry_priorcomplianceFAIL",
        "batch": "batch45", "version": "v6.32.0-synthetic-design-utility.rc1",
        "status": "priorcomplianceFAIL",
        "reason_missing_at_the_time": ("batch45 worker did NOT append to registry/CHANGELOG during its worker phase, "
                                       "although Protocol D required it and b41-b44 appended. Recorded here retroactively "
                                       "for completeness; NOT a re-endorsement."),
        "known_defects": [
            "HepG2 barcode-spread entries duplicated K562 values (diagnostic field).",
            "Independent review authored_at 2026-10-04T04:20:00+08:00 is later than the reviewer actual phase end "
            "2026-10-04T04:12:08.241327+08:00 (clock defect)."],
        "reproducible_evidence": {"short": {"G": 48, "dK562_pred_med": 5.18966233730316,
                                            "dK562_meas_med": 1.8146045162370514},
                                  "long": {"G": 44, "dK562_pred_med": 7.643129444122325,
                                           "dK562_meas_med": 3.4535746217526135}},
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "appended_by": "batch46 worker (retroactive completion of a required append)",
        "appended_at": NOW,
        "note": "original b45 report/review bytes retained; corrections live in reports/breakthrough_20261002/batch46/."}
    b46_entry = {
        "record_type": "within_generation_selection_descriptive",
        "batch": "batch46", "version": C.VERSION,
        "kind": ("retrospective WITHIN-GENERATION selection utility on the author synthetic GA study; "
                 "ZERO fit/solve/regression/calibration/newscore/neural/GPU/web/download; cached source only"),
        "appended_by": "batch46 worker (DeepSeek Flash implementer; parent-authorized append-only)",
        "appended_at": NOW,
        "production_pointer_unchanged": "v6.3 main / v6.4 cellline pilot",
        "source": {"file": "data/breakthrough_20261002/batch44/syn.txt.gz",
                   "sha256_compressed": C.SRC_SHA256_COMPRESSED},
        "design": "244 matched triplets (short 129 / long 115), same library+lineage+generation, one DNA each for "
                  "{max,median,min}; 16 incomplete groups retained.",
        "primary_K562": {
            "all_rows_median_dmm_short": prim["per_library"]["short"]["all_rows_dmm"]["median"],
            "all_rows_median_dmm_long": prim["per_library"]["long"]["all_rows_dmm"]["median"],
            "all_rows_median_dmn_short": prim["per_library"]["short"]["all_rows_dmn"]["median"],
            "all_rows_median_dmn_long": prim["per_library"]["long"]["all_rows_dmn"]["median"],
            "K562_score_selector_mean_utility_short": sel["primary_K562"]["K562_score"]["short"]["mean_utility_vs_uniform"],
            "K562_score_selector_mean_utility_long": sel["primary_K562"]["K562_score"]["long"]["mean_utility_vs_uniform"],
            "K562_score_hit_rate_short": sel["primary_K562"]["K562_score"]["short"]["hit_rate_tie_inclusive"],
            "K562_score_hit_rate_long": sel["primary_K562"]["K562_score"]["long"]["hit_rate_tie_inclusive"],
            "GC_high_selector_mean_utility_short": sel["primary_K562"]["GC_high"]["short"]["mean_utility_vs_uniform"],
            "GC_high_selector_mean_utility_long": sel["primary_K562"]["GC_high"]["long"]["mean_utility_vs_uniform"]},
        "score_ordering_conflicts": prim["score_ordering_check"]["n_conflict"],
        "nonfinite_case": "1 barcode dropout (0 raw RNA) at long:rep01:gen0 _median; primary LIMITED + extended-real direction.",
        "decision": "DESCRIPTIVE_POSITIVE_WITHIN_GENERATION (source-study, retrospective, NOT our model gain)",
        "preserved": ["local 293FT D28 FAIL", "public HEK fragment transfer weak (b41 rho 0.108)",
                      "2000bp unsupported", "batch44 source branch CLOSED"],
        "next": "fresh independent review then parent decision"}
    reg["batch_appends"] = reg.get("batch_appends", []) + [b45_entry, b46_entry]
    reg["last_appended_at"] = NOW
    reg["last_appended_by"] = "batch46 worker"
    assert set(reg.keys()) >= before_keys
    assert len(reg["batch_appends"]) == before_n_appends + 2
    proto = json.loads(REG.read_text(encoding="utf-8"))["production_pointer"]
    assert reg["production_pointer"] == proto
    atomic_write_text(REG, json.dumps(reg, ensure_ascii=False, indent=2))

    lines = [
        "[%s] batch45 v6.32.0-synthetic-design-utility.rc1 -> RETROACTIVE REGISTRY ENTRY marked priorcomplianceFAIL: "
        "batch45 did not append at its worker phase (required by Protocol D); HepG2 barcode_spread duplicated K562 "
        "(diagnostic); independent-review authored_at 04:20:00 later than actual reviewer end 04:12:08.241327. "
        "Author K562 design gains reproducible (short dMeas 1.8146 / long dMeas 3.4536 median, 10-0). Original bytes retained." % NOW,
        "[%s] batch46 %s -> WITHIN-GENERATION selection utility on the SAME author synthetic GA study (ZERO fit/GPU/"
        "new-score): 244 matched triplets (short129/long115) same library+lineage+generation. K562-score fixed selector "
        "mean utility vs uniform-among-these-three short %.4f / long %.4f; hit-rate %.3f / %.3f; GC-high comparator "
        "short %.4f / long %.4f. 12/244 score-order conflicts flagged (labels not repaired). 1 zero-RNA dropout -> "
        "primary LIMITED + extended-real direction. DESCRIPTIVE_POSITIVE (source-study, retrospective, NOT our model). "
        "Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved." % (
            NOW, C.VERSION,
            sel["primary_K562"]["K562_score"]["short"]["mean_utility_vs_uniform"],
            sel["primary_K562"]["K562_score"]["long"]["mean_utility_vs_uniform"],
            sel["primary_K562"]["K562_score"]["short"]["hit_rate_tie_inclusive"],
            sel["primary_K562"]["K562_score"]["long"]["hit_rate_tie_inclusive"],
            sel["primary_K562"]["GC_high"]["short"]["mean_utility_vs_uniform"],
            sel["primary_K562"]["GC_high"]["long"]["mean_utility_vs_uniform"])]
    old = CHG.read_text(encoding="utf-8")
    if not old.endswith("\n"):
        old += "\n"
    atomic_write_text(CHG, old + "\n" + "\n".join(lines) + "\n")
    print("registry appends now", len(reg["batch_appends"]),
          "production", reg["production_pointer"]["model"])
    print("changelog appended", len(lines), "lines")


if __name__ == "__main__":
    main()
