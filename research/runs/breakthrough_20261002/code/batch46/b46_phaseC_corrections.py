# -*- coding: utf-8 -*-
"""batch46 Phase C corrections (NEW files only; b45 bytes unchanged):
 (1) repair the HepG2 barcode-spread calculation by ACTUAL cell, show old vs new.
 (2) b45 review-clock caveat with the actual observed reviewer end.
"""
import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T


def pop_sd(v):
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    return math.sqrt(sum((x - m) ** 2 for x in v) / n)


def spread_by_cell(frags, lib, cell):
    sds, n_frag = [], 0
    for f in frags.values():
        if f["lib"] != lib:
            continue
        vals = f["cells"][cell + "_r1"] + f["cells"][cell + "_r2"]
        sd = pop_sd(vals)
        if sd is not None:
            sds.append(sd)
            n_frag += 1
    return {"median_barcode_sd": statistics.median(sds) if sds else None, "n": n_frag}


def main():
    old = json.loads((C.B45 / "secondary_reliability.json").read_text(encoding="utf-8"))
    frags = T.build_fragments()
    repair = {"batch": C.BATCH, "version": C.VERSION, "phase": "C_correction_hepg2_spread",
              "b45_source_bytes": "reports/breakthrough_20261002/batch45/secondary_reliability.json (READ-ONLY, UNCHANGED)",
              "bug": "b45 barcode_spread used K562_barcode_spread_sd unconditionally, so the HepG2 rows duplicated "
                     "the K562 values. This NEW file recomputes HepG2 spread from ACTUAL HepG2 cells.",
              "entries": {}}
    for lib in ("short", "long"):
        oldK = old["barcode_spread"]["%s:K562" % lib]["median_barcode_sd"]
        oldH = old["barcode_spread"]["%s:HepG2" % lib]["median_barcode_sd"]
        newK = spread_by_cell(frags, lib, "K562")
        newH = spread_by_cell(frags, lib, "HepG2")
        repair["entries"]["%s:K562" % lib] = {"old_b45": oldK, "new_recomputed": newK["median_barcode_sd"],
                                              "n": newK["n"], "matches_old": abs(oldK - newK["median_barcode_sd"]) < 1e-12}
        repair["entries"]["%s:HepG2" % lib] = {"old_b45": oldH, "old_equals_K562": abs(oldH - oldK) < 1e-12,
                                               "new_recomputed": newH["median_barcode_sd"], "n": newH["n"]}
    repair["note"] = ("diagnostic field only; b45 all-evidence numbers are unaffected. b45 old bytes retained "
                      "per policy (never rewrite old reports).")
    C.write_json(C.OUT / "correction_hepg2_barcode_spread.json", repair)

    clock = {"batch": C.BATCH, "version": C.VERSION, "phase": "C_correction_b45_reviewclock",
             "target": "reports/breakthrough_20261002/batch45/independent_review.json (READ-ONLY, UNCHANGED)",
             "issue": "b45 independent_review.json 'authored_at' = 2026-10-04T04:20:00+08:00 is LATER than the "
                      "reviewer's actual phase end. Do NOT treat authored_at as an accurate clock.",
             "actual_observed_reviewer_end": "2026-10-04T04:12:08.241327+08:00",
             "actual_observed_source": "runs/breakthrough_20261002/control_53/status.json reviewer.finished_at",
             "reviewer_os_creation": "2026-10-04T04:08:24.464767+08:00",
             "reviewer_session_id": "d432085f-4fbe-4782-8322-a3297ef16435",
             "reviewer_started_at": "2026-10-04T04:08:25.252195+08:00",
             "policy": "original b45 bytes retained; this caveat is a NEW correction file; no rewrite of old reports."}
    C.write_json(C.OUT / "correction_b45_reviewclock.json", clock)
    print(json.dumps(repair["entries"], indent=1))
    print("wrote correction files")


if __name__ == "__main__":
    main()
