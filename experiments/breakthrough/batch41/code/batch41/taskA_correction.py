#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 Task A (<=8 min, BEFORE any scoring): batch41-only CPU repair/correction
addendum for the batch40 OligoMPRA design-gate discrepancies.

Recomputes the step90 window-overlap count from the AUTHOR RAW DESIGN using the
correct family pattern (base name with trailing _<idx>_<strand> stripped), NOT a
hardcoded 8821. Reconciles the structured join gate counters. Preserves/discloses
the old batch40 inconsistency without modifying old bytes. Independently checks 20
raw design / workbook(s5) / joined joins. ZERO model forward if source/raw/target
fails.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402

HDR_RE = re.compile(r"^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):"
                    r"(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$")
FAM_RE = re.compile(r"_\d+_[+-]$")
WORKBOOK_SHA = "bc6e05853091d2d1245c73afb5551eabe8fb5ad4bfba2bb61fa8b5136e62090e"
WORKBOOK = C.ROOT / "data/breakthrough_20261002/batch36/mapt/media-1.xlsx"


def fam_base(name):
    return FAM_RE.sub("", name)


def main():
    t0 = time.time()
    out = {"batch": C.BATCH, "version": C.VERSION,
           "record_type": "batch41_taskA_repair_addendum_for_batch40",
           "generated_at": C.now_local(), "pid": os.getpid()}
    failures = []

    design_sha = C.sha256_file(C.DESIGN_GZ)
    design_bytes = C.DESIGN_GZ.stat().st_size
    wb_sha = C.sha256_file(WORKBOOK)
    src = {
        "design_file": str(C.DESIGN_GZ.relative_to(C.ROOT)),
        "design_sha256": design_sha, "design_sha256_expected": C.DESIGN_SHA,
        "design_sha_ok": design_sha == C.DESIGN_SHA,
        "design_bytes": design_bytes, "design_bytes_expected": C.DESIGN_BYTES,
        "design_bytes_ok": design_bytes == C.DESIGN_BYTES,
        "workbook_file": str(WORKBOOK.relative_to(C.ROOT)),
        "workbook_sha256": wb_sha, "workbook_sha256_expected": WORKBOOK_SHA,
        "workbook_sha_ok": wb_sha == WORKBOOK_SHA,
    }
    out["source_identity"] = src
    if not (src["design_sha_ok"] and src["design_bytes_ok"] and src["workbook_sha_ok"]):
        failures.append("SOURCE_IDENTITY")

    recs = C.load_design_records()
    design = dict(recs)
    coord, noncoord = [], []
    for h, s in recs:
        (coord if HDR_RE.match(h) else noncoord).append((h, s))
    out["raw_design_census"] = {
        "records": len(recs), "coordinate_oligos": len(coord),
        "non_coordinate_control_like": len(noncoord),
        "noncoord_headers_sample": [h for h, _ in noncoord[:3]],
        "all_300bp": all(len(s) == 300 for _, s in coord),
        "all_span270": all(int(HDR_RE.match(h).group("end")) -
                           int(HDR_RE.match(h).group("start")) == 270 for h, _ in coord),
        "adapter_fwd_all": all(s[:15] == C.FWD15 for _, s in coord),
        "adapter_rev_all": all(s[-15:] == C.REV15 for _, s in coord),
    }
    rc = out["raw_design_census"]
    if not (rc["all_300bp"] and rc["all_span270"] and rc["adapter_fwd_all"] and rc["adapter_rev_all"]):
        failures.append("RAW_DESIGN_STRUCTURE")

    fam = defaultdict(lambda: defaultdict(list))
    for h, s in coord:
        m = HDR_RE.match(h)
        base = fam_base(m.group("name"))
        fam[(base, m.group("chr"))][int(m.group("start"))].append(h)
    pairs_checked = pairs_consistent = pairs_inconsistent = 0
    inconsistent_examples = []
    for key, starts in fam.items():
        for s, hdrs in starts.items():
            if s + 90 in starts:
                for ha in hdrs:
                    for hb in starts[s + 90]:
                        pairs_checked += 1
                        ins_a = design[ha][15:285]
                        ins_b = design[hb][15:285]
                        if ins_a[90:270] == ins_b[0:180]:
                            pairs_consistent += 1
                        else:
                            pairs_inconsistent += 1
                            if len(inconsistent_examples) < 3:
                                inconsistent_examples.append([ha, hb])
    out["step90_overlap_recompute"] = {
        "definition": ("family=(base,chr) where base=header name minus trailing _idx_strand; "
                       "ordered pairs with window start difference == 90; property "
                       "insert_k[90:270]==insert_k+1[0:180] on the 270bp insert raw[15:285]"),
        "families": len(fam),
        "coordinate_records": len(coord),
        "pairs_checked": pairs_checked,
        "pairs_consistent": pairs_consistent,
        "pairs_inconsistent": pairs_inconsistent,
        "consistency_fraction": (pairs_consistent / pairs_checked) if pairs_checked else None,
        "reviewer_expected": "8821/8821 (NOT hardcoded; recomputed from raw design)",
        "inconsistent_examples": inconsistent_examples,
        "hardcoded": False,
    }

    rows = C.read_joined()
    bio = [r for r in rows if r["control"] == "0"]
    ctrl = [r for r in rows if r["control"] == "1"]
    n_bio, n_ctrl = len(bio), len(ctrl)
    bio_matched = sum(1 for r in bio if r["design_match"] == "exact_unique")
    ctrl_matched = sum(1 for r in ctrl if r["design_match"] == "exact_unique")
    design_only_bio = len(coord) - n_bio
    design_only_ctrl = len(noncoord) - n_ctrl
    gate = {
        "biological_rows": n_bio, "controls": n_ctrl,
        "bio_matched": bio_matched,
        "bio_coverage_fraction": (bio_matched / n_bio) if n_bio else None,
        "unmatched_bio": n_bio - bio_matched,
        "ctrl_matched": ctrl_matched,
        "ctrl_coverage_fraction": (ctrl_matched / n_ctrl) if n_ctrl else None,
        "design_only_coordinate_biological": design_only_bio,
        "design_only_scrambled_controls": design_only_ctrl,
        "design_only_total": design_only_bio + design_only_ctrl,
        "corrected_targets": {"bio": "7537/7537=1.0", "unmatched": 0, "controls": "85/85",
                              "design_only": "1407 bio + 15 controls = 1422"},
    }
    gate["reconcile_ok"] = bool(n_bio == 7537 and n_ctrl == 85 and bio_matched == n_bio and
                                ctrl_matched == n_ctrl and design_only_bio == 1407 and
                                design_only_ctrl == 15)
    out["reconciled_gate"] = gate
    if not gate["reconcile_ok"]:
        failures.append("GATE_RECONCILE")

    old_gate = json.load(open(C.ROOT / "reports/breakthrough_20261002/batch40/readiness_gate.json",
                              encoding="utf-8"))
    out["preserved_batch40_inconsistency"] = {
        "batch40_saved_window_overlap_pairs": old_gate["measured_dna"]["window_overlap_pairs"],
        "batch40_saved_window_overlap_consistent": old_gate["measured_dna"]["window_overlap_consistent"],
        "batch40_report_string": "8821/8821 (hardcoded in report_writer.py / version_registry_entry)",
        "batch40_join_coverage_fraction": old_gate["join_coverage_fraction"],
        "batch40_unmatched_real_rows_retained": old_gate["unmatched_real_rows_retained"],
        "disclosure": ("batch40 saved 0 overlap pairs yet printed 8821/8821; coverage was reported "
                       "as 7622/7537 (1.0113>1) and unmatched as -85. Preserved exactly as-is here "
                       "(no old bytes rewritten). The corrected independent recomputation is above."),
        "old_bytes_modified": False,
    }

    s5 = json.load(open(C.S5_RAW, encoding="utf-8"))
    s5_by_label = {r["A"]: r for r in s5}

    def pick(pred, k):
        got = [r for r in bio if pred(r)]
        got.sort(key=lambda r: r["label"])
        return got[:k]

    sel = []
    sel += pick(lambda r: r["chr"] == "chr17", 6)
    others = [c for c in sorted({r["chr"] for r in bio} - {"chr17"})]
    for c in others[:6]:
        sel += pick(lambda r, c=c: r["chr"] == c, 1)
    sel += pick(lambda r: fam_base(HDR_RE.match(r["label"]).group("name")) == "mapt_gap", 4)
    sel += pick(lambda r: r["chr"] != "chr17" and
                fam_base(HDR_RE.match(r["label"]).group("name")) != "mapt_gap", 4)
    seen = set()
    sel20 = []
    for r in sel:
        if r["label"] not in seen:
            seen.add(r["label"])
            sel20.append(r)
    sel20 = sel20[:20]

    checks = []
    for r in sel20:
        lab = r["label"]
        m = HDR_RE.match(lab)
        raw = design.get(lab)
        wb = s5_by_label.get(lab)
        c = {
            "label": lab,
            "parsed_header": bool(m),
            "design_header_present": raw is not None,
            "design_coord_equals_joined": bool(m and raw is not None and
                (m.group("chr"), int(m.group("start")), int(m.group("end")), m.group("strand")) ==
                (r["chr"], int(r["start"]), int(r["end"]), r["strand"])),
            "raw_len_300": bool(raw and len(raw) == 300),
            "raw_fwd15_ok": bool(raw and raw[:15] == C.FWD15),
            "raw_rev15_ok": bool(raw and raw[-15:] == C.REV15),
            "insert_boundary_ok": bool(raw and raw[15:285] == r["insert_270bp"]),
            "insert_len_270": len(r["insert_270bp"]) == 270,
            "wb_row_present": wb is not None,
            "wb_A_parses_same_header": bool(wb and wb["A"] == lab),
            "wb_BCD_equals_joined": bool(wb and (wb["B"], int(wb["C"]), int(wb["D"])) ==
                                        (r["chr"], int(r["start"]), int(r["end"]))),
            "wb_statistic_equals_joined_alpha": bool(wb and
                abs(float(wb["E"]) - float(r["statistic_alpha"])) < 1e-12),
        }
        c["all_ok"] = all(v for k, v in c.items() if isinstance(v, bool))
        checks.append(c)
    n_join_ok = sum(1 for c in checks if c["all_ok"])
    out["source20_join_check"] = {
        "n_checked": len(checks), "n_all_ok": n_join_ok,
        "all_ok": (n_join_ok == len(checks) and len(checks) == 20),
        "no_guessed_reference_dna": True,
        "no_adapter_stripped_from_other_assay": True,
        "adapter_unit": "raw[:15]=FWD15 / raw[-15:]=REV15 / insert=raw[15:285] (author design file)",
        "checks": checks,
    }
    if n_join_ok != len(checks) or len(checks) != 20:
        failures.append("SOURCE20_JOIN")

    out["strand_and_unknown_context"] = {
        "design_strands": dict(Counter(r["strand"] for r in bio)),
        "physical_assay_orientation": "UNKNOWN (design file is + only; single- vs double-strand unresolved)",
        "minimal_promoter": "UNKNOWN (not in design file, workbook, or author call chain)",
        "vector_delivery_integration": "UNKNOWN (not in call chain)",
        "autonomy_vs_fixed_vector": "UNKNOWN (do not infer from fixed vector)",
        "assay_strand_carrythrough": "UNKNOWN",
        "target_definition": ("MPRAnalyze getAlpha activity (uq depth factors; 3 DNA + 3 RNA aggregate); "
                              "upstream DNA>=50 author ascertainment; p-values/z-scores NOT primary truth"),
        "public_cell_name": "HEK293FT (public name only; differs from local 293FT / D28)",
    }
    out["failures"] = failures
    out["forward_allowed"] = (len(failures) == 0)
    out["elapsed_s"] = round(time.time() - t0, 3)

    C.atomic_write_json(C.DATA / "taskA_correction_addendum.json", out)
    if not out["forward_allowed"]:
        C.atomic_write_json(C.DATA / "blocked_finite_proposal.json", {
            "record_type": "finite_proposal_blocked",
            "reason": "Task A source/raw/target check failed", "failures": failures,
            "proposal": ("STOP forward; repair only the failing link in a new batch41 addendum, "
                         "then re-run Task A. No model scoring without valid frozen input.")})
    print(json.dumps({"forward_allowed": out["forward_allowed"], "failures": failures,
                      "step90": out["step90_overlap_recompute"],
                      "gate": out["reconciled_gate"]["reconcile_ok"],
                      "source20_all_ok": out["source20_join_check"]["all_ok"],
                      "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))
    return 0 if out["forward_allowed"] else 1


if __name__ == "__main__":
    sys.exit(main())
