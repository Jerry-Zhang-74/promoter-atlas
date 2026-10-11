#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: LOCAL SEQUENCE / ENDPOINT RECONCILIATION (batch23).

Corrects the batch22 factual error (records hold no DNA sequence for the 14 local
constructs) using the REAL project records that pre-date batch22.
Sequence availability and assay-mapping certainty are SEPARATE axes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (  # noqa: E402
    ROOT, DATA_OUT, REPORT_OUT, CONSTRUCT_SEQ_TSV, CONSTRUCT_SEQ_FASTA,
    CONSTRUCT_PROV, ENDPOINTS_D28, CV_PREDS, EXCLUDED_81, BATCH04_TABLE,
    BATCH21_AUDIT_TSV, BATCH22_HARDCODED_LENGTHS,
    now_local, sha256_file, sha256_text, atomic_write_json, set_thread_limits)


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_fasta(path):
    recs = {}
    name = None
    buf = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(buf)
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        recs[name] = "".join(buf)
    return recs


def mapping_certainty(mapping_evidence, barcode_verified):
    me = mapping_evidence or ""
    base = "barcode_verified" if barcode_verified == "1" else "barcode_unverified"
    if "user_clarified" in me:
        qual = "user_clarified_feature"
    elif me.startswith("filename_and_feature_length"):
        qual = "filename_and_feature_length"
    elif me.startswith("inferred_from_gene_and_feature_length"):
        qual = "inferred_from_gene_and_feature_length"
    elif me.startswith("inferred_from_name_and_combined_features"):
        qual = "inferred_from_name_and_combined_features"
    else:
        qual = me or "unspecified"
    return "%s;%s" % (base, qual)


def main():
    set_thread_limits(4)
    t0_start = now_local()
    seq_rows = read_tsv(CONSTRUCT_SEQ_TSV)
    fasta = read_fasta(CONSTRUCT_SEQ_FASTA)
    prov = json.load(open(CONSTRUCT_PROV, encoding="utf-8"))
    seq_by_id = {r["construct_id"]: r for r in seq_rows}
    endpoints = {r["construct_id"]: r for r in read_tsv(ENDPOINTS_D28)}
    b04 = {r["construct_id"]: r for r in read_tsv(BATCH04_TABLE)}
    b21 = {r["construct_id"]: r for r in read_tsv(BATCH21_AUDIT_TSV)}
    cv = read_tsv(CV_PREDS)
    prov_src = {s["path"]: s for s in prov["sources"]}

    recon = []
    for idx, r in enumerate(seq_rows):
        cid = r["construct_id"]
        seq = r["sequence"]
        seq_sha_calc = sha256_text(seq)
        seq_sha_rec = r["sequence_sha256"]
        fa = fasta.get(cid)
        fasta_ok = (fa is not None and fa.upper() == seq.upper())
        ep = endpoints.get(cid)
        b21r = b21.get(cid)
        b04r = b04.get(cid)
        prov_rec = prov_src.get(r["source_path"])
        b22_len = BATCH22_HARDCODED_LENGTHS[idx] if idx < len(BATCH22_HARDCODED_LENGTHS) else ""
        rec = {
            "construct_id": cid,
            "family": r["family"],
            "documented_length_bp": r["documented_length_bp"],
            "sequence_length_bp": int(r["sequence_length_bp"]),
            "sequence_available": 1,
            "sequence_sha256_recorded": seq_sha_rec,
            "sequence_sha256_computed": seq_sha_calc,
            "sequence_sha_match": int(seq_sha_calc == seq_sha_rec),
            "fasta_record_present": int(fa is not None),
            "fasta_sequence_match": int(bool(fasta_ok)),
            "sequence_field_length": len(seq),
            "sequence_alphabet_ok": int(set(seq.upper()) <= set("ACGT")),
            "sequence_source_path": r["source_path"],
            "sequence_source_sha256_recorded": r["source_sha256"],
            "sequence_source_sha256_in_provenance": (prov_rec or {}).get("sha256", ""),
            "source_molecule_length_bp": r["source_molecule_length_bp"],
            "provenance_molecule_length_bp": (prov_rec or {}).get("molecule_length_bp", ""),
            "feature_names": r["feature_names"],
            "feature_ranges_1based_inclusive": r["feature_ranges_1based_inclusive"],
            "feature_directionality": r["feature_directionality"],
            "sequence_status": r["sequence_status"],
            "mapping_evidence": r["mapping_evidence"],
            "assay_to_plasmid_barcode_verified": r["assay_to_plasmid_barcode_verified"],
            "assay_mapping_certainty": mapping_certainty(r["mapping_evidence"],
                                                         r["assay_to_plasmid_barcode_verified"]),
            "assay_mapping_is_verified": 0,
            "endpoint_row_present": int(ep is not None),
            "endpoint_target_value_log10": (ep or {}).get("target_value", ""),
            "endpoint_source": (ep or {}).get("source_id", ""),
            "endpoint_cell_line": (ep or {}).get("cell_line_reported", ""),
            "endpoint_assay_context": (ep or {}).get("use_scope", ""),
            "endpoint_replicate_kind": (ep or {}).get("repeat_interpretation", ""),
            "endpoint_eligible": (ep or {}).get("endpoint_eligible", ""),
            "in_batch21_audit": int(b21r is not None),
            "batch21_sequence_sha256": (b21r or {}).get("sequence_sha256", ""),
            "batch21_sha_matches": int(bool(b21r) and b21r.get("sequence_sha256") == seq_sha_rec),
            "batch21_length": (b21r or {}).get("sequence_length_bp", ""),
            "batch04_sequence_length_bp": (b04r or {}).get("sequence_length_bp", ""),
            "batch04_sequence_sha256": (b04r or {}).get("sequence_sha256", ""),
            "batch04_sha_matches": int(bool(b04r) and b04r.get("sequence_sha256") == seq_sha_rec),
            "batch22_hardcoded_length_bp": b22_len,
            "batch22_length_correct": int(b22_len == int(r["sequence_length_bp"])),
        }
        recon.append(rec)

    cv_by_cid = {}
    for row in cv:
        if row["protocol"] == "construct_LOO":
            cv_by_cid[row["construct_id"]] = row
    cv_join = []
    for rec in recon:
        cid = rec["construct_id"]
        row = cv_by_cid.get(cid)
        cv_join.append({
            "construct_id": cid,
            "cv_prediction_present": int(row is not None),
            "cv_sequence_sha256": (row or {}).get("sequence_sha256", ""),
            "cv_sha_matches_tsv": int(bool(row) and row.get("sequence_sha256") == rec["sequence_sha256_recorded"]),
            "cv_target_value": (row or {}).get("target_value", ""),
            "cv_target_matches_endpoint": int(bool(row) and bool(rec["endpoint_target_value_log10"]) and
                abs(float(row["target_value"]) - float(rec["endpoint_target_value_log10"])) < 1e-9),
        })

    exc = read_tsv(EXCLUDED_81)
    excluded = []
    for row in exc:
        excluded.append({
            "construct_id": row["construct_id"], "family": row["family"],
            "documented_length_bp": row["documented_length_bp"],
            "sequence_length_bp": row["sequence_length_bp"],
            "sequence_available": 1,
            "exclusion_reason": "user_confirmed_not_transfected (%s)" % row["sequence_status"],
            "sequence_sha256": row["sequence_sha256"],
            "why_not_joined": "no assay endpoint exists (untransfected); not a missing-sequence case",
        })
    a6_note = {
        "construct_id": "A6", "family": "ACTB", "documented_length_bp": "600",
        "sequence_available": 0,
        "why_not_joined": "A6 appears only as an acquisition metadata annotation "
                          "(observations_raw.tsv O013037, sample_alias_rule); provenance.json states no "
                          "explicit 600 bp A6 donor was identified and no A6 sequence was synthesized from "
                          "another construct; no A6 row in construct_endpoints_D28.tsv",
        "evidence": "reports/cellline_v2_20261001/construct_sequence_provenance.json limitations; "
                    "reports/cellline_v2_20261001/observations_raw.tsv; construct_endpoints_D28.tsv",
    }

    files_for_correction = [CONSTRUCT_SEQ_TSV, CONSTRUCT_SEQ_FASTA, CONSTRUCT_PROV,
                            ENDPOINTS_D28, BATCH04_TABLE, BATCH21_AUDIT_TSV]
    mt = {}
    for p in files_for_correction:
        st = os.stat(p)
        mt[str(p.relative_to(ROOT))] = {
            "sha256": sha256_file(p),
            "mtime_local": datetime.datetime.fromtimestamp(st.st_mtime).astimezone()
                             .strftime("%Y-%m-%dT%H:%M:%S%z"),
            "size": st.st_size,
        }
    b22_start = "2026-10-03T15:06:40+0800"
    all_predate = all(v["mtime_local"] < b22_start for v in mt.values())

    correction = {
        "generated_at_local": now_local(),
        "batch": "batch23",
        "version": "v6.17.1-sequence-evidence.rc1",
        "corrects": "reports/breakthrough_20261002/batch22 (v6.17.0-hektransfer.rc1)",
        "original_batch22_claim": {
            "text": "Project records hold no DNA sequence for the 14 ACTB/GAPDH/CMV constructs.",
            "where": [
                "reports/breakthrough_20261002/batch22/report.md section 8",
                "reports/breakthrough_20261002/batch22/figures_manifest.json (fig3 reason)",
                "reports/breakthrough_20261002/batch22/manifests.json (fig3 reason)",
                "reports/breakthrough_20261002/batch22/fig3_data.tsv rows C1..C14",
                "reports/breakthrough_20261002/batch22/independent_review.json (hek293t_suitability)",
                "runs/breakthrough_20261002/code/batch22/b22_report.py line 148 "
                "(n_constructs_with_sequence_in_records=0)",
            ],
            "preserved": True,
        },
        "corrected_fact": "Full native insert sequences for all 14 constructs ARE present in project records "
                          "and have been there since 2026-10-01, before batch21 and long before batch22 "
                          "started (2026-10-03T15:06:40+0800).",
        "authoritative_records": [
            "reports/cellline_v2_20261001/construct_sequences.tsv (construct_id, sequence, sequence_sha256)",
            "reports/cellline_v2_20261001/construct_sequences.fasta (14 records)",
            "reports/cellline_v2_20261001/construct_sequence_provenance.json (source .dna sha256, features)",
            "data/breakthrough_20261002/batch21/construct_audit.tsv (sequence_sha256 + sequence_source_sha256)",
            "reports/breakthrough_20261002/batch04/cellline_14construct_diagnostic_table.tsv (sequence_sha256)",
        ],
        "batch22_start_local": b22_start,
        "source_files_all_predate_batch22_start": bool(all_predate),
        "source_file_evidence": mt,
        "how_batch22_concluded_wrongly": [
            "batch22 fig3 used a HARDCODED length list "
            "CONSTRUCT_LENGTHS=[100,200,300,400,100,200,300,400,100,200,300,400,700,700] "
            "(b22_report.py line 33) and asserted n_constructs_with_sequence_in_records=0 without "
            "reading construct_sequences.tsv or batch21 construct_audit.tsv.",
            "The hardcoded lengths are inconsistent with project records: the real set is "
            "ACTB{100,200,300,400,500,2000}, GAPDH{100,200,300,400,500,2000}, CMVfull=584, CMVcore=204. "
            "batch22 invented 700 bp entries and omitted the 500/2000 bp members -- it did not even "
            "correctly enumerate the constructs it claimed to have no sequence for.",
            "batch22 evidently inspected data/breakthrough_20261002/batch20/examples (duplicated bag "
            "examples and synthetic held-out oligos) plus the batch21 K562 133/164 bp branches, then "
            "generalized to no sequence in project records.",
            "The claim is internally contradicted by batch21 own construct_audit.tsv/json (created "
            "2026-10-03 14:29, before batch22), carrying sequence_sha256 and sequence_source_sha256 for "
            "every one of the 14 constructs.",
        ],
        "what_did_NOT_change": [
            "Sequence AVAILABILITY does not establish assay-to-plasmid mapping (barcode verification = 0 for all 14).",
            "A6 (600 bp annotation) still has no identified donor sequence; the 81 bp CMV-SP1 fragment was not transfected.",
            "CMVcore exact design feature is 204 bp while historical documents call it 200 bp (no clone sequencing).",
            "Original v6.3 TSS +/-200 and upstream-2000 bp genomic windows are NOT replaced by any MPRA 164 bp "
            "window; no deleted genomic sequence was restored and last164 was not used as a verified TSS window.",
            "Local data remain a retrospective HEK293T D28 GFP endpoint with technical (not biological) measurements.",
        ],
        "mapping_qualifications_retained": [
            "filename_and_feature_length", "inferred_from_gene_and_feature_length",
            "inferred_from_name_and_combined_features", "user_clarified_CMVcore_without_enhancer",
            "assay_to_plasmid_barcode_verified=0 for all 14",
        ],
    }

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    seq_out = DATA_OUT / "local_constructs.tsv"
    with open(seq_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["construct_id", "family", "documented_length_bp", "sequence_length_bp",
                    "sequence_sha256", "sequence", "mapping_evidence",
                    "assay_to_plasmid_barcode_verified", "assay_mapping_certainty",
                    "feature_names", "feature_ranges_1based_inclusive", "feature_directionality",
                    "endpoint_target_value_log10"])
        for r in recon:
            w.writerow([r["construct_id"], r["family"], r["documented_length_bp"],
                        r["sequence_length_bp"], r["sequence_sha256_recorded"],
                        seq_by_id[r["construct_id"]]["sequence"], r["mapping_evidence"],
                        r["assay_to_plasmid_barcode_verified"], r["assay_mapping_certainty"],
                        r["feature_names"], r["feature_ranges_1based_inclusive"],
                        r["feature_directionality"], r["endpoint_target_value_log10"]])

    cols = list(recon[0].keys())
    rec_out = REPORT_OUT / "construct_reconciliation.tsv"
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    with open(rec_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in recon:
            w.writerow(r)

    actual_sorted = sorted(r["sequence_length_bp"] for r in recon)
    summary = {
        "generated_at_local": now_local(),
        "started_local": t0_start,
        "batch": "batch23",
        "n_constructs": len(recon),
        "n_with_sequence": int(sum(r["sequence_available"] for r in recon)),
        "n_sequence_sha_match": int(sum(r["sequence_sha_match"] for r in recon)),
        "n_fasta_match": int(sum(r["fasta_sequence_match"] for r in recon)),
        "n_alphabet_ok": int(sum(r["sequence_alphabet_ok"] for r in recon)),
        "n_with_endpoint": int(sum(r["endpoint_row_present"] for r in recon)),
        "n_batch21_audit_join": int(sum(r["in_batch21_audit"] for r in recon)),
        "n_batch21_sha_match": int(sum(r["batch21_sha_matches"] for r in recon)),
        "n_batch04_sha_match": int(sum(r["batch04_sha_matches"] for r in recon)),
        "n_assay_barcode_verified": int(sum(1 for r in recon
                                            if r["assay_to_plasmid_barcode_verified"] == "1")),
        "length_set_actual": actual_sorted,
        "length_set_batch22_hardcoded": sorted(BATCH22_HARDCODED_LENGTHS),
        "length_set_equal": bool(actual_sorted == sorted(BATCH22_HARDCODED_LENGTHS)),
        "n_batch22_lengths_correct": int(sum(r["batch22_length_correct"] for r in recon)),
        "excluded_candidates": excluded,
        "unresolved_candidates": [a6_note],
        "cv_join": cv_join,
        "n_cv_join": int(sum(x["cv_prediction_present"] for x in cv_join)),
        "n_cv_sha_match": int(sum(x["cv_sha_matches_tsv"] for x in cv_join)),
        "n_cv_target_match": int(sum(x["cv_target_matches_endpoint"] for x in cv_join)),
        "sequence_table_out": str(seq_out.relative_to(ROOT)),
        "reconciliation_tsv": str(rec_out.relative_to(ROOT)),
    }

    atomic_write_json(REPORT_OUT / "construct_reconciliation.json", summary)
    atomic_write_json(REPORT_OUT / "correction.json", correction)
    atomic_write_json(REPORT_OUT / "stage1_manifest.json", {
        "generated_at_local": now_local(),
        "outputs": {
            str(seq_out.relative_to(ROOT)): sha256_file(seq_out),
            str(rec_out.relative_to(ROOT)): sha256_file(rec_out),
            "reports/breakthrough_20261002/batch23/construct_reconciliation.json":
                sha256_file(REPORT_OUT / "construct_reconciliation.json"),
            "reports/breakthrough_20261002/batch23/correction.json":
                sha256_file(REPORT_OUT / "correction.json"),
        },
    })
    print("[b23_reconcile] n=%d seq=%d sha_ok=%d fasta_ok=%d endpoint=%d b21_join=%d "
          "barcode_verified=%d b22_lengths_correct=%d/14"
          % (len(recon), summary["n_with_sequence"], summary["n_sequence_sha_match"],
             summary["n_fasta_match"], summary["n_with_endpoint"],
             summary["n_batch21_audit_join"], summary["n_assay_barcode_verified"],
             summary["n_batch22_lengths_correct"]))
    return summary


if __name__ == "__main__":
    main()
