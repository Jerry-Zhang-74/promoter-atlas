#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: RELEVANT PUBLIC DATA ACQUISITION AND TRAINING READINESS (batch23).

Tests the larger target-domain-supervision hypothesis against primary public sources.
Candidate sequences WITHOUT measured cell-specific promoter activity do NOT count as
training rows. Enhancer assays with a fixed exogenous minimal promoter, STAP nuclear
extract, endogenous transcript abundance, lenti-integrated and transient reporter
endpoints all require separate context labels and are NOT silently pooled.

ZERO fits here. Outputs a verified source inventory + data_ready.json +
next_batch_proposal.json. Downloads are logged in an append-only attempts ledger.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b23_common import (  # noqa: E402
    ROOT, DATA_OUT, REPORT_OUT, BATCH22_DATA, BATCH22_SEQ_TABLE, BATCH22_FOLD_ASSIGN,
    now_local, sha256_file, atomic_write_json, set_thread_limits)

SRC = DATA_OUT / "source"
GSE217680 = SRC / "GSE217680_bulk_MPRA_counts_promoter_series.txt.gz"
GSE245191_PART = SRC / "GSE245191_architecture_INCOMPLETE_partial.txt.gz"
LEDGER = REPORT_OUT / "attempts_ledger.jsonl"


def ledger(rec):
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(rec)
    import os
    rec["pid"] = os.getpid()
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def gz_lines(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read().split("\n")


def partial_gz_lines(path):
    """Decompress as much of a truncated gzip stream as possible."""
    buf = open(path, "rb").read()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return d.decompress(buf).decode("utf-8", errors="replace").split("\n")


def analyse_gse217680():
    lines = gz_lines(GSE217680)
    hdr = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:] if l.strip()]
    i = {c: k for k, c in enumerate(hdr)}
    cell = collections.Counter(r[i["cell_line"]] for r in rows)
    cre = collections.Counter(r[i["CRE_id"]] for r in rows)
    cls = collections.Counter(r[i["CRE_class"]] for r in rows)
    rep = collections.Counter(r[i["rep_id"]] for r in rows)
    return {
        "path": str(GSE217680.relative_to(ROOT)), "sha256": sha256_file(GSE217680),
        "columns": hdr, "n_data_rows": len(rows),
        "cell_lines": dict(cell), "CRE_class": dict(cls), "reps": dict(rep),
        "n_distinct_cre_id": len(cre), "cre_ids": sorted(cre),
        "has_sequence_column": int(any("seq" in c.lower() for c in hdr)),
        "rows_usable_as_promoter_activity": 0,
        "reason": "heavily-barcoded counts for 5 FIXED CONTROL promoters "
                  "(noP/minP/UBCp/Pgk1P/eef1aP) only -- not a sequence-varied promoter "
                  "library; no sequence column; cannot yield >=1000 sequence-activity rows",
    }


def analyse_gse245191_partial():
    lines = partial_gz_lines(GSE245191_PART)
    hdr = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:] if l.strip() and len(l.split("\t")) >= 16]
    i = {c: k for k, c in enumerate(hdr)}
    prom = collections.Counter(r[i["promoter"]] for r in rows)
    bio = collections.Counter(r[i["biotype"]] for r in rows)
    libs = sorted({r[i["lib_short_name"]] for r in rows})
    return {
        "path": str(GSE245191_PART.relative_to(ROOT)), "sha256": sha256_file(GSE245191_PART),
        "download_state": "INCOMPLETE (5.7 MB of 33 MB; corrupt/truncated gzip retained for honesty)",
        "partial_head_path": str((SRC / "part_head.gz").relative_to(ROOT)),
        "partial_head_sha256": sha256_file(SRC / "part_head.gz"),
        "columns": hdr,
        "n_data_rows_read": len(rows),
        "n_distinct_promoter": len(prom), "promoter_values": sorted(prom),
        "library_replicate_labels_biotype_col": dict(bio),
        "n_distinct_libraries_seen": len(libs),
        "has_sequence_column": 0,
        "rows_usable_as_promoter_activity": 0,
        "reason": "the only processed supplementary for GSE245191 is the promoter-ARCHITECTURE/"
                  "positional-effects table (5 fixed control promoters minP/UBCp/EEF1A1p/noP/Pgk1p); "
                  "the 0.3k/10k/30k promoter-library samples exist only as raw sequence data with no "
                  "processed sequence-activity table and no sequence definitions",
    }


def reused_reference():
    """The GSE188639 HEK293 cohort already used by batch22 (retrospective reference)."""
    rows = list(csv.DictReader(open(BATCH22_SEQ_TABLE, encoding="utf-8"), delimiter="\t"))
    fa = list(csv.DictReader(open(BATCH22_FOLD_ASSIGN, encoding="utf-8"), delimiter="\t"))
    groups = sorted({r["group_id"] for r in fa})
    return {
        "accession": "GSE188639",
        "title": "A single-cell massively parallel reporter assay detects cell type specific "
                 "cis-regulatory activity (scMPRA core-promoter library)",
        "role": "clearly REUSED retrospective reference (batch22 cohort); NOT a new independent study",
        "processed_file": "data/breakthrough_20261002/batch22/source/"
                          "GSE188639_mixed_cell_bulk_exp.csv.gz",
        "bulk_file_sha256": sha256_file(BATCH22_DATA / "source" / "GSE188639_mixed_cell_bulk_exp.csv.gz"),
        "columns": ["prom_id", "k562 exp", "hek exp", "input", "k562_norm", "hek_norm"],
        "n_library_members": 676, "n_with_published_133bp_sequence": len(rows),
        "n_label_free_groups": len(groups),
        "sequence_available": 1,
        "sequence_definition": "Haberle et al. 2019 hg19 133 bp core promoter",
        "target_units_hek": "log2(hek exp / input)",
        "cell": "HEK293 (series is a HEK293+K562 mixture; processed file reports per-cell-type exp)",
        "assay": "transient bulk MPRA (episomal/plasmid reporter), 2 replicates",
        "note": "674 of 676 members usable after the 3 SCP1 mutants; the K562 columns are measurements "
                "of the SAME sequences -> extra rows do NOT add independent sequence groups (669 total)",
    }


def main():
    set_thread_limits(4)
    t0 = now_local()
    ledger({"event": "stage3_start", "start": t0,
            "sources_targeted": ["GSE188639 (reused)", "GSE245191", "GSE217680",
                                 "GSE306816", "GSE78709"]})
    g217680 = analyse_gse217680()
    g245191 = analyse_gse245191_partial()
    g306816 = {
        "accession": "GSE306816",
        "title": "Systematic evaluation of the impact of promoter proximal short tandem repeats "
                 "on gene expression",
        "primary_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306816",
        "supplementary": ["GSE306816_RAW.tar (22 MB)",
                          "GSE306816_dpSTR_HEK293_BC_cDNA_gDNA_ratio_rep1..3.csv.gz"],
        "cell": "HEK293T (also HeLa/RNH arms)",
        "assay": "transient plasmid MPRA, GFP reporter, cDNA/gDNA barcode ratio, 3 replicates",
        "n_loci_reported": 19818, "sequence_available": "in RAW.tar (not retrieved)",
        "rows_usable_as_promoter_activity": 0, "inclusion": "EXCLUDED",
        "reason": "library = synthetic promoter variants carrying variable STR copy number/motif; the "
                  "target is an STR-driven reporter effect, not the autonomous activity of a natural "
                  "promoter sequence. Transient HEK293T plasmid context is closer to the local "
                  "constructs, but the design is variant-effect and cannot be silently pooled.",
        "possible_use": "a separate repeat-variant question only",
    }
    g78709 = {
        "accession": "GSE78709",
        "title": "Genome-wide mapping of autonomous promoter activity in human cells (SuRE)",
        "primary_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78709",
        "supplementary": ["GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz (1.1 GB)",
                          "GSE78709_gencode.based.tss.SuRE.170828.txt.gz (1.1 MB)",
                          "bigWigs (multi-GB)"],
        "cell": "K562 (this series)",
        "assay": "SuRE: random 0.2-2 kb genomic fragments, plasmid, autonomous transcription, barcoded, "
                 "2 biological replicates",
        "sequence_available": "genomic hg19 coordinates -> sequences derivable",
        "rows_usable_as_promoter_activity": 0,
        "inclusion": "EXCLUDED (as a HEK target-domain source)",
        "reason": "K562 only in this series, not HEK293/HEK293T; processed counts are 1.1 GB + "
                  "multi-GB bigWigs, exceeding the processed-download bound without a HEK file.",
        "possible_use": "genome-wide autonomous-promoter reference in K562",
    }
    reused = reused_reference()
    inventory = {
        "generated_at_local": now_local(), "started_local": t0, "batch": "batch23",
        "hypothesis": "a larger HEK293/HEK293T promoter-autonomy sequence-activity cohort exists that "
                      "can raise target-domain supervision above the batch21 from-scratch gate",
        "criterion": ">=1000 usable sequence-activity rows AND >=500 independent sequence/locus groups "
                     "with measured cell-specific promoter activity and defined sequences",
        "reused_retrospective_reference": reused,
        "leads": [
            {"lead": 1, "accession": "GSE245191",
             "cell": "HEK293T + K562 (genome-integrated)", "assessment": g245191},
            {"lead": 2, "accession": "GSE217680",
             "cell": "HEK293T + HepG2 + K562 (genome-integrated)", "assessment": g217680},
            {"lead": 3, "accession": "GSE306816",
             "cell": "HEK293T (transient plasmid)", "assessment": g306816},
            {"lead": 4, "accession": "GSE78709", "cell": "K562", "assessment": g78709},
        ],
        "cohort_obtainable_within_bound": False,
        "conclusion": "No coherent HEK293/HEK293T promoter-AUTONOMY sequence-activity cohort with "
                      ">=1000 usable rows and >=500 independent sequence groups was obtainable within "
                      "the bound. The two HEK293T bulk-MPRA processed tables (GSE245191, GSE217680) "
                      "contain only 5 fixed control promoters each; the large HEK293T promoter library "
                      "samples exist only as raw sequence data with no processed sequence-activity "
                      "table and no sequence definitions; the HEK293T STR library (GSE306816) is a "
                      "repeat-variant design, not promoter autonomy; SuRE (GSE78709) is K562.",
        "what_this_does_not_prove": [
            "It does NOT prove that data volume is the binding constraint.",
            "It does NOT prove cell transfer must fail.",
            "It does NOT lower or raise the batch21 from-scratch gate; the 673 reused rows are retained.",
        ],
        "attempts": {
            "downloaded": [
                {"file": g217680["path"], "sha256": g217680["sha256"], "bytes": 88412},
                {"file": g245191["path"], "sha256": g245191["sha256"], "bytes": 5735600,
                 "state": "incomplete"},
                {"file": g245191["partial_head_path"], "sha256": g245191["partial_head_sha256"],
                 "bytes": 2500001},
            ],
            "processed_download_mb": round((88412 + 5735600 + 2500001) / 1e6, 3),
            "budget_mb": 750,
        },
    }
    atomic_write_json(REPORT_OUT / "public_source_inventory.json", inventory)
    ledger({"event": "stage3_inventory_done", "at": now_local(), "cohort_obtainable": False,
            "processed_download_mb": inventory["attempts"]["processed_download_mb"]})
    print("[b23_public] cohort_obtainable=%s leads=%d processed_mb=%.2f"
          % (inventory["cohort_obtainable_within_bound"], len(inventory["leads"]),
             inventory["attempts"]["processed_download_mb"]))
    return inventory


if __name__ == "__main__":
    main()
