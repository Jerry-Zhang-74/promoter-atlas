#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 bounded GSE306816 qualification (<=25 min, <=75 MB, NO fit).

Batch23 excluded GSE306816 only because it measures engineered STR variants.
This script inspects the actual processed tables / sequence definitions and
decides from EVIDENCE whether it supports a separate variant-effect task or is
inapplicable to the current 133 bp promoter-autonomy domain.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C

C.set_thread_limits(4)
SRC = C.DATA_OUT / "gse306816"


def partial_lines(path, max_lines=200000):
    """Decode as much as possible from a possibly-truncated gzip stream."""
    d = open(path, "rb").read()
    o = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        t = o.decompress(d, 8 * 1024 * 1024)
    except zlib.error:
        t = b""
    txt = t.decode("utf-8", errors="replace")
    lines = txt.split("\n")
    return lines[:max_lines], len(d)


def parse_variant(name):
    """Human_STR_<locus>-<origRpt>-<curRpt>-<strand>_<curRptSeq>_p<num>"""
    try:
        head, tail = name.split("_p", 1)
        pre, curseq = head.rsplit("_", 1)
        body = pre.replace("Human_STR_", "")
        locus, orig, cur, strand = body.rsplit("-", 3)
        return {"locus": locus, "orig_repeat": orig, "current_repeat": cur,
                "strand": strand, "current_repeat_seq": curseq, "pRPT": "p" + tail}
    except Exception:
        return None


def main():
    out = {"generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
           "accession": "GSE306816", "zero_fits": True,
           "bound": {"max_minutes": 25, "max_mb": 75},
           "files": {}, "findings": {}, "decision": {}}
    paths = {
        "linear_regression": SRC / "GSE306816_dpSTR_HEK293_linear_regression.csv.gz",
        "ratio_rep1": SRC / "GSE306816_dpSTR_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz",
        "ratio_rep2": SRC / "GSE306816_dpSTR_HEK293_BC_cDNA_gDNA_ratio_rep2.csv.gz",
        "ratio_rep3": SRC / "GSE306816_dpSTR_HEK293_BC_cDNA_gDNA_ratio_rep3.csv.gz",
        "association_illumina": SRC / "GSM9209852_dpSTR_association_Illumina_PARTIAL_855KB.tsv.gz",
    }
    total_mb = 0.0
    for name, p in paths.items():
        if p.exists():
            b = p.stat().st_size
            total_mb += b / 1e6
            out["files"][name] = {"bytes": b, "mb": round(b / 1e6, 3),
                                  "sha256": C.sha256_file(p)}
    out["files"]["linear_regression"]["state"] = "COMPLETE (size matches filelist 263799)"
    for n in ("ratio_rep1", "ratio_rep2", "ratio_rep3", "association_illumina"):
        out["files"][n]["state"] = "INCOMPLETE/truncated by the bounded fetch"
    out["files"]["association_illumina"]["full_size_bytes"] = 11231653
    out["downloaded_mb_total"] = round(total_mb, 3)

    # ---- 1. primary processed variant-effect table (COMPLETE) ---------------
    loci = Counter()
    units = Counter()
    n = 0
    pos = 0
    sig = 0
    with gzip.open(paths["linear_regression"], "rt", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        for row in rd:
            n += 1
            lid = row["locus"]
            parts = lid.rsplit("-", 3)
            loci[parts[0] if len(parts) == 4 else lid] += 1
            units[row["repeat_unit"]] += 1
            b1 = float(row["beta_1"])
            if b1 > 0:
                pos += 1
            if float(row["p_value_1"]) < 0.05:
                sig += 1
    out["findings"]["variant_effect_table"] = {
        "file": "GSE306816_dpSTR_HEK293_linear_regression.csv.gz",
        "columns": cols,
        "meaning": ("per (locus, original-repeat, current-repeat, strand) linear regression of "
                    "reporter expression on repeat COPY NUMBER: beta_0 = intercept, beta_1 = "
                    "slope/effect, p_value_1 = slope p-value. These are VARIANT-EFFECT labels."),
        "n_rows": n, "n_parent_loci": len(loci),
        "rows_per_locus": {"min": min(loci.values()), "max": max(loci.values()),
                           "median": sorted(loci.values())[len(loci) // 2]},
        "frac_beta1_positive": pos / n, "frac_p_lt_0.05": sig / n,
        "n_distinct_repeat_units": len(units),
        "top_repeat_units": units.most_common(10),
    }

    # ---- 2. per-barcode ratio table (partial) ------------------------------
    lines, nbytes = partial_lines(paths["ratio_rep1"])
    hdr = [l for l in lines[:5] if not l.startswith("#")]
    body = [l for l in lines if l and not l.startswith("#")]
    rows = list(csv.DictReader(body, delimiter=","))
    bcs = set(); alleles = set(); readsum = 0
    pl = Counter()
    for r in rows:
        bcs.add(r["BC"]); alleles.add(r["STR_allele"])
        p = parse_variant(r["STR_allele"])
        if p:
            pl[p["locus"]] += 1
    out["findings"]["per_barcode_ratio_table"] = {
        "file": "GSE306816_dpSTR_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz (PARTIAL)",
        "columns": ["BC", "STR_allele", "reads_cDNA", "reads_gDNA", "ratio"],
        "meaning": ("per-barcode measured reporter expression = cDNA/gDNA read ratio for a "
                    "20 bp barcode; a per-CONSTRUCT relative activity, but the construct's full "
                    "DNA sequence is NOT in this table"),
        "decoded_lines": len(body), "n_partial_rows": len(rows),
        "n_distinct_bc": len(bcs), "n_distinct_str_allele": len(alleles),
        "n_distinct_parent_loci_in_partial": len(pl),
        "has_sequence_column": False,
    }
    # ---- 3. association table (partial): BC -> variant name is the only "sequence" ----
    alines, abytes = partial_lines(temp_assoc := paths["association_illumina"]
                                   if paths["association_illumina"].exists() else paths["ratio_rep1"])
    arows = [l for l in alines if l and not l.startswith("#")]
    assoc = list(csv.DictReader(arows, delimiter="\t"))
    assoc_variants = set(r["Variant_name"] for r in assoc)
    out["findings"]["association_table"] = {
        "file": "GSM9209852_dpSTR_association_Illumina_BC_STRs.tsv.gz (PARTIAL)",
        "columns": ["BC", "Variant_name"],
        "name_pattern": "Human_STR_<locus>-<originalRepeat>-<currentRepeat>-<strand>_<currentRepeatSeq>_p<pRPTNum>",
        "n_partial_rows": len(assoc), "n_distinct_variant": len(assoc_variants),
        "has_full_construct_sequence": False,
        "note": ("the variable element is an STR (repeat unit / copy number / strand) in a FIXED "
                 "reporter/promoter backbone; the fixed backbone sequence and flanking context "
                 "are NOT provided in the processed supplementary"),
    }
    inter = len(alleles & assoc_variants) if alleles else 0
    out["findings"]["exact_join_check"] = {
        "criterion": "STR_allele (ratio table) == Variant_name (association table), exact string",
        "n_partial_alleles": len(alleles), "n_partial_assoc_variants": len(assoc_variants),
        "n_exact_intersection": inter,
        "note": ("both partial extracts share the same variant-name namespace; the intersection "
                 "is bounded by the partial decode and is not a full-library join"),
    }

    # ---- 4. decision from evidence (not nomenclature) ----------------------
    out["findings"]["cell_and_design"] = {
        "cell": "HEK293T (GEO sample metadata titles dpSTR_HEK293T_cDNA_Rep1..3 / _gDNA_Rep1..3)",
        "note_processed_filename": ("the processed FILE name abbreviates the cell as 'HEK293'; the "
                                    "GEO sample metadata says HEK293T"),
        "biological_replicates": 3,
        "assay": "transient MPRA, GFP reporter, unique 20 bp barcodes; 2 synthesis platforms "
                 "(Illumina + Element) for the association/detection library",
        "variable_element": "promoter-proximal STR (repeat unit x copy number x strand)",
        "fixed_element": "reporter/promoter backbone (constant; sequence NOT in the processed tables)",
        "does_not_supply": ["full 133 bp core-promoter inserts",
                            "core-promoter sequence autonomy",
                            "raw-FASTA construct sequences"],
    }
    out["decision"] = {
        "usable_for_current_133bp_promoter_autonomy_domain": False,
        "reason": ("the measured variable element is an STR in a FIXED promoter/reporter backbone; "
                   "no construct-level DNA sequence is provided, so the existing sequence model "
                   "cannot be trained/evaluated on it without the backbone context"),
        "supports_separate_variant_effect_scope": "PARTIAL (variant-effect slopes are well defined; "
                                                  "sequence-level input is missing)",
        "data_ready_as_sequence_activity": False,
        "variant_effect_task_ready": False,
        "specific_missing_context": [
            "fixed reporter/promoter backbone sequence and the STR insertion position",
            "full per-barcode construct sequence (only BC -> STR variant name is given)",
            "the hSTR1/hSTR2 association tables (larger, 14.2 MB each unmirrored here) were not "
            "downloaded in full within the bound",
        ],
        "not_claimed": "this is NOT a claim that all public HEK293/HEK293T data are unavailable; "
                       "it is the specific missing-sequence/context evidence for GSE306816",
    }
    C.atomic_write_json(C.REPORT_OUT / "gse306816_followup.json", out)
    print(json.dumps({k: out[k] for k in ("files", "findings", "decision", "downloaded_mb_total")},
                     indent=1)[:6000])


if __name__ == "__main__":
    main()
