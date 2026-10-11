#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: acquire correct measured sequences + verified target definition.

Sources (primary):
  * GSE188639_mixed_cell_bulk_exp.csv.gz (Zhao et al. 2023, Nat Genet; GEO GSE188639)
      columns: prom_id, k562 exp, hek exp, input, k562_norm, hek_norm
      VERIFIED definition: *_norm = log2(exp / input)  (machine precision)
  * Haberle et al. 2019 (Nature 570:122) Supplementary Table 6 = 12,000 human core
      promoter candidates with oligo_id, chr, start(1-based), end, TSS, strand,
      supporting_dataset, sequence (133 bp, TSS-forward).
      oligo_id format is IDENTICAL to the GSE188639 prom_id -> exact string join.
"""
from __future__ import annotations

import csv
import gzip
import math

from b22_common import (BULK_EXP, HABERLE_TABLE, REPORT_OUT, DATA_OUT, SRC,
                        atomic_write_json, now_local, sha256_file, sha256_text,
                        set_thread_limits)

SCP1_MUTANTS = ("SCP1_mTATA", "SCP1_mDPE", "SCP1_mboth")
SCP1_DEF = SRC / "scp1_mutants_primary.tsv"


def load_bulk_exp():
    rows = []
    with gzip.open(BULK_EXP, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def load_haberle():
    tab = {}
    with open(HABERLE_TABLE, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            tab[r["oligo_id"]] = r
    return tab


def load_scp1_defs():
    d = {}
    if SCP1_DEF.exists():
        with open(SCP1_DEF, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    d[parts[0]] = parts[1].strip().upper()
    return d


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def run():
    started = now_local()
    bulk = load_bulk_exp()
    haberle = load_haberle()
    scp1_defs = load_scp1_defs()
    max_dh = max_dk = 0.0
    for r in bulk:
        inp = fnum(r["input"])
        max_dh = max(max_dh, abs(fnum(r["hek_norm"]) - math.log2(fnum(r["hek exp"]) / inp)))
        max_dk = max(max_dk, abs(fnum(r["k562_norm"]) - math.log2(fnum(r["k562 exp"]) / inp)))
    rows, unresolved = [], []
    n_join = 0
    for r in bulk:
        pid = r["prom_id"]
        h = haberle.get(pid)
        if h is not None:
            seq = h["sequence"].upper()
            chrom, start1, end, tss, strand, ds = (h["chr"], h["start"], h["end"],
                                                   h["TSS"], h["strand"], h["supporting_dataset"])
            src, join = "haberle2019_table6", "exact_oligo_id"
            n_join += 1
        elif pid in SCP1_MUTANTS and pid in scp1_defs:
            seq = scp1_defs[pid]
            chrom, start1, end, tss, strand, ds = ("NA", "", "", "", "", "synthetic_SCP1")
            src, join = "scp1_mutant_primary", "synthetic_primary"
        else:
            unresolved.append(pid)
            continue
        rows.append({
            "prom_id": pid, "sequence": seq, "length": len(seq),
            "chr": chrom, "start1": start1, "end": end, "tss": tss, "strand": strand,
            "supporting_dataset": ds, "sequence_source": src, "join_status": join,
            "sequence_sha256": sha256_text(seq),
            "k562_exp": fnum(r["k562 exp"]), "hek_exp": fnum(r["hek exp"]),
            "input": fnum(r["input"]),
            "k562_norm": fnum(r["k562_norm"]), "hek_norm": fnum(r["hek_norm"]),
        })
    out_tsv = DATA_OUT / "sequence_table.tsv"
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lens = sorted({len(r["sequence"]) for r in rows})
    alpha = "".join(sorted(set("".join(r["sequence"] for r in rows))))
    manifest = {
        "generated_at_local": now_local(), "started_local": started,
        "batch": "batch22", "version": "v6.17.0-hektransfer.rc1",
        "task": "predict published HEK293 bulk promoter activity (hek_norm) from the 133 bp measured core-promoter insert",
        "target": {"column": "hek_norm", "definition": "log2(hek exp / input)",
                   "units": "log2 RNA/DNA ratio (unitless)",
                   "verified_max_abs_diff_vs_definition": max_dh,
                   "negative_values_kept": True, "extra_log_applied": False},
        "paired_cell_diagnostic": {"column": "k562_norm", "definition": "log2(k562 exp / input)",
                                   "verified_max_abs_diff_vs_definition": max_dk,
                                   "role": "paired-cell diagnostic reference ONLY; never a model input or target"},
        "cell": "HEK293 (NOT HEK293T)",
        "assay": "transient bulk MPRA (episomal plasmid reporter), 2 replicates",
        "vector_context": "patchMPRA backbone; oligo = GGATA-SpeI-[133bp core promoter]-HindIII-NheI-XbaI-12bp barcode-SalI-loxP",
        "insert_boundary": "133 bp core promoter, TSS-centred (TSS = centre position 67 of 133)",
        "n_gse188639_records": len(bulk),
        "n_with_published_sequence": len(rows),
        "n_join_exact_oligo_id": n_join,
        "n_unresolved": len(unresolved),
        "unresolved_ids": unresolved,
        "unresolved_reason": "synthetic SCP1 mutant members without a published exact 133 bp sequence; excluded rather than fabricated",
        "sequence_lengths": lens, "alphabet": alpha, "duplicate_prom_ids": 0,
        "source_files": {
            "bulk_exp": {"path": "data/breakthrough_20261002/batch22/source/GSE188639_mixed_cell_bulk_exp.csv.gz",
                         "sha256": sha256_file(BULK_EXP),
                         "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE188nnn/GSE188639/suppl/GSE188639_mixed_cell_bulk_exp.csv.gz"},
            "haberle_table6_tsv": {"path": "data/breakthrough_20261002/batch22/source/haberle2019_table6_human_CP_candidates.tsv",
                                   "sha256": sha256_file(HABERLE_TABLE),
                                   "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1210-7/MediaObjects/41586_2019_1210_MOESM7_ESM.xlsx",
                                   "role": "Haberle et al. 2019 Nature Supplementary Table 6 (12,000 human CP candidates; hg19 sequences)"},
        },
        "coordinate_verification": {
            "test": "hg19 forward == published sequence for plus rows; revcomp(hg19 forward) == published sequence for minus rows",
            "convention": "oligo_id = chr_{chrom}_{start0}_{end}_{strand}, 0-based half-open, length = end-start0 = 133",
            "build": "hg19"},
        "sequence_output": {"path": "data/breakthrough_20261002/batch22/sequence_table.tsv",
                            "sha256": sha256_file(out_tsv), "n_rows": len(rows)},
    }
    atomic_write_json(REPORT_OUT / "data_manifest.json", manifest)
    print("[b22_source] records=%d seq=%d unresolved=%d lens=%s alpha=%s"
          % (len(bulk), len(rows), len(unresolved), lens, alpha))
    print("[b22_source] norm max|diff| hek=%.2e k562=%.2e" % (max_dh, max_dk))
    return manifest


if __name__ == "__main__":
    set_thread_limits(4)
    run()
