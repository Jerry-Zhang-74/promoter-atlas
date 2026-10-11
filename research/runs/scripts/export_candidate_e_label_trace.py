#!/usr/bin/env python3
"""Read-only Candidate E provenance export; never rewrites source labels.

The full GTEx GCT is streamed on CPU and tissue means/medians are independently
checked against the historical six-decimal long table. Outputs are analysis
intermediates for a downstream workbook, not a redistribution of sample TPMs.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GCT = ROOT / "data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
ANNOTATION = ROOT / "data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
EXPRESSION = ROOT / "labels/build/gene_expression_table.tsv"
LABELS = ROOT / "labels/build/gene_expression_label_table.tsv"
PROMOTERS = ROOT / "labels/build/promoter_label_table.tsv"
FIGURE = ROOT / "figures/labels/candidate_e_es_scatter_main_density.png"
ANCHORS = {"ACTB", "GAPDH", "RPLP0", "EEF1A1", "UBC", "B2M", "RPL8", "PGK1"}
NUMERIC_FIELDS = ["E_raw", "S_raw", "B_raw", "Score_E", "Score_S", "Score_B",
                  "Score_HKG", "Score_HKG_rank", "rank_frac_within_main"]
INTEGER_FIELDS = ["tissue_count", "raw_row_count", "duplicate_tissue_pair_count",
                  "detectable_tissue_count", "B_gate_pass", "is_in_main_subset",
                  "S_defined_main", "rank_by_HKG_within_main"]
TOLERANCE = 0.5001e-6
DETECTABLE_TPM_THRESHOLD = 1.0
B_MIN_FOR_MAIN = 0.30
MIN_DETECTABLE_TISSUES_FOR_S = 3


def collapse_duplicate_values(values):
    positives = [value for value in values if value > 0]
    if positives and any(value == 0 for value in values):
        return max(positives), "prefer_nonzero_max"
    return statistics.fmean(values), "mean"


def median_absolute_deviation(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def assign_scores(rows):
    """Independent implementation of the documented historical ordinal ranks."""
    n = len(rows)
    def rank(field, output):
        values = np.array([-np.inf if row.get(field) is None else row[field] for row in rows])
        indices = np.argsort(values, kind="stable")
        for position, index in enumerate(indices):
            rows[int(index)][output] = position / (n - 1) if n > 1 else 1.0
    for field, output in [("E_raw", "Score_E"), ("S_raw", "Score_S"), ("B_raw", "Score_B")]:
        rank(field, output)
    for row in rows:
        if row["S_raw"] is None:
            row["Score_S"] = None
            row["Score_HKG"] = None
        else:
            row["Score_HKG"] = max(row["Score_E"] * row["Score_S"] * row["Score_B"], 0.0) ** (1 / 3)
    rank("Score_HKG", "Score_HKG_rank")
    defined = [row for row in rows if row["is_in_main_subset"] and row["Score_HKG"] is not None]
    ranked = sorted(defined, key=lambda row: row["Score_HKG"], reverse=True)
    positions = {row["gene_id"]: i + 1 for i, row in enumerate(ranked)}
    for row in rows:
        position = positions.get(row["gene_id"])
        row["rank_by_HKG_within_main"] = position if position is not None else ""
        row["rank_frac_within_main"] = position / len(ranked) if position is not None else ""


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_tsv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_wide(path, metadata, tissues, matrix):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        fields = list(metadata[0])
        writer.writerow(fields + tissues)
        for info, values in zip(metadata, matrix):
            writer.writerow([info[key] for key in fields] + [repr(float(x)) for x in values])


def read_expression(gct_n):
    print("Reading historical gene-tissue table...", flush=True)
    expression = pd.read_csv(EXPRESSION, sep="\t", dtype={
        "gene_id": "category", "tissue": "category", "mean_tpm": np.float64,
        "median_tpm": np.float64, "sample_count": np.int32})
    tissues = expression["tissue"].drop_duplicates().tolist()
    nt = len(tissues)
    assert len(expression) == gct_n * nt, "Expression table is not one tissue block per GCT row"
    genes = expression["gene_id"].cat.codes.to_numpy().reshape(gct_n, nt)
    assert np.all(genes == genes[:, :1]), "A tissue block contains multiple genes"
    tissue_codes = expression["tissue"].cat.codes.to_numpy().reshape(gct_n, nt)
    assert np.all(tissue_codes == tissue_codes[:1]), "Tissue order differs across source rows"
    gene_ids = expression["gene_id"].iloc[::nt].astype(str).tolist()
    means = expression["mean_tpm"].to_numpy().reshape(gct_n, nt).copy()
    medians = expression["median_tpm"].to_numpy().reshape(gct_n, nt).copy()
    counts = expression["sample_count"].to_numpy().reshape(gct_n, nt).copy()
    assert np.isfinite(means).all() and (means >= 0).all()
    return tissues, gene_ids, means, medians, counts


def verify_gct(out, tissues, source_ids, means, medians, counts):
    annotations = {}
    with ANNOTATION.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
            sample = row["SAMPID"]
            if sample in annotations:
                assert annotations[sample][0] == row["SMTSD"], "Conflicting sample tissue annotations"
            annotations[sample] = (row["SMTSD"], line)
    rows = []
    started = time.monotonic()
    with gzip.open(GCT, "rt", encoding="utf-8") as handle:
        version = handle.readline().strip()
        nrows, ncols = map(int, handle.readline().strip().split("\t"))
        header = handle.readline().rstrip("\r\n").split("\t")
        samples = header[2:]
        assert ncols == len(samples) and nrows == len(means)
        groups = [[] for _ in tissues]
        tissue_number = {name: index for index, name in enumerate(tissues)}
        sample_rows = []
        for index, sample in enumerate(samples):
            tissue, annotation_line = annotations.get(sample, ("", ""))
            included = tissue in tissue_number
            if included:
                groups[tissue_number[tissue]].append(index)
            sample_rows.append({"sample_id": sample, "tissue": tissue,
                                "gct_column_1based": index + 3,
                                "sample_annotation_line_1based": annotation_line,
                                "included_in_tissue_aggregation": int(included)})
        write_tsv(out / "sample_tissue_mapping.tsv", sample_rows)
        group_arrays = [np.array(group, dtype=np.int64) for group in groups]
        group_sizes = np.array([len(group) for group in groups], dtype=np.int64)
        assert (group_sizes > 0).all()
        assert np.all(counts == group_sizes), "Historical sample counts do not match GTEx header/annotation"
        flat_indices = np.concatenate(group_arrays)
        group_starts = np.r_[0, np.cumsum(group_sizes)[:-1]]
        write_tsv(out / "tissue_metadata.tsv", [{
            "tissue_column_index_1based": j + 1, "tissue": tissue,
            "sample_count": int(group_sizes[j]), "gct_source_gene_rows": nrows,
            "historical_expression_first_row_1based": j + 2,
        } for j, tissue in enumerate(tissues)])
        for index, line in enumerate(handle):
            gene_version, symbol, values_text = line.rstrip("\r\n").split("\t", 2)
            gene = gene_version.split(".")[0]
            assert gene == source_ids[index], f"GCT/long-table gene order mismatch at row {index + 4}"
            values = np.fromstring(values_text, sep="\t", dtype=np.float64)
            assert len(values) == ncols and np.isfinite(values).all()
            grouped = values[flat_indices]
            raw_means = np.add.reduceat(grouped, group_starts) / group_sizes
            raw_medians = np.array([np.median(values[group]) for group in group_arrays])
            mean_delta = np.abs(raw_means - means[index])
            median_delta = np.abs(raw_medians - medians[index])
            rows.append({
                "source_gene_row_key": f"GTEx_v10_GCT_line_{index + 4}",
                "gene_id": gene, "gtex_gene_id_versioned": gene_version,
                "gtex_description": symbol, "gct_gene_row_1based": index + 1,
                "gct_file_line_1based": index + 4,
                "expression_first_line_1based": index * len(tissues) + 2,
                "expression_last_line_1based": (index + 1) * len(tissues) + 1,
                "tissue_count": len(tissues),
                "max_mean_tpm_abs_error": float(mean_delta.max()),
                "mean_tpm_mismatched_tissues": int((mean_delta > TOLERANCE).sum()),
                "max_median_tpm_abs_error": float(median_delta.max()),
                "median_tpm_mismatched_tissues": int((median_delta > TOLERANCE).sum()),
                "sample_count_all_match": 1,
            })
            if (index + 1) % 5000 == 0:
                print(f"GTEx verified {index + 1}/{nrows} genes; {time.monotonic() - started:.0f}s", flush=True)
        assert len(rows) == nrows
    write_tsv(out / "raw_gct_gene_rows.tsv", rows)
    return rows, {"gct_version": version, "gct_gene_rows": nrows, "gct_sample_columns": ncols,
                  "mapped_sample_columns": int(sum(group_sizes)),
                  "unmapped_sample_columns": int(ncols - sum(group_sizes)),
                  "mean_tpm_mismatch_cells": sum(r["mean_tpm_mismatched_tissues"] for r in rows),
                  "median_tpm_mismatch_cells": sum(r["median_tpm_mismatched_tissues"] for r in rows),
                  "max_mean_tpm_abs_error": max(r["max_mean_tpm_abs_error"] for r in rows),
                  "max_median_tpm_abs_error": max(r["max_median_tpm_abs_error"] for r in rows)}


def promoter_mapping():
    names, types, source_lines = defaultdict(set), defaultdict(set), defaultdict(list)
    with PROMOTERS.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
            gene = row["gene_id"].strip().split(".")[0]
            if not gene:
                continue
            if row["gene_name"].strip():
                names[gene].add(row["gene_name"].strip())
            if row["gene_type"].strip():
                types[gene].add(row["gene_type"].strip())
            source_lines[gene].append(line)
    return names, types, source_lines


def export_labels(out, tissues, source_rows, means):
    print("Recomputing Candidate E labels and joining promoter gene names...", flush=True)
    gene_source = defaultdict(list)
    for index, row in enumerate(source_rows):
        gene_source[row["gene_id"]].append(index)
    recomputed, duplicate_rows, collapsed_matrix = [], [], []
    for gene in sorted(gene_source):
        indices = gene_source[gene]
        collapsed = []
        for j, tissue in enumerate(tissues):
            values = means[indices, j].tolist()
            if len(values) == 1:
                value = values[0]
            else:
                value, rule = collapse_duplicate_values(values)
                duplicate_rows.append({
                    "gene_id": gene, "tissue": tissue, "source_row_count": len(values),
                    "collapse_rule": rule, "collapsed_mean_tpm": value,
                    "source_mean_tpm_values": "|".join(map(str, values)),
                    "gtex_gene_ids": "|".join(source_rows[i]["gtex_gene_id_versioned"] for i in indices),
                    "expression_lines_1based": "|".join(str(i * len(tissues) + j + 2) for i in indices),
                })
            collapsed.append(value)
        logs = [math.log2(1.0 + value) for value in collapsed]
        detectable = [value for value in collapsed if value >= DETECTABLE_TPM_THRESHOLD]
        breadth = len(detectable) / len(tissues)
        main = int(breadth >= B_MIN_FOR_MAIN)
        defined = int(main and len(detectable) >= MIN_DETECTABLE_TISSUES_FOR_S)
        stability = -median_absolute_deviation([math.log2(1.0 + value) for value in detectable]) if defined else None
        recomputed.append({"gene_id": gene, "tissue_count": len(tissues),
                           "raw_row_count": len(indices) * len(tissues),
                           "duplicate_tissue_pair_count": len(tissues) if len(indices) > 1 else 0,
                           "detectable_tissue_count": len(detectable), "B_gate_pass": main,
                           "is_in_main_subset": main, "S_defined_main": defined,
                           "E_raw": statistics.fmean(logs), "S_raw": stability, "B_raw": breadth})
        collapsed_matrix.append(collapsed)
    assign_scores(recomputed)
    rec_by_gene = {row["gene_id"]: row for row in recomputed}
    with LABELS.open(encoding="utf-8", newline="") as handle:
        stored = list(csv.DictReader(handle, delimiter="\t"))
    stored_by_gene = {row["gene_id"]: row for row in stored}
    label_lines = {row["gene_id"]: index + 2 for index, row in enumerate(stored)}
    assert set(stored_by_gene) == set(rec_by_gene), "Stored/recomputed label gene universes differ"
    names, types, promoter_lines = promoter_mapping()
    hkg_symbols = {line.split()[0] for line in (ROOT / "data/external/list_of_HKGs.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    audit = {field: {"field": field, "compared_genes": 0, "missing_pairs": 0,
                      "missing_pattern_mismatches": 0, "value_mismatches": 0, "max_abs_error": 0.0}
             for field in NUMERIC_FIELDS + INTEGER_FIELDS}
    records = []
    all_genes = sorted(set(stored_by_gene) | set(promoter_lines))
    for gene in all_genes:
        label = stored_by_gene.get(gene)
        rec = rec_by_gene.get(gene)
        indices = gene_source.get(gene, [])
        raw_symbols = sorted({source_rows[i]["gtex_description"] for i in indices})
        promoter_names = sorted(names.get(gene, []))
        selected = promoter_names[0] if promoter_names else (raw_symbols[0] if raw_symbols else "")
        row = {"gene_id": gene, "gene_symbol": selected,
               "gene_symbol_source": "promoter_label_table.lexicographic_first" if promoter_names else ("GTEx_Description.lexicographic_first" if raw_symbols else "missing"),
               "promoter_gene_symbols_all": "|".join(promoter_names),
               "gtex_descriptions_all": "|".join(raw_symbols),
               "gene_types_all": "|".join(sorted(types.get(gene, []))),
               "in_gtex_label_database": int(label is not None),
               "has_promoter_record": int(gene in promoter_lines),
               "promoter_record_count": len(promoter_lines.get(gene, [])),
               "promoter_first_source_line_1based": min(promoter_lines[gene]) if gene in promoter_lines else "",
               "label_source_line_1based": label_lines.get(gene, ""),
               "gtex_source_gene_row_count": len(indices),
               "gtex_gene_ids_versioned": "|".join(source_rows[i]["gtex_gene_id_versioned"] for i in indices),
               "gct_file_lines_1based": "|".join(str(source_rows[i]["gct_file_line_1based"]) for i in indices),
               "source_row_keys": "|".join(source_rows[i]["source_gene_row_key"] for i in indices),
               "expression_row_blocks_1based": "|".join(f"{source_rows[i]['expression_first_line_1based']}-{source_rows[i]['expression_last_line_1based']}" for i in indices),
               "is_named_anchor": int(bool(set(promoter_names) & ANCHORS)),
               "in_eisenberg_list_by_promoter_symbol": int(bool(set(promoter_names) & hkg_symbols))}
        issues, deltas = [], []
        for field in NUMERIC_FIELDS + INTEGER_FIELDS:
            raw = label.get(field, "") if label else ""
            row[field] = raw
            calculated = rec[field] if rec else None
            row[field + "_recomputed"] = "" if calculated is None else calculated
            if label is None:
                continue
            left_missing = raw in ("", None)
            right_missing = calculated in ("", None)
            item = audit[field]
            if left_missing and right_missing:
                item["missing_pairs"] += 1
            elif left_missing != right_missing:
                item["missing_pattern_mismatches"] += 1
                issues.append(field)
            else:
                delta = abs(float(raw) - float(calculated))
                item["compared_genes"] += 1
                item["max_abs_error"] = max(item["max_abs_error"], delta)
                deltas.append(delta)
                if delta > (0 if field in INTEGER_FIELDS else TOLERANCE):
                    item["value_mismatches"] += 1
                    issues.append(field)
        plotted = bool(label and label["Score_E"] != "" and label["Score_S"] != "" and label["S_defined_main"] == "1")
        reasons = []
        if label is None:
            reasons.append("no_GTEx_expression_label")
        else:
            if float(label["B_raw"]) < B_MIN_FOR_MAIN:
                reasons.append("B_raw_below_0.30")
            if int(label["detectable_tissue_count"]) < MIN_DETECTABLE_TISSUES_FOR_S:
                reasons.append("detectable_tissues_below_3")
            if label["Score_E"] == "":
                reasons.append("Score_E_missing")
            if label["Score_S"] == "":
                reasons.append("Score_S_undefined")
        row.update({"in_attached_density_plot": int(plotted),
                    "plot_exclusion_reason": "|".join(reasons),
                    "recompute_check": "not_applicable_no_GTEx_label" if label is None else ("FAIL:" + "|".join(issues) if issues else "PASS"),
                    "max_label_abs_error": max(deltas) if deltas else ""})
        records.append(row)
    write_tsv(out / "gene_labels_all_database.tsv", records)
    gtex_records = [row for row in records if row["in_gtex_label_database"]]
    write_tsv(out / "gene_labels_gtex.tsv", gtex_records)
    core_fields = ["gene_id", "gene_symbol", "Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw",
                   "Score_HKG", "detectable_tissue_count", "tissue_count", "in_attached_density_plot",
                   "plot_exclusion_reason", "is_named_anchor", "has_promoter_record", "source_row_keys",
                   "label_source_line_1based", "recompute_check", "max_label_abs_error"]
    write_tsv(out / "gene_labels_workbook_core.tsv", [{key: row[key] for key in core_fields} for row in gtex_records])
    write_tsv(out / "named_anchor_labels.tsv", [row for row in gtex_records if row["is_named_anchor"]])
    write_tsv(out / "label_recompute_audit.tsv", list(audit.values()))
    write_tsv(out / "duplicate_resolution.tsv", duplicate_rows)
    collapsed_info = [{"gene_id": row["gene_id"], "gene_symbol": row["gene_symbol"],
                       "source_row_keys": row["source_row_keys"]} for row in gtex_records]
    write_wide(out / "tissue_mean_tpm_used_for_labels.tsv", collapsed_info, tissues, collapsed_matrix)
    plotted = [row for row in gtex_records if row["in_attached_density_plot"]]
    return {"gene_label_count": len(gtex_records), "all_database_union_gene_count": len(records),
            "promoter_only_genes_without_GTEx_label": len(records) - len(gtex_records),
            "GTEx_genes_without_promoter_record": sum(not row["has_promoter_record"] for row in gtex_records),
            "plotted_gene_count": len(plotted), "nonplotted_GTEx_gene_count": len(gtex_records) - len(plotted),
            "named_anchor_count_in_plot": sum(row["is_named_anchor"] for row in plotted),
            "duplicate_gene_count": sum(len(rows) > 1 for rows in gene_source.values()),
            "duplicate_gene_tissue_pairs": len(duplicate_rows),
            "duplicate_rules": dict(Counter(row["collapse_rule"] for row in duplicate_rows)),
            "label_audit_all_pass": all(row["recompute_check"] == "PASS" for row in gtex_records),
            "label_max_abs_error": max(row["max_label_abs_error"] for row in gtex_records),
            "plotted_data_range": {field: [min(float(row[field]) for row in plotted), max(float(row[field]) for row in plotted)] for field in ["Score_E", "Score_S"]}}


def export_promoter_inheritance(out):
    """Deduplicate the two source windows on disk, keeping explicit row keys."""
    print("Exporting promoter-to-gene inheritance map...", flush=True)
    with LABELS.open(encoding="utf-8", newline="") as handle:
        gene_label_lines = {row["gene_id"]: line for line, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2)}
    temporary = out / "promoter_mapping_work.sqlite"
    if temporary.exists():
        raise FileExistsError(f"Remove or inspect leftover exporter work database first: {temporary}")
    connection = sqlite3.connect(temporary)
    fields = ["record_id", "gene_id", "gene_id_versioned", "gene_symbol", "transcript_id",
              "chrom", "strand", "tss_1based", "available_windows", "source_promoter_lines_1based",
              "label_matched", "Score_E", "Score_S", "Score_B", "Score_HKG", "S_defined_main"]
    connection.execute("CREATE TABLE promoters (" + ",".join(f"{key} TEXT" + (" PRIMARY KEY" if key == "record_id" else "") for key in fields) + ", source_record_count INTEGER, conflict INTEGER)")
    sql = ("INSERT INTO promoters VALUES (" + ",".join("?" for _ in range(len(fields) + 2)) + ") "
           "ON CONFLICT(record_id) DO UPDATE SET "
           "available_windows=promoters.available_windows || '|' || excluded.available_windows, "
           "source_promoter_lines_1based=promoters.source_promoter_lines_1based || '|' || excluded.source_promoter_lines_1based, "
           "source_record_count=promoters.source_record_count+1, "
           "conflict=MAX(promoters.conflict, (promoters.gene_id != excluded.gene_id OR promoters.Score_E != excluded.Score_E OR promoters.Score_S != excluded.Score_S OR promoters.Score_B != excluded.Score_B OR promoters.tss_1based != excluded.tss_1based))")
    try:
        source_count = 0
        with PROMOTERS.open(encoding="utf-8", newline="") as handle:
            batch = []
            for line, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
                gene = row["gene_id"].split(".")[0]
                batch.append([row["record_id"], gene, row["gene_id"], row["gene_name"], row["transcript_id"],
                              row["chrom"], row["strand"], row["tss_1based"], row["window_name"], str(line),
                              row["label_matched"], row["Score_E"], row["Score_S"], row["Score_B"],
                              row["Score_HKG"], row["S_defined_main"], 1, 0])
                source_count += 1
                if len(batch) == 10000:
                    connection.executemany(sql, batch)
                    batch.clear()
            if batch:
                connection.executemany(sql, batch)
        connection.commit()
        assert connection.execute("SELECT SUM(conflict) FROM promoters").fetchone()[0] == 0, "Window records disagree on inherited labels or gene/TSS"
        out_fields = fields + ["source_record_count", "label_source_line_1based", "label_measurement_level"]
        unique, matched = 0, 0
        with (out / "promoter_gene_label_mapping.tsv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(out_fields)
            for record in connection.execute("SELECT " + ",".join(fields) + ",source_record_count FROM promoters ORDER BY record_id"):
                gene, label_match = record[1], record[10] == "1"
                writer.writerow(list(record) + [gene_label_lines.get(gene, ""),
                                                "GTEx_gene_summary_inherited_not_promoter_specific_measurement" if label_match else "no_GTEx_gene_label"])
                unique += 1
                matched += int(label_match)
        return {"source_promoter_window_rows": source_count, "unique_promoter_records": unique,
                "promoters_with_gene_labels": matched, "promoters_without_gene_labels": unique - matched,
                "inheritance_join_key": "gene_id", "source_file": str(PROMOTERS)}
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def export_plot_geometry(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(LABELS, sep="\t")
    frame = frame[(frame["S_defined_main"] == 1) & frame["Score_E"].notna() & frame["Score_S"].notna()]
    fig, ax = plt.subplots(figsize=(10, 7.2))
    hb = ax.hexbin(frame["Score_E"], frame["Score_S"], gridsize=55, mincnt=1, linewidths=0)
    centers, counts = hb.get_offsets(), hb.get_array()
    geometry = {"matplotlib_version_used_for_geometry_check": matplotlib.__version__,
                "points": len(frame), "gridsize": 55, "mincnt": 1,
                "xlim": list(map(float, ax.get_xlim())), "ylim": list(map(float, ax.get_ylim())),
                "occupied_hexagons": len(counts), "min_count": int(counts.min()), "max_count": int(counts.max()),
                "sum_hexagon_counts": int(counts.sum()), "density_unit": "gene count per hexagon"}
    write_tsv(out / "plot_hexbin_counts.tsv", [{"bin_index": i + 1, "Score_E_center": float(point[0]),
               "Score_S_center": float(point[1]), "gene_count": int(count)} for i, (point, count) in enumerate(zip(centers, counts))])
    (out / "plot_geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    plt.close(fig)
    notes_path = out / "provenance.json"
    if notes_path.exists():
        notes = json.loads(notes_path.read_text(encoding="utf-8"))
        notes["reconstructed_plot_geometry"] = geometry
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(geometry, indent=2), flush=True)


def output_manifest(out):
    owned = {"duplicate_resolution.tsv", "gene_labels_all_database.tsv", "gene_labels_gtex.tsv",
             "gene_labels_workbook_core.tsv", "label_recompute_audit.tsv", "named_anchor_labels.tsv",
             "promoter_gene_label_mapping.tsv", "provenance.json", "raw_gct_gene_rows.tsv",
             "sample_tissue_mapping.tsv", "source_manifest.tsv", "tissue_mean_tpm_before_duplicate_collapse.tsv",
             "tissue_mean_tpm_used_for_labels.tsv", "tissue_median_tpm_source_not_used_in_labels.tsv",
             "tissue_metadata.tsv", "plot_geometry.json", "plot_hexbin_counts.tsv"}
    outputs = [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(out.iterdir()) if path.is_file() and path.name in owned]
    write_tsv(out / "export_file_manifest.tsv", outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/label_trace_20261001")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--reuse-verified-gct", action="store_true",
                        help="Reuse prior full GCT verification only after all three raw input SHA256 hashes match")
    parser.add_argument("--geometry-only", action="store_true", help="Reconstruct hexbin geometry using an interpreter with Matplotlib")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.geometry_only:
        export_plot_geometry(out)
        output_manifest(out)
        return
    with gzip.open(GCT, "rt", encoding="utf-8") as handle:
        handle.readline()
        gct_n = int(handle.readline().split("\t")[0])
    tissues, source_ids, means, medians, counts = read_expression(gct_n)
    if args.reuse_verified_gct:
        prior = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        with (out / "export_file_manifest.tsv").open(encoding="utf-8-sig", newline="") as handle:
            output_hashes = {row["file"]: row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}
        for name in ["raw_gct_gene_rows.tsv", "provenance.json"]:
            assert sha256(out / name) == output_hashes[name], f"Cached verification output changed: {name}"
        with (out / "source_manifest.tsv").open(encoding="utf-8-sig", newline="") as handle:
            prior_hashes = {row["source_path_absolute"]: row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}
        for path in [GCT, ANNOTATION, EXPRESSION]:
            assert sha256(path) == prior_hashes[str(path.resolve())], f"Verified source changed: {path}"
        with (out / "raw_gct_gene_rows.tsv").open(encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in source_rows:
            for key in ["gct_gene_row_1based", "gct_file_line_1based", "expression_first_line_1based", "expression_last_line_1based", "tissue_count", "mean_tpm_mismatched_tissues", "median_tpm_mismatched_tissues", "sample_count_all_match"]:
                row[key] = int(row[key])
            for key in ["max_mean_tpm_abs_error", "max_median_tpm_abs_error"]:
                row[key] = float(row[key])
        gct_summary = prior["gct_verification"]
        assert len(source_rows) == gct_n
        print("Reused full GCT verification after exact source SHA256 match", flush=True)
    else:
        source_rows, gct_summary = verify_gct(out, tissues, source_ids, means, medians, counts)
    wide_metadata = [{key: row[key] for key in ["source_gene_row_key", "gene_id", "gtex_gene_id_versioned", "gtex_description", "expression_first_line_1based"]} for row in source_rows]
    write_wide(out / "tissue_mean_tpm_before_duplicate_collapse.tsv", wide_metadata, tissues, means)
    write_wide(out / "tissue_median_tpm_source_not_used_in_labels.tsv", wide_metadata, tissues, medians)
    label_summary = export_labels(out, tissues, source_rows, means)
    promoter_summary = export_promoter_inheritance(out)
    print("Hashing source files...", flush=True)
    source_files = [GCT, ANNOTATION, EXPRESSION, LABELS, PROMOTERS, FIGURE,
                    ROOT / "data/external/list_of_HKGs.txt",
                    ROOT / "runs/scripts/build_candidate_e_labels.py",
                    ROOT / "runs/scripts/plot_candidate_e_es_density_nocaption.py",
                    ROOT / "runs/scripts/plot_candidate_e_es_maps_main_only.py",
                    ROOT / "docs/notes/gtex_processing_notes.md",
                    ROOT / "docs/DATA_CITATION.md", ROOT / "代码提交文件夹/DATA_SOURCES.md",
                    ROOT / "labels/qc/candidate_e_es_plot_summary_main_only.tsv",
                    Path(__file__).resolve()]
    if args.reference_image:
        source_files.append(args.reference_image.resolve())
    manifest = [{"source_path_absolute": str(path.resolve()),
                 "bytes": path.stat().st_size, "sha256": sha256(path),
                 "mtime_ns_not_download_time": path.stat().st_mtime_ns,
                 "download_time": "not_recorded"} for path in source_files]
    write_tsv(out / "source_manifest.tsv", manifest)
    figure_hash = next(row["sha256"] for row in manifest if row["source_path_absolute"] == str(FIGURE.resolve()))
    reference_hash = manifest[-1]["sha256"] if args.reference_image else None
    notes = {
        "created_for_date": "2026-10-01", "source_files_modified": False,
        "scope": "Complete human Candidate E label database used by the attached figure, plus all human promoter-to-gene mappings. These are source labels, not v6.3 predictions or nonhuman ortholog-derived targets.",
        "gct_verification": gct_summary, "label_verification": label_summary,
        "promoter_mapping": promoter_summary,
        "tissues": tissues, "comparison_tolerance": TOLERANCE,
        "reference_figure_sha256": figure_hash,
        "attachment_sha256": reference_hash,
        "attachment_exact_file_match": reference_hash == figure_hash if reference_hash else None,
        "formulas": {
            "input": "mean_tpm is the GTEx tissue arithmetic mean across samples, stored with six decimals in the historical long table; these exact stored values feed label recomputation.",
            "duplicate_gene_tissue": "Remove version suffix with gene_id.split('.')[0]. If any value >0 and any value ==0, take max(nonzero); otherwise arithmetic mean of duplicate mean_tpm values. This is not sample-weighted pooling.",
            "E_raw": "arithmetic mean over all 54 tissues of log2(1 + collapsed_mean_tpm)",
            "B_raw": "count(collapsed_mean_tpm >= 1.0) / 54",
            "S_raw": "negative unscaled median absolute deviation of log2(1+collapsed_mean_tpm), restricted to tissues with TPM >=1; defined only if B_raw >=0.30 and detectable tissue count >=3",
            "Score_E_B_S": "stable ascending ordinal rank / (58988-1); rows first sorted by gene_id. Equal metric values receive different ranks in gene_id order. For S, undefined values enter ranking as -infinity across all 58988 genes, then their Score_S is cleared. This is not average-tie rank or a within-main-subset percentile.",
            "Score_HKG": "(Score_E * Score_S * Score_B) ** (1/3); missing if Score_S missing",
            "Score_HKG_rank": "same global ordinal-rank rule with missing HKG as -infinity; historical code does NOT clear this rank for undefined HKG",
            "plot_membership": "Score_E present AND Score_S present AND S_defined_main == 1. No explicit E/S range filter, promoter-match filter, or protein-coding filter.",
            "plot_geometry": "matplotlib hexbin gridsize=55, mincnt=1, no weights/C, linear count colors, linewidths=0; 10x7.2 inches, PNG 300 dpi. Axes use automatic limits; colorbar labeled gene density is a count per occupied hexagon, not probability density.",
            "anchor_symbols": sorted(ANCHORS),
            "anchor_mapping": "gene_id base -> symbols from promoter_label_table; lexicographically first symbol is rendered; any matching anchor symbol identifies the gene.",
        },
        "stored_source_links": [
            {"source": "GTEx official datasets", "url": "https://gtexportal.org/home/datasets/", "found_in": "代码提交文件夹/DATA_SOURCES.md:11", "online_rechecked_this_export": False},
            {"source": "GENCODE v49", "url": "https://www.gencodegenes.org/human/release_49.html", "found_in": "代码提交文件夹/DATA_SOURCES.md:9", "online_rechecked_this_export": False},
            {"source": "GTEx dbGaP accession", "identifier": "phs000424", "found_in": "docs/DATA_CITATION.md:13-14", "url": None},
        ],
        "limitations": [
            "Original GTEx-to-tissue aggregation script is not retained. This export independently verifies present GCT/sample annotations against every historical tissue mean and median; it does not reconstruct the historical execution log or downloader.",
            "Source download time and historical software versions for aggregation are not recorded; file modification times are not download times.",
            "TPM workbook tables are per-tissue summaries, not sample-level measurements; complete sample TPMs remain in the local GCT. Sample-to-tissue membership and source row/column keys are exported.",
            "Gene expression labels are associated gene-level GTEx summaries joined to promoters, not experimentally measured activity for each promoter/TSS.",
            "GENCODE promoter symbols and GTEx Description symbols may differ. Both are retained; exact figure names use promoter mappings.",
            "Stable ordinal tie ranking in the historical implementation can assign unequal Score_B to genes with identical B_raw; preserved for traceability, not silently corrected.",
            "No stored GTEx paper URL/DOI was identified in the inspected source records; the existing official dataset URL is provided, without inventing a paper link.",
        ],
        "files": {"gene_labels_all_database.tsv": "Union of all GTEx label genes and genes represented by the promoter label table; unmatched genes explicitly flagged.",
                  "gene_labels_gtex.tsv": "All 58988 genes in the plotted label database, including excluded genes; historical and recalculated values side by side.",
                  "gene_labels_workbook_core.tsv": "Compact 19-column subset of all 58988 GTEx gene labels for the workbook.",
                  "named_anchor_labels.tsv": "The eight named figure anchors with the same full audit fields.",
                  "promoter_gene_label_mapping.tsv": "One row per unique promoter record_id, shared gene_id label key, inherited scores, available windows and every source promoter row line.",
                  "tissue_mean_tpm_used_for_labels.tsv": "58988 x 54 matrix after the exact historical duplicate rule; raw untransformed tissue means used for labels.",
                  "tissue_mean_tpm_before_duplicate_collapse.tsv": "59033 x 54 historical mean TPM rows, retaining original GCT gene row/version and source lines.",
                  "tissue_median_tpm_source_not_used_in_labels.tsv": "59033 x 54 historical median TPM rows; retained for audit, not used by Candidate E.",
                  "raw_gct_gene_rows.tsv": "GCT source row map and full independent mean/median check per original row.",
                  "sample_tissue_mapping.tsv": "GCT sample column to annotation line/tissue mapping; no sample-level TPM values copied.",
                  "duplicate_resolution.tsv": "All duplicate gene-tissue inputs, source lines and chosen collapse rules.",
                  "source_manifest.tsv": "SHA256 hashes and absolute paths of existing inputs, scripts and supplied figure.",
                  "label_recompute_audit.tsv": "Comparison of every historical raw metric/score/count/rank with independent recomputation.",
                  "tissue_metadata.tsv": "Tissue names, sample counts and columns."},
    }
    (out / "provenance.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output_manifest(out)
    print(json.dumps({"gct": gct_summary, "labels": label_summary, "attachment_exact_match": notes["attachment_exact_file_match"]}, ensure_ascii=False, indent=2), flush=True)
    if gct_summary["mean_tpm_mismatch_cells"] or gct_summary["median_tpm_mismatch_cells"] or not label_summary["label_audit_all_pass"]:
        raise SystemExit("Export completed with provenance discrepancies: inspect audit tables before use")


if __name__ == "__main__":
    main()
