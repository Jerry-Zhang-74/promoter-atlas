"""Build Candidate E gene- and promoter-level labels.

This script expects a long-format gene-by-tissue expression table with at least:
    gene_id, tissue, mean_tpm

Outputs:
    - gene_expression_label_table.tsv
    - promoter_label_table.tsv
    - promoter_model_input_core.tsv
    - promoter_model_input_extended.tsv
    - QC, sanity, anchor, and manifest files

Candidate E rules:
    - E_raw: mean(log2(1 + mean_tpm)) across all tissues
    - B_raw: detectable tissue fraction using TPM >= 1
    - S_raw: -MAD(log2(1 + mean_tpm)) over detectable tissues only
    - S_raw is defined only when:
        * B_raw >= 0.30
        * detectable_tissue_count >= 3
    - Score_S remains empty when S_raw is undefined
    - Score_HKG remains empty when Score_S is undefined

Duplicate gene+tissue handling:
    - If duplicate values include both zero and non-zero values, keep max(non-zero)
    - Otherwise, use mean(values)
"""

from pathlib import Path
import csv
import math
import statistics
from collections import Counter, defaultdict


ROOT = Path(".")

EXPR_TSV = ROOT / "labels/build/gene_expression_table.tsv"
PROMOTER_TSV = ROOT / "promoters/analysis/promoter_analysis_set.tsv"
if not PROMOTER_TSV.exists():
    PROMOTER_TSV = ROOT / "labels/build/promoter_analysis_set.tsv"

GENE_LABEL_TSV = ROOT / "labels/build/gene_expression_label_table.tsv"
PROMOTER_LABEL_TSV = ROOT / "labels/build/promoter_label_table.tsv"
CORE_INPUT_TSV = ROOT / "labels/build/promoter_model_input_core.tsv"
EXTENDED_INPUT_TSV = ROOT / "labels/build/promoter_model_input_extended.tsv"
MANIFEST_TSV = ROOT / "labels/build/baseline_train_manifest.tsv"

QC_TSV = ROOT / "labels/qc/label_build_qc.tsv"
SANITY_TSV = ROOT / "labels/qc/gene_label_sanity_stats.tsv"
ANCHOR_TSV = ROOT / "labels/qc/hkg_anchor_sanity.tsv"
DUPLICATE_QC_TSV = ROOT / "labels/qc/gene_expression_duplicate_tissue_qc.tsv"

LABEL_NOTES_MD = ROOT / "docs/notes/label_build_notes.md"
SANITY_NOTES_MD = ROOT / "docs/notes/gene_label_sanity_notes.md"

DETECTABLE_TPM_THRESHOLD = 1.0
B_MIN_FOR_MAIN = 0.30
MIN_DETECTABLE_TISSUES_FOR_S = 3
ANCHOR_NAMES = {"ACTB", "GAPDH", "RPLP0", "EEF1A1", "UBC", "B2M", "RPL8"}


def ensure_directories() -> None:
    for path in [
        ROOT / "labels/build",
        ROOT / "labels/qc",
        ROOT / "docs/notes",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    if not EXPR_TSV.exists():
        raise SystemExit("ERROR: missing labels/build/gene_expression_table.tsv")
    if not PROMOTER_TSV.exists():
        raise SystemExit("ERROR: missing promoter analysis set")


def base_gene_id(value: str) -> str:
    return value.split(".")[0] if value else ""


def normalize_fieldname(value: str) -> str:
    return value.strip().strip('"').strip()


def safe_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)

    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]

    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))

    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")

    return numerator / (denom_x * denom_y)


def quantiles(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def pick(p):
        idx = int(round((n - 1) * p))
        idx = min(n - 1, max(0, idx))
        return values[idx]

    return values[0], pick(0.25), pick(0.50), pick(0.75), values[-1]


def median(values):
    return statistics.median(values) if values else 0.0


def mad(values):
    if not values:
        return 0.0
    m = median(values)
    return median([abs(x - m) for x in values])


def collapse_duplicate_values(values):
    nonzero_values = [x for x in values if x > 0.0]
    zero_count = sum(1 for x in values if x == 0.0)

    if nonzero_values and zero_count > 0:
        return max(nonzero_values), "prefer_nonzero_max"

    return statistics.fmean(values), "mean"


def rank_scale(rows, key, out_key, none_value=None):
    n = len(rows)
    if n == 0:
        return

    def sort_value(row):
        value = row.get(key)
        if value is None:
            return none_value
        return value

    order = sorted(range(n), key=lambda i: sort_value(rows[i]))

    if n == 1:
        rows[0][out_key] = 1.0
        return

    for rank, idx in enumerate(order):
        rows[idx][out_key] = rank / (n - 1)


def collect_values(rows, key, require_defined=False):
    output = []
    for row in rows:
        if require_defined and not row.get("S_defined_main", 0):
            continue
        value = row.get(key)
        if value is None:
            continue
        output.append(value)
    return output


def load_expression_table():
    gene_tissue_values = defaultdict(lambda: defaultdict(list))
    gene_total_rows = Counter()
    all_tissues = set()

    with EXPR_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("ERROR: gene_expression_table.tsv has no header")

        normalized = [normalize_fieldname(x) for x in reader.fieldnames]
        if "gene_id" not in normalized:
            raise SystemExit("ERROR: gene_expression_table.tsv missing gene_id")
        if "tissue" not in normalized:
            raise SystemExit("ERROR: gene_expression_table.tsv missing tissue")
        if "mean_tpm" not in normalized:
            raise SystemExit("ERROR: gene_expression_table.tsv missing mean_tpm")

        raw_to_normalized = dict(zip(reader.fieldnames, normalized))

        gene_id_key = None
        tissue_key = None
        mean_tpm_key = None

        for raw_key, normalized_key in raw_to_normalized.items():
            if normalized_key == "gene_id":
                gene_id_key = raw_key
            elif normalized_key == "tissue":
                tissue_key = raw_key
            elif normalized_key == "mean_tpm":
                mean_tpm_key = raw_key

        for row in reader:
            gene_id = base_gene_id(row[gene_id_key])
            tissue = str(row[tissue_key]).strip()
            mean_tpm_raw = str(row[mean_tpm_key]).strip()

            if not gene_id or not tissue:
                continue

            mean_tpm = 0.0 if mean_tpm_raw == "" else float(mean_tpm_raw)

            gene_tissue_values[gene_id][tissue].append(mean_tpm)
            gene_total_rows[gene_id] += 1
            all_tissues.add(tissue)

    return gene_tissue_values, gene_total_rows, all_tissues


def build_gene_rows(gene_tissue_values, gene_total_rows):
    rows_per_gene_counter = Counter()
    unique_tissues_per_gene_counter = Counter()

    duplicate_gene_count = 0
    duplicate_gene_tissue_pairs = 0
    prefer_nonzero_pair_count = 0
    mean_pair_count = 0
    max_raw_rows_per_gene = 0
    max_unique_tissues_per_gene = 0

    duplicate_qc_rows = []
    gene_rows = []

    for gene_id in sorted(gene_tissue_values.keys()):
        tissue_map = gene_tissue_values[gene_id]
        raw_row_count = gene_total_rows[gene_id]
        unique_tissue_count = len(tissue_map)

        rows_per_gene_counter[raw_row_count] += 1
        unique_tissues_per_gene_counter[unique_tissue_count] += 1
        max_raw_rows_per_gene = max(max_raw_rows_per_gene, raw_row_count)
        max_unique_tissues_per_gene = max(max_unique_tissues_per_gene, unique_tissue_count)

        duplicate_tissue_pair_count = 0
        collapsed_values = []

        for tissue, values in sorted(tissue_map.items()):
            if len(values) > 1:
                duplicate_tissue_pair_count += 1
                duplicate_gene_tissue_pairs += 1

                collapsed_value, collapse_rule = collapse_duplicate_values(values)
                if collapse_rule == "prefer_nonzero_max":
                    prefer_nonzero_pair_count += 1
                else:
                    mean_pair_count += 1

                duplicate_qc_rows.append([
                    gene_id,
                    tissue,
                    len(values),
                    collapse_rule,
                    f"{collapsed_value:.6f}",
                    ",".join(f"{x:.6f}" for x in values[:10]),
                ])
            else:
                collapsed_value = values[0]

            collapsed_values.append(collapsed_value)

        if duplicate_tissue_pair_count > 0:
            duplicate_gene_count += 1

        tissue_count = len(collapsed_values)
        if tissue_count == 0:
            continue

        log_values_all = [math.log2(1.0 + x) for x in collapsed_values]
        e_raw = statistics.fmean(log_values_all)

        detectable_values = [x for x in collapsed_values if x >= DETECTABLE_TPM_THRESHOLD]
        detectable_tissue_count = len(detectable_values)
        b_raw = detectable_tissue_count / tissue_count if tissue_count > 0 else 0.0

        is_in_main_subset = 1 if b_raw >= B_MIN_FOR_MAIN else 0

        if is_in_main_subset and detectable_tissue_count >= MIN_DETECTABLE_TISSUES_FOR_S:
            log_detectable = [math.log2(1.0 + x) for x in detectable_values]
            s_raw = -mad(log_detectable)
            s_defined_main = 1
        else:
            s_raw = None
            s_defined_main = 0

        gene_rows.append({
            "gene_id": gene_id,
            "tissue_count": tissue_count,
            "raw_row_count": raw_row_count,
            "duplicate_tissue_pair_count": duplicate_tissue_pair_count,
            "detectable_tissue_count": detectable_tissue_count,
            "B_gate_pass": is_in_main_subset,
            "is_in_main_subset": is_in_main_subset,
            "S_defined_main": s_defined_main,
            "E_raw": e_raw,
            "S_raw": s_raw,
            "B_raw": b_raw,
        })

    return {
        "gene_rows": gene_rows,
        "duplicate_qc_rows": duplicate_qc_rows,
        "rows_per_gene_counter": rows_per_gene_counter,
        "unique_tissues_per_gene_counter": unique_tissues_per_gene_counter,
        "duplicate_gene_count": duplicate_gene_count,
        "duplicate_gene_tissue_pairs": duplicate_gene_tissue_pairs,
        "prefer_nonzero_pair_count": prefer_nonzero_pair_count,
        "mean_pair_count": mean_pair_count,
        "max_raw_rows_per_gene": max_raw_rows_per_gene,
        "max_unique_tissues_per_gene": max_unique_tissues_per_gene,
    }


def assign_scores(gene_rows):
    rank_scale(gene_rows, "E_raw", "Score_E")
    rank_scale(gene_rows, "B_raw", "Score_B")
    rank_scale(gene_rows, "S_raw", "Score_S", none_value=float("-inf"))

    for row in gene_rows:
        if row["S_raw"] is None:
            row["Score_S"] = None

    for row in gene_rows:
        if row["Score_S"] is None:
            row["Score_HKG"] = None
        else:
            product = max(row["Score_E"] * row["Score_S"] * row["Score_B"], 0.0)
            row["Score_HKG"] = product ** (1.0 / 3.0)

    rank_scale(gene_rows, "Score_HKG", "Score_HKG_rank", none_value=float("-inf"))

    main_rows = [row for row in gene_rows if row["is_in_main_subset"] == 1]
    ranked_main = [row for row in main_rows if row["Score_HKG"] is not None]
    ranked_main.sort(key=lambda row: row["Score_HKG"], reverse=True)

    main_rank_map = {}
    for rank, row in enumerate(ranked_main, start=1):
        main_rank_map[row["gene_id"]] = rank

    for row in gene_rows:
        rank = main_rank_map.get(row["gene_id"])
        if rank is None:
            row["rank_by_HKG_within_main"] = ""
            row["rank_frac_within_main"] = ""
        else:
            row["rank_by_HKG_within_main"] = rank
            row["rank_frac_within_main"] = rank / len(ranked_main) if ranked_main else ""

    return main_rows


def write_duplicate_qc(rows):
    with DUPLICATE_QC_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "gene_id",
            "tissue",
            "row_count_for_gene_tissue",
            "collapse_rule",
            "collapsed_mean_tpm",
            "mean_tpm_values_preview",
        ])
        for row in rows:
            writer.writerow(row)


def write_gene_labels(gene_rows):
    gene_rows.sort(key=lambda row: row["gene_id"])

    with GENE_LABEL_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "gene_id",
            "tissue_count",
            "raw_row_count",
            "duplicate_tissue_pair_count",
            "detectable_tissue_count",
            "B_gate_pass",
            "is_in_main_subset",
            "S_defined_main",
            "B_min_used",
            "E_raw",
            "S_raw",
            "B_raw",
            "Score_E",
            "Score_S",
            "Score_B",
            "Score_HKG",
            "Score_HKG_rank",
            "rank_by_HKG_within_main",
            "rank_frac_within_main",
        ])

        for row in gene_rows:
            writer.writerow([
                row["gene_id"],
                row["tissue_count"],
                row["raw_row_count"],
                row["duplicate_tissue_pair_count"],
                row["detectable_tissue_count"],
                row["B_gate_pass"],
                row["is_in_main_subset"],
                row["S_defined_main"],
                f"{B_MIN_FOR_MAIN:.6f}",
                f"{row['E_raw']:.6f}",
                "" if row["S_raw"] is None else f"{row['S_raw']:.6f}",
                f"{row['B_raw']:.6f}",
                f"{row['Score_E']:.6f}",
                "" if row["Score_S"] is None else f"{row['Score_S']:.6f}",
                f"{row['Score_B']:.6f}",
                "" if row["Score_HKG"] is None else f"{row['Score_HKG']:.6f}",
                "" if row["Score_HKG_rank"] is None else f"{row['Score_HKG_rank']:.6f}",
                row["rank_by_HKG_within_main"],
                "" if row["rank_frac_within_main"] == "" else f"{row['rank_frac_within_main']:.6f}",
            ])


def write_subset(path, rows):
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_promoter_labels(gene_rows):
    gene_map = {row["gene_id"]: row for row in gene_rows}

    with PROMOTER_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        raw_header = next(reader)
    normalized_header = [normalize_fieldname(x) for x in raw_header]

    promoter_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    core_rows = []
    extended_rows = []

    window_counts = Counter()
    tier_counts = Counter()
    window_matched_counts = Counter()
    tier_matched_counts = Counter()

    gene_name_map = {}

    out_fields = normalized_header + [
        "gene_id_base",
        "label_matched",
        "tissue_count_label",
        "raw_row_count_label",
        "duplicate_tissue_pair_count",
        "detectable_tissue_count",
        "B_gate_pass",
        "is_in_main_subset",
        "S_defined_main",
        "B_min_used",
        "E_raw",
        "S_raw",
        "B_raw",
        "Score_E",
        "Score_S",
        "Score_B",
        "Score_HKG",
        "Score_HKG_rank",
        "rank_by_HKG_within_main",
        "rank_frac_within_main",
    ]

    with PROMOTER_TSV.open("r", encoding="utf-8", newline="") as fin, PROMOTER_LABEL_TSV.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        next(reader)

        writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()

        for parts in reader:
            promoter_rows += 1
            row = dict(zip(normalized_header, parts))

            gene_id = row.get("gene_id", "")
            gene_name = row.get("gene_name", "")
            gene_id_base = base_gene_id(gene_id)
            hit = gene_map.get(gene_id_base)

            if gene_id_base and gene_name and gene_id_base not in gene_name_map:
                gene_name_map[gene_id_base] = gene_name

            window_name = row.get("window_name", "")
            analysis_tier = row.get("analysis_tier", "")

            window_counts[window_name] += 1
            tier_counts[analysis_tier] += 1

            row["gene_id_base"] = gene_id_base

            if hit is None:
                unmatched_rows += 1
                row["label_matched"] = "0"
                row["tissue_count_label"] = ""
                row["raw_row_count_label"] = ""
                row["duplicate_tissue_pair_count"] = ""
                row["detectable_tissue_count"] = ""
                row["B_gate_pass"] = ""
                row["is_in_main_subset"] = ""
                row["S_defined_main"] = ""
                row["B_min_used"] = f"{B_MIN_FOR_MAIN:.6f}"
                row["E_raw"] = ""
                row["S_raw"] = ""
                row["B_raw"] = ""
                row["Score_E"] = ""
                row["Score_S"] = ""
                row["Score_B"] = ""
                row["Score_HKG"] = ""
                row["Score_HKG_rank"] = ""
                row["rank_by_HKG_within_main"] = ""
                row["rank_frac_within_main"] = ""
            else:
                matched_rows += 1
                row["label_matched"] = "1"
                row["tissue_count_label"] = str(hit["tissue_count"])
                row["raw_row_count_label"] = str(hit["raw_row_count"])
                row["duplicate_tissue_pair_count"] = str(hit["duplicate_tissue_pair_count"])
                row["detectable_tissue_count"] = str(hit["detectable_tissue_count"])
                row["B_gate_pass"] = str(hit["B_gate_pass"])
                row["is_in_main_subset"] = str(hit["is_in_main_subset"])
                row["S_defined_main"] = str(hit["S_defined_main"])
                row["B_min_used"] = f"{B_MIN_FOR_MAIN:.6f}"
                row["E_raw"] = f"{hit['E_raw']:.6f}"
                row["S_raw"] = "" if hit["S_raw"] is None else f"{hit['S_raw']:.6f}"
                row["B_raw"] = f"{hit['B_raw']:.6f}"
                row["Score_E"] = f"{hit['Score_E']:.6f}"
                row["Score_S"] = "" if hit["Score_S"] is None else f"{hit['Score_S']:.6f}"
                row["Score_B"] = f"{hit['Score_B']:.6f}"
                row["Score_HKG"] = "" if hit["Score_HKG"] is None else f"{hit['Score_HKG']:.6f}"
                row["Score_HKG_rank"] = "" if hit["Score_HKG_rank"] is None else f"{hit['Score_HKG_rank']:.6f}"
                row["rank_by_HKG_within_main"] = hit["rank_by_HKG_within_main"]
                row["rank_frac_within_main"] = "" if hit["rank_frac_within_main"] == "" else f"{hit['rank_frac_within_main']:.6f}"

                window_matched_counts[window_name] += 1
                tier_matched_counts[analysis_tier] += 1

                if row.get("core_model_pass", "0") == "1":
                    core_rows.append(dict(row))
                if row.get("extended_context_pass", "0") == "1":
                    extended_rows.append(dict(row))

            writer.writerow(row)

    write_subset(CORE_INPUT_TSV, core_rows)
    write_subset(EXTENDED_INPUT_TSV, extended_rows)

    return {
        "promoter_rows": promoter_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "core_rows": core_rows,
        "extended_rows": extended_rows,
        "window_counts": window_counts,
        "tier_counts": tier_counts,
        "window_matched_counts": window_matched_counts,
        "tier_matched_counts": tier_matched_counts,
        "gene_name_map": gene_name_map,
    }


def write_qc(gene_rows, main_rows, build_stats, promoter_stats, distinct_tissue_count):
    with QC_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["gene_label_rows", len(gene_rows)])
        writer.writerow(["distinct_tissues_in_expression_table", distinct_tissue_count])
        writer.writerow(["duplicate_gene_count", build_stats["duplicate_gene_count"]])
        writer.writerow(["duplicate_gene_tissue_pairs", build_stats["duplicate_gene_tissue_pairs"]])
        writer.writerow(["prefer_nonzero_pair_count", build_stats["prefer_nonzero_pair_count"]])
        writer.writerow(["mean_pair_count", build_stats["mean_pair_count"]])
        writer.writerow(["max_raw_rows_per_gene", build_stats["max_raw_rows_per_gene"]])
        writer.writerow(["max_unique_tissues_per_gene", build_stats["max_unique_tissues_per_gene"]])
        writer.writerow(["main_subset_gene_rows", len(main_rows)])
        writer.writerow(["promoter_rows", promoter_stats["promoter_rows"]])
        writer.writerow(["matched_promoter_rows", promoter_stats["matched_rows"]])
        writer.writerow(["unmatched_promoter_rows", promoter_stats["unmatched_rows"]])
        writer.writerow([
            "matched_promoter_frac",
            f"{(promoter_stats['matched_rows'] / promoter_stats['promoter_rows']):.6f}" if promoter_stats["promoter_rows"] else "0.000000",
        ])
        writer.writerow(["detectable_tpm_threshold_for_B", DETECTABLE_TPM_THRESHOLD])
        writer.writerow(["B_min_for_main", f"{B_MIN_FOR_MAIN:.6f}"])
        writer.writerow(["min_detectable_tissues_for_S", MIN_DETECTABLE_TISSUES_FOR_S])

        for key, count in build_stats["rows_per_gene_counter"].most_common(20):
            writer.writerow([f"raw_rows_per_gene::{key}", count])

        for key, count in build_stats["unique_tissues_per_gene_counter"].most_common(20):
            writer.writerow([f"unique_tissues_per_gene::{key}", count])

        for key in sorted(promoter_stats["window_counts"]):
            writer.writerow([f"window_count::{key}", promoter_stats["window_counts"][key]])

        for key in sorted(promoter_stats["window_matched_counts"]):
            writer.writerow([f"window_matched_count::{key}", promoter_stats["window_matched_counts"][key]])

        for key in sorted(promoter_stats["tier_counts"]):
            writer.writerow([f"analysis_tier_count::{key}", promoter_stats["tier_counts"][key]])

        for key in sorted(promoter_stats["tier_matched_counts"]):
            writer.writerow([f"analysis_tier_matched_count::{key}", promoter_stats["tier_matched_counts"][key]])


def write_sanity(gene_rows, main_rows):
    e_raw_values = collect_values(gene_rows, "E_raw")
    b_raw_values = collect_values(gene_rows, "B_raw")
    s_raw_defined_values = collect_values(gene_rows, "S_raw")
    score_e_values = collect_values(gene_rows, "Score_E")
    score_s_values = collect_values(gene_rows, "Score_S")
    score_b_values = collect_values(gene_rows, "Score_B")
    score_hkg_values = collect_values(gene_rows, "Score_HKG")

    main_e_raw_values = collect_values(main_rows, "E_raw", require_defined=True)
    main_b_raw_values = collect_values(main_rows, "B_raw", require_defined=True)
    main_s_raw_values = collect_values(main_rows, "S_raw", require_defined=True)
    main_score_e_values = collect_values(main_rows, "Score_E", require_defined=True)
    main_score_s_values = collect_values(main_rows, "Score_S", require_defined=True)
    main_score_b_values = collect_values(main_rows, "Score_B", require_defined=True)
    main_score_hkg_values = collect_values(main_rows, "Score_HKG", require_defined=True)

    with SANITY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["subset", "row_type", "name", "count", "mean", "min", "q25", "median", "q75", "max", "value"])

        distributions = [
            ("global", [
                ("E_raw", e_raw_values),
                ("S_raw_defined_only", s_raw_defined_values),
                ("B_raw", b_raw_values),
                ("Score_E", score_e_values),
                ("Score_S", score_s_values),
                ("Score_B", score_b_values),
                ("Score_HKG", score_hkg_values),
            ]),
            ("main_subset", [
                ("E_raw", main_e_raw_values),
                ("S_raw", main_s_raw_values),
                ("B_raw", main_b_raw_values),
                ("Score_E", main_score_e_values),
                ("Score_S", main_score_s_values),
                ("Score_B", main_score_b_values),
                ("Score_HKG", main_score_hkg_values),
            ]),
        ]

        for subset_name, rows in distributions:
            for name, values in rows:
                q0, q25, q50, q75, q100 = quantiles(values)
                mean_value = statistics.fmean(values) if values else 0.0
                writer.writerow([
                    subset_name,
                    "distribution",
                    name,
                    len(values),
                    f"{mean_value:.6f}",
                    f"{q0:.6f}",
                    f"{q25:.6f}",
                    f"{q50:.6f}",
                    f"{q75:.6f}",
                    f"{q100:.6f}",
                    "",
                ])

        global_defined_rows = [row for row in gene_rows if row["S_raw"] is not None]

        global_e_for_s = [row["E_raw"] for row in global_defined_rows]
        global_s_only = [row["S_raw"] for row in global_defined_rows]
        global_b_for_s = [row["B_raw"] for row in global_defined_rows]
        global_score_e_for_s = [row["Score_E"] for row in global_defined_rows]
        global_score_s_only = [row["Score_S"] for row in global_defined_rows]
        global_score_b_for_s = [row["Score_B"] for row in global_defined_rows]
        global_score_hkg_for_s = [row["Score_HKG"] for row in global_defined_rows if row["Score_HKG"] is not None]

        correlations = [
            ("global", [
                ("corr::E_raw__S_raw_defined_only", safe_corr(global_e_for_s, global_s_only)),
                ("corr::E_raw__B_raw", safe_corr(e_raw_values, b_raw_values)),
                ("corr::S_raw__B_raw_defined_only", safe_corr(global_s_only, global_b_for_s)),
                ("corr::Score_E__Score_S_defined_only", safe_corr(global_score_e_for_s, global_score_s_only)),
                ("corr::Score_E__Score_B", safe_corr(score_e_values, score_b_values)),
                ("corr::Score_S__Score_B_defined_only", safe_corr(global_score_s_only, global_score_b_for_s)),
                ("corr::Score_HKG__Score_E", safe_corr(score_hkg_values, score_e_values[:len(score_hkg_values)] if len(score_hkg_values) <= len(score_e_values) else score_e_values)),
                ("corr::Score_HKG__Score_S_defined_only", safe_corr(global_score_hkg_for_s, global_score_s_only[:len(global_score_hkg_for_s)] if len(global_score_hkg_for_s) <= len(global_score_s_only) else global_score_s_only)),
                ("corr::Score_HKG__Score_B", safe_corr(score_hkg_values, score_b_values[:len(score_hkg_values)] if len(score_hkg_values) <= len(score_b_values) else score_b_values)),
            ]),
            ("main_subset", [
                ("corr::E_raw__S_raw", safe_corr(main_e_raw_values, main_s_raw_values)),
                ("corr::E_raw__B_raw", safe_corr(main_e_raw_values, main_b_raw_values)),
                ("corr::S_raw__B_raw", safe_corr(main_s_raw_values, main_b_raw_values)),
                ("corr::Score_E__Score_S", safe_corr(main_score_e_values, main_score_s_values)),
                ("corr::Score_E__Score_B", safe_corr(main_score_e_values, main_score_b_values)),
                ("corr::Score_S__Score_B", safe_corr(main_score_s_values, main_score_b_values)),
                ("corr::Score_HKG__Score_E", safe_corr(main_score_hkg_values, main_score_e_values)),
                ("corr::Score_HKG__Score_S", safe_corr(main_score_hkg_values, main_score_s_values)),
                ("corr::Score_HKG__Score_B", safe_corr(main_score_hkg_values, main_score_b_values)),
            ]),
        ]

        for subset_name, rows in correlations:
            for name, value in rows:
                writer.writerow([
                    subset_name,
                    "correlation",
                    name,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"{value:.6f}",
                ])


def write_anchor_qc(gene_rows, gene_name_map):
    ranked_rows = [row for row in gene_rows if row["Score_HKG"] is not None]
    ranked_rows.sort(key=lambda row: row["Score_HKG"], reverse=True)

    gene_rank_map = {row["gene_id"]: idx + 1 for idx, row in enumerate(ranked_rows)}
    total_ranked = len(ranked_rows)

    anchor_rows = []
    for row in gene_rows:
        gene_id = row["gene_id"]
        gene_name = gene_name_map.get(gene_id, "")
        if gene_name not in ANCHOR_NAMES:
            continue

        rank = gene_rank_map.get(gene_id, "")
        rank_frac = ""
        if rank != "" and total_ranked > 0:
            rank_frac = rank / total_ranked

        anchor_rows.append({
            "gene_name": gene_name,
            "gene_id": gene_id,
            "is_in_main_subset": row["is_in_main_subset"],
            "B_gate_pass": row["B_gate_pass"],
            "rank_by_HKG": rank,
            "rank_frac": rank_frac,
            "rank_by_HKG_within_main": row["rank_by_HKG_within_main"],
            "rank_frac_within_main": row["rank_frac_within_main"],
            "Score_E": row["Score_E"],
            "Score_S": row["Score_S"],
            "Score_B": row["Score_B"],
            "Score_HKG": row["Score_HKG"],
            "E_raw": row["E_raw"],
            "S_raw": row["S_raw"],
            "B_raw": row["B_raw"],
            "tissue_count": row["tissue_count"],
            "detectable_tissue_count": row["detectable_tissue_count"],
            "S_defined_main": row["S_defined_main"],
        })

    anchor_rows.sort(key=lambda row: (row["rank_by_HKG"] == "", row["rank_by_HKG"]))

    with ANCHOR_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "gene_name",
            "gene_id",
            "is_in_main_subset",
            "B_gate_pass",
            "rank_by_HKG",
            "rank_frac",
            "rank_by_HKG_within_main",
            "rank_frac_within_main",
            "Score_E",
            "Score_S",
            "Score_B",
            "Score_HKG",
            "E_raw",
            "S_raw",
            "B_raw",
            "tissue_count",
            "detectable_tissue_count",
            "S_defined_main",
        ])

        for row in anchor_rows:
            writer.writerow([
                row["gene_name"],
                row["gene_id"],
                row["is_in_main_subset"],
                row["B_gate_pass"],
                row["rank_by_HKG"],
                "" if row["rank_frac"] == "" else f"{row['rank_frac']:.6f}",
                row["rank_by_HKG_within_main"],
                "" if row["rank_frac_within_main"] == "" else f"{row['rank_frac_within_main']:.6f}",
                f"{row['Score_E']:.6f}",
                "" if row["Score_S"] is None else f"{row['Score_S']:.6f}",
                f"{row['Score_B']:.6f}",
                "" if row["Score_HKG"] is None else f"{row['Score_HKG']:.6f}",
                f"{row['E_raw']:.6f}",
                "" if row["S_raw"] is None else f"{row['S_raw']:.6f}",
                f"{row['B_raw']:.6f}",
                row["tissue_count"],
                row["detectable_tissue_count"],
                row["S_defined_main"],
            ])


def write_manifest(promoter_stats, distinct_tissue_count, build_stats):
    with MANIFEST_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])

        manifest_rows = [
            ("baseline_name", "promoter_only_candidate_E"),
            ("label_version", "candidate_E_long_table_B_gated_detectable_tissue_MAD"),
            ("gene_expression_input_path", EXPR_TSV.as_posix()),
            ("gene_expression_input_mode", "long_format_gene_tissue_mean_tpm"),
            ("duplicate_resolution_rule", "prefer_nonzero_when_zero_and_nonzero_coexist_else_mean"),
            ("core_input_path", CORE_INPUT_TSV.as_posix()),
            ("extended_input_path", EXTENDED_INPUT_TSV.as_posix()),
            ("core_row_count", str(len(promoter_stats["core_rows"]))),
            ("extended_row_count", str(len(promoter_stats["extended_rows"]))),
            ("recommended_first_training_set", "core_input"),
            ("primary_targets", "Score_E,Score_S,Score_B"),
            ("ranking_only_target", "Score_HKG"),
            ("recommended_window", "up200"),
            ("recommended_filter", "label_matched=1 AND core_model_pass=1"),
            ("detectable_tpm_threshold_for_B", str(DETECTABLE_TPM_THRESHOLD)),
            ("B_min_for_main", f"{B_MIN_FOR_MAIN:.6f}"),
            ("min_detectable_tissues_for_S", str(MIN_DETECTABLE_TISSUES_FOR_S)),
            ("main_subset_definition", "B_raw >= B_min_for_main"),
            ("S_definition_domain", "main_subset AND detectable_tissue_count >= min_detectable_tissues_for_S"),
            ("duplicate_gene_tissue_pairs", str(build_stats["duplicate_gene_tissue_pairs"])),
            ("duplicate_gene_count", str(build_stats["duplicate_gene_count"])),
            ("prefer_nonzero_pair_count", str(build_stats["prefer_nonzero_pair_count"])),
            ("mean_pair_count", str(build_stats["mean_pair_count"])),
            ("distinct_tissues_in_expression_table", str(distinct_tissue_count)),
            ("score_s_missing_when_s_undefined", "true"),
            ("score_hkg_missing_when_score_s_missing", "true"),
        ]

        for key, value in manifest_rows:
            writer.writerow([key, value])


def write_notes():
    label_lines = [
        "# Label Build Notes",
        "",
        "- current label system: candidate_E_long_table_B_gated_detectable_tissue_MAD",
        "- gene expression input is long-format gene x tissue mean_tpm",
        "- repeated gene_id+tissue rows use nonzero-preferred collapse when zero and nonzero coexist",
        "- Score_S remains empty when S_raw is undefined",
        "- Score_HKG remains empty when Score_S is undefined",
    ]
    LABEL_NOTES_MD.write_text("\n".join(label_lines), encoding="utf-8")

    sanity_lines = [
        "# Gene Label Sanity Notes",
        "",
        "- inspect both global and main-subset distributions",
        "- inspect housekeeping anchors before split and baseline training",
        "- use S_defined_main to distinguish defined vs undefined stability targets",
    ]
    SANITY_NOTES_MD.write_text("\n".join(sanity_lines), encoding="utf-8")


def main():
    ensure_directories()
    validate_inputs()

    print("Loading gene expression table...")
    gene_tissue_values, gene_total_rows, all_tissues = load_expression_table()
    print(f"Genes loaded: {len(gene_tissue_values)}")
    print(f"Distinct tissues: {len(all_tissues)}")

    build_stats = build_gene_rows(gene_tissue_values, gene_total_rows)
    gene_rows = build_stats["gene_rows"]

    main_rows = assign_scores(gene_rows)

    write_duplicate_qc(build_stats["duplicate_qc_rows"])
    write_gene_labels(gene_rows)

    print("Reading promoter analysis set...")
    promoter_stats = build_promoter_labels(gene_rows)

    write_qc(gene_rows, main_rows, build_stats, promoter_stats, len(all_tissues))
    write_sanity(gene_rows, main_rows)
    write_anchor_qc(gene_rows, promoter_stats["gene_name_map"])
    write_manifest(promoter_stats, len(all_tissues), build_stats)
    write_notes()

    print("Candidate E rebuild complete.")
    print(f"Gene label rows: {len(gene_rows)}")
    print(f"Main subset gene rows: {len(main_rows)}")
    print(f"Distinct tissues: {len(all_tissues)}")
    print(f"Duplicate gene count: {build_stats['duplicate_gene_count']}")
    print(f"Duplicate gene+tissue pairs: {build_stats['duplicate_gene_tissue_pairs']}")
    print(f"Prefer-nonzero pairs: {build_stats['prefer_nonzero_pair_count']}")
    print(f"Mean-collapse duplicate pairs: {build_stats['mean_pair_count']}")
    print(f"Promoter rows: {promoter_stats['promoter_rows']}")
    print(f"Matched promoter rows: {promoter_stats['matched_rows']}")
    print(f"Unmatched promoter rows: {promoter_stats['unmatched_rows']}")
    print(f"Core model rows: {len(promoter_stats['core_rows'])}")
    print(f"Extended model rows: {len(promoter_stats['extended_rows'])}")


if __name__ == "__main__":
    main()
