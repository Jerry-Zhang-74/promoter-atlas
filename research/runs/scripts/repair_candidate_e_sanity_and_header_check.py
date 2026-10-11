"""Repair sanity statistics and inspect promoter label header integrity."""

from pathlib import Path
import csv
import math
import statistics


ROOT = Path(".")

GENE_LABEL_TSV = ROOT / "labels/build/gene_expression_label_table.tsv"
PROMOTER_LABEL_TSV = ROOT / "labels/build/promoter_label_table.tsv"

SANITY_TSV = ROOT / "labels/qc/gene_label_sanity_stats.tsv"
HEADER_CHECK_TSV = ROOT / "labels/qc/promoter_label_header_check.tsv"
SUMMARY_TSV = ROOT / "labels/qc/repair_candidate_e_sanity_summary.tsv"
NOTES_MD = ROOT / "docs/notes/repair_candidate_e_sanity_and_header_check.md"


def safe_float(value):
    """Convert a string value to float, preserving missing values as None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def safe_int(value):
    """Convert a string value to int, preserving missing values as None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def quantiles(values):
    """Return min, q25, median, q75, max for a numeric list."""
    values = sorted(values)
    n = len(values)

    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def pick(p):
        idx = int(round((n - 1) * p))
        idx = min(n - 1, max(0, idx))
        return values[idx]

    return values[0], pick(0.25), pick(0.50), pick(0.75), values[-1]


def safe_corr(xs, ys):
    """Return Pearson correlation if both vectors are valid."""
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


def paired_values(rows, key_x, key_y):
    """Collect paired non-missing values for two keys."""
    xs = []
    ys = []

    for row in rows:
        x = row.get(key_x)
        y = row.get(key_y)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    return xs, ys


def collect_values(rows, key):
    """Collect non-missing values for a single key."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(value)
    return values


def load_gene_rows():
    """Load gene-level label table into a list of typed records."""
    rows = []

    with GENE_LABEL_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({
                "gene_id": str(row["gene_id"]).strip(),
                "tissue_count": safe_int(row.get("tissue_count")),
                "raw_row_count": safe_int(row.get("raw_row_count")),
                "duplicate_tissue_pair_count": safe_int(row.get("duplicate_tissue_pair_count")),
                "detectable_tissue_count": safe_int(row.get("detectable_tissue_count")),
                "B_gate_pass": safe_int(row.get("B_gate_pass")),
                "is_in_main_subset": safe_int(row.get("is_in_main_subset")),
                "S_defined_main": safe_int(row.get("S_defined_main")),
                "E_raw": safe_float(row.get("E_raw")),
                "S_raw": safe_float(row.get("S_raw")),
                "B_raw": safe_float(row.get("B_raw")),
                "Score_E": safe_float(row.get("Score_E")),
                "Score_S": safe_float(row.get("Score_S")),
                "Score_B": safe_float(row.get("Score_B")),
                "Score_HKG": safe_float(row.get("Score_HKG")),
            })

    return rows


def write_sanity(rows):
    """Recompute sanity statistics using strictly paired non-missing values."""
    main_rows = [row for row in rows if row["S_defined_main"] == 1]

    global_distributions = [
        ("E_raw", collect_values(rows, "E_raw")),
        ("S_raw_defined_only", collect_values(main_rows, "S_raw")),
        ("B_raw", collect_values(rows, "B_raw")),
        ("Score_E", collect_values(rows, "Score_E")),
        ("Score_S", collect_values(main_rows, "Score_S")),
        ("Score_B", collect_values(rows, "Score_B")),
        ("Score_HKG", collect_values(main_rows, "Score_HKG")),
    ]

    main_distributions = [
        ("E_raw", collect_values(main_rows, "E_raw")),
        ("S_raw", collect_values(main_rows, "S_raw")),
        ("B_raw", collect_values(main_rows, "B_raw")),
        ("Score_E", collect_values(main_rows, "Score_E")),
        ("Score_S", collect_values(main_rows, "Score_S")),
        ("Score_B", collect_values(main_rows, "Score_B")),
        ("Score_HKG", collect_values(main_rows, "Score_HKG")),
    ]

    global_corr_pairs = [
        ("corr::E_raw__S_raw_defined_only", *paired_values(main_rows, "E_raw", "S_raw")),
        ("corr::E_raw__B_raw", *paired_values(rows, "E_raw", "B_raw")),
        ("corr::S_raw__B_raw_defined_only", *paired_values(main_rows, "S_raw", "B_raw")),
        ("corr::Score_E__Score_S_defined_only", *paired_values(main_rows, "Score_E", "Score_S")),
        ("corr::Score_E__Score_B", *paired_values(rows, "Score_E", "Score_B")),
        ("corr::Score_S__Score_B_defined_only", *paired_values(main_rows, "Score_S", "Score_B")),
        ("corr::Score_HKG__Score_E", *paired_values(main_rows, "Score_HKG", "Score_E")),
        ("corr::Score_HKG__Score_S_defined_only", *paired_values(main_rows, "Score_HKG", "Score_S")),
        ("corr::Score_HKG__Score_B", *paired_values(main_rows, "Score_HKG", "Score_B")),
    ]

    main_corr_pairs = [
        ("corr::E_raw__S_raw", *paired_values(main_rows, "E_raw", "S_raw")),
        ("corr::E_raw__B_raw", *paired_values(main_rows, "E_raw", "B_raw")),
        ("corr::S_raw__B_raw", *paired_values(main_rows, "S_raw", "B_raw")),
        ("corr::Score_E__Score_S", *paired_values(main_rows, "Score_E", "Score_S")),
        ("corr::Score_E__Score_B", *paired_values(main_rows, "Score_E", "Score_B")),
        ("corr::Score_S__Score_B", *paired_values(main_rows, "Score_S", "Score_B")),
        ("corr::Score_HKG__Score_E", *paired_values(main_rows, "Score_HKG", "Score_E")),
        ("corr::Score_HKG__Score_S", *paired_values(main_rows, "Score_HKG", "Score_S")),
        ("corr::Score_HKG__Score_B", *paired_values(main_rows, "Score_HKG", "Score_B")),
    ]

    with SANITY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["subset", "row_type", "name", "count", "mean", "min", "q25", "median", "q75", "max", "value"])

        for subset_name, distributions in [
            ("global", global_distributions),
            ("main_subset", main_distributions),
        ]:
            for name, values in distributions:
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

        for subset_name, correlations in [
            ("global", global_corr_pairs),
            ("main_subset", main_corr_pairs),
        ]:
            for name, xs, ys in correlations:
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
                    f"{safe_corr(xs, ys):.6f}",
                ])

    return len(main_rows)


def inspect_promoter_header():
    """Check promoter label header integrity and column count."""
    with PROMOTER_LABEL_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)

    suspicious_columns = []
    for idx, column in enumerate(header):
        if "rank_by_HKG_within_main" in column and column != "rank_by_HKG_within_main":
            suspicious_columns.append((idx, column))
        if "Score_HKG_rank" in column and column != "Score_HKG_rank":
            suspicious_columns.append((idx, column))

    with HEADER_CHECK_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["column_index", "column_name", "is_suspicious"])
        for idx, column in enumerate(header):
            is_suspicious = 1 if any(idx == s_idx for s_idx, _ in suspicious_columns) else 0
            writer.writerow([idx, column, is_suspicious])

    return header, suspicious_columns


def write_summary(total_rows, main_rows, header, suspicious_columns):
    """Write a compact repair summary."""
    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["gene_label_rows", total_rows])
        writer.writerow(["main_rows_with_defined_s", main_rows])
        writer.writerow(["promoter_label_column_count", len(header)])
        writer.writerow(["promoter_label_suspicious_column_count", len(suspicious_columns)])

    notes_lines = [
        "# Repair Candidate E Sanity And Header Check",
        "",
        "- recomputed sanity correlations using strictly paired non-missing values",
        "- checked promoter label header integrity after Score_S / Score_HKG missing-value update",
    ]
    NOTES_MD.write_text("\n".join(notes_lines), encoding="utf-8")


def main():
    if not GENE_LABEL_TSV.exists():
        raise SystemExit("ERROR: missing labels/build/gene_expression_label_table.tsv")
    if not PROMOTER_LABEL_TSV.exists():
        raise SystemExit("ERROR: missing labels/build/promoter_label_table.tsv")

    print("Loading gene label table...")
    rows = load_gene_rows()
    print(f"Gene rows loaded: {len(rows)}")

    print("Recomputing sanity statistics...")
    main_row_count = write_sanity(rows)
    print(f"Main rows with defined S: {main_row_count}")

    print("Inspecting promoter label header...")
    header, suspicious_columns = inspect_promoter_header()
    print(f"Promoter label column count: {len(header)}")
    print(f"Suspicious header columns: {len(suspicious_columns)}")

    write_summary(len(rows), main_row_count, header, suspicious_columns)

    print("Repair complete.")
    print(f"Wrote sanity: {SANITY_TSV.as_posix()}")
    print(f"Wrote header check: {HEADER_CHECK_TSV.as_posix()}")
    print(f"Wrote summary: {SUMMARY_TSV.as_posix()}")


if __name__ == "__main__":
    main()
