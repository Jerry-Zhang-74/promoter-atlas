from pathlib import Path
import csv
from collections import Counter, defaultdict

root = Path(".")

expr_tsv = root / "labels/build/gene_expression_table.tsv"
manifest_tsv = root / "labels/build/baseline_train_manifest.tsv"
report_tsv = root / "labels/qc/gene_expression_table_structure_report.tsv"
report_txt = root / "docs/notes/gene_expression_table_structure_report.txt"

for p in [
    root / "labels/qc",
    root / "docs/notes",
]:
    p.mkdir(parents=True, exist_ok=True)

if not expr_tsv.exists():
    raise SystemExit("ERROR: missing labels/build/gene_expression_table.tsv")

def normalize(x):
    return x.strip().strip('"').strip()

def is_float_like(x):
    if x is None:
        return False
    x = str(x).strip()
    if x == "":
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False

print("Inspecting gene expression table...")
print(f"INPUT: {expr_tsv.as_posix()}")

with expr_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    if reader.fieldnames is None:
        raise SystemExit("ERROR: gene_expression_table.tsv has no header")
    raw_fieldnames = list(reader.fieldnames)
    rows = list(reader)

fieldnames = [normalize(x) for x in raw_fieldnames]
fieldname_map = dict(zip(raw_fieldnames, fieldnames))

print("Header column count:", len(fieldnames))
print("Row count:", len(rows))
print("Header:", fieldnames)

if not rows:
    raise SystemExit("ERROR: gene_expression_table.tsv is empty")

has_gene_id = "gene_id" in fieldnames
has_mean_tpm = "mean_tpm" in fieldnames
has_tissue = ("tissue" in fieldnames) or ("tissue_id" in fieldnames) or ("tissue_name" in fieldnames)

numeric_counts = Counter()
non_numeric_counts = Counter()

sample_value_types = []

for raw_k in raw_fieldnames:
    k = fieldname_map[raw_k]
    seen_numeric = 0
    seen_non_numeric = 0
    examples = []
    for r in rows[:200]:
        v = r.get(raw_k)
        if v is None or str(v).strip() == "":
            continue
        if is_float_like(v):
            seen_numeric += 1
        else:
            seen_non_numeric += 1
            if len(examples) < 3:
                examples.append(str(v))
    numeric_counts[k] = seen_numeric
    non_numeric_counts[k] = seen_non_numeric
    sample_value_types.append((k, seen_numeric, seen_non_numeric, examples))

gene_id_raw = None
for raw_k in raw_fieldnames:
    if fieldname_map[raw_k] == "gene_id":
        gene_id_raw = raw_k
        break

if gene_id_raw is None:
    raise SystemExit("ERROR: gene_expression_table.tsv has no gene_id column")

gene_counts = Counter()
gene_examples = defaultdict(list)

for r in rows:
    gid = str(r[gene_id_raw]).split(".")[0]
    gene_counts[gid] += 1
    if len(gene_examples[gid]) < 5:
        gene_examples[gid].append(r)

distinct_genes = len(gene_counts)
row_per_gene_counter = Counter(gene_counts.values())

print("Distinct genes:", distinct_genes)
print("Rows per gene distribution (top 20):", row_per_gene_counter.most_common(20))

top_gene_counts = list(row_per_gene_counter.items())[:20]

likely_numeric_columns = []
for raw_k in raw_fieldnames:
    k = fieldname_map[raw_k]
    if k == "gene_id":
        continue
    if numeric_counts[k] > 0 and non_numeric_counts[k] == 0:
        likely_numeric_columns.append(k)

print("Likely numeric columns:", likely_numeric_columns)

table_mode = "unknown"
if has_mean_tpm:
    table_mode = "long_mean_tpm"
elif len(likely_numeric_columns) >= 10:
    table_mode = "wide_gene_by_tissue"
elif len(likely_numeric_columns) <= 5:
    table_mode = "compressed_or_summary_table"

print("Inferred table mode:", table_mode)

notes = []
notes.append(f"input_path\t{expr_tsv.as_posix()}")
notes.append(f"row_count\t{len(rows)}")
notes.append(f"header_count\t{len(fieldnames)}")
notes.append(f"distinct_genes\t{distinct_genes}")
notes.append(f"has_gene_id\t{int(has_gene_id)}")
notes.append(f"has_mean_tpm\t{int(has_mean_tpm)}")
notes.append(f"has_tissue_like_column\t{int(has_tissue)}")
notes.append(f"inferred_table_mode\t{table_mode}")
notes.append(f"likely_numeric_column_count\t{len(likely_numeric_columns)}")
notes.append(f"likely_numeric_columns\t{','.join(likely_numeric_columns)}")

with report_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["section", "name", "value"])

    writer.writerow(["summary", "input_path", expr_tsv.as_posix()])
    writer.writerow(["summary", "row_count", len(rows)])
    writer.writerow(["summary", "header_count", len(fieldnames)])
    writer.writerow(["summary", "distinct_genes", distinct_genes])
    writer.writerow(["summary", "has_gene_id", int(has_gene_id)])
    writer.writerow(["summary", "has_mean_tpm", int(has_mean_tpm)])
    writer.writerow(["summary", "has_tissue_like_column", int(has_tissue)])
    writer.writerow(["summary", "inferred_table_mode", table_mode])
    writer.writerow(["summary", "likely_numeric_column_count", len(likely_numeric_columns)])
    writer.writerow(["summary", "likely_numeric_columns", ",".join(likely_numeric_columns)])

    for k, count in row_per_gene_counter.most_common(20):
        writer.writerow(["rows_per_gene_distribution", str(k), count])

    for k, n_num, n_non, examples in sample_value_types:
        writer.writerow(["column_type_scan", k, f"numeric={n_num};non_numeric={n_non};examples={' | '.join(examples)}"])

example_gene_ids = []
for gid, count in gene_counts.most_common(10):
    example_gene_ids.append((gid, count))

report_lines = []
report_lines.append("# Gene Expression Table Structure Report")
report_lines.append("")
report_lines.append(f"- input: {expr_tsv.as_posix()}")
report_lines.append(f"- row_count: {len(rows)}")
report_lines.append(f"- header_count: {len(fieldnames)}")
report_lines.append(f"- distinct_genes: {distinct_genes}")
report_lines.append(f"- inferred_table_mode: {table_mode}")
report_lines.append("")
report_lines.append("## Header")
report_lines.append("")
report_lines.append(", ".join(fieldnames))
report_lines.append("")
report_lines.append("## Rows per gene distribution")
report_lines.append("")
for k, count in row_per_gene_counter.most_common(20):
    report_lines.append(f"- rows_per_gene={k}: genes={count}")
report_lines.append("")
report_lines.append("## Likely numeric columns")
report_lines.append("")
for k in likely_numeric_columns:
    report_lines.append(f"- {k}")
report_lines.append("")
report_lines.append("## Column scan")
report_lines.append("")
for k, n_num, n_non, examples in sample_value_types:
    report_lines.append(f"- {k}: numeric={n_num}, non_numeric={n_non}, examples={examples}")
report_lines.append("")
report_lines.append("## Interpretation")
report_lines.append("")
if table_mode == "long_mean_tpm":
    report_lines.append("- this looks like a long-format table keyed by gene_id and mean_tpm")
    if row_per_gene_counter and row_per_gene_counter.most_common(1)[0][0] <= 5:
        report_lines.append("- the rows per gene are very small, which strongly suggests tissue values were compressed or filtered")
elif table_mode == "wide_gene_by_tissue":
    report_lines.append("- this looks like a proper wide gene-by-tissue table")
else:
    report_lines.append("- this does not look like a full tissue matrix; it looks compressed or summary-like")

report_txt.write_text("\\n".join(report_lines), encoding="utf-8")

print("Wrote report:", report_tsv.as_posix())
print("Wrote note:", report_txt.as_posix())
print("Done.")
