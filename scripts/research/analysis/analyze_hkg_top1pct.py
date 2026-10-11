"""Analyze top 1% HKG genes from the formal Candidate E label table.

Outputs:
    - labels/qc/hkg_top1pct_gene_list.tsv
    - labels/qc/hkg_top1pct_summary.tsv
    - labels/qc/hkg_top1pct_gene_type_summary.tsv
    - labels/qc/hkg_top1pct_prefix_summary.tsv
    - docs/notes/temporal_stability_target_note.md
"""

from pathlib import Path
import csv
import math
import re
from collections import Counter, defaultdict


ROOT = Path(".")

GENE_LABEL_TSV = ROOT / "labels/build/gene_expression_label_table.tsv"
PROMOTER_LABEL_TSV = ROOT / "labels/build/promoter_label_table.tsv"
EISENBERG_TXT = ROOT / "data/external/list_of_HKGs.txt"

TOP_LIST_TSV = ROOT / "labels/qc/hkg_top1pct_gene_list.tsv"
SUMMARY_TSV = ROOT / "labels/qc/hkg_top1pct_summary.tsv"
GENE_TYPE_SUMMARY_TSV = ROOT / "labels/qc/hkg_top1pct_gene_type_summary.tsv"
PREFIX_SUMMARY_TSV = ROOT / "labels/qc/hkg_top1pct_prefix_summary.tsv"
NOTE_MD = ROOT / "docs/notes/temporal_stability_target_note.md"


def safe_float(value):
    """Parse float while preserving empty strings as None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def safe_int(value):
    """Parse int while preserving empty strings as None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def leading_prefix(gene_name):
    """Extract a simple leading alphabetic prefix from a gene symbol."""
    if not gene_name:
        return ""
    match = re.match(r"^[A-Za-z]+", gene_name)
    return match.group(0) if match else gene_name


def load_gene_metadata():
    """Build gene_id -> gene_name / gene_type mappings from promoter labels."""
    gene_name_counter = defaultdict(Counter)
    gene_type_counter = defaultdict(Counter)

    with PROMOTER_LABEL_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene_id_raw = str(row.get("gene_id", "")).strip()
            gene_name = str(row.get("gene_name", "")).strip()
            gene_type = str(row.get("gene_type", "")).strip()

            if gene_id_raw == "":
                continue

            gene_id = gene_id_raw.split(".")[0]

            if gene_name != "":
                gene_name_counter[gene_id][gene_name] += 1
            if gene_type != "":
                gene_type_counter[gene_id][gene_type] += 1

    gene_metadata = {}
    for gene_id in sorted(set(gene_name_counter.keys()) | set(gene_type_counter.keys())):
        gene_name = ""
        gene_type = ""

        if gene_name_counter[gene_id]:
            gene_name = gene_name_counter[gene_id].most_common(1)[0][0]
        if gene_type_counter[gene_id]:
            gene_type = gene_type_counter[gene_id].most_common(1)[0][0]

        gene_metadata[gene_id] = {
            "gene_name": gene_name,
            "gene_type": gene_type,
        }

    return gene_metadata


def load_eisenberg_symbols():
    """Load Eisenberg HKG symbols if the file exists."""
    if not EISENBERG_TXT.exists():
        return set()

    symbols = set()
    with EISENBERG_TXT.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            symbol = line.split()[0].strip()
            if symbol != "":
                symbols.add(symbol)

    return symbols


def load_gene_rows(gene_metadata, eisenberg_symbols):
    """Load gene-level labels and attach metadata."""
    rows = []

    with GENE_LABEL_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene_id = str(row["gene_id"]).strip()
            meta = gene_metadata.get(gene_id, {})
            gene_name = meta.get("gene_name", "")
            gene_type = meta.get("gene_type", "")

            rows.append({
                "gene_id": gene_id,
                "gene_name": gene_name,
                "gene_type": gene_type,
                "family_prefix": leading_prefix(gene_name),
                "is_eisenberg_hkg": 1 if gene_name in eisenberg_symbols else 0,
                "tissue_count": safe_int(row.get("tissue_count")),
                "detectable_tissue_count": safe_int(row.get("detectable_tissue_count")),
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


def write_top_list(top_rows):
    """Write the full top-1% HKG gene list."""
    with TOP_LIST_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "rank_hkg_defined",
            "gene_name",
            "gene_id",
            "gene_type",
            "family_prefix",
            "is_eisenberg_hkg",
            "is_in_main_subset",
            "S_defined_main",
            "tissue_count",
            "detectable_tissue_count",
            "E_raw",
            "S_raw",
            "B_raw",
            "Score_E",
            "Score_S",
            "Score_B",
            "Score_HKG",
        ])

        for idx, row in enumerate(top_rows, start=1):
            writer.writerow([
                idx,
                row["gene_name"],
                row["gene_id"],
                row["gene_type"],
                row["family_prefix"],
                row["is_eisenberg_hkg"],
                row["is_in_main_subset"],
                row["S_defined_main"],
                row["tissue_count"],
                row["detectable_tissue_count"],
                "" if row["E_raw"] is None else f"{row['E_raw']:.6f}",
                "" if row["S_raw"] is None else f"{row['S_raw']:.6f}",
                "" if row["B_raw"] is None else f"{row['B_raw']:.6f}",
                "" if row["Score_E"] is None else f"{row['Score_E']:.6f}",
                "" if row["Score_S"] is None else f"{row['Score_S']:.6f}",
                "" if row["Score_B"] is None else f"{row['Score_B']:.6f}",
                "" if row["Score_HKG"] is None else f"{row['Score_HKG']:.6f}",
            ])


def write_summary(all_rows, defined_rows, top_rows, eisenberg_symbols):
    """Write summary statistics for the top-1% analysis."""
    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["all_gene_rows", len(all_rows)])
        writer.writerow(["defined_score_hkg_rows", len(defined_rows)])
        writer.writerow(["top_fraction", "0.01"])
        writer.writerow(["top_gene_count", len(top_rows)])
        writer.writerow(["eisenberg_symbol_count", len(eisenberg_symbols)])
        writer.writerow(["top_gene_eisenberg_overlap_count", sum(r["is_eisenberg_hkg"] for r in top_rows)])
        writer.writerow(["top_gene_unique_gene_type_count", len(set(r["gene_type"] for r in top_rows if r["gene_type"] != ""))])
        writer.writerow(["top_gene_unique_prefix_count", len(set(r["family_prefix"] for r in top_rows if r["family_prefix"] != ""))])


def write_gene_type_summary(top_rows):
    """Summarize gene types in the top-1% set."""
    counter = Counter(row["gene_type"] for row in top_rows if row["gene_type"] != "")

    with GENE_TYPE_SUMMARY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene_type", "count", "fraction_in_top1pct"])
        total = len(top_rows)
        for gene_type, count in counter.most_common():
            writer.writerow([gene_type, count, f"{count / total:.6f}"])


def write_prefix_summary(top_rows):
    """Summarize leading gene-name prefixes in the top-1% set."""
    counter = Counter(row["family_prefix"] for row in top_rows if row["family_prefix"] != "")

    with PREFIX_SUMMARY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["family_prefix", "count", "fraction_in_top1pct"])
        total = len(top_rows)
        for prefix, count in counter.most_common():
            writer.writerow([prefix, count, f"{count / total:.6f}"])


def write_temporal_note():
    """Write a short design note separating tissue stability from temporal durability."""
    lines = [
        "# Temporal Stability Target Note",
        "",
        "Current Candidate E uses cross-tissue stability rather than temporal durability.",
        "",
        "Current tissue-stability definition:",
        "",
        "$$",
        "S_{tissue}(g) = -\operatorname{MAD}\left(\left\{\log_2(1+x_{g,t}) : t \in D(g)\right\}\right)",
        "$$",
        "",
        "where $D(g)$ is the detectable tissue set for gene $g$.",
        "",
        "This captures cross-tissue snapshot consistency, not long-term persistence over time.",
        "",
        "The new design target should be formulated as a separate durability axis:",
        "",
        "$$",
        "D_{time}(p) = -\operatorname{MAD}\left(\left\{\log_2(1+y_{p,\tau}) : \tau \in \mathcal{T}\right\}\right)",
        "$$",
        "",
        "where $p$ is a promoter and $y_{p,\tau}$ is expression measured at time point $\tau$.",
        "",
        "To explicitly penalize hit-and-run behavior, use an early/late retention score:",
        "",
        "$$",
        "R_{late/early}(p) = \frac{y_{p,\tau_{late}} + \varepsilon}{y_{p,\tau_{early}} + \varepsilon}",
        "$$",
        "",
        "Interpretation:",
        "",
        "- high $S_{tissue}$: broad endogenous stability across tissues",
        "- high $D_{time}$: durable expression across time",
        "- high $R_{late/early}$: less hit-and-run behavior",
        "",
        "Recommendation before Phase 5:",
        "",
        "- keep current Candidate E unchanged as the GTEx-grounded tissue-stability label system",
        "- treat temporal durability as a new downstream target, not a replacement for current S",
        "- when time-course assay data becomes available, model durability as an additional output",
    ]
    NOTE_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not GENE_LABEL_TSV.exists():
        raise SystemExit("ERROR: missing labels/build/gene_expression_label_table.tsv")
    if not PROMOTER_LABEL_TSV.exists():
        raise SystemExit("ERROR: missing labels/build/promoter_label_table.tsv")

    print("Loading gene metadata...")
    gene_metadata = load_gene_metadata()

    print("Loading Eisenberg HKG list...")
    eisenberg_symbols = load_eisenberg_symbols()

    print("Loading gene label rows...")
    all_rows = load_gene_rows(gene_metadata, eisenberg_symbols)

    defined_rows = [row for row in all_rows if row["Score_HKG"] is not None]
    defined_rows.sort(
        key=lambda row: (
            row["Score_HKG"],
            row["Score_E"] if row["Score_E"] is not None else -1.0,
            row["Score_S"] if row["Score_S"] is not None else -1.0,
            row["Score_B"] if row["Score_B"] is not None else -1.0,
        ),
        reverse=True,
    )

    top_n = max(1, math.ceil(len(defined_rows) * 0.01))
    top_rows = defined_rows[:top_n]

    print(f"All gene rows: {len(all_rows)}")
    print(f"Defined Score_HKG rows: {len(defined_rows)}")
    print(f"Top 1% gene count: {top_n}")

    write_top_list(top_rows)
    write_summary(all_rows, defined_rows, top_rows, eisenberg_symbols)
    write_gene_type_summary(top_rows)
    write_prefix_summary(top_rows)
    write_temporal_note()

    print("Done.")
    print(f"Wrote list: {TOP_LIST_TSV.as_posix()}")
    print(f"Wrote summary: {SUMMARY_TSV.as_posix()}")
    print(f"Wrote gene-type summary: {GENE_TYPE_SUMMARY_TSV.as_posix()}")
    print(f"Wrote prefix summary: {PREFIX_SUMMARY_TSV.as_posix()}")
    print(f"Wrote note: {NOTE_MD.as_posix()}")


if __name__ == "__main__":
    main()
