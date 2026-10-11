from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

root = Path(".")

gene_label_tsv = root / "labels/build/gene_expression_label_table.tsv"
promoter_label_tsv = root / "labels/build/promoter_label_table.tsv"
hkg_list_txt = root / "data/external/list_of_HKGs.txt"

figure_png = root / "figures/labels/candidate_e_es_scatter.png"
figure_pdf = root / "figures/labels/candidate_e_es_scatter.pdf"
eisenberg_qc_tsv = root / "labels/qc/eisenberg_hkg_overlap_qc.tsv"
anchor_qc_tsv = root / "labels/qc/positive_anchor_es_positions.tsv"
summary_tsv = root / "labels/qc/candidate_e_es_plot_summary.tsv"

for p in [
    root / "figures/labels",
    root / "labels/qc",
]:
    p.mkdir(parents=True, exist_ok=True)

if not gene_label_tsv.exists():
    raise SystemExit("ERROR: missing labels/build/gene_expression_label_table.tsv")
if not promoter_label_tsv.exists():
    raise SystemExit("ERROR: missing labels/build/promoter_label_table.tsv")
if not hkg_list_txt.exists():
    raise SystemExit("ERROR: missing data/external/list_of_HKGs.txt")

# 这里把你的 8 个阳参写死在脚本顶部，方便你之后直接改
# 当前先沿用你前面一直看的 7 个，再给第 8 个留位
POSITIVE_ANCHORS = [
    "ACTB",
    "GAPDH",
    "RPL8",
    "SAFB",
    "ARHGDISA",
    "LDHA",
    "RPL32",
]

def safe_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return float(x)

print("Loading promoter gene-name mapping...")

gene_id_to_names = defaultdict(set)
gene_name_to_ids = defaultdict(set)

with promoter_label_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        gene_id_raw = str(r.get("gene_id", "")).strip()
        gene_name = str(r.get("gene_name", "")).strip()
        if gene_id_raw == "" or gene_name == "":
            continue
        gene_id = gene_id_raw.split(".")[0]
        gene_id_to_names[gene_id].add(gene_name)
        gene_name_to_ids[gene_name].add(gene_id)

print("Loading gene labels...")

gene_rows = []
gene_map = {}

with gene_label_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        gene_id = str(r["gene_id"]).strip()
        Score_E = safe_float(r.get("Score_E"))
        Score_S = safe_float(r.get("Score_S"))
        B_raw = safe_float(r.get("B_raw"))
        Score_HKG = safe_float(r.get("Score_HKG"))
        is_in_main_subset = int(str(r.get("is_in_main_subset", "0") or "0"))
        S_defined_main = int(str(r.get("S_defined_main", "0") or "0"))

        gene_name_candidates = sorted(gene_id_to_names.get(gene_id, []))
        gene_name = gene_name_candidates[0] if gene_name_candidates else ""

        row = {
            "gene_id": gene_id,
            "gene_name": gene_name,
            "Score_E": Score_E,
            "Score_S": Score_S,
            "B_raw": B_raw,
            "Score_HKG": Score_HKG,
            "is_in_main_subset": is_in_main_subset,
            "S_defined_main": S_defined_main,
        }
        gene_rows.append(row)
        gene_map[gene_id] = row

print("Loading Eisenberg HKG list...")

eisenberg_symbols = []
with hkg_list_txt.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        if len(parts) == 0:
            continue
        symbol = parts[0].strip()
        if symbol != "":
            eisenberg_symbols.append(symbol)

eisenberg_symbols = sorted(set(eisenberg_symbols))

eisenberg_gene_ids = set()
missing_symbols = []

for symbol in eisenberg_symbols:
    ids = gene_name_to_ids.get(symbol, set())
    if len(ids) == 0:
        missing_symbols.append(symbol)
    else:
        eisenberg_gene_ids.update(ids)

anchor_gene_ids = set()
missing_anchor_symbols = []

for symbol in POSITIVE_ANCHORS:
    ids = gene_name_to_ids.get(symbol, set())
    if len(ids) == 0:
        missing_anchor_symbols.append(symbol)
    else:
        anchor_gene_ids.update(ids)

print("Preparing scatter subsets...")

all_x = []
all_y = []

main_x = []
main_y = []

eisenberg_x = []
eisenberg_y = []
eisenberg_rows = []

anchor_x = []
anchor_y = []
anchor_rows = []

for r in gene_rows:
    Score_E = r["Score_E"]
    Score_S = r["Score_S"]

    if (
        Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        all_x.append(Score_E)
        all_y.append(Score_S)

    if r["S_defined_main"] == 1 and Score_E is not None and Score_S is not None:
        main_x.append(Score_E)
        main_y.append(Score_S)

    if (
        r["gene_id"] in eisenberg_gene_ids
        and Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        eisenberg_x.append(Score_E)
        eisenberg_y.append(Score_S)
        eisenberg_rows.append(r)

    if (
        r["gene_id"] in anchor_gene_ids
        and Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        anchor_x.append(Score_E)
        anchor_y.append(Score_S)
        anchor_rows.append(r)

print("Writing QC tables...")

with eisenberg_qc_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_name",
        "gene_id",
        "is_in_main_subset",
        "S_defined_main",
        "Score_E",
        "Score_S",
        "B_raw",
        "Score_HKG",
    ])
    for r in sorted(eisenberg_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
        writer.writerow([
            r["gene_name"],
            r["gene_id"],
            r["is_in_main_subset"],
            r["S_defined_main"],
            "" if r["Score_E"] is None else f"{r['Score_E']:.6f}",
            "" if r["Score_S"] is None else f"{r['Score_S']:.6f}",
            "" if r["B_raw"] is None else f"{r['B_raw']:.6f}",
            "" if r["Score_HKG"] is None else f"{r['Score_HKG']:.6f}",
        ])

with anchor_qc_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_name",
        "gene_id",
        "is_in_main_subset",
        "S_defined_main",
        "Score_E",
        "Score_S",
        "B_raw",
        "Score_HKG",
    ])
    for r in sorted(anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
        writer.writerow([
            r["gene_name"],
            r["gene_id"],
            r["is_in_main_subset"],
            r["S_defined_main"],
            "" if r["Score_E"] is None else f"{r['Score_E']:.6f}",
            "" if r["Score_S"] is None else f"{r['Score_S']:.6f}",
            "" if r["B_raw"] is None else f"{r['B_raw']:.6f}",
            "" if r["Score_HKG"] is None else f"{r['Score_HKG']:.6f}",
        ])

with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["metric", "value"])
    writer.writerow(["all_gene_rows", len(gene_rows)])
    writer.writerow(["all_points_with_defined_E_and_S", len(all_x)])
    writer.writerow(["main_points_with_defined_E_and_S", len(main_x)])
    writer.writerow(["eisenberg_symbol_count", len(eisenberg_symbols)])
    writer.writerow(["eisenberg_symbol_missing_count", len(missing_symbols)])
    writer.writerow(["eisenberg_gene_id_hit_count", len(eisenberg_gene_ids)])
    writer.writerow(["eisenberg_points_plotted", len(eisenberg_x)])
    writer.writerow(["positive_anchor_symbol_count", len(POSITIVE_ANCHORS)])
    writer.writerow(["positive_anchor_missing_count", len(missing_anchor_symbols)])
    writer.writerow(["positive_anchor_gene_id_hit_count", len(anchor_gene_ids)])
    writer.writerow(["positive_anchor_points_plotted", len(anchor_x)])

print("Plotting figure...")

plt.figure(figsize=(10, 8))

# 全体背景
plt.scatter(
    all_x,
    all_y,
    s=6,
    alpha=0.18,
    linewidths=0,
)

# Eisenberg HKG
plt.scatter(
    eisenberg_x,
    eisenberg_y,
    s=16,
    alpha=0.65,
    linewidths=0,
)

# 8个阳参
plt.scatter(
    anchor_x,
    anchor_y,
    s=40,
    alpha=1.0,
    marker="x",
)

for r in sorted(anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
    plt.text(
        r["Score_E"],
        r["Score_S"],
        r["gene_name"],
        fontsize=9,
    )

plt.xlabel("Score_E")
plt.ylabel("Score_S")
plt.title("Candidate E: Score_E-Score_S map (S_defined_main only)")
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(figure_png, dpi=200)
plt.savefig(figure_pdf)
plt.close()

print("Done.")
print(f"Wrote figure: {figure_png.as_posix()}")
print(f"Wrote figure: {figure_pdf.as_posix()}")
print(f"Wrote QC: {eisenberg_qc_tsv.as_posix()}")
print(f"Wrote QC: {anchor_qc_tsv.as_posix()}")
print(f"Wrote QC: {summary_tsv.as_posix()}")
print(f"Eisenberg symbols loaded: {len(eisenberg_symbols)}")
print(f"Eisenberg symbols missing: {len(missing_symbols)}")
print(f"Eisenberg gene ids hit: {len(eisenberg_gene_ids)}")
print(f"Positive anchor symbols missing: {len(missing_anchor_symbols)}")
