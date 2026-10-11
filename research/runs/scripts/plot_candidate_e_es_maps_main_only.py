from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

root = Path(".")

gene_label_tsv = root / "labels/build/gene_expression_label_table.tsv"
promoter_label_tsv = root / "labels/build/promoter_label_table.tsv"
hkg_list_txt = root / "data/external/list_of_HKGs.txt"

figure_all_png = root / "figures/labels/candidate_e_es_scatter_all_main_only.png"
figure_all_pdf = root / "figures/labels/candidate_e_es_scatter_all_main_only.pdf"

figure_noe_png = root / "figures/labels/candidate_e_es_scatter_no_e_main_only.png"
figure_noe_pdf = root / "figures/labels/candidate_e_es_scatter_no_e_main_only.pdf"

figure_density_png = root / "figures/labels/candidate_e_es_scatter_main_density.png"
figure_density_pdf = root / "figures/labels/candidate_e_es_scatter_main_density.pdf"

summary_tsv = root / "labels/qc/candidate_e_es_plot_summary_main_only.tsv"
anchor_qc_tsv = root / "labels/qc/positive_anchor_es_positions_main_only.tsv"
eisenberg_qc_tsv = root / "labels/qc/eisenberg_hkg_overlap_main_only_qc.tsv"

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

POSITIVE_ANCHORS = [
    "ACTB",
    "GAPDH",
    "RPLP0",
    "EEF1A1",
    "UBC",
    "B2M",
    "RPL8",
    "PGK1",
]

# Match the visual system used by Figures 11–13 in the Qijie project plan.
INK = "#18333F"
GRAY = "#667B86"
GRID = "#DCE8EC"
PALE = "#EEF6F7"
BLUE = "#316C9B"
TEAL = "#159A93"
MINT = "#77CDBD"
AMBER = "#E59A2F"
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "qijie_density",
    ["#FFFFFF", PALE, MINT, TEAL, BLUE, INK],
)
ANCHOR_LABEL_OFFSETS = {
    "GAPDH": (-7, 7),
    "PGK1": (-7, -4),
    "UBC": (-7, 5),
    "EEF1A1": (-7, 10),
    "RPL8": (-7, -2),
    "RPLP0": (-7, 13),
    "B2M": (-7, -10),
    "ACTB": (-7, 5),
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)

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

        gene_rows.append({
            "gene_id": gene_id,
            "gene_name": gene_name,
            "Score_E": Score_E,
            "Score_S": Score_S,
            "B_raw": B_raw,
            "Score_HKG": Score_HKG,
            "is_in_main_subset": is_in_main_subset,
            "S_defined_main": S_defined_main,
        })

print("Loading Eisenberg list...")

eisenberg_symbols = []
with hkg_list_txt.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        if not parts:
            continue
        eisenberg_symbols.append(parts[0].strip())

eisenberg_symbols = sorted(set(eisenberg_symbols))

eisenberg_gene_ids = set()
missing_eisenberg_symbols = []
for symbol in eisenberg_symbols:
    ids = gene_name_to_ids.get(symbol, set())
    if len(ids) == 0:
        missing_eisenberg_symbols.append(symbol)
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

print("Preparing main-only subsets...")

main_all_x = []
main_all_y = []

main_eisenberg_x = []
main_eisenberg_y = []
main_eisenberg_rows = []

main_anchor_x = []
main_anchor_y = []
main_anchor_rows = []

for r in gene_rows:
    Score_E = r["Score_E"]
    Score_S = r["Score_S"]

    if (
        Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        main_all_x.append(Score_E)
        main_all_y.append(Score_S)

    if (
        r["gene_id"] in eisenberg_gene_ids
        and Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        main_eisenberg_x.append(Score_E)
        main_eisenberg_y.append(Score_S)
        main_eisenberg_rows.append(r)

    if (
        r["gene_id"] in anchor_gene_ids
        and Score_E is not None
        and Score_S is not None
        and r["S_defined_main"] == 1
    ):
        main_anchor_x.append(Score_E)
        main_anchor_y.append(Score_S)
        main_anchor_rows.append(r)

print("Writing QC...")

with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["metric", "value"])
    writer.writerow(["all_gene_rows", len(gene_rows)])
    writer.writerow(["main_only_points_plotted", len(main_all_x)])
    writer.writerow(["eisenberg_symbol_count", len(eisenberg_symbols)])
    writer.writerow(["eisenberg_missing_symbol_count", len(missing_eisenberg_symbols)])
    writer.writerow(["eisenberg_gene_id_hit_count", len(eisenberg_gene_ids)])
    writer.writerow(["main_only_eisenberg_points", len(main_eisenberg_x)])
    writer.writerow(["positive_anchor_symbol_count", len(POSITIVE_ANCHORS)])
    writer.writerow(["positive_anchor_missing_symbol_count", len(missing_anchor_symbols)])
    writer.writerow(["positive_anchor_gene_id_hit_count", len(anchor_gene_ids)])
    writer.writerow(["main_only_anchor_points", len(main_anchor_x)])

with anchor_qc_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_name",
        "gene_id",
        "Score_E",
        "Score_S",
        "B_raw",
        "Score_HKG",
        "is_in_main_subset",
        "S_defined_main",
    ])
    for r in sorted(main_anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
        writer.writerow([
            r["gene_name"],
            r["gene_id"],
            "" if r["Score_E"] is None else f"{r['Score_E']:.6f}",
            "" if r["Score_S"] is None else f"{r['Score_S']:.6f}",
            "" if r["B_raw"] is None else f"{r['B_raw']:.6f}",
            "" if r["Score_HKG"] is None else f"{r['Score_HKG']:.6f}",
            r["is_in_main_subset"],
            r["S_defined_main"],
        ])

with eisenberg_qc_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_name",
        "gene_id",
        "Score_E",
        "Score_S",
        "B_raw",
        "Score_HKG",
        "is_in_main_subset",
        "S_defined_main",
    ])
    for r in sorted(main_eisenberg_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
        writer.writerow([
            r["gene_name"],
            r["gene_id"],
            "" if r["Score_E"] is None else f"{r['Score_E']:.6f}",
            "" if r["Score_S"] is None else f"{r['Score_S']:.6f}",
            "" if r["B_raw"] is None else f"{r['B_raw']:.6f}",
            "" if r["Score_HKG"] is None else f"{r['Score_HKG']:.6f}",
            r["is_in_main_subset"],
            r["S_defined_main"],
        ])

print("Plotting all-main-only figure...")

fig, ax = plt.subplots(figsize=(10, 7.2))
ax.scatter(main_all_x, main_all_y, s=7, alpha=0.16, linewidths=0, color=BLUE)
ax.scatter(main_eisenberg_x, main_eisenberg_y, s=17, alpha=0.58, linewidths=0, color=TEAL)
ax.scatter(main_anchor_x, main_anchor_y, s=52, alpha=1.0, marker="x", color=AMBER, linewidths=1.8)

for r in sorted(main_anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
    ax.annotate(
        r["gene_name"],
        (r["Score_E"], r["Score_S"]),
        xytext=ANCHOR_LABEL_OFFSETS.get(r["gene_name"], (-6, 4)),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )

ax.set_xlabel("表达强度分位得分（Score_E）")
ax.set_ylabel("跨组织稳定度分位得分（Score_S）")
ax.set_title("管家基因候选的表达强度—跨组织稳定度分布", color=INK, weight="bold", pad=14)
style_axes(ax)
fig.tight_layout()
fig.savefig(figure_all_png, dpi=300, facecolor="white")
fig.savefig(figure_all_pdf, facecolor="white")
plt.close(fig)

print("Plotting no-Eisenberg main-only figure...")

fig, ax = plt.subplots(figsize=(10, 7.2))
ax.scatter(main_all_x, main_all_y, s=7, alpha=0.16, linewidths=0, color=BLUE)
ax.scatter(main_anchor_x, main_anchor_y, s=52, alpha=1.0, marker="x", color=AMBER, linewidths=1.8)

for r in sorted(main_anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
    ax.annotate(
        r["gene_name"],
        (r["Score_E"], r["Score_S"]),
        xytext=ANCHOR_LABEL_OFFSETS.get(r["gene_name"], (-6, 4)),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )

ax.set_xlabel("表达强度分位得分（Score_E）")
ax.set_ylabel("跨组织稳定度分位得分（Score_S）")
ax.set_title("管家基因候选分布（不叠加外部清单）", color=INK, weight="bold", pad=14)
style_axes(ax)
fig.tight_layout()
fig.savefig(figure_noe_png, dpi=300, facecolor="white")
fig.savefig(figure_noe_pdf, facecolor="white")
plt.close(fig)

print("Plotting main-only density figure...")

fig, ax = plt.subplots(figsize=(10, 7.2))
hb = ax.hexbin(
    main_all_x,
    main_all_y,
    gridsize=55,
    mincnt=1,
    cmap=DENSITY_CMAP,
    linewidths=0,
)
ax.scatter(
    main_anchor_x,
    main_anchor_y,
    s=62,
    marker="x",
    color=AMBER,
    linewidths=2.0,
    label="常用阳性参照",
    zorder=3,
)

for r in sorted(main_anchor_rows, key=lambda x: (x["gene_name"], x["gene_id"])):
    ax.annotate(
        r["gene_name"],
        (r["Score_E"], r["Score_S"]),
        xytext=ANCHOR_LABEL_OFFSETS.get(r["gene_name"], (-7, 4)),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
        zorder=4,
    )

ax.set_xlabel("表达强度分位得分（Score_E）")
ax.set_ylabel("跨组织稳定度分位得分（Score_S）")
ax.set_title("管家基因候选的表达强度—跨组织稳定度密度图", color=INK, weight="bold", pad=14)
ax.text(
    0.01,
    0.015,
    "仅显示 B_raw ≥ 0.30 且至少 3 个组织可检出的基因；橙色叉号为常用参照基因",
    transform=ax.transAxes,
    fontsize=9,
    color=GRAY,
    ha="left",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID, "alpha": 0.92},
)
style_axes(ax)
legend = ax.legend(frameon=False, loc="lower right")
for text in legend.get_texts():
    text.set_color(INK)
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label("基因密度", color=INK)
cbar.outline.set_edgecolor(GRID)
cbar.ax.tick_params(colors=INK)
fig.tight_layout()
fig.savefig(figure_density_png, dpi=300, facecolor="white")
fig.savefig(figure_density_pdf, facecolor="white")
plt.close(fig)

print("Done.")
print(f"Wrote figure: {figure_all_png.as_posix()}")
print(f"Wrote figure: {figure_noe_png.as_posix()}")
print(f"Wrote figure: {figure_density_png.as_posix()}")
print(f"Wrote QC: {summary_tsv.as_posix()}")
print(f"Wrote QC: {anchor_qc_tsv.as_posix()}")
print(f"Wrote QC: {eisenberg_qc_tsv.as_posix()}")
