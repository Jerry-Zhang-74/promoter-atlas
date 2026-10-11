from pathlib import Path
import csv
import math
import statistics

root = Path(".")

expr_tsv = root / "labels/build/gene_expression_table.tsv"
promoter_tsv = root / "labels/build/promoter_analysis_set.tsv"

gene_label_tsv = root / "labels/build/gene_expression_label_table.tsv"
promoter_label_tsv = root / "labels/build/promoter_label_table.tsv"
core_input_tsv = root / "labels/build/promoter_model_input_core.tsv"
extended_input_tsv = root / "labels/build/promoter_model_input_extended.tsv"
manifest_tsv = root / "labels/build/baseline_train_manifest.tsv"

qc_tsv = root / "labels/qc/label_build_qc.tsv"
sanity_tsv = root / "labels/qc/gene_label_sanity_stats.tsv"
anchor_tsv = root / "labels/qc/hkg_anchor_sanity.tsv"

label_notes_md = root / "docs/notes/label_build_notes.md"
sanity_notes_md = root / "docs/notes/gene_label_sanity_notes.md"

for p in [
    root / "labels/build",
    root / "labels/qc",
    root / "docs/notes",
]:
    p.mkdir(parents=True, exist_ok=True)

if not expr_tsv.exists():
    raise SystemExit("ERROR: missing labels/build/gene_expression_table.tsv")
if not promoter_tsv.exists():
    raise SystemExit("ERROR: missing labels/build/promoter_analysis_set.tsv")

DETECTABLE_TPM = 1.0
MIN_DETECTABLE_TISSUES_FOR_S = 3
ANCHOR_NAMES = {"ACTB", "GAPDH", "RPLP0", "EEF1A1", "UBC", "B2M", "RPL8"}

def base_gene_id(x: str) -> str:
    return x.split(".")[0] if x else ""

def normalize_fieldname(x: str) -> str:
    return x.strip().strip('"').strip()

def corr(a, b):
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den_a = math.sqrt(sum(x * x for x in da))
    den_b = math.sqrt(sum(y * y for y in db))
    if den_a == 0.0 or den_b == 0.0:
        return float("nan")
    return num / (den_a * den_b)

def quantiles(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    def q(p):
        idx = min(n - 1, max(0, round((n - 1) * p)))
        return xs[idx]
    return (xs[0], q(0.25), q(0.50), q(0.75), xs[-1])

def median(xs):
    return statistics.median(xs) if xs else 0.0

def mad(xs):
    if not xs:
        return 0.0
    m = median(xs)
    return median([abs(x - m) for x in xs])

def rank_scale(rows, key, out_key):
    n = len(rows)
    if n == 0:
        return
    order = sorted(range(n), key=lambda i: rows[i][key])
    if n == 1:
        rows[0][out_key] = 1.0
        return
    for rank, idx in enumerate(order):
        rows[idx][out_key] = rank / (n - 1)

print("Loading gene x tissue expression table...")

gene_to_tpms = {}
with expr_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    current_gene = None
    current_vals = []
    for r in reader:
        gene_id = base_gene_id(r["gene_id"])
        mean_tpm = float(r["mean_tpm"])
        if current_gene is None:
            current_gene = gene_id
            current_vals = [mean_tpm]
            continue
        if gene_id == current_gene:
            current_vals.append(mean_tpm)
        else:
            gene_to_tpms[current_gene] = current_vals
            current_gene = gene_id
            current_vals = [mean_tpm]
    if current_gene is not None:
        gene_to_tpms[current_gene] = current_vals

print("Genes loaded:", len(gene_to_tpms))

gene_rows = []

for gene_id, vals in gene_to_tpms.items():
    if not vals:
        continue

    T = len(vals)

    # E: overall expression strength across all tissues
    log_vals_all = [math.log2(1.0 + x) for x in vals]
    E_raw = statistics.fmean(log_vals_all)

    # B: breadth across all tissues
    detectable_vals = [x for x in vals if x >= DETECTABLE_TPM]
    detectable_count = len(detectable_vals)
    B_raw = detectable_count / T

    # D: detectable-tissue MAD stability
    if detectable_count >= MIN_DETECTABLE_TISSUES_FOR_S:
        log_vals_detectable = [math.log2(1.0 + x) for x in detectable_vals]
        S_raw = -mad(log_vals_detectable)
        S_defined = 1
    else:
        S_raw = 0.0
        S_defined = 0

    gene_rows.append({
        "gene_id": gene_id,
        "tissue_count": T,
        "detectable_tissue_count": detectable_count,
        "S_defined": S_defined,
        "E_raw": E_raw,
        "S_raw": S_raw,
        "B_raw": B_raw,
    })

rank_scale(gene_rows, "E_raw", "Score_E")
rank_scale(gene_rows, "S_raw", "Score_S")
rank_scale(gene_rows, "B_raw", "Score_B")

for r in gene_rows:
    product = max(r["Score_E"] * r["Score_S"] * r["Score_B"], 0.0)
    r["Score_HKG"] = product ** (1.0 / 3.0)

rank_scale(gene_rows, "Score_HKG", "Score_HKG_rank")
gene_rows.sort(key=lambda r: r["gene_id"])

with gene_label_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_id",
        "tissue_count",
        "detectable_tissue_count",
        "S_defined",
        "E_raw",
        "S_raw",
        "B_raw",
        "Score_E",
        "Score_S",
        "Score_B",
        "Score_HKG",
        "Score_HKG_rank",
    ])
    for r in gene_rows:
        writer.writerow([
            r["gene_id"],
            r["tissue_count"],
            r["detectable_tissue_count"],
            r["S_defined"],
            f"{r['E_raw']:.6f}",
            f"{r['S_raw']:.6f}",
            f"{r['B_raw']:.6f}",
            f"{r['Score_E']:.6f}",
            f"{r['Score_S']:.6f}",
            f"{r['Score_B']:.6f}",
            f"{r['Score_HKG']:.6f}",
            f"{r['Score_HKG_rank']:.6f}",
        ])

gene_map = {r["gene_id"]: r for r in gene_rows}

print("Reading promoter analysis set with normalized header...")

with promoter_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    raw_header = next(reader)

normalized_header = [normalize_fieldname(x) for x in raw_header]

promoter_rows = 0
matched_rows = 0
unmatched_rows = 0
core_rows = []
extended_rows = []

window_counts = {}
tier_counts = {}
window_matched_counts = {}
tier_matched_counts = {}

gene_name_map = {}

with promoter_tsv.open("r", encoding="utf-8", newline="") as fin, promoter_label_tsv.open("w", encoding="utf-8", newline="") as fout:
    reader = csv.reader(fin, delimiter="\t")
    next(reader)

    out_fields = normalized_header + [
        "gene_id_base",
        "label_matched",
        "tissue_count_label",
        "detectable_tissue_count",
        "S_defined",
        "E_raw",
        "S_raw",
        "B_raw",
        "Score_E",
        "Score_S",
        "Score_B",
        "Score_HKG",
        "Score_HKG_rank",
    ]

    writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t")
    writer.writeheader()

    for parts in reader:
        promoter_rows += 1
        row = dict(zip(normalized_header, parts))

        gene_id = row.get("gene_id", "")
        gene_name = row.get("gene_name", "")
        gid_base = base_gene_id(gene_id)
        hit = gene_map.get(gid_base)

        if gid_base and gene_name and gid_base not in gene_name_map:
            gene_name_map[gid_base] = gene_name

        window_name = row.get("window_name", "")
        analysis_tier = row.get("analysis_tier", "")

        window_counts[window_name] = window_counts.get(window_name, 0) + 1
        tier_counts[analysis_tier] = tier_counts.get(analysis_tier, 0) + 1

        row["gene_id_base"] = gid_base

        if hit is None:
            unmatched_rows += 1
            row["label_matched"] = "0"
            row["tissue_count_label"] = ""
            row["detectable_tissue_count"] = ""
            row["S_defined"] = ""
            row["E_raw"] = ""
            row["S_raw"] = ""
            row["B_raw"] = ""
            row["Score_E"] = ""
            row["Score_S"] = ""
            row["Score_B"] = ""
            row["Score_HKG"] = ""
            row["Score_HKG_rank"] = ""
        else:
            matched_rows += 1
            row["label_matched"] = "1"
            row["tissue_count_label"] = str(hit["tissue_count"])
            row["detectable_tissue_count"] = str(hit["detectable_tissue_count"])
            row["S_defined"] = str(hit["S_defined"])
            row["E_raw"] = f"{hit['E_raw']:.6f}"
            row["S_raw"] = f"{hit['S_raw']:.6f}"
            row["B_raw"] = f"{hit['B_raw']:.6f}"
            row["Score_E"] = f"{hit['Score_E']:.6f}"
            row["Score_S"] = f"{hit['Score_S']:.6f}"
            row["Score_B"] = f"{hit['Score_B']:.6f}"
            row["Score_HKG"] = f"{hit['Score_HKG']:.6f}"
            row["Score_HKG_rank"] = f"{hit['Score_HKG_rank']:.6f}"

            window_matched_counts[window_name] = window_matched_counts.get(window_name, 0) + 1
            tier_matched_counts[analysis_tier] = tier_matched_counts.get(analysis_tier, 0) + 1

            if row.get("core_model_pass", "0") == "1":
                core_rows.append(dict(row))
            if row.get("extended_context_pass", "0") == "1":
                extended_rows.append(dict(row))

        writer.writerow(row)

def write_subset(path, rows):
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

write_subset(core_input_tsv, core_rows)
write_subset(extended_input_tsv, extended_rows)

E_raw_vals = [r["E_raw"] for r in gene_rows]
S_raw_vals = [r["S_raw"] for r in gene_rows]
B_raw_vals = [r["B_raw"] for r in gene_rows]
Score_E_vals = [r["Score_E"] for r in gene_rows]
Score_S_vals = [r["Score_S"] for r in gene_rows]
Score_B_vals = [r["Score_B"] for r in gene_rows]
Score_HKG_vals = [r["Score_HKG"] for r in gene_rows]

with qc_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["metric", "value"])
    writer.writerow(["gene_label_rows", len(gene_rows)])
    writer.writerow(["promoter_rows", promoter_rows])
    writer.writerow(["matched_promoter_rows", matched_rows])
    writer.writerow(["unmatched_promoter_rows", unmatched_rows])
    writer.writerow(["matched_promoter_frac", f"{(matched_rows / promoter_rows):.6f}" if promoter_rows else "0.000000"])
    writer.writerow(["detectable_tpm_threshold_for_B", DETECTABLE_TPM])
    writer.writerow(["min_detectable_tissues_for_S", MIN_DETECTABLE_TISSUES_FOR_S])
    for k in sorted(window_counts):
        writer.writerow([f"window_count::{k}", window_counts[k]])
    for k in sorted(window_matched_counts):
        writer.writerow([f"window_matched_count::{k}", window_matched_counts[k]])
    for k in sorted(tier_counts):
        writer.writerow([f"analysis_tier_count::{k}", tier_counts[k]])
    for k in sorted(tier_matched_counts):
        writer.writerow([f"analysis_tier_matched_count::{k}", tier_matched_counts[k]])

with sanity_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["row_type", "name", "count", "mean", "min", "q25", "median", "q75", "max", "value"])
    for name, values in [
        ("E_raw", E_raw_vals),
        ("S_raw", S_raw_vals),
        ("B_raw", B_raw_vals),
        ("Score_E", Score_E_vals),
        ("Score_S", Score_S_vals),
        ("Score_B", Score_B_vals),
        ("Score_HKG", Score_HKG_vals),
    ]:
        q0, q25, q50, q75, q100 = quantiles(values)
        writer.writerow([
            "distribution",
            name,
            len(values),
            f"{statistics.fmean(values):.6f}",
            f"{q0:.6f}",
            f"{q25:.6f}",
            f"{q50:.6f}",
            f"{q75:.6f}",
            f"{q100:.6f}",
            "",
        ])
    for name, value in [
        ("corr::E_raw__S_raw", corr(E_raw_vals, S_raw_vals)),
        ("corr::E_raw__B_raw", corr(E_raw_vals, B_raw_vals)),
        ("corr::S_raw__B_raw", corr(S_raw_vals, B_raw_vals)),
        ("corr::Score_E__Score_S", corr(Score_E_vals, Score_S_vals)),
        ("corr::Score_E__Score_B", corr(Score_E_vals, Score_B_vals)),
        ("corr::Score_S__Score_B", corr(Score_S_vals, Score_B_vals)),
        ("corr::Score_HKG__Score_E", corr(Score_HKG_vals, Score_E_vals)),
        ("corr::Score_HKG__Score_S", corr(Score_HKG_vals, Score_S_vals)),
        ("corr::Score_HKG__Score_B", corr(Score_HKG_vals, Score_B_vals)),
    ]:
        writer.writerow([
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

ranked = sorted(gene_rows, key=lambda r: r["Score_HKG"], reverse=True)
gene_rank_map = {r["gene_id"]: i + 1 for i, r in enumerate(ranked)}
total_gene_labels = len(ranked)

anchor_hits = []
for r in gene_rows:
    gid = r["gene_id"]
    gname = gene_name_map.get(gid, "")
    if gname in ANCHOR_NAMES:
        rank = gene_rank_map[gid]
        frac = rank / total_gene_labels if total_gene_labels else 1.0
        anchor_hits.append({
            "gene_name": gname,
            "gene_id": gid,
            "rank_by_HKG": rank,
            "rank_frac": frac,
            "Score_E": r["Score_E"],
            "Score_S": r["Score_S"],
            "Score_B": r["Score_B"],
            "Score_HKG": r["Score_HKG"],
            "E_raw": r["E_raw"],
            "S_raw": r["S_raw"],
            "B_raw": r["B_raw"],
            "detectable_tissue_count": r["detectable_tissue_count"],
            "S_defined": r["S_defined"],
        })

anchor_hits.sort(key=lambda x: x["rank_by_HKG"])

with anchor_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "gene_name",
        "gene_id",
        "rank_by_HKG",
        "rank_frac",
        "Score_E",
        "Score_S",
        "Score_B",
        "Score_HKG",
        "E_raw",
        "S_raw",
        "B_raw",
        "detectable_tissue_count",
        "S_defined",
    ])
    for r in anchor_hits:
        writer.writerow([
            r["gene_name"],
            r["gene_id"],
            r["rank_by_HKG"],
            f"{r['rank_frac']:.6f}",
            f"{r['Score_E']:.6f}",
            f"{r['Score_S']:.6f}",
            f"{r['Score_B']:.6f}",
            f"{r['Score_HKG']:.6f}",
            f"{r['E_raw']:.6f}",
            f"{r['S_raw']:.6f}",
            f"{r['B_raw']:.6f}",
            r["detectable_tissue_count"],
            r["S_defined"],
        ])

with manifest_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["key", "value"])
    rows = [
        ("baseline_name", "promoter_only_candidate_D"),
        ("label_version", "candidate_D_detectable_tissue_MAD"),
        ("core_input_path", core_input_tsv.as_posix()),
        ("extended_input_path", extended_input_tsv.as_posix()),
        ("core_row_count", str(len(core_rows))),
        ("extended_row_count", str(len(extended_rows))),
        ("recommended_first_training_set", "core_input"),
        ("primary_targets", "Score_E,Score_S,Score_B"),
        ("ranking_only_target", "Score_HKG"),
        ("recommended_window", "up200"),
        ("recommended_filter", "label_matched=1 AND core_model_pass=1"),
        ("detectable_tpm_threshold_for_B", str(DETECTABLE_TPM)),
        ("min_detectable_tissues_for_S", str(MIN_DETECTABLE_TISSUES_FOR_S)),
    ]
    for k, v in rows:
        writer.writerow([k, v])

label_notes = []
label_notes.append("# Label Build Notes")
label_notes.append("")
label_notes.append("## Label Version")
label_notes.append("")
label_notes.append("- current label system: candidate_D_detectable_tissue_MAD")
label_notes.append("- S is defined only from detectable-tissue dispersion")
label_notes.append("- E is not used inside the formula of S")
label_notes.append("")
label_notes.append("## Definitions")
label_notes.append("")
label_notes.append("- E_raw: mean over all tissues of log2(1 + tissue_mean_tpm)")
label_notes.append(f"- B_raw: fraction of all tissues with tissue_mean_tpm >= {DETECTABLE_TPM}")
label_notes.append("- S_raw: negative MAD of log2(1 + tissue_mean_tpm) over detectable tissues only")
label_notes.append(f"- S_defined: 1 only when detectable_tissue_count >= {MIN_DETECTABLE_TISSUES_FOR_S}")
label_notes.append("- Score_HKG: geometric mean of Score_E, Score_S, and Score_B")
label_notes.append("")
label_notes.append("## Interpretation")
label_notes.append("")
label_notes.append("- B remains a gate-like hidden axis, not the main experimental axis")
label_notes.append("- S focuses on detectable-tissue stability without forcing uniformity")
label_notes.append("- low-breadth genes can still have an S value, but S_defined flags confidence for main interpretation")
label_notes.append("")

label_notes_md.write_text("\n".join(label_notes), encoding="utf-8")

sanity_notes = []
sanity_notes.append("# Gene Label Sanity Notes")
sanity_notes.append("")
sanity_notes.append("## Purpose")
sanity_notes.append("")
sanity_notes.append("- verify the candidate D label system at gene level")
sanity_notes.append("- inspect housekeeping anchors before baseline training")
sanity_notes.append("- freeze baseline manifest for the next decision")
sanity_notes.append("")
sanity_notes.append("## Important")
sanity_notes.append("")
sanity_notes.append("- focus on semantic independence, not perfect statistical independence")
sanity_notes.append("- the key failure mode to avoid is high E automatically implying high S by definition")
sanity_notes.append("")

sanity_notes_md.write_text("\n".join(sanity_notes), encoding="utf-8")

print("Candidate D rebuild complete.")
print(f"Gene labels: {gene_label_tsv.as_posix()}")
print(f"Promoter labels: {promoter_label_tsv.as_posix()}")
print(f"Core input: {core_input_tsv.as_posix()}")
print(f"Extended input: {extended_input_tsv.as_posix()}")
print(f"QC: {qc_tsv.as_posix()}")
print(f"Sanity: {sanity_tsv.as_posix()}")
print(f"Anchors: {anchor_tsv.as_posix()}")
print(f"Manifest: {manifest_tsv.as_posix()}")
print("")
print(f"Gene label rows: {len(gene_rows)}")
print(f"Promoter rows: {promoter_rows}")
print(f"Matched promoter rows: {matched_rows}")
print(f"Unmatched promoter rows: {unmatched_rows}")
print(f"Core model rows: {len(core_rows)}")
print(f"Extended model rows: {len(extended_rows)}")
