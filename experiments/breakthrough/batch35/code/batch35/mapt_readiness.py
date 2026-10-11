# -*- coding: utf-8 -*-
"""Batch35 Part A: MAPT/HEK293FT primary-data readiness (ZERO fits, ZERO model scoring).
Describes only the recoverable measured-sequence library and the MISSING labels.
Never fabricates sequences, never guesses flanks, never joins labels it does not have.
"""
import gzip, hashlib, json, collections, re, time, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = REPO / "data/breakthrough_20261002/batch35/mapt"
OUTDIR = M / "frozen"; OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTS = REPO / "reports/breakthrough_20261002/batch35"; REPORTS.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def arm_of(n):
    if n.startswith("maptprom_"): return "MAPT_promoter_saturation_mutagenesis"
    if n.startswith("SNV_"): return "known_variant_SNV"
    if n.startswith("delete"): return "known_variant_deletion"
    if n.startswith("insert"): return "known_variant_insertion"
    if n.startswith("internalPositive") or n.startswith("internalNegative"): return "internal_control"
    if "oligoMPRANegative" in n: return "negative_control_scrambled"
    return "positive_control_promoter"


recs = []
gz = M / "gsm9640449_VariantMPRAdesignFile.fa.gz"
for line in gzip.open(gz, "rt", encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith(">"):
        recs.append([line[1:].strip(), []])
    elif line:
        recs[-1][1].append(line.strip())
recs = [(n, "".join(s)) for n, s in recs]

fa = OUTDIR / "mapr_gsm9640449_design_36000.fasta"
with open(fa, "w", encoding="utf-8", newline="\n") as f:
    for n, s in recs:
        f.write(">" + n + "\n" + s + "\n")

arms = collections.Counter(arm_of(n) for n, s in recs)
lengths = collections.Counter(len(s) for n, s in recs)
seqs = [s for n, s in recs]
dup_exact = len(seqs) - len(set(seqs))
rcset = set(revcomp(s) for s in set(seqs))
neg = len(set(seqs) & rcset)
alpha = dict(collections.Counter(ch for s in seqs for ch in set(s)))
mpos = sorted(set(int(m.group(1)) for n, s in recs for m in [re.match(r"maptprom_chr17_(\d+)_", n)] if m))
snvpos = sorted(set(int(m.group(1)) for n, s in recs for m in [re.match(r"SNV_chr17_(\d+)_", n)] if m))
chroms_all = sorted(set(n.split("_")[1] for n, s in recs if "_chr" in n))
gz_bytes = os.path.getsize(gz)

report = {
    "batch": "batch35", "version": "v6.24.0-293ft-cohort-readiness.rc1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "study": {"pmid": "41959502", "pmcid": "PMC13060879", "doi": "10.64898/2026.03.06.710116",
              "status": "PREPRINT (bioRxiv 2026-03-09, v2)",
              "data_availability": "NCBI GEO SuperSeries GSE325670 (Homo sapiens, 140 samples); code github.com/HudsonAlpha/Hauser_MAPT_MPRAs",
              "license": "bioRxiv preprint; GEO public data (local research use)"},
    "architecture_finding": {
        "vector": "pLS-SceI lentiMPRA reporter (Gordon et al. style; Addgene 137725)",
        "insert_role": "candidate cis-regulatory elements (CREs/enhancers) cloned into a FIXED lentiMPRA vector; both orientations tested and combined",
        "is_autonomous_promoter_measured": False,
        "note": "Measured quantity is CRE/enhancer reporter activity via the vector reporter, NOT an autonomous isolated promoter. Do NOT label these as promoters.",
        "two_thousand_bp_meaning": "2000 bp = MUTAGENIZED GENOMIC REGION (chr17:45,893,536-45,895,535); each MEASURED insert is a 219 bp oligo centered per base (ref + 3 alt + centered 5bp deletion). NOT a full 2000 bp measured insert.",
        "cell_activity_host": "HEK293FT ARE an activity-assay host (DNA and RNA collected and quantified in HEK293FT; variant logFC reported in HEK293FT cells). HEK293FT is ALSO used to package lentivirus. VERIFIED: not packaging-only.",
        "other_host": "KOLF2.1J h-NGN2 excitatory neurons (14-day); BAC/Oligo and variant/mutagenesis MPRAs run in BOTH",
        "readout": "lentiMPRA barcode RNA/DNA counts; MPRAnalyze (BAC/Oligo) or bcalm logFC (variant/mutagenesis); abs(logFC)>0.1, adj p<=0.05",
        "barcode_bins_warning": "BAC MPRA analyzes 100 bp NONOVERLAPPING genomic bins (barcode assigned by the MIDDLE of the sheared BAC alignment). These are coverage aggregates, NOT per-insert fragment labels."},
    "recovered_measured_sequences": {
        "file": "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz",
        "source_file_sha256": sha(gz), "geo_sample": "GSM9640449 (SuperSeries GSE325670)",
        "n_records": len(recs), "arm_counts": dict(arms), "length_histogram": dict(sorted(lengths.items())),
        "alphabet": alpha, "frozen_fasta": str(fa.relative_to(REPO)).replace("\\", "/"),
        "frozen_fasta_sha256": sha(fa),
        "constant_flank_warning": "Most records are 249 bp = 219 bp genomic oligo + constant library/cloning flank. The genomic insert boundary is NOT recovered here (would need the vector/cloning spec). Flanks are neither stripped nor guessed.",
        "saturation_arm_region": {"chrom": "chr17", "n_genomic_positions": len(mpos),
                                  "span": [min(mpos), max(mpos)] if mpos else None,
                                  "declared_region_hg38": "chr17:45,893,536-45,895,535 (2000 bp)"},
        "known_variant_snv_positions": len(snvpos), "chroms_seen": chroms_all},
    "duplicates_and_rc": {"n_exact_duplicate_sequences": dup_exact,
                          "n_exact_RC_pairs_or_self": neg},
    "labels_status": {
        "per_insert_quantitative_activity_recovered": False,
        "why": "Per-variant/per-oligo activity is NOT a small downloadable table. GEO GSE325670 ships BarcodeDictionary files of 85-582 MB (variant dict = 555 MB) inside a 935 MB RAW.tar; small design FASTAs carry SEQUENCE ONLY. Processed activity needs the author pipeline or the bioRxiv supplementary tables, not retrievable within the 50 MB / bounded-attempt cap.",
        "download_budget_check": {"cap_mb": 50, "largest_needed_but_blocked_mb": 555,
                                  "notable_retrieved_mb": round(gz_bytes / 1e6, 3)}},
    "license": {"model": "n/a (no model used here)",
                "data": "public GEO GSE325670 + bioRxiv preprint; local academic research use"},
    "arm_verdicts": {
        "MAPT_promoter_saturation_mutagenesis (within-locus variant effects, HEK293FT)": {
            "verdict": "LIMITED",
            "have": "measured insert sequences (219 bp oligos, per-base ref+3alt+5bp-del) recovered and frozen; HEK293FT confirmed activity host; exact mutagenized region declared",
            "missing": "quantitative per-variant activity labels (bcalm logFC) not obtainable within the 50 MB cap",
            "domain_claim_allowed": "within-locus 293FT/H293FT regulatory-VARIANT effect ONLY, NOT general novel promoter prediction"},
        "known_variant SNV/InDel (regulatory variant effects, HEK293FT)": {
            "verdict": "LIMITED",
            "have": "measured ref/alt insert sequences recovered and frozen",
            "missing": "per-variant activity labels (bcalm logFC) not obtainable within the cap"},
        "BAC_MPRA (100 bp genomic bins)": {
            "verdict": "UNRESOLVED / NOT USABLE",
            "why": "labels are 100 bp coverage-bin aggregates, not independent per-insert fragment measurements; must NOT be used as fragment training examples"},
        "Oligo_MPRA (270 bp tiling)": {
            "verdict": "UNRESOLVED (this batch)",
            "why": "separate GEO sample; designfile 26 MB + huge dictionary; labels not confirmed within cap"},
        "autonomous promoter activity": {
            "verdict": "NOT READY (out of domain)",
            "why": "measures CRE/enhancer reporter activity in a fixed lentiMPRA vector, not autonomous promoter activity"}},
    "split_proposal": {
        "principle": "Group by shared PARENT LOCUS + spatial block, never by single SNV position.",
        "independent_regions_counted": {"saturation_positions": len(mpos), "snv_positions": len(snvpos),
                                        "note": "2000 positions/siblings are NOT 2000 independent families; they are one locus."},
        "proposed_rule": "Spatial-block parent-group split on chr17: hold out contiguous >=10 kb blocks (and their exact-RC siblings) so overlapping 219 bp tiles and ref/alt sibling oligos cannot cross train/validation/test.",
        "near_homology_check": "pairwise identity / exact-RC before splitting; RC duplicates go to the SAME group",
        "forbidden": ["random SNV split", "treating 2000 positions as 2000 families", "reusing historical tests as independent"],
        "status": "PROPOSED ONLY; requires parent approval; NO training/scoring in this batch"}}

(REPORTS / "mapt_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

rows = [("arm", "verdict", "measured_sequence_available", "quant_labels_available", "cell_activity_host", "notes")]
for k, v in report["arm_verdicts"].items():
    seqav = "yes" if v.get("have") and "sequence" in v["have"] else "no"
    blob = v.get("have", "") + v.get("why", "")
    host = "HEK293FT" if "293FT" in k else ("HEK293FT" if "HEK293FT" in blob else "n/a")
    rows.append((k, v.get("verdict", ""), seqav, "no", host, (v.get("missing") or v.get("why") or "")[:160]))
with open(REPORTS / "primary_data_readiness.tsv", "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(json.dumps({"n_records": len(recs), "arms": dict(arms), "dup_exact": dup_exact,
                  "frozen_fasta_sha256": report["recovered_measured_sequences"]["frozen_fasta_sha256"],
                  "labels": False}, ensure_ascii=False))
