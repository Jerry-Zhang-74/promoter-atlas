# -*- coding: utf-8 -*-
"""BATCH27 B: assemble autonomous_fragment_data_ready.json from the measured artifacts."""
import json, hashlib, os, time
import numpy as np

R = "reports/breakthrough_20261002/batch27/"
D = "data/breakthrough_20261002/batch27/"
p1 = json.load(open(R + "sure_pass1.json", encoding="utf-8"))
sel = json.load(open(R + "sure_selected_ids.json", encoding="utf-8"))
g1 = json.load(open(R + "sure_group_graph.json", encoding="utf-8"))
g2 = json.load(open(R + "sure_tier2_group_graph.json", encoding="utf-8"))
ss = json.load(open(R + "sure_seq_sample.json", encoding="utf-8"))
cov = json.load(open(R + "sure_selected_covariates.json", encoding="utf-8"))
exp = json.load(open(R + "source_exposure_audit.json", encoding="utf-8"))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

ip = []; r1 = []; r2 = []; pl = []
for k, v in cov.items():
    c = v["counts"]
    ip.append(c[0]); r1.append(sum(c[1:5])); r2.append(sum(c[5:7])); pl.append(sum(c[7:9]))
ip = np.array(ip); r1 = np.array(r1); r2 = np.array(r2); pl = np.array(pl)
def sp(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))
m = (r1 >= 1) & (r2 >= 1)
SL = []
for k, v in cov.items():
    SL.append(v["end"] - v["start"])
SL = np.array(SL)
import collections
STR = collections.Counter(v["strand"] for v in cov.values())
CHR = collections.Counter(v["chrom"] for v in cov.values())

out = {
  "batch": "batch27",
  "version": "v6.19.1-promoter-readiness.rc1 (DATA-READY MANIFEST; no model trained, no estimator fit)",
  "written_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
  "source": {
    "geo_accession": "GSE78709",
    "title": "Genome-wide mapping of autonomous promoter activity in human cells (SuRE)",
    "primary_article_pmc": "PMC5498152",
    "assay": "SuRE (Survey of Regulatory Elements): random genomic fragments 0.2-2 kb cloned 5' of a GFP reporter with a 20 bp barcode, expressed in K562; activity = cDNA / input",
    "genome_build": "hg19 / GRCh37 (declared in the primary methods; coordinate convention verified empirically)",
    "files_used": [
      {"name": "GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz",
       "bytes": 1187342132,
       "sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
       "content_length_checked": True, "gzip_crc_eof_verified": True,
       "gzip_verification_artifact": "runs/breakthrough_20261002/batch27/gzip_big.log",
       "download": "8-way HTTP Range, walltime 137.1 s, artifact runs/breakthrough_20261002/batch27/big_dl.log",
       "role": "PRIMARY per-fragment measured counts table"},
      {"name": "GSE78709_SuRE34.counts.JvA170901.txt.gz",
       "bytes": 11378918,
       "sha256": "240c0ed2e9e31599401b35826a769a7f2d6fe0d59e1c33c9c93432b85efe424b",
       "row_count": 781020,
       "role": "secondary per-fragment table (different SuRE library 'SuRE34', K562 with a hemin/solvent contrast); DOWNLOADED AND RETAINED but NOT USED: its header has 11 fields while data rows have 15, so the schema is not self-consistent and no per-column meaning was asserted"},
      {"name": "tss_test.gz", "role": "authors' TSS-aggregated summary; explicitly NOT a measured-fragment target and NOT used"}
    ],
    "human_sequence_retrieved": {
      "provider": "Ensembl REST GRCh37 (/sequence/region/human)",
      "n_regions_requested": ss["n_requested"], "n_regions_retrieved": ss["n_retrieved"],
      "bytes_scale": "<5 MB (2500 regions x <=2 kb)",
      "artifact": R + "sure_seq_sample.json",
      "fasta": ss["fasta"], "fasta_sha256": ss["fasta_sha256"],
      "response_payload_sha256": ss["response_sha256_concat"]}
  },
  "measured_fragment_row_schema": {
    "columns": ["chrom", "start", "end", "strand", "iPCR SuRE23",
                "cDNA Biorep1 PCRrep1..4", "cDNA Biorep2 PCRrep1..2",
                "plasmid library PCRrep1..2"],
    "coordinates": "0-based half-open [start,end) on hg19/GRCh37; verified against Ensembl GRCh37 by exact interval round-trip",
    "unit_of_row": "one unique measured genomic insert; barcode-level rows are NOT present in this table; per-biorep PCR replicate columns are retained so biological replicate identity is preserved"
  },
  "coordinate_verification": {
    "check": "UCSC hg19 0-based half-open [46660147,46660376) on chr17 == Ensembl GRCh37 17:46660148..46660376 (1-based inclusive); retrieved length 229 == end-start",
    "result": "MATCH",
    "note": "the same interval also appears as a data row in the secondary SuRE34 table, independently confirming the convention"
  },
  "eligibility": {
    "tier1_rule": "row parses to 13 fields; start<end; canonical chromosomes chr1-22,X,Y; 200<=length<=2000; finite non-negative counts; input depth = iPCR + 2 plasmid PCR reps + 1 pseudocount >= 10",
    "tier1_frozen_before_any_scoring": True,
    "tier1_genomewide": p1["n_eligible"],
    "tier2_rule": "tier1 AND cDNA detected in BOTH biological replicates (biorep1 sum >= 1 AND biorep2 sum >= 1). Declared after schema inspection and before any scoring; a detection/measurement-QC rule, NOT an activity-magnitude rule",
    "tier2_genomewide": g2["tier2_cohort_size"],
    "tier2_fraction_of_tier1": g2["tier2_fraction_of_tier1"],
    "exclusions_recorded": ["non-canonical contigs: %d rows" % (p1["n_rows"] - p1["n_canonical_chrom"]),
                            "length outside 200-2000 among canonical rows: %d" % (p1["n_canonical_chrom"] - p1["n_len_200_2000"]),
                            "input depth < 10 among in-window rows: %d" % (p1["n_len_200_2000"] - p1["n_input_ge10"])]
  },
  "pass_counts": {
    "total_rows": p1["n_rows"], "parse_failures": p1["parse_fail"],
    "start_lt_end": p1["n_start_lt_end"], "canonical_chrom": p1["n_canonical_chrom"],
    "length_200_2000": p1["n_len_200_2000"],
    "length_hist_100bp_all_rows": p1["len_hist_100bp"],
    "strand_counts": p1["n_strand"], "n_chromosomes_seen": len(p1["n_chrom"])
  },
  "selection": {
    "rule": sel["selection_rule"],
    "n_selected": sel["n_selected"],
    "label_independent": True,
    "files": [R + "sure_selected_ids.json", R + "sure_selected_covariates.json"],
    "chronology": "eligibility + selection frozen at pass1b completion (18:44:26); labels first read in pass2 (18:44:35-18:48:49)"
  },
  "selected_cohort": {
    "n": sel["n_selected"],
    "length_min": int(SL.min()), "length_median": int(np.median(SL)), "length_max": int(SL.max()),
    "strand_counts": {k: int(v) for k, v in STR.items()}, "n_chromosomes": len(CHR),
    "n_overlap_groups_genomewide_containing_selected": g1["n_groups_containing_selected"],
    "n_groups_with_more_than_one_selected_member": g1["n_groups_with_>1_selected_member"],
    "n_overlap_edges_selected_vs_all_eligible": 410913,
    "group_definition": "connected components of genomic-interval overlap (>=1 bp) over ALL tier-1 eligible inserts genome-wide, restricted to components containing >=1 selected insert",
    "artifact": R + "sure_group_graph.json (pass2 output)"
  },
  "label_units_and_transform": {
    "raw_measured_columns": "per-insert integer counts: iPCR; biorep1 cDNA x4 PCR reps; biorep2 cDNA x2 PCR reps; plasmid library x2 PCR reps",
    "documented_transform": "SuRE activity = (cDNA reads per billion) / (input reads per billion); input = iPCR + plasmid-library counts (+1 pseudocount). This is the primary-paper normalisation; it was NOT computed in this batch (no scoring allowed).",
    "units": "dimensionless ratio (normalised cDNA/input); a log2 may be applied at training time only if declared in advance",
    "measured_depth_on_selected_cohort": {
      "n": 25000,
      "frac_biorep1_ge1": round(float((r1 >= 1).mean()), 6),
      "frac_biorep2_ge1": round(float((r2 >= 1).mean()), 6),
      "frac_both_ge1": round(float(m.mean()), 6),
      "n_both_ge1": int(m.sum()),
      "frac_both_zero": round(float(((r1 == 0) & (r2 == 0)).mean()), 6),
      "biorep1_cDNA_total": int(r1.sum()), "biorep2_cDNA_total": int(r2.sum()),
      "median_iPCR": int(np.median(ip)), "median_plasmid_reads": int(np.median(pl))
    },
    "replicate_agreement_selected_cohort": {
      "spearman_biorep1_vs_biorep2_raw_all": round(sp(r1, r2), 4),
      "spearman_log1p_all": round(sp(np.log1p(r1), np.log1p(r2)), 4),
      "spearman_log1p_both_detected": round(sp(np.log1p(r1[m]), np.log1p(r2[m])), 4),
      "interpretation": "raw Spearman is inflated by the 71% of inserts with zero cDNA in both replicates; among inserts detected in both replicates agreement is only ~0.15, so per-insert activity is NOISY at this sequencing depth. This reflects the label-blind hash-selected cohort, not necessarily the assay's deep subset."
    }
  },
  "tier2_cohort": {
    "n": g2["tier2_cohort_size"],
    "n_overlap_components": g2["n_overlap_components"],
    "n_singleton_components": g2["n_singleton_components"],
    "component_size_max": g2["component_size_max"],
    "n_overlap_edges": g2["n_overlap_edges"],
    "len_min": g2["len_min"], "len_median": g2["len_median"], "len_max": g2["len_max"],
    "n_chromosomes": g2["n_chrom_present"],
    "cohort_table": D + "sure_tier2_cohort.tsv.gz",
    "biorep1_cDNA_total": g2["biorep1_cDNA_total"], "biorep2_cDNA_total": g2["biorep2_cDNA_total"],
    "artifact": R + "sure_tier2_group_graph.json",
    "caveat": "tier-2 uses cDNA detection in both replicates (presence/absence, not activity magnitude); the parent must approve this eligibility before any training run"
  },
  "sequence_retrieval": {
    "sampling_rule": ss["sampling_rule"],
    "n_requested": ss["n_requested"], "n_retrieved": ss["n_retrieved"],
    "sequence_retrieval_fraction": ss["sequence_retrieval_fraction"],
    "length_match_fraction_of_retrieved": ss["length_match_fraction_of_retrieved"],
    "n_length_mismatch": ss["n_length_mismatch"],
    "n_containing_N": ss["n_retrieved_containing_N"],
    "n_exact_duplicate_sequence_pairs_in_sample": ss["n_exact_duplicate_sequence_pairs_in_sample"],
    "n_reverse_complement_pairs_in_sample": ss["n_reverse_complement_pairs_in_sample"],
    "n_http_batches": ss["n_http_batches"], "n_failed_batches": ss["n_failed_batches"],
    "note": "sampled (2500 of 25000) because protocol caps sequence retrieval; the sampled fraction is used to ESTIMATE exact/RC duplication over the cohort"
  },
  "near_homology_bounds": {
    "measured": "exact duplicate sequence pairs and reverse-complement pairs within the 2500-insert sequence sample",
    "n_exact_duplicate_pairs": ss["n_exact_duplicate_sequence_pairs_in_sample"],
    "n_rc_pairs": ss["n_reverse_complement_pairs_in_sample"],
    "not_measured": "all-pairs mismatch near-homology over the full 25000-insert cohort and all-member near-homology over the tier-2 genome-wide cohort",
    "consequence": "the group graph is a LOWER BOUND on dependency: overlap components are exact, but sequence-level near-homology could only INCREASE the number of merges, never decrease it; a split built on these groups is therefore exact for overlap and optimistic for unknown near-homology"
  },
  "split_feasibility": {
    "feasible": True,
    "proposal": "assign whole overlap components (never individual inserts) to train/validation/test by sha256('batch28_sure_seed<seed>|component_root_key'), 70/15/15 by component; components never span chromosomes, so a chromosome-held-out arm is also constructible and should be reported as the stricter variant",
    "not_forced": "no split was chosen to favour any score; no model was applied in this batch",
    "risk": "unknown sequence-level near-homology between components (see near_homology_bounds)"
  },
  "source_exposure_audit": exp,
  "data_ready_gate": {
    "criterion_n_full_measured_inserts_ge_1000": sel["n_selected"] >= 1000,
    "n_full_measured_inserts": sel["n_selected"],
    "criterion_n_independent_groups_ge_500": g2["n_overlap_components"] >= 500,
    "n_independent_groups": g2["n_overlap_components"],
    "correct_assay_targets_sequence_mapping": True,
    "feasible_untouched_grouping": True,
    "gate_met": bool(sel["n_selected"] >= 1000 and g2["n_overlap_components"] >= 500),
    "caveats": [
      "the gate is a FEASIBILITY gate, not evidence of predictive accuracy",
      "the hash-ordered label-blind 25000-insert cohort carries only 754 inserts with both replicates detected; the tier-2 genome-wide cohort (%d inserts) is the recommended training substrate" % g2["tier2_cohort_size"],
      "per-insert replicate agreement is low at moderate depth; a depth-stratified target definition must be predeclared"
    ]
  },
  "prior_sources_kept_separate": [
    "HEK293 rho 0.5264 (batch24/H1) and local-14 rho 0.4286: retrospective/OOD evidence, not comparable to any new SuRE score",
    "HEK293 is not HEK293T; RNA/DNA reporters are not D28-GFP; 14-construct barcode mappings are not independent confirmations",
    "v6.3 TSS+-200/upstream-2000 features are a different task from MPRA-164 and from native SuRE fragments",
    "no old K562 model is claimed to be independently tested on these source-exposed inserts"
  ]
}
json.dump(out, open(R + "autonomous_fragment_data_ready.json", "w", encoding="utf-8"), indent=2, default=str)
print("wrote", R + "autonomous_fragment_data_ready.json")
print("gate_met", out["data_ready_gate"]["gate_met"])
