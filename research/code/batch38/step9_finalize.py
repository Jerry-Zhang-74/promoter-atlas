# -*- coding: utf-8 -*-
"""Batch38 finalize: A audit_decision, B cache_reconciliation, C readiness, D report/figures.
ZERO fits / ZERO GPU. Old assets read-only."""
import json, os, csv, hashlib, datetime
def ts2(): return datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def jdump(o,p,indent=1):
    with open(p,"w",encoding="utf-8") as f:
        json.dump(o,f,indent=indent,ensure_ascii=False,allow_nan=False)
def nan2null(x):
    if isinstance(x,float) and (x!=x or x in (float('inf'),float('-inf'))): return None
    return x
OUTD="data/breakthrough_20261002/batch38"
REP="reports/breakthrough_20261002/batch38"
FIG="figures/batch38"
os.makedirs(REP,exist_ok=True); os.makedirs(FIG,exist_ok=True)
s6=json.load(open(os.path.join(OUTD,"step6_summary.json"),encoding="utf-8"))
s7=json.load(open(os.path.join(OUTD,"step7_groups.json"),encoding="utf-8"))
sC=json.load(open(os.path.join(OUTD,"workbook_s2_s5_summary.json"),encoding="utf-8"))
b37m=json.load(open("reports/breakthrough_20261002/batch37/metrics.json",encoding="utf-8"))
DESIGN="data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz"
KV="data/breakthrough_20261002/batch36/mapt/cohort_KV_SNV_HEK293FT.tsv"
SAT="data/breakthrough_20261002/batch36/mapt/cohort_SAT_HEK293FT.tsv"
KEY="data/breakthrough_20261002/batch38/variant_reference_key.tsv"
DRM="data/breakthrough_20261002/batch38/design.rmd"
HSR="data/breakthrough_20261002/batch38/heks_snvs.R"
FIV="data/breakthrough_20261002/batch38/fiveallelesmpradesign.tsv"
XLS="data/breakthrough_20261002/batch36/mapt/media-1.xlsx"
sources={}
for p in [DESIGN,KV,SAT,KEY,DRM,HSR,FIV,XLS]:
    sources[os.path.basename(p)]={"path":p.replace("\\","/"),"sha256":sha(p),
      "mtime":datetime.datetime.fromtimestamp(os.path.getmtime(p)).astimezone().isoformat()}

# ---------------- A: audit_decision.json ----------------
audit_dec={
 "batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1",
 "record_type":"audit_decision","phase":"A_source_identity_and_contrast_audit","generated_at":ts2(),
 "zero_gpu":True,"zero_fits":True,"calibration":0,"optimizer_steps":0,"neural_forward":0,
 "frozen_before_metrics":True,
 "parent_acceptance":"runs/breakthrough_20261002/batch38/PARENT_ACCEPTANCE.txt",
 "frozen_rules":{
  "rule_scope":"SEQUENCE IDENTITY ONLY; no cached model scores and no label values were consulted to choose any rule or interpretation",
  "inputs":"author design FASTA all 36000 physical records (35100 distinct headers + 900 extra records duplicating a header), author variant_reference_key.tsv, author design.rmd, author heks_snvs.R, batch36 SAT/KV cohorts unchanged",
  "rules":[
   "R1 ref header = design header with trailing _altseq[_R_to_A] removed and _referenceseq appended, MUST equal author variant_reference_key.tsv control for that id",
   "R2 load ALL physical design records; if a header has >1 record verify sequences byte-identical (no hidden conflict)",
   "R3 for each pair compute every mismatch index between alt oligo and key/derived control oligo",
   "R4 classify: IDENTICAL_ALT_EQ_CONTROL if 0 mismatches; SINGLE_SUB_AT_125 if exactly one mismatch at 125; SINGLE_SUB_AT_124 if at 124; else OTHER",
   "R5 ALT allele index i where alt[i]==base_alt and control[i]!=base_alt",
   "R6 no base editing, no inventing refs, no choosing RC/sign/offset to maximise rho, no dropping the 21.5%"]
 },
 "sources":sources,
 "mapping_reconstruction":{
   "author_key_coverage":{"kv_rows":8513,"key_present":s6["key_agree"]+s6["key_disagree"],
     "control_eq_derived_ref":s6["key_agree"],"disagree":s6["key_disagree"],"missing":s6["key_missing"]},
   "author_key_example":{"id":"SNV_chr17_44904682_altseq_C_to_G","control":"SNV_chr17_44904682_referenceseq"},
   "design_dup_header_conflicts":s6["dup_conflict"],
   "conclusion":"batch37 derived reference header (<chr>_<pos>_referenceseq) equals the author variant_reference_key.tsv control for all 8513 KV rows. This is NOT a pipeline mis-mapping."
 },
 "offset_finding":{
   "design_facts":{"oligo_len_bp":249,
     "layout":"15bp fadapter AGGACCGGATCAACT + 219bp variable insert + 15bp radapter CATTGCGTGAACCGA",
     "alt_allele_index_observed":125,"reference_base_index_observed":124,
     "ref124_eq_base_ref":s6["ref124_eq_base_ref"],"alt125_eq_base_alt":s6["alt125_eq_base_alt"],
     "paired_rows":8513,"ref125":"genome base at POS+1, not the header base_ref"},
   "author_root_cause":{
     "file":"Variation_and_MAPT_promoter_mutagenesis_MPRA/design.rmd",
     "sha256":sources["design.rmd"]["sha256"],
     "SNV_code":"middlepos <- ceiling(region_length/2) = 110; altseq = paste0(substr(refseq,1,middlepos), ALT, substr(refseq,middlepos+2,nchar(refseq))) -> replaces 1-based 111 = design index 125",
     "SNV_refcheck":"substr(referenceseq,middlepos,middlepos)==REF -> checks 1-based 110 = design index 124",
     "maptprom_SAT_code":"half_region <- (region_length-1)/2 = 109; altseq1 = paste0(substr(refseq,1,109), ALT, substr(refseq,111,...)) -> replaces 1-based 110 = design index 124 (CORRECT)",
     "shared_reference_window":"start=POS-109, end=POS+109 for BOTH; SNV reference records == maptprom reference records at shared positions",
     "conclusion":"The SNV library places the ALT allele ONE bp 3-prime of the reference-base centre (design index 125 vs SAT/maptprom index 124). Source-reproducible AUTHOR-SIDE design inconsistency / off-by-one, not a downstream pipeline error."
   }
 },
 "classification_whole_source":{
   "n_rows":8513,"SINGLE_SUB_AT_125":s6["class"].get("SINGLE_SUB_AT_125",0),
   "IDENTICAL_ALT_EQ_CONTROL":s6["class"].get("IDENTICAL_ALT_EQ_CONTROL",0),
   "OTHER":sum(v for k,v in s6["class"].items() if k not in ("SINGLE_SUB_AT_125","IDENTICAL_ALT_EQ_CONTROL")),
   "diff_position_histogram":s6["diffpos_hist"],
   "alt_allele_placement_histogram":s6["placement"]
 },
 "case_categories":{
   "pipeline_wrong_mapping":0,"author_design_inconsistency":8513,
   "coordinate_convention_vs_genome":8513,
   "alternative_allele_already_reference_at_substituted_coord":1833,
   "truly_identical_oligos":1833,"unresolved":0,
   "note":"Every KV row shares the author design.rmd +1 ALT placement. The 1833 IDENTICAL rows are the subset where the ALT allele equals the genome base at the substituted coordinate POS+1, so the alt oligo is byte-identical to its control oligo. No case is left unresolved for the source-identity question."
 },
 "bcalm_contrast":{
   "source":"Variation_and_MAPT_promoter_mutagenesis_MPRA/bcalm/heks_snvs.R",
   "sha256":sources["heks_snvs.R"]["sha256"],
   "key_file":"variant_reference_key.tsv columns id,control; the SNV bcalm renames id->ID, control->REF and ALT:=id",
   "contrast":"ALT oligo (id) vs control (REF = pos_referenceseq); design column alt = grepl(alt, colnames); mpralm model_type=corr_groups, aggregate=none, normalize=TRUE, coef=2",
   "numerator":"ALT oligo","denominator":"control = reference-sequence oligo",
   "sign":"positive logFC => higher activity of the ALT oligo relative to its reference control",
   "paper_text":"variant effects quantified by bcalm with a |logFC| cut-off of 0.1 by comparing to reference sequence controls",
   "log_base":"UNRESOLVED from the script alone (mpralm topTable default); sign direction stated but numeric base of logFC NOT assumed",
   "conclusion":"batch37 paired contrast (alt oligo vs pos_referenceseq) MATCHES the author call chain and control definition. Model d=score(alt)-score(control) and the label use the SAME two oligos."
 },
 "decision":{
   "overall":"SOURCE_RESOLVED",
   "corpus_correction":"NONE required. The reference mapping is author-confirmed; the 21.5% are author-design no-op oligo pairs, not pipeline errors, and NOT to be silently dropped.",
   "degenerate_21_5pct":"1833/8513 rows: ALT allele == genome base at POS+1 -> alt oligo sequence-identical to control oligo (no encoded change); the author design.rmd still emits a control for them and bcalm reports an unchanged-sequence effect.",
   "preflight_reinterpretation":"the header/coordinate join is 8513/8513 = 100% and author-confirmed; the earlier 78.47% valid fraction measures the fraction with a NET single-base sequence change, not join coverage.",
   "retention":"all 8513 rows retained in the primary all-rows view; the 6680 single-substitution rows are a source-valid secondary stratum, NOT post-hoc promotion.",
   "no_retroactive_b37_pass":"This decision does NOT convert batch37 verdict to PASS; it only resolves the audit caveat. batch37 metrics/verdict unchanged.",
   "no_base_edit":True,"no_invented_ref":True,"no_rc_sign_offset_tuned_for_rho":True,"no_rows_dropped":True
 },
 "groups":{
   "interval_union_groups":s7["interval_union_groups"],"interval_min":s7["min"],"interval_max":s7["max"],
   "interval_size_histogram":s7["size_hist"],"n_groups_ge10pairs":s7["n_ge10"],
   "gap_clusters_2kb":s7["gap_clusters_2kb"],
   "exact_sequence_groups":s7["exact_groups"],"exact_max_multiplicity":s7["exact_max_mult"],
   "rc_merged_groups":s7["rc_groups"],"rc_max_multiplicity":s7["rc_max_mult"],
   "note":"interval-union (+-109bp) gives 180 groups sized 11..271 (all >=10 VERIFIED). exact-sequence and RC-merged groupings are 8513 singletons (all alt inserts unique). >30 groups is proximity, NOT verified CRE-family independence."
 },
 "sat_control_check":{"n_checked":len(s6["sat_checks"]),"rows":s6["sat_checks"],
   "note":"mixed SAT rows: SNV mismatch at design index 124 (alt124=base_alt, ref124=base_ref); deletions centred 5bp removals; each has a variant_reference_key control. Confirms maptprom/SAT centring 124 differs from the SNV library 125."}
}
jdump(audit_dec,os.path.join(REP,"audit_decision.json"))
print("wrote audit_decision.json")

# ---------------- B: cache_reconciliation.json ----------------
B={
 "batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1","record_type":"cache_reconciliation",
 "generated_at":ts2(),"optional":True,"applied":"no_correction_required",
 "rationale":"Part A decision is deterministic and AUTHOR-CONFIRMED: the reference mapping equals the author variant_reference_key.tsv control for 8513/8513 rows, so there is NO corrected pair set. No label/contrast change is warranted and none is made.",
 "source_of_numbers":"reports/breakthrough_20261002/batch37/metrics.json (frozen, read-only; no rescoring; batch37 unique_scores untouched)",
 "strata":{
  "original_primary_all_rows":{"n":b37m["kv_all"]["n_rows"],"n_label_finite":b37m["kv_all"]["n_label_finite"],
    "n_pred_ties_zero_d":b37m["kv_all"]["n_pred_ties_zero_d"],
    "spearman_d_logFC":nan2null(b37m["kv_all"]["spearman_d_logFC"]),
    "deltaGC_baseline_rho":nan2null(b37m["kv_all"]["spearman_deltaGC_logFC"])},
  "source_valid_diagnostic_secondary_single_sub":{"n":b37m["kv_valid_only"]["n_rows"],
    "n_label_finite":b37m["kv_valid_only"]["n_label_finite"],
    "spearman_d_logFC":nan2null(b37m["kv_valid_only"]["spearman_d_logFC"]),
    "deltaGC_baseline_rho":nan2null(b37m["kv_valid_only"]["spearman_deltaGC_logFC"])},
  "identical_alt_eq_control":{"n":b37m["kv_degenerate_only"]["n_rows"],
    "spearman_d_logFC":nan2null(b37m["kv_degenerate_only"]["spearman_d_logFC"]),"note":"undefined: all d=0"}
 },
 "reconciled":{"n":b37m["kv_all"]["n_rows"],"spearman_d_logFC":nan2null(b37m["kv_all"]["spearman_d_logFC"]),
   "identical_to_original":True,"reason":"author-confirmed mapping -> no pair/label change"},
 "null_for_undefined":{"kv_degenerate_only_spearman_d_logFC":"null (undefined: constant predictor, all d=0; reported null WITH this reason, NOT a real 0)"},
 "families":"no new family CI claimed; the 180 interval-union groups are proximity components. Simple pooled/SAT/KV metrics sufficient for the future protocol.",
 "old_assets_unchanged":True
}
jdump(B,os.path.join(REP,"cache_reconciliation.json"))
print("wrote cache_reconciliation.json")

# ---------------- C: readiness_C.json ----------------
C={
 "batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1","record_type":"fragment_activity_readiness",
 "generated_at":ts2(),"workbook":"data/breakthrough_20261002/batch36/mapt/media-1.xlsx",
 "workbook_sha256":sha(XLS),
 "sheets_audited":["s2-BAC MPRA Neurons","s3-BAC MPRA HEKs","s4-OligoMPRA Neurons","s5-OligoMPRA Heks"],
 "sheet_structure":{
  "s2_BAC_MPRA_Neurons":{"n_rows_incl_header":sC["s2"]["n_rows_incl_header"],"header":sC["s2"]["header"],
    "unit":"100bp bin (start..end=start+99)","per_oligo_activity":False},
  "s3_BAC_MPRA_HEKs":{"n_rows_incl_header":sC["s3"]["n_rows_incl_header"],"header":sC["s3"]["header"],
    "unit":"100bp bin","per_oligo_activity":False},
  "s4_OligoMPRA_Neurons":{"n_rows_incl_header":sC["s4"]["n_rows_incl_header"],"header":sC["s4"]["header"],
    "unit":"~270bp oligo (orientation in label)","per_oligo_activity":True},
  "s5_OligoMPRA_Heks":{"n_rows_incl_header":sC["s5"]["n_rows_incl_header"],"header":sC["s5"]["header"],
    "unit":"~270bp oligo (orientation in label)","per_oligo_activity":True}
 },
 "question_answer":"YES - the HEK293FT (HEKs) OligoMPRA arm (s5) carries a PER-OLIGO (~270bp) reporter-activity statistic (statistic / zscore / mad.score / pval.empirical / pval.mad / pval.zscore), unlike the BAC 100bp-bin arms (s2/s3) and unlike the variant logFC arms.",
 "unit_sample_rep_orientation_control_minP_vector":{
   "unit":"270bp oligo","sample":"HEK293FT (HEKs)","rep":"aggregated (not per-replicate in the sheet)",
   "orientation":"present in the label (+/-)","control":"column control present (0 in shown rows) - exact meaning UNKNOWN",
   "minP":"empirical/mad/zscore p-values present","vector":"NOT stated in the sheet; a fixed reporter vector alone cannot prove autonomous vs nonautonomous activity"
 },
 "clarity":"CLARITY = has PER-OLIGO activity (present); DEFINITION/NORMALISATION = UNKNOWN (needs the Oligo MPRA MPRAflow/MPRAnalyze call chain, not fetched within the 4-request source cap).",
 "next_join":{"design_file":"GSM9609016_designfile.fa.gz (GEO GSE325670, ~25.9 MB) - the referenced 26MB Oligo MPRA design file",
   "status":"NOT local; NOT fetched (the 4 authorised author-repo source requests for batch38 are exhausted)",
   "feasible":"YES bounded: ~25.9MB fits the 35MB budget (batch38 used ~2.24MB) but exceeds the <=4 request cap -> deferred to the next batch source budget",
   "join_key":"s5 label name_orientation::chr:start-end -> design oligo sequence -> per-oligo activity target"},
 "not_done":["NO DNA fabricated from bins or unknown reference","NO scoring/fitting of this arm","NO claim that all SuRE/MPRA is out of scope from one assay"],
 "decision":"READINESS = per-oligo 270bp HEK293FT activity EXISTS in the author workbook. One finite next step: fetch GSM9609016_designfile.fa.gz (26MB) plus the oligo MPRA MPRAnalyze script under a fresh source budget, then join sequences to the s5 statistic for a genuine per-fragment activity target (subject to author call-chain review)."
}
jdump(C,os.path.join(REP,"readiness_C.json"))
print("wrote readiness_C.json", ts2())
