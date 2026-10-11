# -*- coding: utf-8 -*-
"""BATCH27 D: concrete proposed next protocol (parent must approve; worker does NOT train)."""
import json, time
R = "reports/breakthrough_20261002/batch27/"
now = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")
dr = json.load(open(R + "autonomous_fragment_data_ready.json", encoding="utf-8"))
g2 = json.load(open(R + "sure_tier2_group_graph.json", encoding="utf-8"))

prop = {
 "proposal_id": "batch28_v6.20.0-surelong.rc1",
 "written_at": now,
 "status": "PROPOSED ONLY - parent must approve and launch; the worker trained nothing and ran no estimator",
 "task": "Predict measured autonomous promoter activity of NATIVE K562 SuRE genomic inserts (200-2000 bp) from DNA sequence alone.",
 "source": {
   "geo": "GSE78709", "primary_article": "PMC5498152",
   "table": "GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz",
   "sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
   "genome_build": "hg19/GRCh37", "sequence_retrieval": "Ensembl GRCh37 REST (verified for 2500/2500 sampled inserts)"
 },
 "cohort": {
   "tier1_eligible_genomewide": dr["eligibility"]["tier1_genomewide"],
   "tier2_cohort": g2["tier2_cohort_size"],
   "tier2_rule": g2["tier2_rule"],
   "independent_overlap_groups": g2["n_overlap_components"],
   "component_size_max": g2["component_size_max"],
   "length_range_bp": [g2["len_min"], g2["len_max"]], "length_median_bp": g2["len_median"],
   "require_parent_approval_of_tier2": True
 },
 "split": {
   "unit": "whole overlap component (never a single insert)",
   "rule": "u = sha256('batch28_sure_seed20261101|' + component_root_key)[:16]/2^64; u<0.70 train, <0.85 val, else test",
   "blocked_arm": "report additionally a chromosome-held-out arm (train chr1-16, val chr17-19, test chr20-22,X,Y) as the STRICTER variant; the primary split stays the component-based one",
   "untouched_test": True, "near_homology_caveat": dr["near_homology_bounds"]["consequence"]
 },
 "targets_and_masks": {
   "unit": "one measured insert",
   "raw": "per-insert integer counts (iPCR, biorep1 cDNA x4 PCR reps, biorep2 cDNA x2 PCR reps, plasmid x2)",
   "transform": "activity_r = (cDNA_r / N_cDNA_r) / ((iPCR + plasmid) / N_input); target = 0.5*(log2(a1+p) + log2(a2+p)), p = pseudocount predeclared as the cohort-wide median nonzero count; NO per-insert calibration",
   "mask": "insert is used only if biorep1 cDNA >= 1 AND biorep2 cDNA >= 1 (tier-2)",
   "sequence_input": "full measured insert; pad/truncate to a FIXED 2048 bp with an explicit per-base mask channel (1=measured base, 0=pad); never use the last 133/164 bp of anything and never invent or pad deleted reference bases",
   "aux_features": "none that use position, TSS annotation or gene identity (keeps the task composition-only and comparable to the baseline)"
 },
 "architecture": {
   "A_baseline": "composition ridge: GC, length, 4-mer/5-mer frequencies + batch16-style count features on the SAME split (this is the matched baseline)",
   "B_cnn": "1D CNN, input 2048 x 5 (A,C,G,T,mask), 6 residual blocks, width 128, global max+mean pooling, 1 scalar output",
   "C_control": "same CNN, train labels shuffled within the training split only (sanity control)"
 },
 "executions": {
   "max_total_including_baseline": 4,
   "planned": ["E1 ridge baseline", "E2 CNN seed 20261101", "E3 CNN seed 20261102", "E4 shuffled-label CNN seed 20261101"],
   "gate": "runs/breakthrough_20261002/code/batch27/exec_gate.py must be used; a 5th reservation must be rejected before the callback",
   "gpu_budget": "<=3 GPUh total on cuda:0 RTX5060; batch<=32; contemporaneous NVIDIA/model/input/PID evidence required; no retrospective telemetry"
 },
 "hyperparameters_fixed_in_advance": {
   "optimizer": "Adam", "lr": 0.001, "weight_decay": 0.0, "batch": 32,
   "epochs": 12, "early_stop": "none - selection is by best validation epoch, not by patience",
   "loss": "masked Huber (delta=1.0) on the transformed target",
   "seed_policy": "seeds fixed per attempt, model init and data order deterministic",
   "no_tuning_after_test": True
 },
 "validation_selection": {
   "metric": "group-balanced MAE on validation groups (mean |err| within component, then mean over components)",
   "rule": "select the lowest; ties -> lower seed",
   "test_read_once": "single read after selection; the test set is not used for any choice"
 },
 "practical_ranking_criteria": {
   "primary_spearman_ge": 0.30,
   "group_balanced_mae_improvement_over_matched_baseline_ge": 0.10,
   "top_decile_precision": "report precision and enrichment of the predicted top decile of inserts vs the measured top decile (k = ceil(0.1 n), ties by sha256 of the insert key)",
   "both_seeds_above_baseline": True,
   "shuffled_control": "must produce a defined but near-zero correlation; if constant predictions occur, report as UNDEFINED (as in batch26) rather than omitting",
   "promotion": "not automatic - parent reviews the independent review and decides"
 },
 "expected_difficulty_note": "per-insert replicate agreement in this cohort is low at moderate sequencing depth (see replicated_agreement in the data-ready manifest); a modest ceiling is expected and should be stated before the run, not after",
 "explicitly_out_of_scope": ["no use of the batch26 STR/minP or HEK H1/R1 domains", "no HepG2/HEK counts",
                             "no TSS-window surrogate target", "no fine-tuning of any old model on these inserts",
                             "no claim that an old K562 model is independently tested here"]
}
json.dump(prop, open(R + "next_batch_proposal.json", "w", encoding="utf-8"), indent=2)
print("wrote next_batch_proposal.json")
