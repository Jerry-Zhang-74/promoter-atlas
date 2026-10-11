# -*- coding: utf-8 -*-
"""batch50 evidence: figure_inputs.json + 3 figures (PNG+SVG). ZERO fit/solve/prediction/neural/GPU/web."""
import json, hashlib
from pathlib import Path
ROOT = Path("E:/promoter_atlas")
OUT  = ROOT / "reports/breakthrough_20261002/batch50"
FIG  = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
S = {
 "b31_metrics":  sha("reports/breakthrough_20261002/batch31/metrics.json"),
 "b31_pairs":    sha("reports/breakthrough_20261002/batch31/tables/local_pair_directions.tsv"),
 "b31_scores":   sha("reports/breakthrough_20261002/batch31/tables/local_supported_scores.tsv"),
 "b33_status":   sha("reports/breakthrough_20261002/batch33/status.json"),
 "b41_review":   sha("reports/breakthrough_20261002/batch41/independent_review.json"),
 "b42_metrics":  sha("reports/breakthrough_20261002/batch42/metrics.json"),
 "b43_decision": sha("reports/breakthrough_20261002/batch43/decision.json"),
 "b43_corr":     sha("reports/breakthrough_20261002/batch43/correction_note.json"),
 "b49_breadth":  sha("reports/breakthrough_20261002/batch49/breadth_counts.json"),
 "b49_usage":    sha("reports/breakthrough_20261002/batch49/USAGE_zh.md"),
 "b49_minB10":   sha("data/breakthrough_20261002/batch49/out/gene_rank_minB_1.0.json"),
 "stage_ledger": sha("reports/breakthrough_20261002/stage_email/stage_email_ledger.jsonl"),
}
m31 = json.loads((ROOT/"reports/breakthrough_20261002/batch31/metrics.json").read_text(encoding="utf-8"))
st33= json.loads((ROOT/"reports/breakthrough_20261002/batch33/status.json").read_text(encoding="utf-8"))
r41 = json.loads((ROOT/"reports/breakthrough_20261002/batch41/independent_review.json").read_text(encoding="utf-8"))
m42 = json.loads((ROOT/"reports/breakthrough_20261002/batch42/metrics.json").read_text(encoding="utf-8"))
d43 = json.loads((ROOT/"reports/breakthrough_20261002/batch43/decision.json").read_text(encoding="utf-8"))
bc49= json.loads((ROOT/"reports/breakthrough_20261002/batch49/breadth_counts.json").read_text(encoding="utf-8"))
mb10= json.loads((ROOT/"data/breakthrough_20261002/batch49/out/gene_rank_minB_1.0.json").read_text(encoding="utf-8"))
fams = m31["local"]["families"]
print("chunk1 ok")

A = {
 "actb":  fams["ACTB"]["direction_accuracy"],  "actb_n":  fams["ACTB"]["n_pairs"],
 "gapdh": fams["GAPDH"]["direction_accuracy"], "gapdh_n": fams["GAPDH"]["n_pairs"],
 "cmv":   fams["CMV"]["direction_accuracy"],   "cmv_n":   fams["CMV"]["n_pairs"],
 "actb_inc": fams["ACTB"]["baseline_increasing_len_accuracy"],
 "actb_dec": fams["ACTB"]["baseline_decreasing_len_accuracy"],
 "gapdh_inc":fams["GAPDH"]["baseline_increasing_len_accuracy"],
 "gapdh_dec":fams["GAPDH"]["baseline_decreasing_len_accuracy"],
 "cmv_inc":  fams["CMV"]["baseline_increasing_len_accuracy"],
 "cmv_dec":  fams["CMV"]["baseline_decreasing_len_accuracy"],
 "macro_acc":m31["local"]["macro_family_accuracy"],
 "macro_inc":m31["local"]["macro_baseline_increasing"],
 "macro_dec":m31["local"]["macro_baseline_decreasing"],
}
A["actb_corr"]  = int(round(A["actb"]*A["actb_n"]))
A["gapdh_corr"] = int(round(A["gapdh"]*A["gapdh_n"]))
A["cmv_corr"]   = int(round(A["cmv"]*A["cmv_n"]))
A["pooled_correct"] = A["actb_corr"] + A["gapdh_corr"] + A["cmv_corr"]
A["pooled_total"]   = A["actb_n"] + A["gapdh_n"] + A["cmv_n"]
assert abs(A["actb"])<1e-12 and abs(A["gapdh"]-0.2)<1e-12 and abs(A["cmv"]-1.0)<1e-12
assert abs(A["macro_acc"]-0.4)<1e-12 and abs(A["macro_inc"]-0.5666666666666667)<1e-12
assert A["actb_corr"]==0 and A["gapdh_corr"]==2 and A["cmv_corr"]==1
assert A["pooled_correct"]==3 and A["pooled_total"]==21
p41 = r41["independent_numbers"]["pooled_rho"]
t41 = r41["independent_numbers"]["top10_precision"]
ci41= r41["independent_numbers"]["parm_minus_h1_ci_cond"]
B = {"parm_rho":p41["parm"],"h1_rho":p41["h1"],"gc_rho":p41["gc"],
     "parm_top":t41["parm"],"h1_top":t41["h1"],"gc_top":t41["gc"],
     "parm_minus_h1":r41["independent_numbers"]["parm_minus_h1"],
     "ci_lo":ci41[0],"ci_hi":ci41[1]}
assert abs(B["parm_rho"]-0.10837903133655544)<1e-15
assert abs(B["parm_top"]-0.10344827586206896)<1e-15
assert abs(B["h1_top"]-0.13660477453580902)<1e-15
assert abs(B["gc_top"]-0.15517241379310345)<1e-15
print("chunk2 ok")

C = {"R0_rho":m42["R0"]["pooled_spearman"],"R1_rho":m42["R1"]["pooled_spearman"],
     "Rnull_rho":m42["Rnull"]["pooled_spearman"],
     "R0_mae":m42["R0"]["group_balanced_mae"],"R1_mae":m42["R1"]["group_balanced_mae"],
     "Rnull_mae":m42["Rnull"]["group_balanced_mae"],
     "const_mae":m42["zero_fit_train_weighted_mean_baseline"]["group_balanced_mae"],
     "const_val":m42["zero_fit_train_weighted_mean_baseline"]["const"],
     "n_test":m42["n_test_rows"],"n_groups":m42["n_test_groups"],"n_chroms":m42["n_test_chroms"],
     "d_rho":m42["decision"]["delta_rho"],"verdict":m42["decision"]["verdict"],
     "ci_avail":m42["decision"]["bootstrap_available"],"ci_reason":m42["decision"]["bootstrap_reason"]}
assert abs(C["R1_rho"]-0.09105537269295995)<1e-15 and abs(C["R0_rho"]-0.15840880496925192)<1e-15
assert abs(C["Rnull_rho"]+0.05438566408664658)<1e-15
assert abs(C["const_mae"]-0.09772764731372789)<1e-15 and C["n_groups"]==21 and C["ci_avail"] is False
D = d43["decomposition_answer"]["TEST_row_weighted_within_frac"]
E = {"R0":D["R0"],"R1":D["R1"],"Rnull":D["Rnull"],"const":D["const"]}
assert abs(E["R0"]-0.8496576753380212)<1e-15
b49_def = {"default_frontier":bc49["thresholds"]["0.30"]["n_frontier"],
           "default_eligible":bc49["thresholds"]["0.30"]["eligible"],
           "minB10_below_E":mb10["counts"]["below_E"],
           "minB10_eligible":mb10["counts"]["eligible"],
           "minB10_sum_ok":mb10["counts"]["counts_sum_equals_full_source"]}
assert b49_def["default_frontier"]==12 and b49_def["default_eligible"]==18947
assert b49_def["minB10_below_E"]==0 and b49_def["minB10_eligible"]==8968 and b49_def["minB10_sum_ok"] is True
def rec(val, units, pop, held, src, ptr, null_reason=None, note=""):
    return {"value": (None if val is None else (round(float(val),6) if isinstance(val,(int,float)) else val)),
       "null_reason": null_reason, "units":units,"population":pop,"heldout_status":held,
       "source_file":src,"json_pointer":ptr,"source_sha256":S[src],"note":note}
print("chunk3 ok")

fi = {"report_version":"v6.36.0-morning-route-evidence.rc1",
 "deliverable":"THREE_figures + ROUTE_zh (report revision only; NOT a model version)",
 "rule":"every displayed number maps to exact source file + JSON pointer/table row + SHA + units/population/heldout status",
 "new_predictions_or_metrics_estimation": False}
B31F="/local/families/"
fi["fig1_local293ft_applicability"]={
 "endpoint":"D28 GFP-positive gated median / same-batch WT-negative median (log10); b31 local",
 "ACTB_PARM_direction_accuracy": rec(A["actb"],"fraction of ordered pairs","5 constructs/10 pairs","retrospective_D28_exposed","b31_metrics",B31F+"ACTB/direction_accuracy"),
 "GAPDH_PARM_direction_accuracy":rec(A["gapdh"],"fraction of ordered pairs","5 constructs/10 pairs","retrospective_D28_exposed","b31_metrics",B31F+"GAPDH/direction_accuracy"),
 "CMV_PARM_direction_accuracy":  rec(A["cmv"],"fraction of ordered pairs","2 constructs/1 pair","retrospective_D28_exposed","b31_metrics",B31F+"CMV/direction_accuracy"),
 "ACTB_baseline_increasing_len": rec(A["actb_inc"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"ACTB/baseline_increasing_len_accuracy"),
 "ACTB_baseline_decreasing_len": rec(A["actb_dec"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"ACTB/baseline_decreasing_len_accuracy"),
 "GAPDH_baseline_increasing_len":rec(A["gapdh_inc"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"GAPDH/baseline_increasing_len_accuracy"),
 "GAPDH_baseline_decreasing_len":rec(A["gapdh_dec"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"GAPDH/baseline_decreasing_len_accuracy"),
 "CMV_baseline_increasing_len":  rec(A["cmv_inc"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"CMV/baseline_increasing_len_accuracy"),
 "CMV_baseline_decreasing_len":  rec(A["cmv_dec"],"fraction","fixed zero-fit baseline","n/a_baseline","b31_metrics",B31F+"CMV/baseline_decreasing_len_accuracy"),
 "macro_family_accuracy":rec(A["macro_acc"],"mean of 3 family accuracies","3 families","retrospective_D28_exposed","b31_metrics","/local/macro_family_accuracy",None,"3 families NOT a generalization CI"),
 "ACTB_correct_of_total":"0/10","GAPDH_correct_of_total":"2/10","CMV_correct_of_total":"1/1",
 "pooled_correct_over_total":rec(A["pooled_correct"]/A["pooled_total"],"fraction","21 non-independent pairs","retrospective_D28_exposed","b31_metrics","/local/families",None,"pairs are NOT independent n"),
 "A20_G20_2000bp":rec(None,"n/a","2 constructs (A20,G20)","n/a_unscored","b31_metrics","/local/unsupported","len=2000 > official L_max=600; not scored; observed only"),
 "provenance_limit":"3 technical tubes NOT biological n; family counts NOT independent pairs; assay->plasmid barcode UNVERIFIED",
}
print("chunk4 ok")

fi["fig2_public_transfer_sameassay"]={
 "b41_scope":"public HEK293FT OligoMPRA getAlpha activity; 270bp insert raw[15:285]; 7537 bio; frozen PARM/H1/GC; ZERO fit",
 "b41_pooled_rho_PARAM":rec(B["parm_rho"],"Spearman rho (rank)","7537 bio rows","frozen_transfer_public","b41_review","/independent_numbers/pooled_rho/parm"),
 "b41_pooled_rho_H1":   rec(B["h1_rho"],"Spearman rho","7537 bio rows","frozen_transfer_public","b41_review","/independent_numbers/pooled_rho/h1"),
 "b41_pooled_rho_GC":   rec(B["gc_rho"],"Spearman rho","7537 bio rows","frozen_baseline_public","b41_review","/independent_numbers/pooled_rho/gc"),
 "b41_top10_precision_PARAM":rec(B["parm_top"],"precision","k=ceil(0.1*n) ranks","frozen_transfer_public","b41_review","/independent_numbers/top10_precision/parm"),
 "b41_top10_precision_H1":   rec(B["h1_top"],"precision","k ranks","frozen_transfer_public","b41_review","/independent_numbers/top10_precision/h1"),
 "b41_top10_precision_GC":   rec(B["gc_top"],"precision","k ranks","frozen_baseline_public","b41_review","/independent_numbers/top10_precision/gc"),
 "b41_parm_minus_h1":rec(B["parm_minus_h1"],"rho difference","paired 306 components","conditional_public","b41_review","/independent_numbers/parm_minus_h1"),
 "b41_parm_minus_h1_CI_lo":rec(B["ci_lo"],"rho diff","conditional component bootstrap","conditional_public","b41_review","/independent_numbers/parm_minus_h1_ci_cond/0"),
 "b41_parm_minus_h1_CI_hi":rec(B["ci_hi"],"rho diff","conditional component bootstrap","conditional_public","b41_review","/independent_numbers/parm_minus_h1_ci_cond/1"),
 "b41_gate":"PARM pooled rho >= 0.30; PARM does NOT clear -> LIMITED",
 "b42_scope":"retrospective assay-specific pilot; frozen union split; 3 fixed fits; R0 composition ridge / R1 full-seq ridge / Rnull permuted-label R1; same alpha units",
 "b42_R0_groupMAE":rec(C["R0_mae"],"alpha units","TEST 21 groups","exposed_retrospective","b42_metrics","/R0/group_balanced_mae"),
 "b42_R1_groupMAE":rec(C["R1_mae"],"alpha units","TEST 21 groups","exposed_retrospective","b42_metrics","/R1/group_balanced_mae"),
 "b42_Rnull_groupMAE":rec(C["Rnull_mae"],"alpha units","TEST 21 groups","exposed_retrospective","b42_metrics","/Rnull/group_balanced_mae"),
 "b42_const_groupMAE":rec(C["const_mae"],"alpha units","TEST 21 groups","exposed_retrospective","b42_metrics","/zero_fit_train_weighted_mean_baseline/group_balanced_mae"),
 "b42_R0_rho":rec(C["R0_rho"],"Spearman rho","TEST 318 rows","exposed_retrospective","b42_metrics","/R0/pooled_spearman"),
 "b42_R1_rho":rec(C["R1_rho"],"Spearman rho","TEST 318 rows","exposed_retrospective","b42_metrics","/R1/pooled_spearman"),
 "b42_Rnull_rho":rec(C["Rnull_rho"],"Spearman rho","TEST 318 rows","exposed_retrospective","b42_metrics","/Rnull/pooled_spearman"),
 "b42_const_rho":rec(None,"Spearman rho","constant predictor","n/a","b42_metrics","/zero_fit_train_weighted_mean_baseline","Spearman undefined for a constant column"),
 "b42_R1_minus_R0_rho":rec(C["d_rho"],"rho difference","TEST 21 groups","exposed_retrospective","b42_metrics","/decision/delta_rho"),
 "b42_verdict":C["verdict"],
 "b42_CI":rec(None,"CI","TEST 21 groups","n/a","b42_metrics","/decision/bootstrap_available","21<30 TEST groups -> conditional CI/significance NOT reported"),
 "b43_correction":"cached PARM/H1 model-score-vs-alpha MAE (chr17 .466390 / non_chr17 .353606) = INVALID cross-unit; NOT displayed",
 "b31_HEK673_context":rec(m31["hek673"]["rho"]["PARM"],"Spearman rho","673 frags/669 groups","retrospective_exposure_UNKNOWN","b31_metrics","/hek673/rho/PARM",None,"contextual only; NOT a local293FT validation; no shared-axis claim"),
}
print("chunk5 ok")

fi["fig2_public_transfer_sameassay"]["b43_within_frac"]={
 k:rec(v,"fraction of TEST MSE","TEST rows","exposed_retrospective","b43_decision","/decomposition_answer/TEST_row_weighted_within_frac/"+k) for k,v in E.items()}
fi["fig3_route_capability"]={
 "fragment_software":"b49 CLI fragment task: 1..600 -> public HEK293 PARM score; >600 (incl 2000) -> null UNSUPPORTED_LENGTH",
 "fragment_cell_units":"PARM_HEK293_model_score; NOT 293FT/GFP activity; local 293FT UNVALIDATED",
 "local_failure":"D28 FAIL (fig1: within-family PARM direction 3/21; technical tubes; barcode UNKNOWN)",
 "public_failure":"b41 LIMITED (PARM rho .108379 < 0.30); b42 LIMITED (R1 .091055 < R0 .158409; 21<30 no CI)",
 "catalog_axes":"E_raw (mean log2(1+tpm)), S_raw (minus MAD across detectable tissues); B_raw gate only; no composite",
 "catalog_frontier_default":rec(b49_def["default_frontier"],"count","eligible 18947","frozen_catalog","b49_breadth","/thresholds/0.30/n_frontier"),
 "catalog_minB_1.00":{
   "full_source":rec(mb10["counts"]["full_source"],"rows","all catalog","frozen","b49_minB10","/counts/full_source"),
   "undefined_S":rec(mb10["counts"]["undefined_S"],"rows","all catalog","frozen","b49_minB10","/counts/undefined_S"),
   "below_B":rec(mb10["counts"]["below_B"],"rows","all catalog","frozen","b49_minB10","/counts/below_B"),
   "below_E":rec(mb10["counts"]["below_E"],"rows","no minE cutoff","frozen","b49_minB10","/counts/below_E"),
   "eligible":rec(mb10["counts"]["eligible"],"rows","B_raw>=1.00","frozen","b49_minB10","/counts/eligible"),
   "sum_equals_full_source":mb10["counts"]["counts_sum_equals_full_source"]},
 "next_learning_experiment_needs":"genuinely MEASURED same-assay / same-parent / common-3prime varying-length sequences with authoritative actual DNA/context and groups",
 "existing_vs_missing":{
   "existing":"public HEK293FT 7537 bio (270bp same-insert library, previously exposed in b41); local 14 constructs (technical replicates)",
   "missing_NEW_independent":"matched-cell 293FT same-assay/same-parent varying-length sequence-activity supervision with verified DNA/barcode identity and >=30 held-out groups",
   "not_authorized":"automatic 2 fits / tuning against exposed TEST / re-open closed GSE301246v2 fetch"},
}
fi["source_sha256"]=S
(OUT/"figure_inputs.json").write_text(json.dumps(fi,ensure_ascii=False,indent=1),encoding="utf-8")
print("figure_inputs.json written OK")
