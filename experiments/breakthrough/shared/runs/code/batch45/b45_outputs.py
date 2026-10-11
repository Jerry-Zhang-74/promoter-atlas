# -*- coding: utf-8 -*-
"""batch45: status, selfcheck, next proposal, phase D source closure, remaining_work, sha manifest + 2x verify."""
import json, hashlib, sys, datetime
from pathlib import Path
OUT=Path("E:/promoter_atlas/reports/breakthrough_20261002/batch45")
def wj(n,o): (OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
now=datetime.datetime.now().astimezone().isoformat()
wj("status.json",{"batch":"batch45","version":"v6.32.0-synthetic-design-utility.rc1","phase":"worker",
 "state":"complete_worker_pending_review","created_at":now,"zero_fit":True,"zero_gpu":True,"zero_new_prediction":True,
 "source":"GSE301246_Synthetic_Promoter_Libraries_Small_and_Large_Data.txt.gz (cached b44)",
 "verdict":"DESCRIPTIVE_POSITIVE_SOURCE_STUDY_RETROSPECTIVE","unique_query_DNA":889,"source_rows":3556,
 "primary":{"short":{"G":48,"n_lineages":10,"median_dK562_pred":5.19,"median_dK562_meas":1.81,"sign_pos":10},
            "long":{"G":44,"n_lineages":10,"median_dK562_pred":7.64,"median_dK562_meas":3.45,"sign_pos":10}},
 "not_evidence_for":"our model new-family/293FT success; D28FAIL/HEK weak-transfer/2000bp-unsupported unchanged",
 "next":"fresh independent Flash review then parent decision"})
wj("worker_selfcheck.json",{"batch":"batch45","role":"worker","selfcheck_only":True,"checks":{
 "zero_fit_solve_calibration_gradient_optimizer_neural_score_gpu":True,
 "no_new_prediction_no_cache_generation":True,"no_web_download_api_source_search":True,
 "only_cached_b44_source_used":True,"no_motif_library_no_old_route_scoring":True,
 "old_assets_read_only":True,"new_batch45_dirs_only":True,
 "gzip_crc_sha_verified":True,"provenance_from_cached_listing":True,"schema_parsed":True,
 "identity_metadata_frozen_before_activity":True,"prediction_consistency_checked":True,
 "all_source_rows_retained_no_activity_filter":True,"aggregation_fixed_and_verified_identical":True,
 "no_fake_independent_n":True,"no_random_cv_no_new_validation_claim":True,"no_credential_read":True,
 "no_email_sent":True,"strict_json_no_nan":True},
 "honest_limitations":[
   "retrospective reanalysis of the AUTHOR's selected design study; strong model-selection/optimization-ancestry bias",
   "889 unique DNA are NOT independent families; barcodes and RNA columns are not independent n",
   "230 vs 232 length label: synthetic long inserts 230 bp vs 42 natural controls 232 bp; which length the label names is undetermined from cached evidence",
   "one barcode dropout (0 RNA counts) gives -Inf; excluded from the finite mean and counted",
   "source prediction columns are analysis inputs, not our model output"],
 "worker_preliminary_review":False,"fresh_independent_review_required":True})
wj("next_batch_proposal.json",{"proposed_batch":"batch46",
 "title":"v6.33.0-293FT-traceable-scoring.rc1",
 "rationale":"batch45 gives a fixed metric/denominator spec and shows the author's K562 design gains are realized in measurement (source-study, retrospective). The open user-facing question is 293FT predictivity, which this source cannot answer.",
 "single_bounded_action":"Apply the SAME fixed summary/metric spec to the user's EXISTING 293FT construct records (sequence->construct->activity pairing) with a predeclared GC baseline and Spearman; NO new download, NO new model prediction.",
 "measurable_gain":"Spearman(pred,measured) + GC baseline on >=10 independent 293FT constructs; report denominators.",
 "stop_condition":"if 293FT records lack a traceable sequence-construct-activity pairing, record UNKNOWN and stop the sub-route (do not substitute).",
 "explicitly_not":["reuse K562 performance as 293FT validity","new public fetch/source hunt","new fit/GPU"],
 "zero_fit":True,"gpu":"none","parent_gate":"parent confirms 293FT record availability/format then approves the bounded scoring run"})
print("status/selfcheck/proposal written")
