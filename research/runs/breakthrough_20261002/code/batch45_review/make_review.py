# -*- coding: utf-8 -*-
"""Write canonical batch45 independent_review.json (reviewer-owned)."""
import json
OUT=r"E:/promoter_atlas/reports/breakthrough_20261002/batch45/independent_review.json"
R={
 "record_type":"independent_review","canonical":True,"batch":"batch45",
 "version_reviewed":"v6.32.0-synthetic-design-utility.rc1",
 "authored_at":"2026-10-04T04:20:00+08:00",
 "reviewer":{
   "role":"fresh separate DeepSeek Flash independent reviewer",
   "requested_model":"deepseek-flash","response_model_observed":"deepseek-flash",
   "model_source":"runs/breakthrough_20261002/control_53/reviewer.jsonl init line {session_id d432085f-4fbe-4782-8322-a3297ef16435, model deepseek-flash}; this is MY OWN review-session log, NOT the worker launch file",
   "session_id":"d432085f-4fbe-4782-8322-a3297ef16435",
   "harness_process":{"pid":53696,"name":"claude.exe","parent_pid":50288,
      "created":"2026-10-04T04:08:24.4647670+08:00",
      "source":"live OS WMI Get-CimInstance Win32_Process ProcessId=53696 while alive (Get-Date 04:08:54)",
      "true_os_creation":"2026-10-04T04:08:24.4647670+08:00"},
   "worker_session":{"session_id":"04baab3b-6530-4e2c-8c80-3bdfdb795e68","pid":51292,
      "started_at":"2026-10-04T04:03:18.738310+08:00","finished_at":"2026-10-04T04:08:24.445411+08:00",
      "model":"deepseek-flash","n_assistant_responses":143,
      "source":"control_53/worker.jsonl init + control_53/status.json"},
   "separate_session_from_worker":True,
   "reviewer_clock_at_authorship":"2026-10-04T04:20:00+08:00",
   "web_used":False,"fits_or_solves_performed":0,"gpu_replay_used":False,
   "review_workspace":"read-only over old artifacts; canonical write ONLY reports/breakthrough_20261002/batch45/independent_review.json"},
 "scope":"batch45 retrospective re-analysis of the OFFICIAL GSE301246 synthetic-promoter design table (author-published PARM prediction columns + real measured K562/HepG2 normalized RNA/DNA). Reviewer: ZERO estimator fit/solve/calibration/scoring/neural/GPU/web/new-source.",
 "method":"Read disk PROTOCOL/PARENT_ACCEPTANCE/REVIEW_PROTOCOL + source_closure_b44 FIRST; then independently re-derived every headline number from data/breakthrough_20261002/batch44/syn.txt.gz with reviewer-owned code (own seed 20261004), recomputed SHA/CRC/offsets, 20 sampled barcode-to-sequence rows, the fixed mean-normalized-log2 aggregation, all primary lineage deltas, all secondary Spearman/GC/top10, replicate concordance, category/MUTATED tables, re-verified the output SHA manifest read-only, and parsed control_53 worker/reviewer JSONL + the live OS process table for identity.",
 "method_flags":{"zero_fits":True,"zero_gpu":True,"zero_scoring":True,"zero_new_prediction":True,
   "web_used":False,"new_source_used":False,"only_cached_b44_source":True,"reviewer_owned_code":True,
   "prior_rejected_protocol_used":"NO"},
 "verdict":{
   "integrity":"PASS",
   "compliance":"FAIL",
   "scientific_gain":"LIMITED-NOT-ACCEPTED",
   "scientific_gain_classification":"Honest, fully reproducible RETROSPECTIVE re-analysis of the already-selected author design study. It shows the author published K562 design gains are realized in the measured activity and that ranking utility exists (K562 rho 0.77-0.85, HepG2 0.57-0.75; GC baseline lower; top10 precision ~0.22-0.30 vs chance 0.10), with correct limits. It is NOT new model/family/293FT gain and is correctly NOT promoted. Useful as a fixed metric/denominator spec, not a breakthrough.",
   "integrity_pass_reason":"Every headline and every inferential number reproduces bit-exact from the raw cached source with reviewer-owned code: compressed sha256 c7ce4b50...244d6 and decompressed e7fc77...c223, gzip CRC/EOF ok, 3556 rows / 889 unique query DNA / 889 unique info / 3556 barcodes (4 per fragment), info-query 1:1, query.seq an exact substring of full.oligo at offset 18 (18bp 5prime + 50bp 3prime) in ALL rows, zero K562/HepG2 prediction conflicts across barcodes, the fixed mean-normalized-log2 summary equals the author averaged-column-over-barcodes (max abs diff 4.9e-15 / 5.1e-15), primary deltas short G=48 (dPred med 5.1897, dMeas med 1.8146, 10-0) and long G=44 (dPred med 7.6431, dMeas med 3.4536, 10-0), all secondary rho/GC/top10 and replicate concordance (0.99587/0.95748/0.99697/0.97968) reproduced, all 21 MUTATED-vs-strongest pairs reproduced. All 3556 rows retained incl 1 -Inf barcode dropout (counted, not fabricated). Source predictions correctly treated as author inputs. No DNA/adapter/RC guessed; 230-vs-232 length mismatch retained explicitly as UNKNOWN. Output SHA manifest verified (21/21 read-only).",
   "compliance_fail_reason":"Two real deviations, both narrow: (i) NO batch45 append to reports/breakthrough_20261002/version_registry.json or CHANGELOG.txt, although Protocol deliverable D requires the registry/changelog atomic append and every comparable prior zero-fit batch (b41-b44) appended during its worker phase; production pointer v6.3/v6.4 IS correctly left unchanged (verified). (ii) In canonical reports/breakthrough_20261002/batch45/secondary_reliability.json the barcode_spread entries keyed short:HepG2 / long:HepG2 are populated with the K562 barcode SD (true HepG2 values 0.28786 / 0.28261 vs reported 0.19733 / 0.19222 which equal the K562 values), i.e. a mislabeled diagnostic field - the same class as the b43 figure mislabel that drew a compliance FAIL. All evidential/headline numbers are unaffected and the field is not cited in report.md.",
   "compliance_fail_is_narrow":True},
 "preserved_verdicts":{
   "local_293FT_D28":"FAIL (unchanged)","HEK_fragment_transfer_weak":"unchanged (b41 rho 0.108)",
   "2000bp":"UNSUPPORTED (unchanged)","batch42_within_group_negative":"preserved",
   "batch44_source_branch":"CLOSED (wrong-file allocation; donor DNA + union grouping UNKNOWN; 100/20/30 gate UNKNOWN)",
   "author_published_predictions":"author inputs, NOT our model output; no model gain"},
 "integrity_checks":[
  {"check":"compressed sha256","observed":"c7ce4b506a133afa5aee04c478c821b16dc64fdbfd612a8722779d06766244d6","result":"PASS"},
  {"check":"decompressed sha256 / bytes","observed":"e7fc776ccb982130d3eb225cc6f47ae22baa3752b43d60bdf41a8db265e8c223 / 2265693","result":"PASS"},
  {"check":"gzip EOF/CRC, compressed bytes","observed":"eof True unused 0 / 385134","result":"PASS"},
  {"check":"row/unique counts","observed":"3556 rows, 889 unique query, 889 unique info, 3556 barcodes","result":"PASS"},
  {"check":"barcodes per fragment","observed":"all fragments 4 barcodes {4:889}","result":"PASS"},
  {"check":"info-query 1:1 (no shared DNA across info)","observed":"True","result":"PASS"},
  {"check":"query.seq exact substring of full.oligo","observed":"offset 18 in all 3556 rows; 5prime 18bp + 3prime 50bp","result":"PASS"},
  {"check":"20 independent barcode-to-sequence rows (seed 20261004)","observed":"20/20 offset 18, 5p=18, 3p=50, qlen 70/230/232","result":"PASS"},
  {"check":"prediction consistency across barcodes","observed":"K562 conflicts 0, HepG2 conflicts 0","result":"PASS"},
  {"check":"fixed summary == author averaged-over-barcodes","observed":"max abs diff K 4.9e-15 / H 5.1e-15; author averaged==mean(rep1,rep2) 1.0e-14","result":"PASS"},
  {"check":"primary lineage delta recompute (short G=48 / long G=44)","observed":"short dPred 5.18966 dMeas 1.81460 10/0; long dPred 7.64313 dMeas 3.45357 10/0","result":"PASS"},
  {"check":"secondary Spearman/GC/top10 (12 groups)","observed":"all 12 reproduce (e.g. long:K562:syn rho 0.84730 gc 0.42703 prec 0.225)","result":"PASS"},
  {"check":"replicate concordance","observed":"shortK 0.99587 shortH 0.95748 longK 0.99697 longH 0.97968","result":"PASS"},
  {"check":"MUTATED vs lineage-matched strongest","observed":"21 pairs, median dMeas -2.339, 20 neg / 1 pos","result":"PASS"},
  {"check":"category table + controls","observed":"design 389/372, best 0/10, top_individual 11/0, MUTATED 11/10, control 44/42; controls 86","result":"PASS"},
  {"check":"output SHA manifest read-only, twice","observed":"21/21 files match sha+bytes; self/mutable excluded; rounds 1 and 2 OK","result":"PASS"},
  {"check":"barcode_spread cell labeling","observed":"short:HepG2 0.19733 and long:HepG2 0.19222 equal the K562 values; true HepG2 SD 0.28786 / 0.28261","result":"FAIL (minor, diagnostic field)"},
  {"check":"version_registry / CHANGELOG batch45 append","observed":"no v6.32 / batch45 entry; last appended batch44","result":"FAIL (procedural)"},
  {"check":"production pointer preserved","observed":"registry production_pointer v6.3 main / v6.4 cellline pilot unchanged","result":"PASS"},
  {"check":"old bytes preserved","observed":"batch44 review/report (03:45/03:38), batch43, registry+CHANGELOG (03:39) all predate batch45; no out-of-scope writes","result":"PASS"},
  {"check":"forbidden metrics absent","observed":"no MAE/R2/CI/pvalue/GFP/probability/causal claim; MAE/R2 appear only in the explicit negation note","result":"PASS"},
  {"check":"strict JSON schema (no NaN)","observed":"all batch45 json parse; no NaN literal","result":"PASS"},
  {"check":"figures axis/metric consistency (b43 precedent)","observed":"Fig1 pred(top)/measured(bottom) separate-unit labels; Fig2 (a)pred-vs-meas (b)rep1-vs-rep2 (c)pred-vs-GC rho; Fig3 (a)counts (b)top10 precision - titles match the plotted metric","result":"PASS"}],
 "findings":[
  {"id":"F1","severity":"minor","type":"labeling","where":"secondary_reliability.json barcode_spread",
   "detail":"HepG2 barcode-spread entries duplicate K562 values (code uses K562_barcode_spread_sd unconditionally). Diagnostic only; not cited in report.md."},
  {"id":"F2","severity":"minor","type":"procedural","where":"version_registry.json / CHANGELOG.txt",
   "detail":"No batch45 append, though Protocol D requires it and b41-b44 appended. Production pointer correctly unchanged."}],
 "reproducibility":{
  "short":{"G":48,"n_lineages":10,"dPred_med":5.18966233730316,"dMeas_med":1.8146045162370514,"sign":"10 plus / 0 minus"},
  "long":{"G":44,"n_lineages":10,"dPred_med":7.643129444122325,"dMeas_med":3.4535746217526135,"sign":"10 plus / 0 minus"}},
 "scientific_limits_affirmed":[
  "Retrospective re-analysis of an author-selected study; strong model-selection + shared genetic-optimization ancestry bias.",
  "889 unique DNA are NOT independent families; 3556 barcodes and 2 RNA columns are NOT independent n; 10 GA search lineages share one PARM model and pipeline.",
  "Source prediction columns are published analysis inputs, NOT our forward-model output; a positive source-study result is NOT a gain of our wrapper.",
  "K562 native design vs measured HepG2 cross-cell: direction transfer supported descriptively, NOT causal, NOT equal strength; no causal-cell claim.",
  "230 vs 232 bp label unresolved from cached evidence (explicitly retained, not padded/clipped).",
  "Nothing here rescues local 293FT D28 FAIL / weak HEK fragment transfer / 2000bp unsupported."],
 "next_recommendation":"The worker next_batch_proposal (batch46 v6.33.0-293FT-traceable-scoring.rc1, ZERO fit, apply the frozen metric/denominator spec to the existing local 293FT construct records with a predeclared GC baseline and a stop condition if no traceable sequence-construct-activity pairing exists) is a genuinely different, bounded, non-replaying action and does NOT reuse K562 performance as 293FT validity. REQUIRES PARENT SCIENCE ASSESSMENT before dispatch. Additionally, before or during batch46 the parent should have the worker (a) correct the barcode_spread HepG2 labeling in a NEW file (do not rewrite the frozen batch45 artifacts) and (b) append the batch45 record to the registry/changelog if that append was not merely deferred to the parent stage. No extra stage email.",
 "preserved_assets":{"batch45_immutable":"21/21 sha manifest verified; old batch43/batch44 artifacts byte-unchanged","no_promotion":True}}
with open(OUT,"w",encoding="utf-8") as f: json.dump(R,f,ensure_ascii=False,indent=2)
print("wrote",OUT)
d=json.load(open(OUT,encoding="utf-8"))
assert set(d["verdict"])>={"integrity","compliance","scientific_gain"}
assert d["verdict"]["integrity"] in ("PASS","FAIL") and d["verdict"]["compliance"] in ("PASS","FAIL")
print("strict parse OK; top verdict =",{k:d["verdict"][k] for k in ["integrity","compliance","scientific_gain"]})
