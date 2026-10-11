# -*- coding: utf-8 -*-
import json, os, hashlib, datetime
def sha(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()
H = {
 "batch26_sha_manifest.json": "a723b5320150fa1a546255b9641c4d315deffb2b674bbdb57796de0a96eac8f2",
 "batch26_ledger": "07b2a0ec067bd4ce6db9c99965b44bb0c491c1d97e3c5b1b512b7d6ae447c018",
 "sure_primary_source.gz": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
 "sure_seq_sample.fa": "331e8c0b42b0cf6442d3235ef12df49855971d3c22986be3739dfd06f5a43b94",
 "sure_tier2_cohort.tsv.gz": "d4243c839b86b2f9888aad7ed9400899f28692b3ea169435ba6ce00840ed8ef9",
 "batch26_test_metrics.json": "3afdad2f345b12e4783bb1ac06c8f4ee513c51b39896e0aabad9a87f133186be",
 "batch26_selection_manifest.json": "cd5bb2de08b6d68a28ddb4e8bb51c45b8560bd32fd4f08c11653a00e29370281"
}
out = {
 "batch": "batch27",
 "version": "v6.19.1-promoter-readiness.rc1",
 "task": "zero-fit evidence cleanup + robustness of batch26 STR locus-holdout, plus preparation of a native long-fragment K562 autonomous-promoter dataset (SuRE/GSE78709)",
 "review_kind": "fresh independent DeepSeek Flash review (read-only verification; no worker artifact rewritten; ZERO estimator fits, ZERO new inference)",
 "reviewer": "fresh independent reviewer, distinct from the batch27 worker",
 "reviewer_agent_pid": 49372,
 "reviewed_at": "2026-10-03T19:15:00+08:00",
 "review_protocol": "runs/breakthrough_20261002/batch27/REVIEW_PROTOCOL.txt",
 "worker_protocol": "runs/breakthrough_20261002/batch27/PROTOCOL.txt",
 "zero_extra_fits": True,
 "fits_performed_by_reviewer": 0,
 "neural_inference_performed_by_reviewer": 0,
 "does_not_send_email": True,
 "does_not_read_credentials": True,
 "review_schema_note": "top-level verdict is a dict {integrity, scientific_gain, compliance} to match the batch26 canonical schema; no conflicting nested keys."
}
out["parent_policies_read"] = ["runs/breakthrough_20261002/PROTOCOL.txt","runs/breakthrough_20261002/REVIEW_PROTOCOL.txt","runs/breakthrough_20261002/USABILITY_CONTINUATION_20261003.txt","runs/breakthrough_20261002/PERFORMANCE_PRIORITY_20261003.txt","runs/breakthrough_20261002/NEXT_SEQUENCE_RUN_20261003.txt","runs/breakthrough_20261002/MODEL_POLICY.txt","runs/breakthrough_20261002/BUDGET_POLICY.txt","runs/breakthrough_20261002/VERSION_POLICY.txt","runs/breakthrough_20261002/STAGE_EMAIL_POLICY.txt","runs/breakthrough_20261002/batch27/PARENT_FINDINGS_ADDENDUM_20261003.txt"]
out["verdict"] = {
 "integrity": "PASS (all headline numbers independently reproduced from saved on-disk artifacts; batch26 old files byte-unchanged; no fabrication; chronology and download/gzip evidence verified). Two disclosure caveats: (a) the known-contamination list is representative-choice-dependent and therefore understated - the worker sha-min representative gives 2 excluded TEST groups while a lex-min representative gives 3; (b) one addendum sentence overstates monotonicity (see findings).",
 "scientific_gain": "EVIDENCE-ONLY; NO MODEL GAIN; NO PROMOTION; production pointer stays v6.3/v6.4. The batch26 STR ranking gain is robust to removal of the KNOWN cross-split near-homolog groups: with the worker 2 excluded groups A1-A0 delta-rho +0.0524 [0.0341,0.0721] (group bootstrap 1000) and with my independently-derived 3 excluded groups delta-rho stays about +0.053; this is a POST-HOC sensitivity on known contamination, NOT strict-family isolation and NOT a new independent test. The SuRE/GSE78709 K562 autonomous-promoter per-fragment table was resolved and its counts independently re-derived: 153,562,178 measured rows; tier-1 eligible 8,033,394; tier-2 (cDNA>=1 in BOTH bio-reps) 253,629 inserts forming 186,969 overlap components (reviewer re-count identical); 2500/2500 label-free sampled inserts retrieved with exact length. The FEASIBILITY data-ready gate is MET, but a feasibility gate is NOT evidence of predictive accuracy; the actual bounded training (batch28, <=4 executions, <=3 GPUh) is only PROPOSED and requires parent approval, and tier-2 eligibility is a label-informed QC rule needing explicit approval.",
 "compliance": "PASS WITH CAVEATS for batch27 itself (zero estimator fits, zero inference, no GPU; downloads bounded - SuRE primary 137.1 s wall vs <=30 min, sequence retrieval 310 s, only GSE78709 + Ensembl GRCh37 (plus a UCSC sequence-API probe) as sources, no broad third-source drift; complete gzip EOF/CRC and SHA256 verified; label-free hash selection frozen at 18:44:26 before labels read at 18:44:35; old batch26 artifacts preserved byte-for-byte; no email, no credentials, no git). Caveats: (1) the contaminating-group edge list used an UNPREDECLARED single-representative rule, so the known contamination set is understated and not robust (2 groups sha-min vs 3 groups lex-min); (2) insert sequence was retrieved uniformly on the FORWARD strand (Ensembl region suffix colon-1) for all 2500 sampled inserts including the 1251 minus rows, so the assayed orientation of minus-strand inserts is unresolved for any future model; (3) sure_seqsample.py reports n_retrieved_containing_N from an initialized constant never updated in-loop (true value independently checked = 0, so no false claim, but the field is not actually measured by the artifact code); (4) version_registry.json top-level last_appended_at/last_appended_by were not advanced by the batch27 append (the d_sure_k562_batch27 entry itself carries appended_at). The batch26 fit-budget FAIL (5 executions vs declared 4) belongs to batch26 and is correctly PRESERVED, not repaired, in batch27."
}
out["independently_reproduced"] = {
 "method": "reviewer scripts under runs/breakthrough_20261002/code/batch27/review/ reading only saved artifacts (no fits, no inference, no extra network)",
 "batch26_full_test_metrics": "re-derived from data/breakthrough_20261002/batch26/test_predictions.tsv: A0 ridge rho 0.418832/gbmae 0.398604, A1 rho 0.471871/gbmae 0.355892, A2 rho 0.444303/gbmae 0.363241, A3 rho UNDEFINED (pred_nunique 1) - EXACT match to batch26 test_metrics.json and to homology_sensitivity.json full metrics",
 "contamination_sensitivity": "independently rebuilt label-free edge list from str_variant_rows.tsv + split_manifest.tsv; with the worker sha-min representative reproduced EXACTLY (3 same-length Hamming<=16 edges, 1 variable-length cov90/ident90 edge, 3 cross-split edges, 2 excluded TEST groups fb7db8d0/fb8bf6cd); with a lex-min representative the set grows to 6 Hamming edges, 0 varlen, 5 cross-split edges, 3 excluded TEST groups (adds 836c95fa9896a50086c332d1f2d3509d)",
 "contamination_robustness": "with my 3 excluded groups: A0 0.418433, A1 0.471489 (A1-A0 +0.0531), A2 0.444066 (A2-A0 +0.0256) - ranking gain persists; direction robust, exact counts not",
 "variant_bin_diagnostic": "independently recomputed from saved predictions: all sign_agreement and delta_spearman values match variant_bin_diagnostic.json to full precision (e.g. A1 by-abs-delta 0.4824/0.5293/0.5656/0.6036)",
 "chronology": "transcript control_33/worker.jsonl: selection_manifest write tool_use 2026-10-03T09:56:46.291Z (=17:56:46+08) BEFORE first b26_eval.py tool_use 09:56:51.799Z (=17:56:51+08); first eval result at 09:59:53.362Z (=17:59:53+08); test_metrics.json mtime 18:07:53 = later regeneration; selection_manifest.json mtime 17:56:46.543",
 "sure_source_file": "bytes 1,187,342,132; SHA256 8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85 (verified); zlib full decompress 6,743,871,060 bytes, eof=True, unused=0 (CRC/EOF OK); header = 13-col per-fragment counts table, data rows 13 fields (measured-fragment table, NOT a TSS/bigWig aggregate)",
 "sure_tier2_recount": "streamed sure_tier2_cohort.tsv.gz: 253,629 rows; biorep1 cDNA total 1,981,009; biorep2 total 2,811,119; independent union-find over interval overlap = 186,969 components, 149,807 singletons, max component 1,605 - EXACT match to the worker artifacts",
 "sure_sequence_sample": "sure_seq_sample.fa: 2500 records, SHA256 331e8c0b... matches claim; lengths 202-1998; 0 records containing N; 0 exact-duplicate pairs; 0 reverse-complement pairs (all independently recomputed)",
 "hash_selection": "selected 25,000 keys; largest selected sha256 threshold 00cc6943... = 3.119e-3 of 2^256, expected about 25,057 of 8,033,394 eligible - consistent with the 25,000-smallest-sha256 label-free selection",
 "exec_gate": "reviewer re-ran the mock test suite (copied to review/ to avoid overwriting the canonical artifact): 9/9 pass, dummy callbacks only, zero estimator solve; gate semantics confirmed (failed retry consumes 2; 5th call of max-4 rejected before callback; 40-thread concurrent registration serialized to 40 consumed / 80 ledger lines; crash-reserved stays consumed; valid resume identity consumes 0; bad identity rejected)",
 "artifact_hash_audit": "19/19 SHA256 entries in reports/breakthrough_20261002/batch27/status.json verified on disk; the 4 special-path entries (tier2_cohort, sure primary source, version card, worker addendum) verified separately - all match",
 "old_artifact_immutability": "batch26 sha_manifest.json unchanged (a723b532...ac8f2); all 13 listed batch26 report hashes and 3 figure hashes and the model card (models/cards/promoter_atlas_v6_19_0_strvariant_rc1_card.txt = 24df5890...) match on disk; attempts/_ledger.jsonl unchanged (07b2a0ec...); all 8 b26 code hashes match the addendum current_ondisk_code_sha256"
}
out["checks"] = {
 "zero_fits_batch27": {"ok": True, "detail": "worker streamed/parsed + diagnostics only; reviewer ran no estimator"},
 "batch26_fit_budget_FAIL_preserved": {"ok": True, "detail": "ledger 6 lines unchanged; disclosure present; addendum asserts FAIL; no re-run/reset"},
 "five_executions_disclosed": {"ok": True, "detail": "A0 ridge solve x2 + A1/A2/A3 neural = 5 vs declared 4"},
 "code_hash_corrections_are_addenda": {"ok": True, "detail": "b26_train.py prereg a4a7c6f7 vs executed 272e924e; b26_figs.py 8980d3db vs on-disk bc962af3; both recorded only in the linked batch27 addendum; original manifest byte-unchanged"},
 "first_test_read_after_selection": {"ok": True, "detail": "17:56:46 selection < 17:56:51 first eval; 18:07:53 is regeneration"},
 "contamination_list_label_free": {"ok": True, "detail": "edges from sequence + split only; built before subset scores; no activity used"},
 "known_homolog_test_groups_excluded": {"ok": "PARTIAL", "detail": "worker excluded 2; independent lex-min representative yields 3 (adds 836c95fa...); set understated/rep-dependent"},
 "original_scores_kept": {"ok": True, "detail": "batch26 test_metrics.json/test_predictions.tsv unchanged; sensitivity reported separately"},
 "subset_not_new_validation": {"ok": True, "detail": "manifest+addendum label it post-hoc sensitivity, not new independent test"},
 "sure_primary_measured_fragment_table": {"ok": True, "detail": "13-col per-fragment counts; TSS/bigWig NOT used"},
 "sure_gzip_content_length_sha": {"ok": True, "detail": "size+sha matched; full decompress CRC/EOF OK"},
 "sure_sequence_hg19": {"ok": True, "detail": "Ensembl GRCh37, 2500/2500 exact length; coordinate convention documented"},
 "sure_orientation_resolved": {"ok": False, "detail": "all inserts retrieved on forward strand; 1251 minus-strand inserts assayed orientation unresolved"},
 "no_target_driven_sample": {"ok": True, "detail": "25000 cohort selected by sha256 only; tier-2 detection rule is label-informed QC and gated on parent approval"},
 "k562_not_hek293t_not_str": {"ok": True, "detail": "explicitly kept separate; no cross-domain score comparison"},
 "independent_groups": {"ok": True, "detail": "186,969 overlap components re-counted exactly; flagged as LOWER BOUND (sequence near-homology unresolved)"},
 "execution_gate_mock_tests": {"ok": True, "detail": "9/9 dummy-callback tests pass; no estimator solve"},
 "download_time_bounds": {"ok": True, "detail": "SuRE primary 137.1s (<30min); sequence 310s; <=1.5GB compressed; no raw FASTQ / no genome download"},
 "no_broad_source_drift": {"ok": True, "detail": "only GSE78709 suppl + Ensembl GRCh37 (+UCSC API probe); GSE245191 not reopened; no HepG2/HEK counts added"},
 "next_proposal_bounded": {"ok": True, "detail": "batch28 <=4 executions, <=3 GPUh, fixed HPs, matched baseline + shuffle control, parent approval required"},
 "worker_selfcheck_not_independent": {"ok": True, "detail": "selfcheck.json is worker-authored; this file is the independent review"}
}
out["findings"] = [
 {"id": "F1", "severity": "medium", "area": "A2 contamination", "text": "The count of KNOWN cross-split near-homolog test groups is not robust to the (unpredeclared) choice of per-group representative sequence. The worker used a sha256-min representative and reports 3 cross-split edges giving 2 excluded TEST groups. A lexicographic-min representative produces 6 same-length Hamming<=16 edges and 5 cross-split edges giving 3 excluded TEST groups (adds 836c95fa9896a50086c332d1f2d3509d). The scientific conclusion (ranking gain persists) is unchanged (A1-A0 delta-rho about +0.052 either way), but the exact known-contamination set is understated and the representative rule should have been predeclared; a union over multiple representative choices (or all-member) would be a stronger bound."},
 {"id": "F2", "severity": "low", "area": "A3 reporting", "text": "The addendum states sign agreement rises monotonically with observed abs(delta) across predeclared motif-length and absolute-delta bins. Only the absolute-delta bins rise monotonically (A1: 0.4824/0.5293/0.5656/0.6036); the motif-length bins are NOT monotonic (A1: 0.5564/0.5576/0.5451/0.5237) and mean abs(delta) across motif bins is not monotonic either. The underlying variant_bin_diagnostic.json numbers are correct; only the prose overstates."},
 {"id": "F3", "severity": "medium", "area": "B sequence", "text": "Insert sequence was retrieved uniformly on the forward genomic strand (Ensembl region suffix colon-1) for all 2500 sampled inserts, including the 1251 minus-strand rows. The strand is preserved in the FASTA headers, but the assayed orientation (fragment cloned 5-prime of the reporter) for minus-strand inserts is not represented in the retrieved sequence and is not resolved in the manifest. Any future model must decide/verify orientation before training; this should be stated as a residual limitation."},
 {"id": "F4", "severity": "low", "area": "B QC field", "text": "b27_sure_seqsample.py initializes n_with_N = 0 and never updates it, yet it is emitted as n_retrieved_containing_N = 0. The reviewer independently confirmed the TRUE count is 0, so the value is correct, but the field is an uncomputed constant rather than a measured count."},
 {"id": "F5", "severity": "low", "area": "B eligibility", "text": "The tier-2 eligibility rule (cDNA>=1 in BOTH biological replicates) is label-informed (uses measured RNA). It is a presence/detection QC rule, not magnitude selection, and it is explicitly declared and gated on parent approval - acceptable, but the no-target-driven-sample requirement is satisfied only for the 25,000-insert tier-1 cohort, not for the 253,629-insert tier-2 training substrate."},
 {"id": "F6", "severity": "info", "area": "registry", "text": "version_registry.json gained a d_sure_k562_batch27 data_versions entry (appended_at 18:57:37, registry_sha256_before_append recorded) but the top-level last_appended_at/last_appended_by still read 18:08:48 / batch26 worker. Minor bookkeeping inconsistency; no data lost and all prior entries preserved."},
 {"id": "F7", "severity": "info", "area": "readiness honesty", "text": "Readiness is reported honestly: the label-blind 25,000-insert cohort contains only 754 inserts detected in both replicates, replicate agreement among both-detected inserts is low (about 0.15 Spearman), and the manifest states the gate is FEASIBILITY, not accuracy, with a modest expected ceiling. No HEK293T/generalization claim is made."}
]
out["limitations_of_this_review"] = [
 "No network sequence re-validation was performed by the reviewer; the reviewer trusted the Ensembl GRCh37 retrieval contents and verified only internal consistency (length, N, dup/RC) and the request orientation from the source code.",
 "The reviewer did not re-stream the 1.1 GB source to re-derive the tier-1 8,033,394 eligible count or the 25,000-smallest-sha selection; it verified the size/SHA/full-gzip integrity and the statistical consistency of the selected-hash threshold instead.",
 "The selected-cohort group statistics (23,086 groups containing selected; 1,535 with more than one member) were not recomputed; the tier-2 component count WAS recomputed exactly.",
 "The reviewer used a different representative tie-break in the contamination audit by design, to test robustness; neither the worker nor the reviewer representative choice is a full all-member near-homology bound."
]
out["next_steps"] = {
 "immediate": "Parent reads this review and the bounded next_batch_proposal.json (batch28_v6.20.0-surelong.rc1).",
 "requires_parent_decision": ["approve or amend the tier-2 eligibility rule (cDNA detected in both bio-reps) before any training", "approve the component-based split + chromosome-held-out arm", "authorize <=4 executions / <=3 GPUh on cuda:0"],
 "recommended_remediations_before_or_during_batch28": ["predeclare the group-representative rule (or use an all-member union) for near-homology exclusion", "resolve/declare insert orientation handling for minus-strand inserts", "predeclare the RNA/input normalization and depth stratification before fitting", "fix the n_retrieved_containing_N computation and advance registry last_appended_at metadata"],
 "must_not": ["no promotion of any candidate; production pointer stays v6.3 main / v6.4 cellline pilot", "no HEK293T or prospective-validation claim", "batch26 fit-budget FAIL remains recorded"]
}
out["worker_selfcheck_present"] = True
out["worker_selfcheck_is_not_independent"] = True
out["reviewer_scripts"] = [
 "runs/breakthrough_20261002/code/batch27/review/rv_repro_contam.py",
 "runs/breakthrough_20261002/code/batch27/review/rv_repro_contam_shamin.py",
 "runs/breakthrough_20261002/code/batch27/review/rv_a3.py",
 "runs/breakthrough_20261002/code/batch27/review/rv_tier2.py",
 "runs/breakthrough_20261002/code/batch27/review/rv_test_exec_gate.py"
]
out["review_artifact_hashes"] = H
out["ongoing_task_active"] = True
out["experiment_complete"] = False
p = "reports/breakthrough_20261002/batch27/independent_review.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
d = json.load(open(p, encoding="utf-8"))
assert set(["integrity","scientific_gain","compliance"]) <= set(d["verdict"].keys())
print("WROTE", p)
print("parsed ok; verdict keys:", list(d["verdict"].keys()))
print("top-level keys:", len(d.keys()))
print("size bytes:", os.path.getsize(p))
