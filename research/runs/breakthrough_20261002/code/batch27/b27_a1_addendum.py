# -*- coding: utf-8 -*-
"""BATCH27 A1: batch26 code/manifest addendum + first-test-read chronology. NO fits, no rewrites."""
import hashlib, json, os, subprocess
def sh(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def mt(p): return subprocess.check_output(["stat","-c","%y",p]).decode().strip()
C26="runs/breakthrough_20261002/code/batch26"
R26="reports/breakthrough_20261002/batch26"
code={fn:sh(os.path.join(C26,fn)) for fn in sorted(os.listdir(C26)) if fn.endswith(".py")}
man=json.load(open(os.path.join(R26,"sha_manifest.json"),encoding="utf-8"))
out={
 "batch":"batch27","purpose":"linked addendum correcting/annotating batch26 provenance; batch26 files NOT modified",
 "original_manifest":{"path":R26+"/sha_manifest.json","sha256":sh(os.path.join(R26,"sha_manifest.json")),"original_generated_at":man["generated_at"]},
 "current_ondisk_code_sha256":code,
 "b26_train_py":{"preregistered_sha256":man["preregistered_code_sha256"]["b26_train.py"],
   "executed_and_current_sha256":code["b26_train.py"],
   "matches":code["b26_train.py"]==man["preregistered_code_sha256"]["b26_train.py"],
   "save_fix":"after the FIRST A0 ridge solve crashed at the artifact-save line (np.concatenate ndim bug) a save-path fix was applied to b26_train.py at ~17:51:32; the executed numeric fit path is present on disk and reproduces; the save-fix is MISSING from sha_manifest.post_prereg_code_changes (documentation gap, not a fit-semantics change)."},
 "b26_figs_py":{"manifest_current_code_sha256":man["current_code_sha256"].get("b26_figs.py"),
   "ondisk_sha256":code.get("b26_figs.py"),
   "matches":man["current_code_sha256"].get("b26_figs.py")==code.get("b26_figs.py"),
   "note":"reporting-only utility; no fit path."},
 "resume_check_malformed_key":{"attempt":"A1_cnn_true_seed20261026","key":"true (boolean)",
   "cause":"'best_val_group_balanced_mae' in ck used as a dict key","still_valid":"resume_inference_max_abs_diff=0.0"},
 "fit_budget":{"declared":4,"observed_executions":5,"compliance":"FAIL",
   "breakdown":"A0 ridge closed-form solve ran twice (2nd after artifact-save crash) + A1/A2 true CNN + A3 shuffled CNN = 5",
   "ledger":"runs/breakthrough_20261002/batch26/attempts/_ledger.jsonl (append-only; original records preserved)",
   "disclosure":"reports/breakthrough_20261002/batch26/fit_accounting_disclosure.json",
   "preserved":True},
 "first_test_read_chronology":{
   "question":"selection-before-FIRST-test-read vs later regenerated timestamp",
   "evidence":[
     {"t":"2026-10-03T17:56:46+08:00","event":"selection_manifest.json written (test_targets_opened=false)","src":"mtime + selection_manifest.json"},
     {"t":"2026-10-03T17:56:51+08:00","event":"FIRST held-out test evaluation run (b26_eval.py)","src":"worker transcript control_33/worker.jsonl tool_use ts 09:56:51.799Z"},
     {"t":"2026-10-03T17:59:55+08:00","event":"worker reads test_metrics.json (exists)","src":"transcript ts 09:59:55.524Z"},
     {"t":"2026-10-03T18:01:19+08:00","event":"held-out evaluation RE-RUN after edits","src":"transcript ts 10:01:19.522Z"},
     {"t":"2026-10-03T18:07:53+08:00","event":"final test_metrics.json / test_predictions.tsv rewrite (regenerated)","src":"file mtime"}],
   "resolution":"FIRST test read = 17:56:51, AFTER selection_manifest (17:56:46). The 18:07:53 timestamp is the LAST regeneration, not the first evaluation. Parent observation of test_metrics.json at ~17:59:52 is consistent with the first eval output. No leakage implied; history unchanged."},
 "elapsed_note":"zero estimator fits performed"}
json.dump(out,open("reports/breakthrough_20261002/batch27/batch26_manifest_addendum.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("wrote batch26_manifest_addendum.json")
print(json.dumps({"train_prereg":out["b26_train_py"]["preregistered_sha256"][:12],"train_disk":out["b26_train_py"]["executed_and_current_sha256"][:12],"figs_manifest":out["b26_figs_py"]["manifest_current_code_sha256"][:12],"figs_disk":out["b26_figs_py"]["ondisk_sha256"][:12]}))
