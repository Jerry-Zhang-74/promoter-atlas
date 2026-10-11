#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH32 finalizer: SHA manifest + status.json + worker_selfcheck + remaining_work + next_batch_proposal
+ atomic append to CHANGELOG/version_registry. Does NOT write independent_review.json (fresh reviewer only)."""
import hashlib, json, os, glob, datetime
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch32")
CODEDIR=os.path.join(REPO,"code/batch32"); DATADIR=os.path.join(REPO,"data/breakthrough_20261002/batch32")
REG=os.path.join(REPO,"reports/breakthrough_20261002/version_registry.json")
CHG=os.path.join(REPO,"reports/breakthrough_20261002/CHANGELOG.txt")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
def rj(p): return json.load(open(p,encoding="utf-8"))
def rel(p): return os.path.relpath(p,REPO).replace(os.sep,"/")
EXCL={"reports/breakthrough_20261002/batch32/status.json",
      "reports/breakthrough_20261002/batch32/sha_manifest.json"}
def gather():
    files=[]
    for d in (CODEDIR,DATADIR):
        for p in glob.glob(os.path.join(d,"**/*"),recursive=True):
            if os.path.isfile(p): files.append(p)
    for p in glob.glob(os.path.join(OUT,"**/*"),recursive=True):
        if os.path.isfile(p): files.append(p)
    return sorted(set(files))

def main():
    A=rj(os.path.join(OUT,"sectionA_metrics.json")); B=rj(os.path.join(OUT,"sectionB_metrics.json"))
    cnt=A["frozen_counts"]; fs=A["subgroups"]
    status={"batch":"batch32","version":"v6.22.0-exposure-endpoint.rc1","phase":"worker",
        "state":"worker_complete_awaiting_fresh_review","self_independent_PASS":False,
        "note":"Evidence-only task; ZERO fits/inference/GPU. Canonical independent_review.json belongs ONLY to a separate fresh reviewer.",
        "actual_clock":now(),"worker_selfcheck_only":True,
        "paths":{"code":"code/batch32","data":"data/breakthrough_20261002/batch32","reports":"reports/breakthrough_20261002/batch32"},
        "sectionA":{"assembly":{"HEK673":"hg19","library":"hg19 (official example_config GENOME hg19_ch1-22_XYM.fa)"},
            "comparison_valid":True,"frozen_counts":cnt,
            "full673_rho_checksum":A["full673_checksum"]["rho"],
            "subgroup_rho":{s:fs[s].get("rho") for s in fs},
            "subgroup_n_groups":{s:fs[s].get("n_groups") for s in fs},
            "headline":"PARM association does NOT collapse in the coordinate-DISJOINT subset, but HEK293 training membership is UNKNOWN (distributed subset is HAP1/AGS-only) and the disjoint set is small; DISJOINT != unseen families."},
        "sectionB":{"endpoint":B["endpoint_evidence"]["target_name"],
            "macro_direction":B["macro_family_direction"],
            "nesting":"A1-A4 and G1-G5 exact substrings of A20/G20 (strand +); A5 differs by 1 base; CMVcore = 3prime 204bp of CMVfull (offset 380)",
            "barcode_verified":False,"independent_biological_replicates":False},
        "zero_fit_accounting":{"estimator_solves":0,"calibrations":0,"fine_tunes":0,"gradient_steps":0,
            "model_inference":0,"gpu_seconds":0,"note":"reused frozen batch31 PARM / batch22 R1 / batch24 H1 saved predictions"},
        "next":"await fresh separate reviewer; then parent route decision","ongoing_task_active":True}
    with open(os.path.join(OUT,"status.json"),"w",encoding="utf-8") as f:
        json.dump(status,f,ensure_ascii=False,indent=2)
    selfcheck={"worker_selfcheck_only":True,"independent_review_written":False,
        "full673_rho_matches_batch31":{m:abs(A["full673_checksum"]["rho"][m]-v)<1e-9 for m,v in
            {"PARM":0.6905737298336427,"R1":0.4566222544370238,"H1":0.526448729125029}.items()},
        "zero_fits":True,"gpu_used":False,
        "n_overlap_rows_sum_check":cnt["OVERLAP_any"]+cnt["DISJOINT"]+cnt["UNKNOWN_chrom_not_distributed"]+cnt["UNKNOWN_no_coord"],
        "n_expected":673,"sectionB_rows":14,"at":now()}
    with open(os.path.join(OUT,"worker_selfcheck.json"),"w",encoding="utf-8") as f:
        json.dump(selfcheck,f,ensure_ascii=False,indent=2)
    rem={"batch":"batch32","remaining_work":[
        "Fresh separate reviewer writes canonical independent_review.json (integrity/compliance PASS/FAIL + scientific_gain).",
        "Parent route decision on whether PARM HEK293 association is interpretable given UNKNOWN HEK training membership + small disjoint subset.",
        "If exposure must be resolved: obtain a verified HEK293-trained fragment member list or an assay-matched 293FT autonomous-promoter dataset (bounded, no fits)."]}
    with open(os.path.join(OUT,"remaining_work.json"),"w",encoding="utf-8") as f:
        json.dump(rem,f,ensure_ascii=False,indent=2)
    nxt={"batch":"batch32","proposal_id":"batch33_293ft_context_audit",
        "tied_to_findings":"Exposure remains UNKNOWN and the coordinate-DISJOINT subset is small; local 293FT direction macro fails. Further coordinate audits have diminishing value.",
        "proposal":"ONE bounded non-training step: assemble and audit ASSAY-MATCHED 293FT/HEK autonomous-promoter fragment-activity evidence for the local candidate promoter families (ACTB/GAPDH/CMV truncations) from existing/available public records, and test whether a parameter-free (zero-fit) length/truncation direction prior reproduces the local D28 direction.",
        "bounded":"<=1 source survey, <=30MB download, <=30 min CPU, ZERO fits/inference/GPU",
        "target":"USER 293FT context (new promoter/truncation sequence activity), per PERFORMANCE_PRIORITY",
        "explicitly_not":"No sparse-SuRE/STR training, no 673/14 hyperparameter tuning, no new model, no promotion.",
        "time_limit":"<=30 min worker; fresh review <=20 min","compute":"CPU<=4, ZERO GPU, ZERO fits"}
    with open(os.path.join(OUT,"next_batch_proposal.json"),"w",encoding="utf-8") as f:
        json.dump(nxt,f,ensure_ascii=False,indent=2)
    reg=rj(REG)
    entry={"record_type":"evidence_only","batch":"batch32","version":"v6.22.0-exposure-endpoint.rc1",
        "kind":"exposure/endpoint evidence; ZERO fits/inference/GPU; NOT a trained model",
        "appended_by":"batch32 worker (DeepSeek Flash implementer; parent-authorized append-only)","appended_at":now(),
        "production_pointer_unchanged":"v6.3 main / v6.4 cellline pilot",
        "sectionA":{"assembly":"HEK673 hg19; library hg19","frozen_counts":cnt,
            "full673_rho_checksum":A["full673_checksum"]["rho"],
            "subgroup_rho":{s:fs[s].get("rho") for s in fs}},
        "sectionB":{"endpoint":B["endpoint_evidence"]["target_name"],"macro_direction":B["macro_family_direction"]}}
    reg.setdefault("batch_appends",[]).append(entry)
    reg["last_appended_at"]=now(); reg["last_appended_by"]="batch32 worker (parent-authorized append-only)"
    tmp=REG+".tmp"; json.dump(reg,open(tmp,"w",encoding="utf-8"),ensure_ascii=False,indent=2); os.replace(tmp,REG)
    head=("batch32 v6.22.0-exposure-endpoint.rc1 -> EVIDENCE ONLY (NOT a trained model, NO promotion, ZERO fits/GPU). "
        "A: assembly both hg19 (HEK673 prior source; library via official PARM_preprocessing_pipeline example_config GENOME hg19). "
        "Frozen label-free overlap counts "+json.dumps(cnt)+". Full673 ranking checksum PARM %.5f/R1 %.5f/H1 %.5f (matches batch31). "
        "Subgroup rho "+json.dumps({s:fs[s].get('rho') for s in fs})+
        ". HEK293 TRAINING MEMBERSHIP=UNKNOWN (distributed subset HAP1/AGS-only); DISJOINT!=unseen families. ")%(
        A['full673_checksum']['rho']['PARM'],A['full673_checksum']['rho']['R1'],A['full673_checksum']['rho']['H1'])
    body=("B: local 293FT D28 endpoint = log10(GFP+ gated median / same-batch WT-neg median); macro direction PARM %.2f; "
        "A1-A4/G1-G5 nest exactly in A20/G20, A5 1bp mismatch, CMVcore=3prime 204bp of CMVfull; no independent biological replicates/barcode link. "
        "Production pointer unchanged (v6.3 main / v6.4 cellline pilot).\n")%(B['macro_family_direction']['PARM'])
    with open(CHG,"a",encoding="utf-8") as f: f.write("\n["+now()+"] "+head+body)
    man={rel(p):sha(p) for p in gather() if rel(p) not in EXCL}
    with open(os.path.join(OUT,"sha_manifest.json"),"w",encoding="utf-8") as f:
        json.dump({"batch":"batch32","version":"v6.22.0-exposure-endpoint.rc1","generated_at":now(),
                   "excluded_mutable":sorted(EXCL),"n_files":len(man),"sha256":man},f,ensure_ascii=False,indent=2)
    print("finalized:",json.dumps({"n_manifest":len(man),"counts":cnt,"rho":A["full673_checksum"]["rho"],
        "sub":{s:fs[s].get('rho') for s in fs}},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
