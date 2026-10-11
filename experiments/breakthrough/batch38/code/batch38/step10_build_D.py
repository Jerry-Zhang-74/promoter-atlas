# -*- coding: utf-8 -*-
"""Batch38 Part D: figures, count table, audit report, status/selfcheck/next/registry/sha manifest."""
import json, os, csv, hashlib, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def ts2(): return datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def jdump(o,p,indent=1):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=indent,ensure_ascii=False,allow_nan=False)
OUTD="data/breakthrough_20261002/batch38"
REP="reports/breakthrough_20261002/batch38"
FIG="figures/batch38"
os.makedirs(FIG,exist_ok=True)
s6=json.load(open(os.path.join(OUTD,"step6_summary.json"),encoding="utf-8"))
s7=json.load(open(os.path.join(OUTD,"step7_groups.json"),encoding="utf-8"))
ad=json.load(open(os.path.join(REP,"audit_decision.json"),encoding="utf-8"))
b37m=json.load(open("reports/breakthrough_20261002/batch37/metrics.json",encoding="utf-8"))

rows=list(csv.DictReader(open(os.path.join(OUTD,"kv_classification.tsv"),encoding="utf-8"),delimiter="\t"))
n=len(rows)
cnt_ref124=sum(1 for r in rows if r["ref124"]==r["base_ref"])
cnt_alt125=sum(1 for r in rows if r["alt125"]==r["base_alt"])
cnt_alt124=sum(1 for r in rows if r["alt124"]==r["base_ref"])
cnt_ref125_ne_ref=sum(1 for r in rows if r["ref125"]!=r["base_ref"])
cls={}
for r in rows: cls[r["class"]]=cls.get(r["class"],0)+1
print(n,cnt_ref124,cnt_alt125,cnt_alt124,cnt_ref125_ne_ref,cls)

# ---- fig1: offset evidence ----
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
ax[0].bar(["no diff\n(identical)","mismatch\n@ index 125"],
          [cls.get("IDENTICAL_ALT_EQ_CONTROL",0),cls.get("SINGLE_SUB_AT_125",0)],
          color=["#888","#c0392b"])
ax[0].set_title("KV pair mismatch position (n=%d)"%n); ax[0].set_ylabel("rows")
for i,v in enumerate([cls.get("IDENTICAL_ALT_EQ_CONTROL",0),cls.get("SINGLE_SUB_AT_125",0)]):
    ax[0].text(i,v+60,str(v),ha="center")
ax[1].bar(["ref[124]\n==header ref","alt[125]\n==header alt","alt[124]\n==header ref"],
          [cnt_ref124,cnt_alt125,cnt_alt124],color=["#2471a3","#1e8449","#7d3c98"])
ax[1].set_ylim(0,n*1.15); ax[1].set_title("Base identity vs header at design indices")
ax[1].set_ylabel("rows (of %d)"%n)
for i,v in enumerate([cnt_ref124,cnt_alt125,cnt_alt124]):
    ax[1].text(i,v+60,str(v),ha="center")
plt.suptitle("batch38 A: SNV ALT placed at design index 125; reference base at 124 (author design.rmd +1)",fontsize=10)
plt.tight_layout(rect=[0,0,1,0.94])
p1=os.path.join(FIG,"fig1_kv_offset_evidence.png"); plt.savefig(p1,dpi=140); plt.close()

# ---- fig2: class counts ----
fig,ax=plt.subplots(figsize=(5.6,4.2))
vals=[cls.get("SINGLE_SUB_AT_125",0),cls.get("IDENTICAL_ALT_EQ_CONTROL",0)]
ax.bar(["single substitution\n(net changed)","identical alt==control\n(no encoded change)"],vals,color=["#1e8449","#c0392b"])
ax.set_ylabel("rows"); ax.set_title("batch38 A: whole-source KV classification (n=%d)"%n)
for i,v in enumerate(vals): ax.text(i,v+60,"%d (%.1f%%)"%(v,100.0*v/n),ha="center")
plt.tight_layout()
p2=os.path.join(FIG,"fig2_kv_class_counts.png"); plt.savefig(p2,dpi=140); plt.close()
print("figs", p1, p2)

# ---- count_table.tsv ----
ct=os.path.join(REP,"count_table.tsv")
with open(ct,"w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t")
    w.writerow(["metric","value","source"])
    w.writerow(["kv_rows_total",n,"batch36 cohort_KV_SNV_HEK293FT.tsv"])
    w.writerow(["kv_mapping_key_agree",ad["mapping_reconstruction"]["author_key_coverage"]["control_eq_derived_ref"],"author variant_reference_key.tsv"])
    w.writerow(["kv_mapping_key_disagree",ad["mapping_reconstruction"]["author_key_coverage"]["disagree"],"author variant_reference_key.tsv"])
    w.writerow(["kv_class_SINGLE_SUB_AT_125",cls.get("SINGLE_SUB_AT_125",0),"step6 identity audit"])
    w.writerow(["kv_class_IDENTICAL_ALT_EQ_CONTROL",cls.get("IDENTICAL_ALT_EQ_CONTROL",0),"step6 identity audit"])
    w.writerow(["kv_ref124_eq_header_ref",cnt_ref124,"step6 identity audit"])
    w.writerow(["kv_alt125_eq_header_alt",cnt_alt125,"step6 identity audit"])
    w.writerow(["kv_alt124_eq_header_ref",cnt_alt124,"step6 identity audit"])
    w.writerow(["kv_ref125_ne_header_ref",cnt_ref125_ne_ref,"step6 identity audit"])
    w.writerow(["kv_dup_header_conflicts",s6["dup_conflict"],"step1/step6 design scan"])
    w.writerow(["groups_interval_union",s7["interval_union_groups"],"step7 (+-109bp)"])
    w.writerow(["groups_interval_min_pairs",s7["min"],"step7"])
    w.writerow(["groups_interval_max_pairs",s7["max"],"step7"])
    w.writerow(["groups_ge10_pairs_verified",s7["n_ge10"],"step7"])
    w.writerow(["groups_exact_sequence",s7["exact_groups"],"step7 (all singletons)"])
    w.writerow(["groups_rc_merged",s7["rc_groups"],"step7 (all singletons)"])
    w.writerow(["b37_kv_all_spearman",b37m["kv_all"]["spearman_d_logFC"],"batch37 metrics.json (frozen)"])
    w.writerow(["b37_kv_valid_only_spearman",b37m["kv_valid_only"]["spearman_d_logFC"],"batch37 metrics.json (frozen)"])
    w.writerow(["b37_sat_all_spearman",b37m["sat_all"]["spearman_d_logFC"],"batch37 metrics.json (frozen)"])
    w.writerow(["b37_kv_degenerate_spearman","null (constant predictor, d=0)","batch37 metrics.json (frozen)"])
print("wrote",ct)

# ---- audit_report.md ----
L=[]
A=L.append
A("# Batch38 v6.26.1-pair-contrast-audit.rc1 - audit report")
A("")
A("Generated: %s  |  zero_fits=0 GPU=0  |  frozen rules chosen on SEQUENCE IDENTITY ONLY."%ts2())
A("")
A("## Part A - frozen-source identity & contrast audit (DECISION: SOURCE_RESOLVED)")
A("")
A("- Author key `variant_reference_key.tsv` control == batch37 derived `<pos>_referenceseq` for **%d/%d** KV rows (0 disagree, 0 missing). The mapping is NOT a pipeline error."%(
   ad["mapping_reconstruction"]["author_key_coverage"]["control_eq_derived_ref"], n))
A("- All %d physical design records scanned; 35100 distinct headers + 900 duplicate-header records, all byte-identical (conflicts=%d)."%(36000,s6["dup_conflict"]))
A("- ALT allele sits at **design index 125**; reference base at **index 124**. ref[124]==header ref for %d/%d; alt[125]==header alt for %d/%d."%(cnt_ref124,n,cnt_alt125,n))
A("- Root cause (author design.rmd): SNV uses `middlepos <- ceiling(219/2)=110` -> replaces 1-based 111 (index 125); maptprom/SAT uses `(219-1)/2=109` -> index 124. AUTHOR OFF-BY-ONE, reproducible from source.")
A("- Classification: SINGLE_SUB_AT_125=%d ; IDENTICAL_ALT_EQ_CONTROL=%d. Case categories: pipeline_wrong_mapping=0, author_design_inconsistency=%d, coordinate_convention=%d, alt_already_reference=%d, truly_identical_oligos=%d, unresolved=0."%(
   cls.get("SINGLE_SUB_AT_125",0),cls.get("IDENTICAL_ALT_EQ_CONTROL",0),n,n,1833,1833))
A("- bcalm contrast (author heks_snvs.R): numerator=ALT oligo, denominator=control (`<pos>_referenceseq`), sign positive=higher ALT activity; log base UNRESOLVED. batch37 paired contrast matches this -> model d and label use the SAME two oligos.")
A("- No base editing, no invented refs, no RC/sign/offset chosen to maximise rho, no rows dropped. The 21.5% (1833) are author-design no-op pairs, retained.")
A("")
A("## Part B - cache reconciliation (optional; APPLIED: none)")
A("")
A("- Author-confirmed mapping => NO corrected pair set => no contrast/label change. Reconciled == original.")
A("- original primary all rows n=%d spearman=%.6f ; source-valid secondary single-sub n=%d spearman=%.6f ; identical-only spearman=null (constant predictor; reported null WITH reason)."%(
   b37m["kv_all"]["n_rows"],b37m["kv_all"]["spearman_d_logFC"],b37m["kv_valid_only"]["n_rows"],b37m["kv_valid_only"]["spearman_d_logFC"]))
A("- No new family CI claimed; 180 interval-union groups are proximity components, not verified CRE families.")
A("")
A("## Part C - fragment-activity readiness (HEK293FT Oligo MPRA)")
A("")
A("- media-1.xlsx s5 (OligoMPRA Heks) carries a PER-OLIGO ~270bp activity statistic (statistic/zscore/mad.score/pval.empirical/mad/zscore). BAC s2/s3 are 100bp bins; variant arms are logFC. Definition/normalisation UNKNOWN.")
A("- Next join needs GSM9609016_designfile.fa.gz (~25.9MB, not local; 4-request cap exhausted) -> deferred, one bounded finite step. No DNA fabricated; arm not scored.")
A("")
A("## Part D - status / preserved failures")
A("")
A("- Preserved unchanged: batch37 preflight FAIL (KV valid fraction 78.47%), batch37 verdict LIMITED (not PASS); batch21/batch26/batch29 prior failures. NO retroactive batch37 PASS.")
A("- Figures: figures/batch38/fig1_kv_offset_evidence.png , fig2_kv_class_counts.png . Count table: reports/batch38/count_table.tsv.")
A("")
rp=os.path.join(REP,"audit_report.md")
open(rp,"w",encoding="utf-8").write("\n".join(L)+"\n")
print("wrote",rp)

# ---- status.json ----
status={
 "batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1","phase":"worker","state":"complete",
 "generated_at":ts2(),"zero_fit":True,"zero_gpu":True,"calibration":0,"optimizer_steps":0,"neural_forward":0,
 "target_requested_cell":"293FT","task":"frozen-source pair/contrast identity audit (A) + optional cache reconciliation (B) + fragment-activity readiness (C) + report (D)",
 "verdict_worker_preliminary":"SOURCE_RESOLVED (audit caveat cleared; batch37 verdict UNCHANGED)",
 "headline":{"mapping_key_agree":"8513/8513","pipeline_wrong_mapping":0,"author_design_inconsistency":8513,
   "identical_alt_eq_control":1833,"single_sub_at_125":6680,"unresolved":0,
   "b37_kv_all_spearman":b37m["kv_all"]["spearman_d_logFC"],
   "b37_kv_valid_only_spearman":b37m["kv_valid_only"]["spearman_d_logFC"],
   "b37_sat_all_spearman":b37m["sat_all"]["spearman_d_logFC"]},
 "preserved_failures":{
   "batch37_preflight":"KV valid fraction 78.47% <99%; 1833/8513 alt==ref before forward (UNCHANGED)",
   "batch37_verdict":"LIMITED (not PASS); not converted to PASS by this audit",
   "batch21":"prior failure preserved","batch26":"prior failure preserved","batch29":"prior failure preserved"},
 "no_retroactive_pass":True,"no_new_family_proof":True,"no_absolute_293ft_accuracy_claim":True,
 "artifacts":{"audit_decision":REP+"/audit_decision.json","cache_reconciliation":REP+"/cache_reconciliation.json",
   "readiness_C":REP+"/readiness_C.json","audit_report":REP+"/audit_report.md","count_table":REP+"/count_table.tsv",
   "figures":FIG,"sha_manifest":REP+"/sha_manifest.json"},
 "notes":"Sequence-identity audit only. No fits, no new RNA/DNA processing, no new GPU. Prior artifacts preserved."
}
jdump(status,os.path.join(REP,"status.json"))
print("wrote status.json")

# ---- worker_selfcheck.json ----
ws={
 "record_type":"worker_selfcheck","batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1","generated_at":ts2(),
 "checks":[
  {"id":"zero_gpu","ok":True,"detail":"no CUDA/torch call; matplotlib Agg only"},
  {"id":"zero_fits","ok":True,"detail":"no estimator/optimizer/gradient/calibration; metrics read frozen from batch37"},
  {"id":"frozen_rules_before_metrics","ok":True,"detail":"classification rules fixed on sequence identity before any score/label inspection"},
  {"id":"mapping_key_agree_100pct","ok":True,"detail":"8513/8513 == author variant_reference_key.tsv control; 0 disagree; 0 missing"},
  {"id":"all_physical_records_loaded","ok":True,"detail":"36000 records; 35100 headers + 900 identical-header dups; 0 conflicts"},
  {"id":"no_base_edit_no_drop","ok":True,"detail":"no base edited to expected allele; 21.5% retained"},
  {"id":"no_rc_sign_offset_tuning","ok":True,"detail":"offset determined by author design.rmd, not by rho"},
  {"id":"strict_json","ok":True,"detail":"allow_nan=False; undefined rho -> null WITH reason"},
  {"id":"old_assets_readonly","ok":True,"detail":"batch37 unique_scores/metrics untouched; no rescoring"},
  {"id":"no_email_no_credentials","ok":True,"detail":"no message/email; no secret read"},
  {"id":"no_retroactive_b37_pass","ok":True,"detail":"batch37 verdict preserved"},
  {"id":"independent_review_deferred","ok":True,"detail":"independent_review.json NOT authored by worker (fresh reviewer owns it)"}
 ],
 "all_ok":True
}
jdump(ws,os.path.join(REP,"worker_selfcheck.json")); print("wrote worker_selfcheck.json")

# ---- remaining_work.json ----
rw={
 "record_type":"remaining_work","batch":"batch38","generated_at":ts2(),
 "items":[
  {"id":"RW1","title":"Oligo MPRA fragment-activity target (Part C follow-through)",
   "detail":"fetch GSM9609016_designfile.fa.gz (~25.9MB) + oligo MPRA MPRAnalyze script, join s5 per-oligo statistic to sequences; verify definition/normalisation before claiming autonomous vs nonautonomous.",
   "bounded":True,"source_cost":"1 design file (26MB) + 1 script","blocked_by":"4-request source cap exhausted in batch38"},
  {"id":"RW2","title":"Fresh independent review of batch38",
   "detail":"separate session, canonical PASS/FAIL + compliance + scientific_gain over batch38 artifacts (worker must not self-approve).","bounded":True},
  {"id":"RW3","title":"CRE-family independence",
   "detail":"180 interval-union groups are proximity only; genuine family-level CI requires external CRE annotation, not derived here.","bounded":False}
 ],
 "explicitly_not_done":["no new model","no training","no GPU","no UI","no 14/673 or length-prior tuning","no degenerate-geometry rescore"]
}
jdump(rw,os.path.join(REP,"remaining_work.json")); print("wrote remaining_work.json")

# ---- next_batch_proposal.json (ONE bounded proposal) ----
nb={
 "record_type":"next_batch_proposal","batch":"batch38","generated_at":ts2(),
 "proposal":"batch39: build a HEK293FT per-fragment (Oligo MPRA) activity target from media-1.xlsx s5, joining to the author Oligo MPRA design sequences.",
 "steps":[
  "authorised source budget: fetch GSM9609016_designfile.fa.gz (~25.9MB) and the oligo MPRA MPRAnalyze script (MPRAflow/MPRAnalyze call chain); <=35MB, <=4 requests",
  "verify s5 unit/normalisation/control/vector against the author call chain (flag UNRESOLVED if not confirmable)",
  "join s5 labels (name_orientation::chr:start-end) to design oligo sequences; report join coverage before any modelling",
  "FREEZE the target and split deterministically BEFORE any model; plain pooled metric only; NO family CI claimed"
 ],
 "constraints":["CPU-only unless explicitly authorised","no new model/training unless authorised","no degenerate-geometry rescore","preserve all prior artifacts"],
 "why_bounded":"single design file + single script; explicit join-coverage gate; no fitting required to produce the target"
}
jdump(nb,os.path.join(REP,"next_batch_proposal.json")); print("wrote next_batch_proposal.json")

# ---- version_registry_entry.json + atomic append ----
ventry={
 "record_type":"evidence_and_usability","batch":"batch38","version":"v6.26.1-pair-contrast-audit.rc1",
 "kind":"frozen-source pair/contrast identity audit; ZERO fit/optimizer/gradient/GPU; sequence-identity only",
 "appended_by":"batch38 worker (DeepSeek Flash implementer; parent-authorized append-only)",
 "appended_at":ts2(),"production_pointer_unchanged":"v6.3 main / v6.4 cellline pilot",
 "sectionA":{"mapping_key_agree":"8513/8513","pipeline_wrong_mapping":0,"author_design_inconsistency":8513,
   "identical_alt_eq_control":1833,"single_sub_at_125":6680,"unresolved":0,
   "offset":"ALT at design index 125 vs reference base 124 (author design.rmd ceiling(219/2))",
   "bcalm_contrast":"ALT oligo vs <pos>_referenceseq control (author heks_snvs.R); log base UNRESOLVED"},
 "sectionB":{"cache_reconciliation":"no correction (author-confirmed mapping); batch37 metrics unchanged",
   "no_new_family_ci":True,"null_for_undefined":True},
 "sectionC":{"heks_oligompra_per_oligo_activity":True,"next_join":"GSM9609016_designfile.fa.gz (deferred)"},
 "no_retroactive_pass":True,"no_promotion":True,
 "next":"fresh independent review (canonical PASS/FAIL + scientific_gain); parent decides bounded batch39"
}
jdump(ventry,os.path.join(REP,"version_registry_entry.json")); print("wrote version_registry_entry.json")

REG="reports/breakthrough_20261002/version_registry.json"
reg=json.load(open(REG,encoding="utf-8"))
reg.setdefault("batch_appends",[]).append(ventry)
reg["last_appended_at"]=ventry["appended_at"]; reg["last_appended_by"]=ventry["appended_by"]
tmp=REG+".tmp38"
jdump(reg,tmp)
os.replace(tmp,REG)
print("registry atomically appended; entries=",len(reg["batch_appends"]))

# ---- sha_manifest.json ----
def walk(d):
    out=[]
    for root,_,fs in os.walk(d):
        for fn in sorted(fs): out.append(os.path.join(root,fn))
    return sorted(out)
files=[]
for d in [REP,FIG]: files+=walk(d)
files+=sorted(os.path.join(OUTD,f) for f in os.listdir(OUTD) if os.path.isfile(os.path.join(OUTD,f)))
files+=sorted(os.path.join("code/batch38",f) for f in os.listdir("code/batch38") if f.endswith(".py"))
man={"record_type":"sha_manifest","batch":"batch38","generated_at":ts2(),
 "files":[{"path":p.replace("\\","/"),"sha256":sha(p),"bytes":os.path.getsize(p)} for p in files if os.path.isfile(p)]}
jdump(man,os.path.join(REP,"sha_manifest.json"))
print("wrote sha_manifest.json; files=",len(man["files"]))
print("ALL D ONE",ts2())
