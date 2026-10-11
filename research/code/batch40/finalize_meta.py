import json, hashlib, time, os
from pathlib import Path
ROOT=Path('E:/promoter_atlas'); D=ROOT/'data/breakthrough_20261002/batch40'; R=ROOT/'reports/breakthrough_20261002/batch40'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
rg=json.load(open(R/'readiness_gate.json',encoding='utf-8'))
gate=json.load(open(D/'join_gate.json',encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
man={'batch':'batch40','version':'v6.27.1-oligo-design-gate.rc1','generated_at':GEN,
 'excludes':['sha_manifest.json','status.json','worker_selfcheck.json','remaining_work.json','next_batch_proposal.json',
             'independent_review.json','independent_reviewer.jsonl','review artifacts','mail artifacts'],
 'production_pointer_note':'v6.3 main / v6.4 cellline pilot unchanged','files':{}}
inc=['code/batch40/fetch_candidate.py','code/batch40/census.py','code/batch40/join_gate.py',
 'code/batch40/validate_measured_dna.py','code/batch40/finalize.py','code/batch40/grouping.py',
 'code/batch40/report_writer.py','code/batch40/finalize_meta.py',
 'data/breakthrough_20261002/batch40/source/source_fetch_log.json',
 'data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz',
 'data/breakthrough_20261002/batch40/census.json','data/breakthrough_20261002/batch40/join_gate.json',
 'data/breakthrough_20261002/batch40/measured_dna_validation.json','data/breakthrough_20261002/batch40/grouping.json',
 'reports/breakthrough_20261002/batch40/joined_target.tsv','reports/breakthrough_20261002/batch40/design_270bp_inserts.fa',
 'reports/breakthrough_20261002/batch40/stratified_joins.tsv','reports/breakthrough_20261002/batch40/readiness_gate.json',
 'reports/breakthrough_20261002/batch40/target_context_spec.json','reports/breakthrough_20261002/batch40/source20_check.json',
 'reports/breakthrough_20261002/batch40/addendum_correcting_b39.json','reports/breakthrough_20261002/batch40/source_audit_table.tsv',
 'reports/breakthrough_20261002/batch40/data_readiness_report.md','reports/breakthrough_20261002/batch40/version_registry_entry.json']
for f in inc:
    p=ROOT/f
    man['files'][f]={'sha256':sha(p),'bytes':p.stat().st_size} if p.exists() else {'sha256':None,'reason':'missing'}
json.dump(man,open(R/'sha_manifest.json','w',encoding='utf-8'),indent=1)
print('manifest',len(man['files']))

ws={'batch':'batch40','version':'v6.27.1-oligo-design-gate.rc1','generated_at':GEN,
 'zero_fits':True,'zero_gpu':True,'model_inference':0,'frozen_before_metrics':True,
 'independent_review_created':False,'notes':'worker selfcheck only; not independent',
 'design_records':gate['design_records'],'s5_bio':7537,'s5_controls':85,
 'bio_matched':gate['real_matched'],'ctrl_matched':gate['ctrl_matched'],
 'join_coverage':gate['real_match_fraction'],'unexplained_conflicts':gate['unmatched_coord_overlap_hits'],
 'gate':rg['gate'],'verdict':rg['verdict']}
json.dump(ws,open(R/'worker_selfcheck.json','w',encoding='utf-8'),indent=1)
st={'batch':'batch40','phase':'worker','state':'complete','version':'v6.27.1-oligo-design-gate.rc1','generated_at':GEN,
 'verdict':'READY','gate':'PASS','zero_fits':True,'zero_gpu':True,'reports_dir':'reports/breakthrough_20261002/batch40',
 'next':'independent review then parent decision on bounded frozen-model-comparison proposal'}
json.dump(st,open(R/'status.json','w',encoding='utf-8'),indent=1)
rw={'batch':'batch40','generated_at':GEN,'remaining':[
 'parent authorization required before any model fit/comparison uses this frozen target',
 'minimal promoter / vector / assay autonomy still UNKNOWN (not in design file or call chain)',
 'physical-assay orientation (single-strand?) still UNRESOLVED',
 'local293FT D28 and b37 preflight FAIL remain unconverted; no predictive claim made']}
json.dump(rw,open(R/'remaining_work.json','w',encoding='utf-8'),indent=1)
print('selfcheck+status+remaining done')

nb={'record_type':'next_batch_proposal','batch':'batch40','generated_at':GEN,
 'proposal':'batch41: frozen model comparison on the recovered OligoMPRA target - fixed software input (design_270bp_inserts.fa 270bp insert -> statistic alpha) and fixed task scope (label-free group CV vs a single frozen baseline), no tuning; parent must separately authorize before execution.',
 'task':'frozen supervised comparison (would consume fits/GPU - NOT authorized here)',
 'input_contract':'design_270bp_inserts.fa (270bp insert keyed to s5 statistic alpha); grouping = 306 label-free components',
 'target':'s5 statistic = MPRAnalyze getAlpha activity (uq)',
 'acceptance':'predefined metric + fixed split + baseline; report honestly; no fabrication',
 'fit_gpu_accounting_here':'0 fits, 0 GPU (this batch data-readiness only)','approval':'parent must separately approve execution'}
json.dump(nb,open(R/'next_batch_proposal.json','w',encoding='utf-8'),indent=1)
vre={'record_type':'evidence_and_usability','batch':'batch40','version':'v6.27.1-oligo-design-gate.rc1',
 'kind':'HEK293FT OligoMPRA measured-design recovery + join gate; ZERO fit/optimizer/gradient/GPU',
 'appended_by':'batch40 worker (DeepSeek Flash implementer; parent-authorized append-only)','appended_at':GEN,
 'production_pointer_unchanged':'v6.3 main / v6.4 cellline pilot',
 'sectionA':{'source_requests':1,'bytes':429038,'design_file':'GSM9580234_design_rmIllegalChars.fa.gz sha a61f3d76...',
   'design_records':9044,'coord_oligos':8944,'scrambled':100,'all_300bp':True,
   'join':'7537/7537 bio + 85/85 controls exact_unique','confidence':1.0,'unexplained_conflicts':0,'window_overlap_consistency':'8821/8821'},
 'sectionB':{'statistic':'MPRAnalyze getAlpha activity (unchanged)','norm':'uq depth factors','minP':'UNKNOWN','270bp_insert':'evidenced (FWD15+270+REV15)','orientation':'+ only in design; physical assay UNRESOLVED'},
 'sectionC':{'groups':'306 overlap components; not CRE families','merges':'0 (inserts unique; no RC collisions)'},
 'gate':'PASS','verdict':'READY','no_predictive_gain':True,'no_promotion':True,
 'preserved':['b37 preflight FAIL','local293FT D28 FAIL','parent minP correction','b39 bytes'],
 'next':'fresh independent review then parent decision on bounded frozen-model-comparison proposal'}
json.dump(vre,open(R/'version_registry_entry.json','w',encoding='utf-8'),indent=1)
reg=ROOT/'reports/breakthrough_20261002/version_registry.json'
o=json.load(open(reg,encoding='utf-8')); o.setdefault('batch_appends',[])
if not any(e.get('batch')=='batch40' for e in o['batch_appends']): o['batch_appends'].append(vre)
o['last_appended_at']=GEN
o['last_appended_by']='batch40 worker (DeepSeek Flash implementer; parent-authorized append-only)'
tmp=reg.with_suffix('.json.tmp'); tmp.write_text(json.dumps(o,indent=1,ensure_ascii=False),encoding='utf-8'); os.replace(tmp,reg)
ch=ROOT/'reports/breakthrough_20261002/CHANGELOG.txt'
with open(ch,'a',encoding='utf-8') as f:
    f.write('%s batch40 v6.27.1-oligo-design-gate.rc1: GSM9580234_design_rmIllegalChars.fa.gz recovered (1 req, 429038B, sha a61f3d76); 7537/7537 bio + 85/85 ctrl exact-unique join to the OligoMPRA design (9044 recs, all 300bp=FWD15+270+REV15); gate PASS -> READY; b39 collision numerator corrected to 0 verified; adapter unit now evidence-supported; minP/vector UNKNOWN; zero fit/GPU.\n'%GEN)
print('registry appends',len(o['batch_appends']))
