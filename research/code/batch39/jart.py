import json,time
from pathlib import Path
ROOT=Path('E:/promoter_atlas');R=ROOT/'reports/breakthrough_20261002/batch39';D=ROOT/'data/breakthrough_20261002/batch39'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
s5=json.load(open(D/'s5_audit.json',encoding='utf-8'))['summary'];gate=json.load(open(R/'readiness_gate.json',encoding='utf-8'))
json.dump({'batch':'batch39','zero_fits':True,'zero_gpu':True,'model_inference':0,'frozen_before_metrics':True,
 'independent_review_created':False,'notes':'worker selfcheck only; not independent','s5_rows':s5['n_data_rows'],
 'real':s5['n_real'],'controls':s5['n_scrambled'],'join_coverage':gate['join_coverage_fraction'],'verdict':'NOT_READY'},
 open(R/'worker_selfcheck.json','w',encoding='utf-8'),indent=1)
json.dump({'batch':'batch39','phase':'worker','state':'complete','version':'v6.27.0-hekft-fragment-readiness.rc1',
 'generated_at':GEN,'verdict':'NOT_READY','zero_fits':True,'zero_gpu':True,
 'reports_dir':'reports/breakthrough_20261002/batch39','next':'independent review then parent decision'},
 open(R/'status.json','w',encoding='utf-8'),indent=1)
json.dump({'batch':'batch39','generated_at':GEN,'remaining':[
 'fetch+gate GSM9580234_design_rmIllegalChars.fa.gz (1 request)',
 'if binds: freeze seq->alpha target + label-free groups; if not: escalate with exact evidence',
 'minP/vector/autonomy still UNKNOWN','no model training authorized']},
 open(R/'remaining_work.json','w',encoding='utf-8'),indent=1)
json.dump({'record_type':'next_batch_proposal','batch':'batch39','generated_at':GEN,
 'proposal':'batch40: single-source gate on companion OligoMPRA design file GSM9580234_design_rmIllegalChars.fa.gz (429,038 B, 1 request) with the batch39 frozen join gate; >99% unique bind -> freeze 270bp seq->alpha target + groups; else escalate.',
 'task':'data readiness (NO fit/score/GPU)','input_contract':'label name_+::chr:start-end+ -> design header name_index_+ (author generateDesignfile.rmd)',
 'target':'s5 statistic = MPRAnalyze alpha (per-fragment activity)','grouping_limitation':'one broader locus; proximity components not CRE families',
 'acceptance':'coverage >0.99 unique, zero conflicting labels, boundaries evidenced; else NOT READY',
 'fit_gpu_accounting':'0 fits, 0 GPU','approval':'parent must separately approve execution'},
 open(R/'next_batch_proposal.json','w',encoding='utf-8'),indent=1)
ent={'record_type':'evidence_and_usability','batch':'batch39','version':'v6.27.0-hekft-fragment-readiness.rc1',
 'kind':'HEK293FT OligoMPRA fragment data-readiness; ZERO fit/optimizer/gradient/GPU',
 'appended_by':'batch39 worker (DeepSeek Flash implementer; parent-authorized append-only)','appended_at':GEN,
 'production_pointer_unchanged':'v6.3 main / v6.4 cellline pilot',
 'sectionA':{'source_requests':5,'bytes':25959333,'design_file':'GSM9609016_designfile.fa.gz (title Barcode Association Sequencing)',
   'design_is_oligompra_design':False,'exact_join':'1/5868 chr17, 0/1669 non-chr17','adapter_records':0},
 'sectionB':{'statistic':'MPRAnalyze getAlpha activity','norm':'uq depth factors per DNA/RNA',
   'reps':'one condition alpha (3 DNA + 3 RNA)','controls':'scram names','minP':'UNKNOWN',
   'paper270_vs_span271':'270bp; 271 is counting artifact','orientation':'+ only (unresolved)'},
 'sectionC':{'groups':'306 overlapping-window components; not CRE families'},
 'gate':'FAIL','verdict':'NOT_READY','no_predictive_gain':True,'no_retroactive_pass':True,'no_promotion':True,
 'preserved':['parent minP correction','batch37 preflight FAIL','local293FT D28 FAIL'],
 'next':'fresh independent review (canonical PASS/FAIL + scientific_gain); parent decides bounded batch40'}
json.dump(ent,open(R/'version_registry_entry.json','w',encoding='utf-8'),indent=1)
print('jart done')
