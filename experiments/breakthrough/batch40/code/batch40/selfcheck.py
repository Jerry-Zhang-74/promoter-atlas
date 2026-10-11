import json, csv, glob, os
from pathlib import Path
ROOT=Path('E:/promoter_atlas'); D=ROOT/'data/breakthrough_20261002/batch40'; R=ROOT/'reports/breakthrough_20261002/batch40'
res={}
# JSON parse all batch40 json
bad=[]
for f in glob.glob(str(D/'*.json'))+glob.glob(str(R/'*.json'))+glob.glob(str(D/'source/*.json')):
    try: json.load(open(f,encoding='utf-8'))
    except Exception as e: bad.append((os.path.basename(f),repr(e)))
res['json_parse_failures']=bad
# joined tsv
rows=list(csv.DictReader(open(R/'joined_target.tsv',encoding='utf-8'),delimiter='\t'))
res['joined_rows']=len(rows)
res['bio']=sum(1 for r in rows if r['control']=='0'); res['ctrl']=sum(1 for r in rows if r['control']=='1')
res['all_matched']=all(r['design_match']=='exact_unique' for r in rows)
res['all_insert270']=all(len(r['insert_270bp'])==270 for r in rows)
res['all_adapters']=all(r['adapter_fwd_ok']=='1' and r['adapter_rev_ok']=='1' for r in rows)
res['cols']=list(rows[0].keys())
# targets untouched (alpha nonzero preserved incl negatives/NA)
res['stat_numeric']=sum(1 for r in rows if r['statistic_alpha'] not in ('','NA'))
res['stat_nonpos']=sum(1 for r in rows if r['statistic_alpha'] not in ('','NA') and float(r['statistic_alpha'])<=0)
# stratified
sl=open(R/'stratified_joins.tsv',encoding='utf-8').read().splitlines()
res['stratified_rows']=len(sl)-1
# fa
fa=open(R/'design_270bp_inserts.fa',encoding='utf-8').read().splitlines()
res['fasta_headers']=sum(1 for l in fa if l.startswith('>'))
# manifest self-exclusion
man=json.load(open(R/'sha_manifest.json',encoding='utf-8'))
res['manifest_files']=len(man['files'])
res['manifest_excludes_self']=('sha_manifest.json' not in man['files'])
res['manifest_self_excluded_labels']=man['excludes']
res['independent_review_exists']=(R/'independent_review.json').exists()
# no forbidden fit/GPU code strings
import re
forbidden=[]
for f in glob.glob('code/batch40/*.py'):
    t=open(f,encoding='utf-8').read()
    for kw in ['torch','tensorflow','cuda','optimizer','backward(','fit(']:
        if kw in t: forbidden.append((os.path.basename(f),kw))
res['forbidden_terms_in_batch40_code']=forbidden
json.dump(res,open(D/'selfcheck_result.json','w',encoding='utf-8'),indent=1)
print(json.dumps(res,indent=1))
