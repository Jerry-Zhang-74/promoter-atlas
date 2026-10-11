import json, re, collections, time
from pathlib import Path
ROOT=Path('E:/promoter_atlas')
D=ROOT/'data/breakthrough_20261002/batch39'
R=ROOT/'reports/breakthrough_20261002/batch39'
R.mkdir(parents=True,exist_ok=True)
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
s5=json.load(open(D/'s5_rows_raw.json',encoding='utf-8'))
pat=re.compile(r'^(?P<name>.+?)::(?P<c>chr[0-9XYM]+):(?P<s>\d+)-(?P<e>\d+)(?P<st>[+-])$')
rows=[]
for r in s5:
    m=pat.match(r['A']); ctrl=int(r['F'])
    rec=dict(label=r['A'],ctrl=ctrl,statistic=r['E'],zscore=r['G'],mad_score=r['H'],
             pval_empirical=r['I'],pval_mad=r['J'],pval_zscore=r['K'])
    if m:
        rec.update(parsed=True,name=m.group('name'),chrom=m.group('c'),start=int(m.group('s')),
                   end=int(m.group('e')),strand=m.group('st'),span=int(m.group('e'))-int(m.group('s')))
    else:
        rec.update(parsed=False,name=r['A'],chrom=None,start=None,end=None,strand=None,span=None)
    rows.append(rec)
bychr=collections.defaultdict(list)
for r in rows:
    if r['parsed']: bychr[r['chrom']].append((r['start'],r['end'],r['label']))
groups=[]
for c,iv in bychr.items():
    iv=sorted(iv); cur=[c,iv[0][0],iv[0][1],[iv[0][2]]]
    for s,e,lab in iv[1:]:
        if s<=cur[2]: cur[2]=max(cur[2],e); cur[3].append(lab)
        else: groups.append(cur); cur=[c,s,e,[lab]]
    groups.append(cur)
sizes=collections.Counter(len(g[3]) for g in groups)
grouping={'n_groups':len(groups),'size_dist':dict(sorted(sizes.items())),
          'largest':sorted(((len(g[3]),g[0],g[1],g[2]) for g in groups),reverse=True)[:12],
          'note':'components of OVERLAPPING 270bp windows (step90); NOT independent CRE families'}
json.dump(grouping,open(D/'grouping.json','w',encoding='utf-8'),indent=1)
with open(R/'joined_target.tsv','w',encoding='utf-8',newline='') as f:
    h=['label','name','chr','start','end','strand','span','statistic_activity','control_flag',
       'zscore','mad_score','pval_empirical','pval_mad','pval_zscore','design_match_status',
       'measured_dna_status','design_reason']
    f.write('\t'.join(h)+'\n')
    for r in rows:
        if r['parsed']: dstat='UNRESOLVED'; reason='no_measured_design_record_in_GSM9609016_designfile'
        else: dstat='CONTROL_SCRAMBLED'; reason='author_scrambled_negative_control'
        f.write('\t'.join(str(x) for x in [r['label'],r['name'],r['chrom'],r['start'],r['end'],
                 r['strand'],r['span'],r['statistic'],r['ctrl'],r['zscore'],r['mad_score'],
                 r['pval_empirical'],r['pval_mad'],r['pval_zscore'],dstat,'null',reason])+'\n')
n_real=sum(1 for r in rows if r['parsed']); n_ctrl=sum(1 for r in rows if not r['parsed'])
gate={'batch':'batch39','version':'v6.27.0-hekft-fragment-readiness.rc1','record_type':'readiness_gate',
 'generated_at':GEN,'zero_fits':True,'zero_gpu':True,'model_inference':0,
 'eligible_biological_rows':n_real,'rows_with_unique_correct_measured_design':1,
 'join_coverage_fraction':round(1/n_real,6),'required_threshold':0.99,
 'target_statistic_definition_evidenced':True,'physical_sequence_boundaries_evidenced':False,
 'gate':'FAIL',
 'reason':'GSM9609016_designfile.fa.gz is a GEO Barcode Association Sequencing reference: chr17-only, 3946070 genome windows (spans 146-350bp), 0 records contain the plasmid adapter AGGACCGGATCAACT, and it contains essentially none of the s5 OligoMPRA labels (exact join 1/5868 chr17; 0 of 1669 non-chr17; best +/-1/270bp offset <=2). The measured 270bp OligoMPRA design DNA keyed to s5 labels is NOT present.',
 'unresolved_target_boundaries':True,'measured_dna':None,
 'measured_dna_reason':'not recovered; no fabrication permitted',
 'controls_retained':n_ctrl,'unmatched_real_rows_retained':n_real,'verdict':'NOT_READY',
 'missing_item':'the actual OligoMPRA measured design DNA keyed to s5 labels (companion file GSM9580234_design_rmIllegalChars.fa.gz, 429038 B, not authorized this batch)'}
json.dump(gate,open(R/'readiness_gate.json','w',encoding='utf-8'),indent=1)
json.dump(grouping,open(D/'grouping.json','w',encoding='utf-8'),indent=1)
print('A done; groups',grouping['n_groups'],'sizes',grouping['size_dist'])
