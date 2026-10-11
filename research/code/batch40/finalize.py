import gzip, re, json, collections, time
from pathlib import Path
ROOT = Path('E:/promoter_atlas')
SRC = ROOT/'data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz'
D = ROOT/'data/breakthrough_20261002/batch40'
R = ROOT/'reports/breakthrough_20261002/batch40'
D.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
GEN = time.strftime('%Y-%m-%dT%H:%M:%S%z')
hdr_re = re.compile(r'^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$')
FWD='AGGACCGGATCAACT'; REV='CATTGCGTGAACCGA'

# load design
design={}
with gzip.open(SRC,'rt',encoding='utf-8',errors='replace') as f:
    nm=None; ch=[]
    for line in f:
        if line.startswith('>'):
            if nm is not None: design[nm]=''.join(ch)
            nm=line[1:].rstrip('\n'); ch=[]
        else: ch.append(line.strip())
    if nm is not None: design[nm]=''.join(ch)

s5=json.load(open(ROOT/'data/breakthrough_20261002/batch39/s5_rows_raw.json',encoding='utf-8'))

# build joined rows
joined=[]
for r in s5:
    lab=r['A']; ctrl=int(r['F'])
    m=hdr_re.match(lab)
    h=design.get(lab)
    raw = h if h else None
    ins = raw[15:285] if raw else None
    joined.append({
        'label':lab,'chr':(m.group('chr') if m else ''),'start':(m.group('start') if m else ''),
        'end':(m.group('end') if m else ''),'strand':(m.group('strand') if m else ''),
        'control':ctrl,'design_header':lab,'design_match':'exact_unique' if h else 'UNMATCHED',
        'raw_300bp':raw or '','insert_270bp':ins or '',
        'adapter_fwd_ok':('1' if raw and raw[:15]==FWD else '0') if raw else '',
        'adapter_rev_ok':('1' if raw and raw[-15:]==REV else '0') if raw else '',
        'statistic_alpha':r['E'],'z':r['G'],'mad':r['H'],'p_empirical':r['I'],'p_mad':r['J'],'p_z':r['K'],
    })
cols=['label','chr','start','end','strand','control','design_header','design_match','raw_300bp','insert_270bp','adapter_fwd_ok','adapter_rev_ok','statistic_alpha','z','mad','p_empirical','p_mad','p_z']
with open(R/'joined_target.tsv','w',encoding='utf-8',newline='') as f:
    f.write('\t'.join(cols)+'\n')
    for j in joined:
        f.write('\t'.join(str(j[c]) for c in cols)+'\n')

# FASTA of frozen 270bp inserts (matched rows)
with open(R/'design_270bp_inserts.fa','w',encoding='utf-8') as f:
    for j in joined:
        if j['insert_270bp']:
            f.write('>'+j['label']+'\n'+j['insert_270bp']+'\n')

# stratified joins (>=20)
bychrom=collections.defaultdict(list)
for j in joined:
    if j['control']==0: bychrom[j['chr']].append(j)
fam=collections.defaultdict(list)
for j in joined:
    if j['control']==0:
        mm=hdr_re.match(j['label']); base=re.sub(r'_\d+_\+$','',mm.group('name'))
        fam[base].append(j)
picks=[]
def add(stratum,rows,n):
    for r in rows[:n]: picks.append((stratum,r))
add('main_chr17', bychrom.get('chr17',[]), 6)
for c in ['chr11','chr21','chr4','chr14','chr18','chr1','chr8','chr22']:
    add('other_chrom_'+c, bychrom.get(c,[]), 1)
bigf=sorted(fam.items(), key=lambda kv:-len(kv[1]))
add('large_family_'+bigf[0][0], sorted(bigf[0][1], key=lambda x:int(x['start'])), 3)
add('controls', [j for j in joined if j['control']==1], 4)
# consecutive-window overlap pair within a family (evidence of step90)
seqfam=[x for x in sorted(bigf[0][1], key=lambda x:int(x['start']))]
if len(seqfam)>=2: picks.append(('window_overlap_pair', seqfam[0])); picks.append(('window_overlap_pair', seqfam[1]))
with open(R/'stratified_joins.tsv','w',encoding='utf-8',newline='') as f:
    f.write('stratum\tlabel\tchr\tstart\tend\tstrand\tcontrol\tdesign_match\tinsert_len\tadapter_ok\tstatistic_alpha\tinsert_head30\n')
    for st,j in picks:
        f.write('\t'.join([st,j['label'],j['chr'],j['start'],j['end'],j['strand'],str(j['control']),
                           j['design_match'],str(len(j['insert_270bp'])),j['adapter_fwd_ok']+j['adapter_rev_ok'],
                           j['statistic_alpha'], j['insert_270bp'][:30]])+'\n')

# grouping: reuse b39 algorithm on the 7537 (identical label-free intervals); then merge shared exact/RC DNA
bio=[j for j in joined if j['control']==0]
byc=collections.defaultdict(list)
for j in bio: byc[j['chr']].append((int(j['start']),int(j['end']),j['insert_270bp']))
groups=[]
for c,iv in sorted(byc.items()):
    iv=sorted(iv); cur=None
    for s,e,ins in iv:
        if cur and s<=cur[1]:
            cur[1]=max(cur[1],e); cur[4].append((s,e,ins))
        else:
            if cur: groups.append(cur)
            cur=[c,s,e,[],[]]
            cur[4].append((s,e,ins))
    if cur: groups.append(cur)
# per-chrom composition
comp=collections.Counter(g[0] for g in groups)
sizes=collections.Counter(len(g[4]) for g in groups)
# merge shared exact or RC DNA within a component
def rc(x): return x.translate(str.maketrans('ACGTacgt','TGCAtgca'))[::-1]
merges=0; components_with_shared=0
for g in groups:
    ins=[x[2] for x in g[4]]
    seen={}
    coll=False
    for x in ins:
        k=x
        if k in seen or rc(x) in seen: coll=True
        seen[k]=1
    if coll: components_with_shared+=1
grouping={
 'n_components':len(groups),
 'chromosome_composition':dict(comp.most_common()),
 'size_dist':dict(sorted(sizes.items())),
 'components_with_shared_exact_or_rc_ins':components_with_shared,
 'merge_note':'reused label-free overlapping-270bp-window components (b39 algorithm); shared exact/RC 270bp inserts merged only where actually recovered',
 'source_parent_note':'group names share a bed-name prefix (source parent); still NOT independent CRE families',
 'note':'components of OVERLAPPING 270bp windows (step90); NOT independent CRE families; 1669 non-chr17 rows mean the dataset is not exclusively MAPT.',
}
json.dump(grouping, open(D/'grouping.json','w',encoding='utf-8'), indent=1)

print('joined', len(joined), 'matched', sum(1 for j in joined if j['design_match']=='exact_unique'))
print('picks', len(picks), 'groups', len(groups), 'comp', dict(comp.most_common()))
print('strata', collections.Counter(p[0] for p in picks))
