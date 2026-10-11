# batch51 v6.37.0-hek-reference-nested-cohort.rc1 coordinate-only nested cohort builder.
# ZERO fit/solve/calibration/modelscore/GPU. Label-free grouping. Frozen hash split.
# Author assembly = hg19 (GEO GSE301246 MH4 sample metadata). Local indexed hg19 ABSENT -> no DNA.
import gzip, hashlib, json, time, collections
from pathlib import Path

ROOT = Path('E:/promoter_atlas')
DATA = ROOT/'data/breakthrough_20261002/batch51'
BED  = DATA/'GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz'
OUT  = DATA/'cohort'
OUT.mkdir(parents=True, exist_ok=True)

T0=time.time()
def log(*a): print('[%7.1fs]'%(time.time()-T0),*a, flush=True)

MINL, MAXL = 100, 600
CAP_COMPONENTS = 500
CAP_ENDPOINTS  = 100000
CAP_PAIRS      = 100000
SALT_CAP   = 'b51-parent-cap-v1|'
SALT_SPLIT = 'b51-nested-v1|'

def h(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

rows=[]; nrows=0; n_na=0; header=None
with gzip.open(BED,'rt',encoding='utf-8') as f:
    header=f.readline().rstrip('\n').split('\t')
    for line in f:
        a=line.rstrip('\n').split('\t'); nrows+=1
        ch,s,e,st=a[1],int(a[2]),int(a[3]),a[4]
        try: hekv=float(a[5])
        except: hekv=float('nan'); n_na+=1
        try: rpmv=float(a[7])
        except: rpmv=float('nan')
        try: lnav=float(a[6])
        except: lnav=float('nan')
        cid='%s:%d-%d:%s'%(ch,s,e,st)
        rows.append((ch,s,e,st,a[44],int(a[45]),int(a[46]),a[47],a[48],hekv,rpmv,lnav,cid))
log('parsed rows',nrows,'header_cols',len(header),'HEK_NA',n_na)

elig=[i for i,r in enumerate(rows) if MINL<=(r[2]-r[1])<=MAXL]
lengths=sorted(rows[i][2]-rows[i][1] for i in elig)
log('eligible rows',len(elig),'length min/median/max',lengths[0],lengths[len(lengths)//2],lengths[-1])

ep2rows=collections.defaultdict(list); row2ep={}
for i in elig:
    ch,s,e,st=rows[i][0],rows[i][1],rows[i][2],rows[i][3]
    pos = e if st=='+' else s
    k=(ch,st,pos); ep2rows[k].append(i); row2ep[i]=k
log('distinct endpoints',len(ep2rows))

parent=list(range(len(rows)))
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[max(ra,rb)]=min(ra,rb)

by_feat=collections.defaultdict(list)
for i in elig: by_feat[rows[i][4]].append(i)
feat_edges=0
for fidx,idxs in by_feat.items():
    a=idxs[0]
    for b in idxs[1:]: union(a,b); feat_edges+=1
log('FEAT-name union edges',feat_edges,'distinct FEAT',len(by_feat),
    'FEATtype',dict(collections.Counter(rows[i][8] for i in elig)))

by_chr=collections.defaultdict(list)
for i in elig: by_chr[rows[i][0]].append(i)
ov_edges=0
for ch,idxs in by_chr.items():
    idxs.sort(key=lambda i:(rows[i][1], rows[i][2]))
    cur_rep=None; cur_max=-1
    for i in idxs:
        s,e=rows[i][1],rows[i][2]
        if cur_rep is None or s>=cur_max:
            cur_rep=i; cur_max=e
        else:
            union(cur_rep,i); ov_edges+=1
            if e>cur_max: cur_max=e
log('interval-overlap union edges',ov_edges)

comp=collections.defaultdict(list)
for i in elig: comp[find(i)].append(i)
comps=list(comp.values())
log('initial components across full shard',len(comps))

def comp_stats(idxs):
    eps=collections.defaultdict(set)
    for i in idxs: eps[row2ep[i]].add(rows[i][2]-rows[i][1])
    pairs=0; epwp=0
    for ep,ls in eps.items():
        if len(ls)>=2: epwp+=1; pairs+=(len(ls)-1)
    return len(eps),epwp,pairs

cinfo=[]
for idxs in comps:
    cid_min=min(rows[i][12] for i in idxs)
    n_ep,epwp,pairs=comp_stats(idxs)
    cinfo.append({'cid_min':cid_min,'members':idxs,'n_ep':n_ep,'n_pairs':pairs,'h':h(SALT_CAP+cid_min)})
log('total components',len(cinfo),'total pairs(all)',sum(c['n_pairs'] for c in cinfo),
    'comps_with_pairs',sum(1 for c in cinfo if c['n_pairs']>0))

cinfo.sort(key=lambda c:(c['h'], c['cid_min']))
chosen=[]; tot_ep=0; tot_pairs=0; excluded_budget=0; excluded_count=0
for c in cinfo:
    if len(chosen)>=CAP_COMPONENTS: excluded_count+=1; continue
    if tot_ep + c['n_ep'] > CAP_ENDPOINTS or tot_pairs + c['n_pairs'] > CAP_PAIRS:
        excluded_budget+=1; continue
    chosen.append(c); tot_ep+=c['n_ep']; tot_pairs+=c['n_pairs']
log('chosen comps',len(chosen),'endpoints',tot_ep,'pairs',tot_pairs,
    'excluded_budget',excluded_budget,'excluded_count',excluded_count)

def split_of(uid):
    v=int(h(SALT_SPLIT+uid)[:8],16)%100
    return 'TRAIN' if v<70 else ('DEV' if v<85 else 'TEST')

members=[]; groups=[]; pair_rows=[]
for c in chosen:
    uid=c['cid_min']; sp=split_of(uid)
    groups.append((uid,sp,len(c['members']),c['n_ep'],c['n_pairs']))
    for i in c['members']:
        r=rows[i]
        members.append((r[12],uid,sp,r[0],r[1],r[2],r[3],r[2]-r[1],r[4],r[7],r[8]))
    eps=collections.defaultdict(dict)
    for i in c['members']:
        ep=row2ep[i]; L=rows[i][2]-rows[i][1]; eps[ep][L]=i
    for ep,ld in eps.items():
        ls=sorted(ld.keys())
        for k in range(len(ls)-1):
            si=ld[ls[k]]; li=ld[ls[k+1]]
            pair_rows.append((uid,sp,ep[0],ep[1],ep[2],rows[si][12],ls[k],rows[li][12],ls[k+1]))

f=open(OUT/'members.tsv','w',encoding='utf-8',newline='\n')
f.write('coord_id\tunion_id\tsplit\tchr\tstart\tend\tstrand\tlength\tFEAT\tFEATsense\tFEATtype\n')
for m in members: f.write('\t'.join(str(x) for x in m)+'\n')
f.close()
f=open(OUT/'groups.tsv','w',encoding='utf-8',newline='\n')
f.write('union_id\tsplit\tn_members\tn_endpoints\tn_adjacent_pairs\n')
for g in groups: f.write('\t'.join(str(x) for x in g)+'\n')
f.close()
split_manifest={'salt_split':SALT_SPLIT,'salt_cap':SALT_CAP,
  'bins':{'TRAIN':'hash mod100 <70','DEV':'70-84','TEST':'>=85'},'counts':{}}
for g in groups: split_manifest['counts'][g[1]]=split_manifest['counts'].get(g[1],0)+1
json.dump(split_manifest,open(OUT/'split_manifest.json','w',encoding='utf-8'),indent=2)
log('members/groups/split persisted; component split counts',split_manifest['counts'])

cid2lab={r[12]:r[9] for r in rows}
pair_rows.sort()
trd=open(OUT/'pairs_train_dev.tsv','w',encoding='utf-8',newline='\n')
trd.write('union_id\tsplit\tchr\tstrand\tthree_prime_end\tshort_coord_id\tshort_len\tlong_coord_id\tlong_len\tshort_log2norm_hek\tlong_log2norm_hek\tdelta_short_minus_long\n')
sealed=open(DATA/'SEALED_test_labels.tsv','w',encoding='utf-8',newline='\n')
sealed.write('union_id\tshort_coord_id\tshort_len\tlong_coord_id\tlong_len\tshort_log2norm_hek\tlong_log2norm_hek\tdelta_short_minus_long\n')
ntr=ndv=nte=0
for pr in pair_rows:
    uid,sp,ch,st,pos,scid,sl,lcid,ll=pr
    slab=cid2lab.get(scid,float('nan')); llab=cid2lab.get(lcid,float('nan'))
    d=(slab-llab) if (slab==slab and llab==llab) else float('nan')
    if sp=='TEST':
        sealed.write('%s\t%s\t%d\t%s\t%d\t%.6f\t%.6f\t%.6f\n'%(uid,scid,sl,lcid,ll,slab,llab,d)); nte+=1
    else:
        trd.write('%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\t%d\t%.6f\t%.6f\t%.6f\n'%(uid,sp,ch,st,pos,scid,sl,lcid,ll,slab,llab,d))
        if sp=='TRAIN': ntr+=1
        else: ndv+=1
trd.close(); sealed.close()
log('pairs TRAIN',ntr,'DEV',ndv,'TEST_sealed',nte,'total',ntr+ndv+nte)

json.dump({'code':'build_cohort.py','nrows_total':nrows,'n_eligible':len(elig),
  'n_endpoints_total':len(ep2rows),'n_components_full':len(comps),
  'chosen_components':len(chosen),'chosen_endpoints':tot_ep,'chosen_pairs':tot_pairs,
  'excluded_budget':excluded_budget,'excluded_count':excluded_count,
  'pairs_TRAIN':ntr,'pairs_DEV':ndv,'pairs_TEST_sealed':nte,
  'components_TRAIN':split_manifest['counts'].get('TRAIN',0),
  'components_DEV':split_manifest['counts'].get('DEV',0),
  'components_TEST':split_manifest['counts'].get('TEST',0)},
  open(OUT/'cohort_summary.json','w',encoding='utf-8'),indent=2)
log('DONE')
