# batch51 finalize: endpoints, 20 raw checks, reference provenance, readiness, manifest, registry append.
# ZERO fit/solve/calibration/modelscore/GPU. Read-only over cohort outputs. Real clock at write.
import gzip, json, hashlib, os, time, collections, statistics
from datetime import datetime
from pathlib import Path

ROOT=Path('E:/promoter_atlas')
DATA=ROOT/'data/breakthrough_20261002/batch51'
OUT=DATA/'cohort'
REP=ROOT/'reports/breakthrough_20261002/batch51'
REP.mkdir(parents=True, exist_ok=True)
def now(): return datetime.now().astimezone().isoformat()
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rsha(p):
    h=hashlib.sha256(); h.update(open(p,'rb').read()); return h.hexdigest()

BED=DATA/'GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz'
bed_sha=sha256(BED); bed_bytes=BED.stat().st_size

# ---- reload cohort ----
members=[]
with open(OUT/'members.tsv',encoding='utf-8') as f:
    hdr=f.readline()
    for l in f: members.append(l.rstrip('\n').split('\t'))
members_by_uid=collections.defaultdict(list)
for m in members: members_by_uid[m[1]].append(m)
groups=[]
with open(OUT/'groups.tsv',encoding='utf-8') as f:
    f.readline()
    for l in f: groups.append(l.rstrip('\n').split('\t'))
gmap={g[0]:g[1] for g in groups}

pairs=[]
with open(OUT/'pairs_train_dev.tsv',encoding='utf-8') as f:
    f.readline()
    for l in f: pairs.append(l.rstrip('\n').split('\t'))
sealed=[]
with open(DATA/'SEALED_test_labels.tsv',encoding='utf-8') as f:
    f.readline()
    for l in f: sealed.append(l.rstrip('\n').split('\t'))
summ=json.load(open(OUT/'cohort_summary.json',encoding='utf-8'))

# ---- endpoints.tsv for chosen cohort ----
ep=collections.defaultdict(lambda:{'L':set(),'rows':0,'split':None})
uid_of={}
for m in members:
    cid,uid,sp,ch,s,e,st,L=m[0],m[1],m[2],m[3],int(m[4]),int(m[5]),m[6],int(m[7])
    pos=e if st=='+' else s
    k=(ch,st,pos)
    ep[k]['L'].add(L); ep[k]['rows']+=1; ep[k]['split']=sp; uid_of[k]=uid
with open(OUT/'endpoints.tsv','w',encoding='utf-8',newline='\n') as f:
    f.write('chr\tstrand\tthree_prime_end\tn_rows\tn_distinct_lengths\tlengths\tsplit\tunion_id\n')
    for k in sorted(ep.keys()):
        d=ep[k]; ls=sorted(d['L'])
        f.write('%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n'%(k[0],k[1],k[2],d['rows'],len(ls),','.join(map(str,ls)),d['split'],uid_of[k]))
n_ep_chosen=len(ep)

# ---- 20 raw row/endpoint checks (stratified: plus/minus/multi-FEAT/duplicate/NA/length-ends) ----
mem_by_ep=collections.defaultdict(list)
for m in members:
    ch,st=m[3],m[6]; s,e=int(m[4]),int(m[5]); pos=e if st=='+' else s
    mem_by_ep[(ch,st,pos)].append(m)
keys=sorted(ep.keys()); incl=set(); chk=[]
def pick(pred,n,tag):
    got=[]
    for k in keys:
        if k in incl: continue
        if pred(k): incl.add(k); got.append((k,tag))
        if len(got)>=n: break
    return got
chk+=pick(lambda k:k[1]=='+',6,'plus')
chk+=pick(lambda k:k[1]=='-',6,'minus')
chk+=pick(lambda k:len(ep[k]['L'])>=2,4,'multi_len')
_minL=min(min(ep[k]['L']) for k in keys); _maxL=max(max(ep[k]['L']) for k in keys)
chk+=pick(lambda k:_minL in ep[k]['L'],1,'len_min_%d'%_minL)
chk+=pick(lambda k:_maxL in ep[k]['L'],1,'len_max_%d'%_maxL)
chk+=pick(lambda k:len(ep[k]['L'])==1,2,'single_len')

checks=[]
for k,tag in chk:
    ch,st,pos=k; ls=sorted(ep[k]['L'])
    mem=mem_by_ep[k]
    conv=all((int(m[5]) if st=='+' else int(m[4]))==pos for m in mem)
    nfeat=len(set(m[8] for m in mem))
    ok_len=all(100<=L<=600 for L in ls)
    checks.append({'endpoint':f'{ch}|{st}|{pos}','tag':tag,'n_rows':ep[k]['rows'],
        'distinct_lengths':ls,'lengths_in_100_600':ok_len,'three_prime_convention_ok':conv,
        'n_distinct_FEAT_at_endpoint':nfeat,'split':ep[k]['split'],'union_id':uid_of[k]})
# globals: multiple-FEAT, duplicate, NA, duplicate-conflict
g_src_multi=0  # verified on full source during recon (0 coords carry >1 FEAT)
checks.append({'global':'multiple_FEAT_per_coordinate_source','value':g_src_multi,
    'detail':'no source coordinate maps to >1 FEAT; all FEAT associations included, none dropped'})
checks.append({'global':'duplicate_coordinate_ids_in_members','n_members':len(members),
    'n_distinct_cids':len(set(m[0] for m in members)),
    'duplicates':len(members)-len(set(m[0] for m in members))})
pairs_shorter=[p for p in pairs if int(p[6])>=int(p[8])]
checks.append({'global':'all_pairs_short_len_lt_long_len','n_pairs':len(pairs),'violations':len(pairs_shorter)})
na_pairs=sum(1 for p in pairs if p[9]=='nan' or p[10]=='nan' or p[11]=='nan')
checks.append({'global':'pair_labels_NA_count','n_pairs':len(pairs),'NA':na_pairs})
json.dump({'n_endpoint_checks':len(chk),'n_checks_total':len(checks),'checks':checks,'generated_at':now()},
          open(DATA/'raw_checks_20.json','w',encoding='utf-8'),indent=2)

# ---- TRAIN label summary (after freeze; TEST sealed not summarised) ----
tr_delta=[float(p[11]) for p in pairs if p[1]=='TRAIN' and p[11]!='nan']
tr_short=[float(p[9]) for p in pairs if p[1]=='TRAIN' and p[9]!='nan']
train_label_summary={'n_TRAIN_pairs':sum(1 for p in pairs if p[1]=='TRAIN'),
  'n_TRAIN_pairs_finite_delta':len(tr_delta),
  'delta_mean':round(statistics.fmean(tr_delta),6) if tr_delta else None,
  'delta_std':round(statistics.pstdev(tr_delta),6) if tr_delta else None,
  'delta_min':min(tr_delta) if tr_delta else None,'delta_max':max(tr_delta) if tr_delta else None,
  'short_Log2Norm_HEK293_mean':round(statistics.fmean(tr_short),6) if tr_short else None}

# ---- reference provenance ----
prov={'source_bed':str(BED.relative_to(ROOT)).replace(os.sep,'/'),'source_sha256':bed_sha,'source_bytes':bed_bytes,
  'author_series':'GSE301246','author_pmid':41639451,'author_lab':'van Steensel (NKI)',
  'library':'PARM_Training_Library_v2_MH4','assay':'MPRA random genomic fragments, HEK293 (ATCC CRL-1573), 24h, rep1+rep2',
  'target_column':'Log2Norm_HEK293','normalization_column':'pDNA_T1_T2_sum',
  'target_formula_author':'Log2RPM_replicate = log2((cDNA_RPM/iPCR_RPM)+pseudocount); pseudocount = 0.1 quantile of non-zero ratios; then shift by abs(min); mean over replicates (author Snakefile+normalize_counts.py). Log2Norm/LnNorm columns are parallel normalizations in the v2 release.',
  'author_assembly_declared':'hg19','author_assembly_evidence':'GEO GSE301246_family.soft.gz: 71x "!Sample_data_processing = Assembly: hg19", incl. HEK293 Library_MH4 24h rep1/rep2 and MH4 barcode_fragment_mapping/pDNA samples',
  'coordinate_convention':'0-based half-open BED(start,end); strand +/-',
  'local_reference_present':{'path':'legacy_data/GRCh38.primary_assembly.genome.fa','assembly':'GRCh38','chr1_len':248956422,'indexed':True},
  'local_reference_matches_author_assembly':False,
  'reference_blocker':'author declares hg19; only GRCh38 locally indexed; whole-genome fetch prohibited by protocol; correct-assembly sequence extraction NOT possible',
  'dna_extracted':False,'fasta_written':False,
  'residual_uncertainty':'author metadata hg19 is the best available declaration for the v2 MH4 samples; it is the same series that released v2_MH4 files. Empirical re-derivation would require a local hg19 genome (absent).',
  'donor':'HG02601 (per author method); actual donor DNA not distributed -> reference-proxy estimand only, not donor identity'}
json.dump(prov,open(DATA/'reference_provenance.json','w',encoding='utf-8'),indent=2)

# ---- download manifest (bytes) ----
dl={'file':'GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz',
  'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE301nnn/GSE301246/suppl/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz',
  'content_length':103144961,'mime':'application/x-gzip','etag':None,'accept_ranges':'bytes',
  'requests':[{'kind':'HEAD','bytes':0},{'kind':'range_probe','bytes':65536,'content_range':'bytes 0-65535/103144961'},
              {'kind':'range_bulk','bytes':103079425,'content_range':'bytes 65536-103144960/103144961'}],
  'aux_primary_document_requests':[{'kind':'GET','url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE301nnn/GSE301246/soft/GSE301246_family.soft.gz','bytes':5672,'http':200}],
  'http_requests_bed_file':3,'http_requests_total_network':4,'logical_attempts':1,'failed_attempts':0,
  'incremental_bytes_bed':103144961,'incremental_bytes_aux':5672,'incremental_bytes':103150633,
  'historical_bytes_batch44':104192526,
  'combined_bytes':207343159,'incremental_cap':130000000,'combined_cap':234192526,
  'actual_bytes_on_disk':bed_bytes,'sha256':bed_sha,'gzip_full_decompress_ok':True,'decompressed_bytes':329876998,
  'header_verified_before_bulk':True,'probe_matches_cached_v2hdr':True}
json.dump(dl,open(DATA/'download_manifest.json','w',encoding='utf-8'),indent=2)

# ---- readiness gate ----
withpairs=collections.Counter()
for g in groups:
    if int(g[4])>0: withpairs[g[1]]+=1
comps=collections.Counter(g[1] for g in groups)
readiness={
 'gate_1_target_Log2Norm_HEK293_known':{'pass':True,'detail':'column present; author formula documented'},
 'gate_2_correct_assembly_representation_explicit_proxy':{'pass':False,
   'detail':'author assembly hg19; local indexed hg19 ABSENT; GRCh38 local != hg19; no DNA extracted (whole-genome fetch prohibited)'},
 'gate_3_97pct_endpoint_unique_row_match':{'pass':True,'detail':'all 761658 source coordinates distinct; every endpoint derived from a measured row; unexplained conflicts 0'},
 'gate_4_component_counts':{'pass':(withpairs['TRAIN']>=100 and withpairs['DEV']>=20 and withpairs['TEST']>=30),
   'TRAIN':withpairs['TRAIN'],'DEV':withpairs['DEV'],'TEST':withpairs['TEST'],
   'need':{'TRAIN':100,'DEV':20,'TEST':30},'detail':'union components with >=1 measured adjacent pair'},
 'OVERALL_READY':False,
 'reason':'assembly/sequence representation missing (gate_2). Coordinate cohort, groups and frozen split are delivered; no fit proposal may claim READY.'}
json.dump(readiness,open(DATA/'readiness.json','w',encoding='utf-8'),indent=2)

# ---- feature-readiness (CPU, NO solve) ----
n_ep=n_ep_chosen; n_pairs=len(pairs)+len(sealed)
onehot_bytes=n_ep*600*4*4
frc=[f'feature-readiness (CPU only, NO fit/solve):',
 f'  endpoints(chosen cohort)={n_ep}',
 f'  pairs total (TRAIN+DEV+TEST)={n_pairs}  (TRAIN {summ["pairs_TRAIN"]}, DEV {summ["pairs_DEV"]}, TEST sealed {summ["pairs_TEST_sealed"]})',
 f'  one-hot feature matrix if L_max=600 float32: {n_ep} x 600 x 4 = {onehot_bytes} bytes (~{onehot_bytes/1e6:.1f} MB)',
 f'  -> cannot build: no correct-assembly (hg19) DNA available locally; reference provenance blocker stands.',
 f'  recommended if unblocked: RAM >= {max(2,round(onehot_bytes/1e9*2,1))} GB incl. both arms; CPU-only; group weights from n_members.']
(ROOT/'data/breakthrough_20261002/batch51/feature_readiness_checks.txt').write_text('\n'.join(frc)+'\n',encoding='utf-8')

# ---- report + status ----
rep_md=f"""# batch51 v6.37.0-hek-reference-nested-cohort.rc1 -- worker report

Clock: {now()}

## Source contract (Phase A)
- Author series GSE301246 (PMID 41639451, van Steensel lab). Library v2 MH4; assay HEK293 MH4 24h rep1/rep2.
- Target column `Log2Norm_HEK293`; normalization column `pDNA_T1_T2_sum`.
- Formula (author Snakefile + normalize_counts.py): `Log2RPM = log2((cDNA_RPM/iPCR_RPM)+pseudocount)`, pseudocount=0.1 quantile of non-zero ratios, shifted by abs(min), mean over replicates.
- **Author assembly = hg19** (GEO soft: 71x `Assembly: hg19`, on the MH4 samples themselves). Local reference is GRCh38 (chr1=248,956,422) -> wrong assembly. No hg19 locally.
- Coordinates: 0-based half-open BED.

## Download (Phase B)
- File `GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz`, 103,144,961 B, sha256 {bed_sha[:16]}...
- 3 HTTP requests (HEAD + range probe + range bulk), 1 logical attempt, 0 failures, ~12 s. Probe byte-identical to cached v2hdr (sha 4865521f...).
- incremental {dl['incremental_bytes']} B (cap 130,000,000) | historical {dl['historical_bytes_batch44']} B | combined {dl['combined_bytes']} B (cap 234,192,526). gzip CRC/EOF OK.

## Cohort (Phase C)
- 761,658 source rows retained on disk; eligible (100<=L<=600) {summ['n_eligible']}; lengths {summ.get('length_min',104)}..599.
- distinct endpoints (full shard) {summ['n_endpoints_total']}; components (FEAT-union + same-chr interval overlap) {summ['n_components_full']}.
- Cap: <=500 components by smallest sha256('b51-parent-cap-v1|'+canonicalcoordID) -> {summ['chosen_components']} comps, {summ['chosen_endpoints']} endpoints, {summ['chosen_pairs']} adjacent pairs. Excluded: budget {summ['excluded_budget']}, over-500 {summ['excluded_count']}.
- Adjacent distinct lengths per 3' endpoint; short/long share oriented 3' endpoint. No activity/positive/rank filter; no DNA (no guessed sequence).
- Frozen split sha256('b51-nested-v1|'+unionID)%100: TRAIN {summ['components_TRAIN']} comps / DEV {summ['components_DEV']} / TEST {summ['components_TEST']}.
- TEST labels sealed: data/breakthrough_20261002/batch51/SEALED_test_labels.tsv (NOT summarised).

## Readiness (Phase D)
- gate1 target formula: PASS
- gate2 correct assembly representation: **FAIL** (author hg19; local GRCh38; no hg19)
- gate3 endpoint/row uniqueness: PASS (100%)
- gate4 component counts with pairs: PASS (TRAIN {withpairs['TRAIN']}/DEV {withpairs['DEV']}/TEST {withpairs['TEST']})
- **OVERALL READY = False.** Coordinate cohort delivered; sequence representation blocked.

## Next (requires parent decision)
1. Supply a local indexed **hg19** reference (or author-authorized interval extraction). Whole-genome fetch remains prohibited.
2. Then extract only the cohort intervals (<=600 bp x {n_ep} endpoints), verify short==exact suffix of long, and re-open the readiness gate.
3. Only then: two-arm CPU experiment -- arm A absolute activity model; arm B direct-delta model on feature(short)-feature(long) with zero free intercept; same frozen representation+regularization+TRAIN-only normalizer+group weights; compare TRAIN/DEV, then once TEST after both model writes + gate ledger; fixed zero-change and length-sign baselines. NO fits in batch51.
"""
(REP/'report.md').write_text(rep_md,encoding='utf-8')

status={'batch':'batch51','version':'v6.37.0-hek-reference-nested-cohort.rc1','record_type':'worker_self_check',
  'phase':'worker','state':'cohort_built_gate_failed_assembly','self_independent_PASS':False,
  'note':'coordinate cohort + frozen split built; correct-assembly sequence blocked; OVERALL_READY=False',
  'actual_clock':now(),'model':'deepseek-flash',
  'deliverables':{'source_bed':'data/breakthrough_20261002/batch51/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz',
    'cohort':'data/breakthrough_20261002/batch51/cohort','sealed_test_labels':'data/breakthrough_20261002/batch51/SEALED_test_labels.tsv',
    'reference_provenance':'data/breakthrough_20261002/batch51/reference_provenance.json',
    'download_manifest':'data/breakthrough_20261002/batch51/download_manifest.json',
    'readiness':'data/breakthrough_20261002/batch51/readiness.json'},
  'zero_fit_accounting':{'fits':0,'solves':0,'calibration':0,'modelscore':0,'gpu':0,'downloads_beyond_v2fold0':0},
  'production_pointer':'unchanged (v6.3 main / v6.4 cellline pilot)',
  'next':'independent fresh review; then parent decides hg19 supply vs alternative route',
  'ongoing_task_active':True,'experiment_complete':False}
json.dump(status,open(REP/'status.json','w',encoding='utf-8'),indent=2)

# ---- registry append (atomic, preserve original) ----
regp=ROOT/'reports/breakthrough_20261002/version_registry.json'
reg=json.load(open(regp,encoding='utf-8'))
entry={'record_type':'cohort_preparation_batch','batch':'batch51','version':'v6.37.0-hek-reference-nested-cohort.rc1',
  'kind':'REAL coordinate cohort from authentic GSE301246 v2_MH4 fold0 HEK shard + frozen split; sequence representation BLOCKED (author hg19, only GRCh38 local). NOT a model, ZERO fit/solve/score/GPU.',
  'appended_by':'batch51 worker (DeepSeek Flash implementer; parent-authorized append-only)','appended_at':now(),
  'production_pointer_unchanged':'v6.3 main / v6.4 cellline pilot',
  'source':{'bed':str(BED.relative_to(ROOT)).replace(os.sep,'/'),'sha256':bed_sha,'bytes':bed_bytes,'assembly_author':'hg19'},
  'cohort':{'chosen_components':summ['chosen_components'],'endpoints':summ['chosen_endpoints'],'adjacent_pairs':summ['chosen_pairs'],
    'split_components':{'TRAIN':summ['components_TRAIN'],'DEV':summ['components_DEV'],'TEST':summ['components_TEST']}},
  'readiness':{'OVERALL_READY':False,'gate_failed':'correct_assembly_representation'},
  'outputs':{'members':'data/breakthrough_20261002/batch51/cohort/members.tsv','groups':'data/breakthrough_20261002/batch51/cohort/groups.tsv',
    'pairs':'data/breakthrough_20261002/batch51/cohort/pairs_train_dev.tsv','endpoints':'data/breakthrough_20261002/batch51/cohort/endpoints.tsv',
    'report':'reports/breakthrough_20261002/batch51/report.md'}}
reg.setdefault('batch_appends',[])
_already=[e for e in reg['batch_appends'] if e.get('version')==entry['version'] and e.get('batch')=='batch51']
if _already:
    print('registry append skipped (idempotent): entry already present for',entry['version'])
else:
    reg['batch_appends'].append(entry)
    reg['last_appended_at']=now(); reg['last_appended_by']='batch51 worker DeepSeek Flash'
tmp=regp.with_suffix('.json.tmp')
json.dump(reg,open(tmp,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
os.replace(tmp,regp)

# ---- manifest (exclude itself/status/review/verify/mail) ----
immut=[OUT/'members.tsv',OUT/'groups.tsv',OUT/'split_manifest.json',OUT/'pairs_train_dev.tsv',OUT/'endpoints.tsv',
       DATA/'SEALED_test_labels.tsv',DATA/'reference_provenance.json',DATA/'download_manifest.json',
       DATA/'raw_checks_20.json',DATA/'readiness.json',DATA/'feature_readiness_checks.txt']
man={'generated_at':now(),'excludes':['MANIFEST.json','status.json','review/','verify/','mail/'],
  'files':{}}
for p in immut:
    rel=str(p.relative_to(ROOT)).replace(os.sep,'/')
    man['files'][rel]={'bytes':p.stat().st_size,'sha256':sha256(p)}
json.dump(man,open(DATA/'MANIFEST.json','w',encoding='utf-8'),indent=2)

# ---- two read-only SHA checks ----
errs=[]
for rel,meta in man['files'].items():
    p=ROOT/rel
    if sha256(p)!=meta['sha256']: errs.append(rel)
print('MANIFEST files:',len(man['files']),'sha_check1_mismatches:',len(errs))
errs2=[r for r,m in man['files'].items() if rsha(ROOT/r)!=m['sha256']]
print('sha_check2_mismatches:',len(errs2))
print('READY:',readiness['OVERALL_READY'],'| comps with pairs:',dict(withpairs))
print('DONE',now())

