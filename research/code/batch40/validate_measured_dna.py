import gzip, re, json, collections
p = 'data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz'
hdr_re = re.compile(r'^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$')
FWD='AGGACCGGATCAACT'; REV='CATTGCGTGAACCGA'
recs=[]
with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
    nm=None; ch=[]
    def fl(n,s):
        recs.append((n,s))
    for line in f:
        if line.startswith('>'):
            if nm is not None: fl(nm,''.join(ch))
            nm=line[1:].rstrip('\n'); ch=[]
        else: ch.append(line.strip())
    if nm is not None: fl(nm,''.join(ch))

# 1) adapter boundary exactness: seq = FWD + insert(270) + REV
bad_bound=0; inserts={}
for h,s in recs:
    if len(s)!=300: bad_bound+=1; continue
    if s[:15]!=FWD or s[-15:]!=REV: bad_bound+=1
    inserts[h]=s[15:285]
# 2) window overlap coherence within a family: window k+1 should be shifted by 90 -> insert_k[90:270]==insert_{k+1}[0:180]
fam=collections.defaultdict(list)
for h,s in recs:
    m=hdr_re.match(h)
    if not m: continue
    base=re.sub(r'_\d+$','',m.group('name'))
    fam[(base,m.group('chr'))].append((int(m.group('start')),int(m.group('end')),h,inserts[h]))
n_pairs=0; n_ok=0; off_hist=collections.Counter()
for k,lst in fam.items():
    lst.sort()
    bystart={st:h for st,en,h,ins in lst}
    for st,en,h,ins in lst:
        nxt=st+90
        if nxt in bystart:
            h2=bystart[nxt]; ins2=inserts[h2]
            n_pairs+=1
            if ins[90:270]==ins2[:180]:
                n_ok+=1
            else:
                # find offset
                found=None
                for o in range(-5,6):
                    if o==0: continue
                off_hist['MISMATCH']+=1
# 3) global duplicate insert check (exact and RC) among the 8944 coord records
coord_ins=[inserts[h] for h,s in recs if hdr_re.match(h)]
cnt=collections.Counter(coord_ins)
dup_ins=[k for k,c in cnt.items() if c>1]
def rc(x): return x.translate(str.maketrans('ACGTacgt','TGCAtgca'))[::-1]
# RC collisions: insert equals RC of another (different) insert
rcset=collections.Counter(rc(x) for x in coord_ins)
rc_collisions=sum(1 for x in set(coord_ins) if rc(x) in cnt and x!=rc(x))
# lowercase fraction (soft-mask) sanity
import statistics
lower_frac=[sum(1 for c in ins if c.islower())/len(ins) for ins in list(inserts.values())[:500]]
out={
 'records':len(recs),
 'adapter_boundary_bad':bad_bound,
 'insert_len_unique':sorted(set(len(v) for v in inserts.values())),
 'family_groups':len(fam),
 'overlap_pairs_checked':n_pairs,
 'overlap_pairs_consistent':n_ok,
 'overlap_consistency_fraction':round(n_ok/n_pairs,6) if n_pairs else None,
 'coord_records':len(coord_ins),
 'distinct_inserts':len(cnt),
 'exact_dup_insert_groups':len(dup_ins),
 'exact_dup_insert_examples':[x[:30] for x in dup_ins[:3]],
 'rc_collision_inserts':rc_collisions,
 'mean_lowercase_fraction':round(statistics.mean(lower_frac),4),
}
json.dump(out, open('data/breakthrough_20261002/batch40/measured_dna_validation.json','w',encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
