import gzip,sys,json,hashlib,os
from collections import defaultdict,Counter
p=sys.argv[1]; out=sys.argv[2]
recs=[]  # (chrom,start,end,strand,feat, cellvals...)
with gzip.open(p,'rt',encoding='utf-8') as f:
    hdr=f.readline().rstrip('\n').split('\t')
    ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        c=line.rstrip('\n').split('\t')
        recs.append((c[ix['chr']],int(c[ix['start']]),int(c[ix['end']]),c[ix['strand']],c[ix['FEAT']]))
L=[e-s for _,s,e,_,_ in recs]
m=Counter(L)
# group by common oriented 3' end
g=defaultdict(list)
for ch,s,e,st,feat in recs:
    idn="%s:%d-%d:%s"%(ch,s,e,st)
    g[(ch,st,(e if st=='+' else s))].append((s,e,feat,idn))
def sha_part(cid):
    h=hashlib.sha256(("b44-nested-v1|"+cid).encode()).digest()
    return int.from_bytes(h[:8],'big')%100
groups={}   # key -> list of (len, id)
dup_groups=0; elig_groups=0; pairs=0
part=Counter(); grp_sizes=Counter()
endpoints=0; eligible_endpoints=0
conflicts=0
for k,v in g.items():
    lens=[(e-s,s,e,feat,idn) for s,e,feat,idn in v]
    ids=[x[4] for x in lens]
    if len(ids)!=len(set(ids)): dup_groups+=1
    dl=sorted(set(e-s for (s,e,feat,idn) in v))
    dl=[x for x in dl if 100<=x<=600]
    if len(dl)>=2:
        elig_groups+=1
        pairs+=len(dl)-1
        cid=min(x[4] for x in lens)
        b=sha_part(cid)
        part['TRAIN' if b<70 else ('DEV' if b<85 else 'TEST')]+=1
        grp_sizes[min(len(dl),6)]+=1
        eligible_endpoints+=len([x for x in lens if 100<=x[2]-x[1]<=600])
    endpoints+=len(lens)
res={
 "shard":os.path.basename(p),
 "shard_sha256":hashlib.sha256(open(p,'rb').read()).hexdigest(),
 "total_fragment_rows":len(recs),
 "length_min":min(L),"length_median":sorted(L)[len(L)//2],"length_max":max(L),
 "distinct_3prime_end_keys":len(g),
 "keys_with_exact_duplicate_ids":dup_groups,
 "eligible_groups_ge2_distinct_len_100_600":elig_groups,
 "adjacent_pairs":pairs,
 "eligible_endpoint_records":eligible_endpoints,
 "union_group_partition_projection":dict(part),
 "group_size_hist(<=6)":dict(grp_sizes),
 "note":"HAP1/AGS shard only (Log2Norm/Log2RPM_HAP1, _AGS). PROTOCOL-EXCLUDED as HEK labels; structural projection only."
}
open(out,'w',encoding='utf-8').write(json.dumps(res,indent=2,ensure_ascii=True))
print(json.dumps(res,indent=2))
