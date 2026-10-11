import json, collections
bio=None
# recompute from joined_target.tsv
import csv
rows=[]
with open('reports/breakthrough_20261002/batch40/joined_target.tsv',encoding='utf-8') as f:
    r=csv.DictReader(f,delimiter='\t')
    for x in r: rows.append(x)
byc=collections.defaultdict(list)
for j in rows:
    if j['control']=='0':
        byc[j['chr']].append((int(j['start']),int(j['end']),j['insert_270bp'],j['label']))
groups=[]
for c in sorted(byc):
    iv=sorted(byc[c]); cur=None
    for s,e,ins,lab in iv:
        if cur is not None and s<=cur[2]:
            cur[2]=max(cur[2],e); cur[3].append((s,e,ins,lab))
        else:
            if cur is not None: groups.append(cur)
            cur=[c,s,e,[(s,e,ins,lab)]]
    if cur is not None: groups.append(cur)
comp=collections.Counter(g[0] for g in groups)
sizes=collections.Counter(len(g[3]) for g in groups)
def rc(x): return x.translate(str.maketrans('ACGTacgt','TGCAtgca'))[::-1]
comps_shared=0; merges=0
for g in groups:
    seen={}; coll=0
    for s,e,ins,lab in g[3]:
        if ins in seen or rc(ins) in seen: coll+=1
        seen[ins]=1
    if coll: comps_shared+=1; merges+=coll
largest=sorted(((len(g[3]),g[0],g[1],g[2]) for g in groups),reverse=True)[:12]
out={
 'n_components':len(groups),
 'chromosome_composition':dict(comp.most_common()),
 'size_dist':dict(sorted(sizes.items())),
 'largest_components':largest,
 'components_with_shared_exact_or_rc_ins':comps_shared,
 'merge_count':merges,
 'note':'components of OVERLAPPING 270bp windows (step90); NOT independent CRE families. Shared exact/RC 270bp inserts merged only where actually recovered (none among distinct labels). 1669 non-chr17 rows -> dataset not exclusively MAPT.',
}
json.dump(out,open('data/breakthrough_20261002/batch40/grouping.json','w',encoding='utf-8'),indent=1)
print(json.dumps(out,indent=1))
