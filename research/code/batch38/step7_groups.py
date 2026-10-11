# -*- coding: utf-8 -*-
import csv, collections, re, json
KV="data/breakthrough_20261002/batch36/mapt/cohort_KV_SNV_HEK293FT.tsv"
rows=list(csv.DictReader(open(KV,encoding="utf-8"),delimiter="\t"))
def rc(s): return s.translate(str.maketrans("ACGTN","TGCAN"))[::-1]
pos=sorted(int(r["pos"]) for r in rows)
# interval union (half=109 as batch37)
half=219//2
iv=sorted((p-half,p+half) for p in pos)
groups=[];cs,ce=iv[0]
for s,e in iv[1:]:
    if s<=ce: ce=max(ce,e)
    else: groups.append((cs,ce));cs,ce=s,e
groups.append((cs,ce))
sizes=[sum(1 for p in pos if cs<=p<=ce) for cs,ce in groups]
print("interval-union groups:",len(groups),"min size",min(sizes),"max",max(sizes))
print("size histogram (bucket->count):",dict(sorted(collections.Counter(sizes).items())))
print("n groups >=10 pairs:",sum(1 for x in sizes if x>=10))
# 2kb gap clusters
ps=sorted(set(pos)); n=1
for a,b in zip(ps,ps[1:]):
    if b-a>2000:n+=1
print("2kb-gap clusters:",n)
# exact / RC / shared-ref groups
alt_ins=[r["variable_insert"] for r in rows]
ref_ins=None
exact=collections.Counter(alt_ins)
# RC groups: union-find on sequences and their RC
parent={}
def find(x):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
    return x
def uni(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
for s in set(alt_ins): find(s)
for s in set(alt_ins): uni(s,rc(s))
rcgroups=collections.Counter(find(s) for s in alt_ins)
print("exact-seq groups:",len(exact),"max multiplicity",max(exact.values()))
print("RC-merged groups:",len(rcgroups),"max",max(rcgroups.values()))
hist=collections.Counter(rcgroups.values())
print("RC group size histogram:",dict(sorted(hist.items())))
# shared ref groups
ref_cnt=collections.Counter()
print("n distinct alt inserts:",len(exact))
json.dump({"interval_union_groups":len(groups),"min":min(sizes),"max":max(sizes),
           "size_hist":{str(k):v for k,v in sorted(collections.Counter(sizes).items())},
           "n_ge10":sum(1 for x in sizes if x>=10),"gap_clusters_2kb":n,
           "exact_groups":len(exact),"exact_max_mult":max(exact.values()),
           "rc_groups":len(rcgroups),"rc_max_mult":max(rcgroups.values()),
           "rc_size_hist":{str(k):v for k,v in sorted(hist.items())}},
          open("data/breakthrough_20261002/batch38/step7_groups.json","w"),indent=1)
