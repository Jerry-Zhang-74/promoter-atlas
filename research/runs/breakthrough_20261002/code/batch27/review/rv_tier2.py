# -*- coding: utf-8 -*-
"""REVIEWER independent recount of tier-2 cohort overlap components. ZERO fits."""
import gzip, collections, sys
p="data/breakthrough_20261002/batch27/sure_tier2_cohort.tsv.gz"
rows=[]
b1=0;b2=0;lens=collections.Counter();n=0
by_chrom=collections.defaultdict(list)
with gzip.open(p,'rt',encoding='utf-8') as f:
    hdr=f.readline()
    for line in f:
        n+=1
        c=line.rstrip('\n').split('\t')
        chrom=c[0]; st=int(c[1]); en=int(c[2]); strand=c[3]
        b1+=int(c[5]); b2+=int(c[6]); L=int(c[7])
        lens[min(L//100*100,1900)]+=1
        by_chrom[chrom].append((st,en))
print("n",n,"biorep1_total",b1,"biorep2_total",b2)
# union-find components by interval overlap within chromosome
parent={}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
ncomp=0; maxsize=0
idx=0
for chrom,ivs in by_chrom.items():
    ivs.sort()
    # sweep: union intervals that overlap (>=1bp)
    active=[]  # list of (end, node)
    import heapq
    heap=[]
    nodes=[]
    for k,(st,en) in enumerate(ivs):
        node=(chrom,k); parent[node]=node; nodes.append(node)
    # sort by start already; use heap of (-end,node) for active overlaps
    for k,(st,en) in enumerate(ivs):
        node=(chrom,k)
        # pop intervals whose end <= st (no overlap)
        while heap and -heap[0][0] <= st:
            heapq.heappop(heap)
        # all remaining in heap overlap with this (their end>st)
        # union with just one representative is enough? No - need overlap relation; 
        # remaining heap items all have end>st and start<=st, so all overlap this node -> union all
        for e2,nd in heap:
            union(nd,node)
        heapq.heappush(heap,(-en,node))
# count roots
roots=set()
sizes=collections.Counter()
for node in parent:
    r=find(node); roots.add(r)
for node in parent:
    sizes[find(node)]+=1
print("n_overlap_components",len(roots))
print("n_singletons",sum(1 for v in sizes.values() if v==1))
print("max_component",max(sizes.values()))
print("len_min",min(lens), "median_100bins", sorted(k for k in lens))
