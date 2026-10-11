import gzip,sys,json,hashlib
from collections import defaultdict
p=sys.argv[1]
rows=[]
with gzip.open(p,'rt',encoding='utf-8') as f:
    hdr=f.readline().rstrip('\n').split('\t')
    ix={c:i for i,c in enumerate(hdr)}
    ci,si,ei,ti=ix['chr'],ix['start'],ix['end'],ix['strand']
    fi=ix['FEAT']
    for line in f:
        c=line.rstrip('\n').split('\t')
        rows.append((c[ci],int(c[si]),int(c[ei]),c[ti],c[fi]))
import statistics
L=[e-s for _,s,e,_,_ in rows]
print("rows",len(rows),"len min/med/mean/max",min(L),statistics.median(L),round(statistics.mean(L),1),max(L))
# common oriented 3' end
g=defaultdict(list)
for ch,s,e,st,feat in rows:
    g[(ch,st,(e if st=='+' else s))].append((s,e,feat))
groups=[(k,v) for k,v in g.items() if len(set(e-s for s,e,_ in v))>1]
print("shared-3'end groups with >=2 distinct lengths:",len(groups))
# apply 100<=short<long<=600 and exact suffix (automatic given same end), adjacent distinct lengths
adj=0; elig=0; ex_pairs=0
for k,v in groups:
    lens=sorted(set(e-s for s,e,_ in v))
    lens=[x for x in lens if 100<=x<=600]
    if len(lens)>=2:
        elig+=1
        ex_pairs+=len(lens)-1  # adjacent distinct lengths ascending
print("groups with >=2 distinct lengths in [100,600]:",elig,"adjacent pairs:",ex_pairs)
# union-merge by genomic overlap within chrom (rough): use FEAT windows as parents
from collections import Counter
# per (chrom,3'end) how many distinct lengths
print("example eligible groups:")
k=0
for kk,v in groups:
    lens=sorted(set(e-s for s,e,_ in v))
    lens2=[x for x in lens if 100<=x<=600]
    if len(lens2)>=2:
        print("  ",kk,[(s,e,e-s) for s,e,_ in sorted(v,key=lambda z:z[1]-z[0])][:6]); k+=1
        if k>=5: break
