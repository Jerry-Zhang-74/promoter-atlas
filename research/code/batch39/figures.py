import json, collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('E:/promoter_atlas')
D=ROOT/'data/breakthrough_20261002/batch39'
F=ROOT/'figures/batch39'; F.mkdir(parents=True,exist_ok=True)
s5=json.load(open(D/'s5_audit.json',encoding='utf-8'))['summary']
g=json.load(open(D/'grouping.json',encoding='utf-8'))
# Fig1: tiles per chromosome
ch=s5['chroms']
ks=sorted(ch,key=lambda k:-ch[k])
plt.figure(figsize=(9,4)); plt.bar(ks,[ch[k] for k in ks],color='#3b6ea5')
plt.ylabel('s5 OligoMPRA tiles'); plt.title('s5 OligoMPRA HEK tiles per chromosome (n=7537; all + strand, 270bp)')
plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(F/'fig1_tiles_by_chrom.png',dpi=110); plt.close()
# Fig2: group size distribution (log)
sizes=collections.Counter(); 
for k,v in g['size_dist'].items(): sizes[int(k)]=v
xs=sorted(sizes)
plt.figure(figsize=(8,4)); plt.bar(xs,[sizes[x] for x in xs],color='#7a4fa3',width=3)
plt.xlabel('connected-component size (overlapping 270bp windows)'); plt.ylabel('n components')
plt.title('Label-free interval-overlap groups, %d components / 7537 tiles'%g['n_groups'])
plt.tight_layout(); plt.savefig(F/'fig2_group_sizes.png',dpi=110); plt.close()
print('figures written')
