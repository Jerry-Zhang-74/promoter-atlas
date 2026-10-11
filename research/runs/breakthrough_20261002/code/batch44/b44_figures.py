# -*- coding: utf-8 -*-
import json,os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG=os.path.join('reports','breakthrough_20261002','batch44','figures'); os.makedirs(FIG,exist_ok=True)
st=json.load(open('data/breakthrough_20261002/batch44/v1_fold0_structural.json',encoding='utf-8'))
# fig1: fragment length distribution (obtained shard)
import gzip
L=[]
with gzip.open('data/breakthrough_20261002/batch44/v1_pHY3_fold0.bed.gz','rt',encoding='utf-8') as f:
    hdr=f.readline().rstrip('\n').split('\t'); ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        c=line.split('\t'); L.append(int(c[ix['end']])-int(c[ix['start']]))
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
ax[0].hist(L,bins=range(80,610,10),color="#4c72b0")
ax[0].set_title("GSE301246 v1_pHY3 fold0 (HAP1/AGS): measured insert length")
ax[0].set_xlabel("insert length (bp)"); ax[0].set_ylabel("fragments")
ax[0].axvline(100,color='r',ls='--',lw=1); ax[0].axvline(600,color='r',ls='--',lw=1)
ax[0].text(0.02,0.95,"n=%d  min=%d med=%d max=%d"%(st['total_fragment_rows'],st['length_min'],st['length_median'],st['length_max']),
           transform=ax[0].transAxes,fontsize=8,va='top')
h=st['group_size_hist(<=6)']; ks=sorted(int(k) for k in h)
ax[1].bar(ks,[h[str(k)] for k in ks],color="#dd8452")
ax[1].set_title("nested same-3'end groups by #distinct lengths[100,600]")
ax[1].set_xlabel("distinct lengths in group (>=2)"); ax[1].set_ylabel("groups")
for i,k in enumerate(ks): ax[1].text(k,h[str(k)],"%.0f"%h[str(k)],ha='center',va='bottom',fontsize=8)
fig.suptitle("batch44 — PARM focused-library shard STRUCTURE (HAP1/AGS labels; PROTOCOL-EXCLUDED as HEK; not a dataset)",fontsize=8.5)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(os.path.join(FIG,"fig1_fragment_structure.png"),dpi=130); plt.close(fig)
print("wrote fig1")
