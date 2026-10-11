# -*- coding: utf-8 -*-
"""BATCH27 B: stream one pass over the SuRE K562 counts file. No model applied."""
import gzip, json, time, collections, hashlib, heapq, os
P="data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
OUT="reports/breakthrough_20261002/batch27/sure_pass1.json"
CANON=set(["chr%d"%i for i in range(1,23)]+["chrX","chrY"])
t0=time.time()
N=0; parse_fail=0
chrom_ct=collections.Counter(); strand_ct=collections.Counter()
lenhist=collections.Counter()   # 100bp bins
n_startok=0; n_canon=0; n_lenrange=0; n_input10=0; n_rna_any=0
elig_keys=set()
heap=[]  # (hash_int, stable)
elig=0
sel_cap=25000
with gzip.open(P,"rt",encoding="utf-8") as f:
    hdr=f.readline().rstrip("\n").split("\t")
    for line in f:
        N+=1
        c=line.rstrip("\n").split("\t")
        if len(c)!=13: parse_fail+=1; continue
        chrom,start,end,strand=c[0],c[1],c[2],c[3]
        chrom_ct[chrom]+=1; strand_ct[strand]+=1
        try:
            st=int(start); en=int(end)
        except Exception:
            parse_fail+=1; continue
        L=en-st
        if 0<L<4000: lenhist[min(L//100*100,3900)]+=1
        if not (st<en): continue
        n_startok+=1
        if chrom not in CANON: continue
        n_canon+=1
        if not (200<=L<=2000): continue
        n_lenrange+=1
        try:
            iPCR=int(c[4]); r1=sum(int(c[k]) for k in (5,6,7,8)); r2=sum(int(c[k]) for k in (9,10))
            pl=sum(int(c[k]) for k in (11,12))
        except Exception:
            continue
        if min(iPCR,r1,r2,pl)<0: continue
        inp=iPCR+pl+1  # pseudo count 1 per paper
        if inp<10: continue
        n_input10+=1
        rna=r1+r2
        if rna>=1: n_rna_any+=1
        # eligible
        elig+=1
        key="%s:%d-%d:%s"%(chrom,st,en,strand)
        h=int(hashlib.sha256(key.encode()).hexdigest(),16)
        if len(heap)<sel_cap:
            heapq.heappush(heap,(-h,key))
        elif h< -heap[0][0]:
            heapq.heapreplace(heap,(-h,key))
        if N%20000000==0: print("[pass] N=%d elig=%d %.0fs"%(N,elig,time.time()-t0),flush=True)
json.dump({"file":"GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz","n_rows":N,"parse_fail":parse_fail,
 "n_strand":dict(strand_ct),"n_chrom":dict(chrom_ct),
 "n_start_lt_end":n_startok,"n_canonical_chrom":n_canon,"n_len_200_2000":n_lenrange,
 "n_input_ge10":n_input10,"n_rna_ge1":n_rna_any,"n_eligible":elig,
 "len_hist_100bp":{str(k):v for k,v in sorted(lenhist.items())},
 "elapsed_s":round(time.time()-t0,1)},open(OUT,"w",encoding="utf-8"),indent=2)
print("[pass1] N=%d elig=%d len(heap)=%d %.1fs"%(N,elig,len(heap),time.time()-t0),flush=True)
# save selected stable ids (ascending hash) for retrieval/grouping
sel=sorted([(-h,k) for h,k in heap])
json.dump({"selection_rule":"25000 smallest sha256('chrom:start-end:strand') among eligible rows (label-free)",
 "n_selected":len(sel),"selected":[k for _,k in sel]},
 open("reports/breakthrough_20261002/batch27/sure_selected_ids.json","w",encoding="utf-8"),indent=1)
print("[pass1] selected saved %d"%len(sel),flush=True)
