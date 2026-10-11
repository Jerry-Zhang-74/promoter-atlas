# -*- coding: utf-8 -*-
"""BATCH27 A2: contamination-aware sensitivity for the batch26 STR split. ZERO estimator fits."""
from __future__ import annotations
import csv, json, os, time, hashlib, collections, difflib
import numpy as np
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "4"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
D = os.path.join(ROOT, "data", "breakthrough_20261002", "batch26")
R = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch27")
os.makedirs(R, exist_ok=True)
MODELS = ["A0_ridgeC", "A1_cnn_true_seed20261026", "A2_cnn_true_seed20261027", "A3_cnn_shuffle_seed20261026"]
CODES = {"A":0,"C":1,"G":2,"T":3,"N":4}
def sha_b(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def _rank(a):
    a=np.asarray(a,float); order=np.argsort(a,kind="mergesort"); r=np.empty(len(a)); sa=a[order]; i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sa[j+1]==sa[i]: j+=1
        r[order[i:j+1]]=0.5*(i+j)+1.0; i=j+1
    return r
def spearman(a,b):
    if len(a)<2: return float("nan")
    ra,rb=_rank(a),_rank(b); ra=ra-ra.mean(); rb=rb-rb.mean()
    d=np.sqrt((ra*ra).sum()*(rb*rb).sum()); return float((ra*rb).sum()/d) if d>0 else float("nan")
def mae(a,b): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(b,float))))
def r2(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); ss=((a-b)**2).sum(); tot=((a-a.mean())**2).sum()
    return float(1.0-ss/tot) if tot>0 else float("nan")
def gbmae(y,p,loci):
    vals=[np.mean(np.abs(y[loci==g]-p[loci==g])) for g in np.unique(loci)]
    return (float(np.mean(vals)), int(len(vals))) if vals else (float("nan"),0)
def topdec(y,p,vhash,frac=0.1):
    n=len(y); k=int(np.ceil(frac*n))
    o=np.lexsort((vhash,-np.asarray(p,float))); ot=np.lexsort((vhash,-np.asarray(y,float)))
    inter=len(set(o[:k].tolist())&set(ot[:k].tolist()))
    return {"k":k,"n":n,"intersection":inter,"precision":inter/k,"baseline_precision":k/n,"enrichment":(inter/k)/(k/n) if k else float("nan")}
def metrics(y,p,loci,vhash):
    gb,ng=gbmae(y,p,loci)
    return {"spearman":spearman(y,p),"mae":mae(y,p),"r2":r2(y,p),"group_balanced_mae":gb,"n_groups":ng,
            "pred_std":float(np.std(p)),"pred_nunique":int(len(np.unique(p))),"top_decile":topdec(y,p,vhash)}
def boot_delta(preds_abc,y,loci,n=1000,seed=20261031):
    rng=np.random.RandomState(seed); ug=np.unique(loci)
    idx={g:np.where(loci==g)[0] for g in ug}
    ra=[];rb=[];rc=[];d1=[];d2=[]
    A,B,C=preds_abc
    for _ in range(n):
        pick=rng.randint(0,len(ug),len(ug))
        rows=np.concatenate([idx[ug[i]] for i in pick])
        r0=spearman(A[rows],y[rows]); r1=spearman(B[rows],y[rows]); r2c=spearman(C[rows],y[rows])
        if np.isfinite(r0): ra.append(r0)
        if np.isfinite(r1): rb.append(r1)
        if np.isfinite(r2c): rc.append(r2c)
        if np.isfinite(r1) and np.isfinite(r0): d1.append(r1-r0)
        if np.isfinite(r2c) and np.isfinite(r0): d2.append(r2c-r0)
    def ci(v): return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] if v else None
    return {"n_resamples":n,"rho_ci95":{"A0_ridgeC":ci(ra),"A1_cnn_true_seed20261026":ci(rb),"A2_cnn_true_seed20261027":ci(rc)},
            "delta_rho_mean":{"A1_vs_A0":float(np.mean(d1)) if d1 else float("nan"),"A2_vs_A0":float(np.mean(d2)) if d2 else float("nan")},
            "delta_rho_ci95":{"A1_cnn_true_seed20261026_vs_A0":ci(d1),"A2_cnn_true_seed20261027_vs_A0":ci(d2)}}
t0=time.time()
rows=list(csv.DictReader(open(os.path.join(D,"str_variant_rows.tsv"),encoding="utf-8"),delimiter="\t"))
man=list(csv.DictReader(open(os.path.join(D,"split_manifest.tsv"),encoding="utf-8"),delimiter="\t"))
preds=list(csv.DictReader(open(os.path.join(D,"test_predictions.tsv"),encoding="utf-8"),delimiter="\t"))
g2split={m["group_id"]:int(m["split"]) for m in man}
by_group=collections.defaultdict(set)
for r in rows: by_group[r["group_id"]].add(r["vreg_seq"])
reps={g:min(ss,key=sha_b) for g,ss in by_group.items()}
glist=sorted(reps)
print("[load] rows=%d groups=%d preds=%d"%(len(rows),len(glist),len(preds)),flush=True)
bylen=collections.defaultdict(list)
for g in glist: bylen[len(reps[g])].append(g)
edges=[]
for L,gs in sorted(bylen.items()):
    if len(gs)<2: continue
    M=np.array([[CODES.get(c,4) for c in reps[g]] for g in gs],dtype=np.int8)
    for i in range(len(gs)):
        d=(M!=M[i]).sum(axis=1)
        for j in range(i+1,len(gs)):
            if int(d[j])<=16: edges.append((gs[i],gs[j],"same_length_hamming_le16",int(d[j])))
print("[ham16] edges=%d %.1fs"%(len(edges),time.time()-t0),flush=True)
K=20
kmers={g:set(reps[g][i:i+K] for i in range(len(reps[g])-K+1)) for g in glist}
idx=collections.defaultdict(list)
for g in glist:
    for km in kmers[g]: idx[km].append(g)
pairs=set()
for km in idx:
    gs=idx[km]
    if len(gs)<2 or len(gs)>8: continue
    u=sorted(gs)
    for i in range(len(u)):
        for j in range(i+1,len(u)): pairs.add((u[i],u[j]))
align_edges=0
for a,b in sorted(pairs):
    sa,sb=reps[a],reps[b]; ja=len(kmers[a]&kmers[b]); jb=len(kmers[a]|kmers[b])
    if jb and ja/jb>=0.7:
        sm=difflib.SequenceMatcher(None,sa,sb,autojunk=False)
        matched=sum(bl.size for bl in sm.get_matching_blocks())
        if matched/min(len(sa),len(sb))>=0.9 and matched/max(len(sa),len(sb))>=0.9:
            edges.append((a,b,"variable_length_alignment_cov90_ident90",None)); align_edges+=1
print("[varlen] candidates=%d align_edges=%d %.1fs"%(len(pairs),align_edges,time.time()-t0),flush=True)
cross=[e for e in edges if g2split[e[0]]!=g2split[e[1]]]
contam=set()
for g1,g2,kind,m in cross:
    if g2split[g1]==2: contam.add(g1)
    if g2split[g2]==2: contam.add(g2)
print("[contam] cross_edges=%d excluded_test_groups=%d"%(len(cross),len(contam)),flush=True)
pred_full={m:np.array([float(p[m]) for p in preds]) for m in MODELS}
y=np.array([float(p["true"]) for p in preds])
gl=np.array([p["group_id"] for p in preds]); loci=np.array([p["locus_id"] for p in preds])
vhash=np.array([int(hashlib.sha256(p["variant"].encode()).hexdigest()[:16],16) for p in preds])
keep=np.array([g not in contam for g in gl])
res={"batch":"batch27","purpose":"contamination-aware sensitivity of batch26 STR locus-holdout (ZERO fits)",
 "n_test_rows":int(len(preds)),"n_test_groups":int(len(np.unique(gl))),"n_test_loci":int(len(np.unique(loci))),
 "n_same_length_h16_edges_all":int(sum(1 for e in edges if e[2]=="same_length_hamming_le16")),
 "n_varlen_align_edges_all":int(align_edges),"n_cross_split_edges":int(len(cross)),
 "cross_split_edges":[{"test_group":(e[0] if g2split[e[0]]==2 else e[1]),"other_group":(e[1] if g2split[e[0]]==2 else e[0]),
   "other_split":int(g2split[e[1]] if g2split[e[0]]==2 else g2split[e[0]]),"kind":e[2],"metric":e[3]} for e in cross],
 "excluded_test_groups":sorted(contam),"n_excluded_test_groups":len(contam),
 "n_retained_test_groups":int(len(np.unique(gl[keep]))),"n_retained_test_rows":int(keep.sum()),
 "full_test_metrics":{m:metrics(y,pred_full[m],loci,vhash) for m in MODELS},
 "filtered_test_metrics":{m:metrics(y[keep],pred_full[m][keep],loci[keep],vhash[keep]) for m in MODELS},
 "definition":"LABEL-FREE edge list built from sequence/split only BEFORE any activity score; excluded = whole test groups with >=1 cross-split near-homology edge to a train/val representative.",
 "note":"post-hoc sensitivity on KNOWN contaminated groups only; NOT strict family isolation (non-representative homology unresolved) and NOT a new independent test. Original scores preserved."}
print("[metrics] full+filtered computed; bootstrap 1000 on retained groups...",flush=True)
res["filtered_group_bootstrap_1000"]=boot_delta([pred_full["A0_ridgeC"][keep],pred_full["A1_cnn_true_seed20261026"][keep],pred_full["A2_cnn_true_seed20261027"][keep]],y[keep],loci[keep],n=1000)
res["elapsed_s"]=round(time.time()-t0,1)
json.dump(res,open(os.path.join(R,"homology_sensitivity.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("[done] %.1fs"%res["elapsed_s"],flush=True)
