# -*- coding: utf-8 -*-
"""b45 REVIEWER-owned independent re-derivation. ZERO fit/GPU/web. Own code, no b45 imports."""
import gzip, hashlib, math, re, collections, json, random, statistics, zlib
ROOT=r"E:/promoter_atlas"
SRC=ROOT+"/data/breakthrough_20261002/batch44/syn.txt.gz"
OUT={}
raw=open(SRC,"rb").read()
d=zlib.decompressobj(16+zlib.MAX_WBITS); dec=d.decompress(raw)
OUT["comp_bytes"]=len(raw); OUT["dec_bytes"]=len(dec)
OUT["sha_comp"]=hashlib.sha256(raw).hexdigest()
OUT["sha_dec"]=hashlib.sha256(dec).hexdigest()
OUT["gzip_eof"]=d.eof; OUT["gzip_unused"]=len(d.unused_data)
rows=[]
with gzip.open(SRC,"rt",encoding="utf-8",errors="replace") as f:
    hdr=f.readline().rstrip("\n").split("\t")
    for line in f:
        rows.append(line.rstrip("\n").split("\t"))
OUT["n_cols"]=len(hdr); OUT["header"]=hdr; OUT["n_rows"]=len(rows)
iq={}; q_all=collections.defaultdict(set); bc=set(); bi=collections.Counter()
offc=collections.Counter(); qlen=collections.Counter(); olen=collections.Counter()
libc=collections.Counter(); libq=collections.defaultdict(set)
predK=collections.defaultdict(set); predH=collections.defaultdict(set)
for r in rows:
    q=r[2]; q_all[r[1]].add(q); iq.setdefault(q,set()).add(r[1])
    bc.add(r[3]); bi[(r[5],q)]+=1
    qlen[len(q)]+=1; olen[len(r[0])]+=1; libc[r[5]]+=1; libq[r[5]].add(q)
    if q in r[0]: offc[r[0].index(q)]+=1
    predK[q].add(r[18]); predH[q].add(r[17])
OUT["n_unique_query"]=len(iq); OUT["n_unique_info"]=len(q_all); OUT["n_unique_barcode"]=len(bc)
OUT["barcodes_per_fragment_dist"]=dict(collections.Counter(bi.values()))
OUT["query_info_is_1to1"]=(len(iq)==len(q_all)==889) and all(len(v)==1 for v in iq.values()) and all(len(v)==1 for v in q_all.values())
OUT["query_length_dist"]={str(k):v for k,v in sorted(qlen.items())}
OUT["full_oligo_length_dist"]={str(k):v for k,v in sorted(olen.items())}
OUT["substring_offsets"]=dict(offc)
OUT["library_row_counts"]=dict(libc)
OUT["library_unique_query"]={k:len(v) for k,v in libq.items()}
OUT["K562_pred_conflicts"]=sum(1 for v in predK.values() if len(v)>1)
OUT["HepG2_pred_conflicts"]=sum(1 for v in predH.values() if len(v)>1)
print("chunk1 ok")
rng=random.Random(20261004)
pick=rng.sample(range(len(rows)),20)
checks=[]
for i in sorted(pick):
    r=rows[i]
    off=r[0].index(r[2]) if r[2] in r[0] else -1
    f5=r[0][:off] if off>=0 else None; f3=r[0][off+len(r[2]):] if off>=0 else None
    checks.append({"row":i,"info":r[1],"library":r[5],"bc":r[3],"qlen":len(r[2]),"oligo_len":len(r[0]),
                   "offset":off,"5p_len":len(f5) if f5 is not None else None,"3p_len":len(f3) if f3 is not None else None,
                   "K562_norm_r1":r[11],"HepG2_norm_r1":r[13]})
OUT["barcode_seq_20"]=checks
OUT["all_offset18"]=(sum(offc.values())==len(rows) and list(offc.keys())==[18])
per=collections.defaultdict(lambda:{"K1":[],"K2":[],"H1":[],"H2":[],"avgK":[],"avgH":[],"info":None,"lib":None})
for r in rows:
    q=r[2]; p=per[q]; p["info"]=r[1]; p["lib"]="long" if "232bp" in r[5] else "short"
    for idx,k in [(11,"K1"),(12,"K2"),(13,"H1"),(14,"H2"),(15,"avgK"),(16,"avgH")]:
        try: v=float(r[idx])
        except: v=None
        if v is not None and math.isfinite(v): p[k].append(v)
def m(v):
    f=[x for x in v if x is not None and math.isfinite(x)]
    return sum(f)/len(f) if f else None
frag={}
for q,p in per.items():
    frag[q]={"q":q,"info":p["info"],"lib":p["lib"],
             "K562":m([m(p["K1"]),m(p["K2"])]),"HepG2":m([m(p["H1"]),m(p["H2"])]),
             "K_avgcol":m(p["avgK"]),"H_avgcol":m(p["avgH"]),
             "GC":(sum(1 for c in q.upper() if c in "GC")/len(q)) if q else None,
             "predK":None,"predH":None}
for r in rows:
    try: frag[r[2]]["predK"]=float(r[18])
    except: pass
    try: frag[r[2]]["predH"]=float(r[17])
    except: pass
dmax=0.0
for r in rows:
    try: dmax=max(dmax, abs(float(r[15])-(float(r[11])+float(r[12]))/2), abs(float(r[16])-(float(r[13])+float(r[14]))/2))
    except: pass
OUT["author_avg_eq_mean_maxabsdiff"]=dmax
dK=dh=0.0
for q,p in frag.items():
    if p["K_avgcol"] is not None and p["K562"] is not None: dK=max(dK,abs(p["K_avgcol"]-p["K562"]))
    if p["H_avgcol"] is not None and p["HepG2"] is not None: dh=max(dh,abs(p["H_avgcol"]-p["HepG2"]))
OUT["fixedsummary_vs_avgcol_K"]=dK; OUT["fixedsummary_vs_avgcol_H"]=dh
OUT["n_fragments"]=len(frag)
print("chunk2 ok", len(frag))
def spearman(x,y):
    pr=[(a,b) for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    n=len(pr)
    if n<3: return (None,n)
    xs=[a for a,_ in pr]; ys=[b for _,b in pr]
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[o[j+1]]==v[o[i]]: j+=1
            avg=(i+j)/2.0+1.0
            for k in range(i,j+1): r[o[k]]=avg
            i=j+1
        return r
    rx=rk(xs); ry=rk(ys); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx=math.sqrt(sum((a-mx)**2 for a in rx)); dy=math.sqrt(sum((b-my)**2 for b in ry))
    return (None,n) if dx==0 or dy==0 else (num/(dx*dy),n)
SYNTH={"design","best","top_individual","MUTATED"}
def cat(info):
    if info.startswith("control"): return "control"
    if "MUTATED" in info: return "MUTATED"
    if "top_individual" in info: return "top_individual"
    if "_best" in info: return "best"
    return "design"
sec={}
for L in ["short","long"]:
    for cell,pk,mk in [("K562","predK","K562"),("HepG2","predH","HepG2")]:
        for sg in ["synthetic_designed","natural_control","ALL"]:
            it=[p for p in frag.values() if p["lib"]==L and (sg=="ALL" or (("synthetic_designed" if cat(p["info"]) in SYNTH else "natural_control")==sg))]
            rho,n=spearman([p[pk] for p in it],[p[mk] for p in it])
            rhog,ng=spearman([p["GC"] for p in it],[p[mk] for p in it])
            blk={"n":len(it),"rho":rho,"rho_n":n,"gc_rho":rhog}
            ok=[i for i,p in enumerate(it) if p[pk] is not None and p[mk] is not None and math.isfinite(p[pk]) and math.isfinite(p[mk])]
            if len(ok)>=30:
                ids=[it[i]["info"] for i in ok]; pr=[it[i][pk] for i in ok]; ms=[it[i][mk] for i in ok]
                op=sorted(range(len(ok)),key=lambda i:(-pr[i],ids[i])); om=sorted(range(len(ok)),key=lambda i:(-ms[i],ids[i]))
                k=math.ceil(0.1*len(ok)); tp=set(op[:k]); tm=set(om[:k]); inter=len(tp&tm)
                blk["top10"]={"k":k,"overlap":inter,"precision":inter/k,"enrichment":(inter/k)/(k/len(ok))}
            sec[L+":"+cell+":"+sg]=blk
OUT["secondary"]=sec
rc={}
for L in ["short","long"]:
    for cell,a,b in [("K562","K1","K2"),("HepG2","H1","H2")]:
        it=[p for p in per.values() if (("long" if "232bp" in p["lib"] else "short")==L)]
        rho,n=spearman([m(p[a]) for p in it],[m(p[b]) for p in it]); rc[L+":"+cell]={"rho":rho,"n":n}
OUT["replicate_concordance"]=rc
print("chunk3 ok")
pat=re.compile(r"^K562_fitness_(?P<L>232bp_)?replicate(?P<rep>\d+)_generation(?P<g>\d+)_individual(?P<i>\d+)_(?P<sel>max|median|min|top_individual|best|best_MUTATED)$")
byinfo={p["info"]:p for p in frag.values()}
grid=collections.defaultdict(dict)
for info,p in byinfo.items():
    mm=pat.match(info)
    if mm: grid[(p["lib"],mm["rep"],mm["sel"])][int(mm["g"])]=info
prim={}
for L in ["short","long"]:
    reps=sorted({k[1] for k in grid if k[0]==L and k[2]=="max"})
    common=None
    for rep in reps:
        gs=set(grid.get((L,rep,"max"),{}).keys()); common=gs if common is None else common&gs
    G=max(common); dl=[]; dm=[]; dhp=[]; dhm=[]
    for rep in reps:
        p0=byinfo[grid[(L,rep,"max")][0]]; pG=byinfo[grid[(L,rep,"max")][G]]
        dl.append(pG["predK"]-p0["predK"]); dm.append(pG["K562"]-p0["K562"])
        dhp.append(pG["predH"]-p0["predH"]); dhm.append(pG["HepG2"]-p0["HepG2"])
    prim[L]={"G":G,"n_reps":len(reps),"common_gens":sorted(common),
             "dPred_med":statistics.median(dl),"dPred_min":min(dl),"dPred_max":max(dl),
             "dMeas_med":statistics.median(dm),"dMeas_min":min(dm),"dMeas_max":max(dm),
             "dMeas_pos":sum(1 for x in dm if x>0),"dMeas_neg":sum(1 for x in dm if x<0),
             "dHepPred_med":statistics.median(dhp),"dHepMeas_med":statistics.median(dhm)}
OUT["primary"]=prim
json.dump(OUT,open(ROOT+"/runs/breakthrough_20261002/code/batch45_review/rev_out.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("DONE")
print(json.dumps({k:OUT[k] for k in ["sha_comp","sha_dec","n_rows","n_unique_query","n_unique_info","n_unique_barcode","query_info_is_1to1","all_offset18","K562_pred_conflicts","HepG2_pred_conflicts","n_fragments"]},indent=1))
print("PRIMARY", json.dumps(prim,indent=1))
