#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH32 Section A: does the frozen PARM HEK673 association survive exclusion of
public-library loci that are DEMONSTRABLY coordinate-overlapping?
ZERO fits / ZERO inference / ZERO GPU. Reads only existing frozen assets + the official
PARM_preprocessing_pipeline example promoter_library coordinate files. Label-free overlap
flags are frozen BEFORE any score is joined."""
import csv, gzip, json, math, os, hashlib, bisect
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
B32  = os.path.join(REPO, "data/breakthrough_20261002/batch32")
LIB32 = os.path.join(B32, "inputs/promoter_library")
LIB31 = os.path.join(REPO, "data/breakthrough_20261002/batch31/_external/preproc")
SEQ   = os.path.join(REPO, "data/breakthrough_20261002/batch22/sequence_table.tsv")
RANK  = os.path.join(REPO, "reports/breakthrough_20261002/batch31/tables/hek673_ranking.tsv")
OUT   = os.path.join(REPO, "reports/breakthrough_20261002/batch32")
TAB   = os.path.join(OUT, "tables")
SEED  = 20261032
OFFICIAL_SIZE = {"chr1":1087089,"chr2":745867,"chr3":621356,"chr4":426383,"chr5":516526,
 "chr6":552616,"chr7":549748,"chr8":426890,"chr9":426029,"chr10":406556,"chr11":652586,
 "chr12":585562,"chr13":186860,"chr14":386419,"chr15":381502,"chr16":477456,"chr17":691599,
 "chr18":179799,"chr19":747363,"chr20":291554,"chr21":130061,"chr22":263404,"chrX":378049}

def spearman(x,y):
    if len(x)<3: return None
    r=stats.spearmanr(x,y)[0]
    return None if (r is None or math.isnan(r)) else float(r)
def read_tsv(p):
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f,delimiter="\t"))

def pick_lib_file(chrom):
    cands=[os.path.join(d,f"{chrom}.txt.gz") for d in (LIB32,LIB31)]
    cands=[p for p in cands if os.path.exists(p)]
    for p in cands:
        try:
            sz=os.path.getsize(p)
            with gzip.open(p,"rt",encoding="utf-8") as f: n=sum(1 for _ in f)
            if n>0 and sz>=0.98*OFFICIAL_SIZE.get(chrom,0): return p,sz,n,"complete"
            return p,sz,n,"partial"
        except Exception:
            last=(p,os.path.getsize(p),0,"unreadable")
    return (None,0,0,"absent")

def load_library():
    lib={}; meta={}
    for chrom in OFFICIAL_SIZE:
        p,sz,n,status=pick_lib_file(chrom)
        if p is None or status!="complete":
            meta[chrom]={"path":p,"size":sz,"n_rows":n,"status":(status)}
            continue
        starts=[]; ends=[]; strands=[]
        with gzip.open(p,"rt",encoding="utf-8") as f:
            rd=csv.reader(f,delimiter="\t"); header=next(rd)
            for row in rd:
                if not row: continue
                starts.append(int(row[1])); ends.append(int(row[2])); strands.append(row[3])
        starts=np.array(starts); ends=np.array(ends); strands=np.array(strands)
        o=np.argsort(starts,kind="mergesort")
        lib[chrom]={"starts":starts[o],"ends":ends[o],"strands":strands[o]}
        meta[chrom]={"path":os.path.relpath(p,REPO).replace(os.sep,"/"),"size":sz,"n_rows":len(starts),
                     "status":"complete","header":header}
    return lib,meta

def query_overlap(lib,chrom,a0,a1,strand):
    e=lib.get(chrom)
    if e is None: return None,None,None
    starts=e["starts"]; ends=e["ends"]; strands=e["strands"]
    ml=int(np.max(ends-starts)) if len(starts) else 0
    lo=bisect.bisect_left(starts,a0-ml); hi=bisect.bisect_left(starts,a1)
    n_any=0; n_ss=0; exact=0
    for i in range(lo,hi):
        if starts[i]<a1 and ends[i]>a0:
            n_any+=1
            if strands[i]==strand: n_ss+=1
            if starts[i]==a0 and ends[i]==a1: exact+=1
    return n_any,n_ss,exact

def main():
    os.makedirs(TAB,exist_ok=True)
    lib,libmeta=load_library()
    avail=sorted(lib.keys())
    seq=read_tsv(SEQ)
    flags=[]; counts={"OVERLAP_any":0,"OVERLAP_same_strand":0,"DISJOINT":0,
                      "UNKNOWN_chrom_not_distributed":0,"UNKNOWN_no_coord":0}
    for r in seq:
        pid=r["prom_id"]; chrom=(r.get("chr") or "").strip()
        strand=(r.get("strand") or "").strip()
        okc = chrom in libmeta and libmeta[chrom]["status"]=="complete" and (r.get("start1") or "").isdigit() and (r.get("end") or "").isdigit()
        rec={"prom_id":pid,"chrom":chrom,"start0":(int(r["start1"])-1 if (r.get("start1") or "").isdigit() else ""),
             "end":(int(r["end"]) if (r.get("end") or "").isdigit() else ""),"strand":strand}
        if not okc:
            rec.update({"chrom_available":0,"n_overlap_any":"","n_overlap_ss":"","exact":"",
                        "overlap_any":"","overlap_ss":"","exact_match":"","subset":"UNKNOWN_chrom_not_distributed"})
            counts["UNKNOWN_chrom_not_distributed"]+=1
        else:
            a0=int(r["start1"])-1; a1=int(r["end"])
            na,nss,ex=query_overlap(lib,chrom,a0,a1,strand)
            if na is None:
                rec.update({"chrom_available":0,"n_overlap_any":"","n_overlap_ss":"","exact":"",
                            "overlap_any":"","overlap_ss":"","exact_match":"","subset":"UNKNOWN_no_coord"})
                counts["UNKNOWN_no_coord"]+=1
            else:
                sub="OVERLAP_any" if na>0 else "DISJOINT"
                counts[sub]+=1
                if nss>0: counts["OVERLAP_same_strand"]+=1
                rec.update({"chrom_available":1,"n_overlap_any":na,"n_overlap_ss":nss,"exact":ex,
                            "overlap_any":int(na>0),"overlap_ss":int(nss>0),
                            "exact_match":int(ex>0),"subset":sub})
        flags.append(rec)
    with open(os.path.join(TAB,"hek673_overlap_flags.tsv"),"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(flags[0].keys()),delimiter="\t"); w.writeheader(); w.writerows(flags)
    with open(os.path.join(OUT,"overlap_group_counts.json"),"w",encoding="utf-8") as f:
        json.dump({"frozen_before_scores":True,"n_rows":len(flags),"subset_row_counts":counts,
                   "library_chrom_status":libmeta,"chroms_available":avail,
                   "chroms_not_distributed":[c for c in OFFICIAL_SIZE if c not in lib]},f,ensure_ascii=False,indent=2)
    flagmap={r["prom_id"]:r for r in flags}
    rank=read_tsv(RANK)
    ids=[r["prom_id"] for r in rank]
    missing=[i for i in ids if i not in flagmap]
    y=np.array([float(r["measured_hek_norm"]) for r in rank])
    groups=[r["group_id"] for r in rank]
    preds={"PARM":np.array([float(r["PARM_HEK293"]) for r in rank]),
           "R1":np.array([float(r["R1_oof"]) for r in rank]),
           "H1":np.array([float(r["H1_oof"]) for r in rank])}
    def topk(vals,idlist,k):
        order=sorted(range(len(idlist)),key=lambda i:(-vals[i],hashlib.sha256(idlist[i].encode()).hexdigest()))
        return set(idlist[i] for i in order[:k])
    k=int(math.ceil(0.1*len(ids)))
    obs_top=topk(list(y),ids,k)
    full={"n":len(ids),"missing_flag":len(missing),"top_decile_k":k,
          "rho":{m:spearman(list(preds[m]),list(y)) for m in preds},
          "top_decile":{m:{"precision":len(topk(list(preds[m]),ids,k)&obs_top)/k} for m in preds}}
    sub_idx={s:[i for i,pid in enumerate(ids) if flagmap[pid]["subset"]==s] for s in
             ("OVERLAP_any","DISJOINT","UNKNOWN_chrom_not_distributed")}
    ss_disj=[i for i,pid in enumerate(ids)
             if flagmap[pid].get("subset")=="OVERLAP_any" and flagmap[pid].get("overlap_ss")==0]

    def subgroup_metrics(idx):
        if not idx: return {"n":0,"n_groups":0}
        yy=[y[i] for i in idx]; gg=[groups[i] for i in idx]
        res={"n":len(idx),"n_groups":len(set(gg)),"rho":{m:spearman([preds[m][i] for i in idx],yy) for m in preds}}
        if len(idx)>=10:
            kk=int(math.ceil(0.1*len(idx)))
            ot=topk([y[i] for i in idx],[ids[i] for i in idx],kk)
            res["top_decile_k"]=kk
            res["top_decile"]={m:{"precision":len(topk([preds[m][i] for i in idx],[ids[i] for i in idx],kk)&ot)/kk} for m in preds}
        return res
    def boot(idx,draws=2000):
        if not idx: return None
        gg=[groups[i] for i in idx]; ug=sorted(set(gg))
        if len(ug)<100: return {"note":"<100 groups: descriptive only, no bootstrap/CI","n_groups":len(ug),"n":len(idx)}
        idx_by={g:[i for i in idx if groups[i]==g] for g in ug}
        rng=np.random.default_rng(SEED); dR1=[]; dH1=[]; rP=[]; rR=[]; rH=[]
        for _ in range(draws):
            pick=rng.integers(0,len(ug),len(ug))
            sel=np.concatenate([idx_by[ug[j]] for j in pick])
            yy=[y[i] for i in sel]
            rp=spearman([preds["PARM"][i] for i in sel],yy)
            r1=spearman([preds["R1"][i] for i in sel],yy)
            h1=spearman([preds["H1"][i] for i in sel],yy)
            if rp is not None: rP.append(rp)
            if r1 is not None: rR.append(r1)
            if h1 is not None: rH.append(h1)
            if rp is not None and r1 is not None: dR1.append(rp-r1)
            if rp is not None and h1 is not None: dH1.append(rp-h1)
        def ci(a):
            if not a: return None
            a=np.array(a); return {"mean":float(a.mean()),"lo":float(np.percentile(a,2.5)),"hi":float(np.percentile(a,97.5)),"n":int(a.size)}
        return {"draws":draws,"n_groups":len(ug),"n":len(idx),"rho_PARM":ci(rP),"rho_R1":ci(rR),"rho_H1":ci(rH),
                "PARM_minus_R1":ci(dR1),"PARM_minus_H1":ci(dH1),
                "note":"paired group resample with multiplicities (group=669 frozen groups); rows carried with their group"}
    results={}
    for s in ("OVERLAP_any","DISJOINT","UNKNOWN_chrom_not_distributed"):
        res=subgroup_metrics(sub_idx[s]); res["bootstrap"]=boot(sub_idx[s]); results[s]=res
    res=subgroup_metrics(ss_disj); res["bootstrap"]=boot(ss_disj); results["SSDISJOINT_within_overlap_any"]=res
    out={"section":"A",
        "assembly":{"HEK673":"hg19 (prior source Haberle2019 table6 / FANTOM5 CAGE)",
            "library":"hg19 (official example_config.yaml GENOME=.../hg19_ch1-22_XYM.fa; pipeline reads genome[chrom][start:end] via pysam -> 0-based half-open)",
            "comparison_valid":True},
        "convention":{"HEK673":"0-based half-open [start0,end)=[start1-1,end), strand as stored",
            "library":"0-based half-open [start,end), strand as stored","inclusivity":"1bp"},
        "library_source":{"repo":"vansteensellab/PARM_preprocessing_pipeline",
            "path":"example_data/promoter_library/chr*.txt.gz",
            "columns":"chr,start,end,strand,iPCR,pDNA_pHY3_T2,HAP1_B1,HAP1_B2,AGS_B1,AGS_B2 (HAP1/AGS only)"},
        "full673_checksum":full,"frozen_counts":counts,"subgroups":results,
        "caveats":["Distributed library is an OFFICIAL SUBSET with HAP1/AGS columns only; HEK293 training membership is UNKNOWN (HAP1/AGS columns do not establish it).",
            "DISJOINT here = no overlap with the distributed subset on a distributed chromosome; NOT demonstrated disjoint from the full training library, NOT proof of unseen families / no training exposure.",
            "Gene-level overlap not determinable (library files carry coordinates only, no gene identifiers).",
            "Residual near-homology / family membership remains UNKNOWN.",
            "Interval overlap = coordinate overlap, not confirmed oligo/sequence identity (library sequences not distributed)."]}
    with open(os.path.join(OUT,"sectionA_metrics.json"),"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({"chroms_available":avail,"chroms_missing":[c for c in OFFICIAL_SIZE if c not in lib],
        "counts":counts,"full673_rho":full["rho"],
        "subgroups":{kk:{"n":v.get("n"),"n_groups":v.get("n_groups"),"rho":v.get("rho")} for kk,v in results.items()}},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
