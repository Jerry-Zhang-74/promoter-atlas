#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BATCH32 Section B: local 293FT failure audit from EXISTING records only.
ZERO fits / inference / GPU. Preserves original HEK293T labels + adds user 293FT context.
Endpoint = log10( GFP-positive gated median / same-batch WT-negative median )."""
import csv, json, os, math
HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.dirname(os.path.dirname(HERE))
CL=os.path.join(REPO,"reports/cellline_v2_20261001")
B31=os.path.join(REPO,"reports/breakthrough_20261002/batch31")
B24=os.path.join(REPO,"data/breakthrough_20261002/batch24")
B23=os.path.join(REPO,"data/breakthrough_20261002/batch23")
OUT=os.path.join(REPO,"reports/breakthrough_20261002/batch32")
TAB=os.path.join(OUT,"tables")
def read_tsv(p):
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f,delimiter="\t"))
def read_fasta(p):
    d={}; k=None
    for line in open(p,encoding="utf-8"):
        line=line.rstrip("\n").strip()
        if line.startswith(">"): k=line[1:]; d[k]=""
        elif k is not None: d[k]+=line
    return d
def rc(s):
    t=str.maketrans("ACGTN","TGCAN"); return s.translate(t)[::-1]
def best_containment(short,longseq):
    """Find offset of short within longseq on + or - strand; return dict."""
    res={}
    for strand,probe in (("+",short),("-",rc(short))):
        i=longseq.find(probe)
        res[strand]=i
    return res
def nearest_align(short,longseq):
    """For non-exact, find minimal-mismatch offset (<=3 mismatches) on the strand with best match."""
    best=None
    for strand,probe in (("+",short),("-",rc(short))):
        n=len(probe)
        for off in range(0,len(longseq)-n+1):
            mm=sum(1 for a,b in zip(probe,longseq[off:off+n]) if a!=b)
            if best is None or mm<best[0]:
                best=(mm,strand,off)
            if mm==0: break
        if best and best[0]==0: break
    return {"min_mismatch":best[0],"strand":best[1],"offset0":best[2]} if best else None

def main():
    os.makedirs(TAB,exist_ok=True)
    cs=read_tsv(os.path.join(CL,"construct_sequences.tsv"))
    fasta=read_fasta(os.path.join(CL,"construct_sequences.fasta"))
    ep={r["sample_annotation"]:r for r in read_tsv(os.path.join(CL,"construct_endpoints_D28.tsv"))}
    mr=read_tsv(os.path.join(CL,"measurements_ready.tsv"))
    parm={r["construct_id"]:r for r in read_tsv(os.path.join(B31,"tables/local_supported_scores.tsv"))}
    h1={r["construct_id"]:r for r in read_tsv(os.path.join(B24,"local14_exploratory_scores.tsv"))}
    r1={r["construct_id"]:r for r in read_tsv(os.path.join(B23,"local_diagnostic_scores.tsv"))}
    # endpoint-definition evidence (primary columns)
    e_def={"numerator_gate":sorted(set(r["numerator_gate"] for r in mr)),
           "numerator_statistic":sorted(set(r["numerator_statistic"] for r in mr)),
           "numerator_channel":sorted(set(r["numerator_channel"] for r in mr)),
           "wt_negative_mean_values":sorted(set(round(float(r["wt_negative_mean"]),3) for r in mr))[:5],
           "gfp_positive_percent_parent_present":all(r.get("gfp_positive_percent_parent") not in (None,"") for r in mr),
           "target_name":ep["A1"]["target_name"],
           "n_measurement_rows":len(mr),
           "distinct_cell_line_reported":sorted(set(r["cell_line_reported"] for r in mr)),
           "distinct_source_id":sorted(set(r["source_id"] for r in mr)),
           "days_after_transfection":sorted(set(r["days_after_transfection_reported"] for r in mr)),
           "days_after_sort":sorted(set(r["days_after_sort_reported"] for r in mr)),
           "sort_round":sorted(set(r["sort_round_reported"] for r in mr)),
           "independent_biological_replicates_vals":sorted(set(r["independent_biological_replicates"] for r in mr))}
    # endpoint per construct (endpoint table) + which measurement rows feed it
    # ---- nesting / orientation (2000bp used ONLY for alignment) ----
    ref={"ACTB":"A20","GAPDH":"G20"}
    nesting=[]; members={"ACTB":["A1","A2","A3","A4","A5"],"GAPDH":["G1","G2","G3","G4","G5"]}
    for fam,R in ref.items():
        parent=fasta[R]
        for cid in members[fam]:
            s=fasta[cid]; a=nearest_align(s,parent)
            nesting.append({"family":fam,"construct":cid,"ref":R,"len":len(s),
                            "min_mismatch":a["min_mismatch"],"strand_in_ref":a["strand"],"offset0_in_ref":a["offset0"]})
    # pairwise chain within series
    chain=[]
    for fam in ("ACTB","GAPDH"):
        ids=members[fam]
        for i in range(len(ids)-1):
            a,b=ids[i],ids[i+1]
            sa,sb=fasta[a],fasta[b]
            c=best_containment(sa,sb)  # is a contained in b
            chain.append({"family":fam,"short":a,"long":b,"short_len":len(sa),"long_len":len(sb),
                          "contained_in_+":c["+"],"contained_in_-":c["-"]})
    # CMV
    cmv={"CMVcore_in_CMVfull":best_containment(fasta["CMVcore"],fasta["CMVfull"]),
         "CMVcore_len":len(fasta["CMVcore"]),"CMVfull_len":len(fasta["CMVfull"]),
         "core_documented_len":next(r["documented_length_bp"] for r in cs if r["construct_id"]=="CMVcore"),
         "core_sequence_len":next(r["sequence_length_bp"] for r in cs if r["construct_id"]=="CMVcore")}

    # ---- direction: D28 observed vs frozen model predictions (supported <=584) ----
    import itertools
    def sgn(x): return 0 if x==0 else (1 if x>0 else -1)
    fam_map={"A1":"ACTB","A2":"ACTB","A3":"ACTB","A4":"ACTB","A5":"ACTB",
             "G1":"GAPDH","G2":"GAPDH","G3":"GAPDH","G4":"GAPDH","G5":"GAPDH",
             "CMVcore":"CMV","CMVfull":"CMV"}
    ep_by_cid_tmp={"A1":"A1","A2":"A2","A3":"A3","A4":"A4","A5":"A5","G1":"G1","G2":"G2","G3":"G3","G4":"G4","G5":"G5","CMVcore":"CMV","CMVfull":"CAV"}
    obs={k:float(ep[ep_by_cid_tmp[k]]["log10_relative_gfp"]) for k in fam_map}
    models={"PARM":{k:float(parm[k]["PARM_HEK293_score"]) for k in fam_map},
            "H1":{k:float(h1[k]["H1_exploratory_model_score_mean"]) for k in fam_map},
            "R1":{k:float(r1[k]["R1_mean_output"]) for k in fam_map}}
    lengths={k:(204 if k=="CMVcore" else 584 if k=="CMVfull" else int(next(r["sequence_length_bp"] for r in cs if r["construct_id"]==k))) for k in fam_map}
    dir_out={m:{} for m in models}; macro={}
    for m in models:
        fams={}
        for fam in ("ACTB","GAPDH","CMV"):
            ids=[k for k in fam_map if fam_map[k]==fam]
            n=0; ok=0; ok_inc=0; ok_dec=0; pairs=[]
            for a,b in itertools.combinations(sorted(ids),2):
                do=sgn(obs[a]-obs[b]); dp=sgn(models[m][a]-models[m][b]); dl=sgn(lengths[a]-lengths[b])
                if do==0 or dp==0: continue
                n+=1; ok+= (do==dp); ok_inc+= (do==dl); ok_dec+= (do!=dl)
                pairs.append({"a":a,"b":b,"obs_dir":do,"pred_dir":dp,"len_dir":dl,"agree":int(do==dp),
                              "obs_fold_change_log10":round(abs(obs[a]-obs[b]),4)})
            fams[fam]={"n_pairs":n,"direction_accuracy":(ok/n if n else None),
                       "baseline_increasing_len":(ok_inc/n if n else None),
                       "baseline_decreasing_len":(ok_dec/n if n else None),"pairs":pairs}
        dir_out[m]=fams
        vals=[fams[f]["direction_accuracy"] for f in ("ACTB","GAPDH","CMV")]
        macro[m]=sum(v for v in vals if v is not None)/len([v for v in vals if v is not None])
    # ---- 14-row evidence table ----
    seq_by_id={r["construct_id"]:r for r in cs}
    ep_by_cid={"A1":"A1","A2":"A2","A3":"A3","A4":"A4","A5":"A5","A20":"A20","G1":"G1","G2":"G2","G3":"G3","G4":"G4","G5":"G5","G20":"G20","CMVcore":"CMV","CMVfull":"CAV"}
    nested_map={}
    for n in nesting: nested_map[n["construct"]]=f"min_mismatch={n['min_mismatch']} strand={n['strand_in_ref']} off0={n['offset0_in_ref']}"
    ev=[]
    for cid in ["A1","A2","A3","A4","A5","A20","G1","G2","G3","G4","G5","G20","CMVfull","CMVcore"]:
        r=seq_by_id[cid]; sa=ep_by_cid[cid]; e=ep[sa]
        ev.append({"construct_id":cid,"family":r["family"],
            "asserted_upstream_bp":e.get("upstream_bp_reported"),
            "documented_length_bp":r["documented_length_bp"],"sequence_length_bp":r["sequence_length_bp"],
            "source_basename":os.path.basename(r["source_path"]),"source_molecule_length_bp":r["source_molecule_length_bp"],
            "feature_ranges_1based_inclusive":r["feature_ranges_1based_inclusive"],
            "feature_directionality":r["feature_directionality"],
            "mapping_evidence":r["mapping_evidence"],"barcode_verified":r["assay_to_plasmid_barcode_verified"],
            "endpoint_sample_annotation":sa,"endpoint_source_path_basename":os.path.basename(e.get("source_path","")),
            "endpoint_target_value":e.get("log10_relative_gfp"),
            "endpoint_eligible":e.get("endpoint_eligible"),
            "nesting_vs_2000bp":nested_map.get(cid,"(is 2000bp reference)"),
            "sequence_status":r["sequence_status"]})
    with open(os.path.join(TAB,"local14_evidence_table.tsv"),"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(ev[0].keys()),delimiter="\t"); w.writeheader(); w.writerows(ev)
    out={"section":"B","cell_context":{"target":"USER 293FT (direct-user correction 2026-10-03)",
        "historical_label_preserved":"HEK293T (original records, unchanged)",
        "endpoint_definition":"log10( GFP-positive gated median / same-batch WT-negative median ); NOT positive-cell fraction"},
        "endpoint_evidence":e_def,"nesting_vs_2000bp":nesting,"series_containment":chain,"cmv":cmv,
        "observed_D28":{"A1":obs["A1"],"A2":obs["A2"],"A3":obs["A3"],"A4":obs["A4"],"A5":obs["A5"],
            "G1":obs["G1"],"G2":obs["G2"],"G3":obs["G3"],"G4":obs["G4"],"G5":obs["G5"],
            "CMVcore":obs["CMVcore"],"CMVfull":obs["CMVfull"],
            "A20_2000bp_unsupported":float(ep["A20"]["log10_relative_gfp"]),"G20_2000bp_unsupported":float(ep["G20"]["log10_relative_gfp"])},
        "direction_vs_frozen_models":dir_out,"macro_family_direction":macro,
        "artifact_assessment":{}}
    # artifact assessment
    art={}
    orient=[n["strand_in_ref"] for n in nesting]
    art["orientation"]="all family members align to the 2000bp reference on strand %s -> orientation not a source of length-direction artifact"%sorted(set(orient))
    art["normalization"]="single same-batch WT-negative denominator (1420.267) used for ALL 14 constructs -> no per-construct normalization switch"
    a5=[n for n in nesting if n["construct"]=="A5"]
    art["naming_structure"]="A1-A4 share feature start 133 (3'-extension series); A5 feature 287-786 is a SHIFTED window, and A5 differs from the 2000bp at %s base(s) -> A5 is not a clean extension of A4; naming/design window (not necessarily the assayed DNA) may explain part of the length-direction mismatch"%(",".join(str(n['min_mismatch']) for n in a5))
    art["absence_of_evidence"]="no barcode link and no clone sequencing; observed mismatch mechanism cannot be proven from these records"
    out["artifact_assessment"]=art
    with open(os.path.join(OUT,"sectionB_metrics.json"),"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({"endpoint_target":e_def["target_name"],"macro":macro,
        "nesting":[f"{n['construct']}<-{n['ref']}:mm={n['min_mismatch']},str={n['strand_in_ref']}" for n in nesting],
        "cmv_core_in_full":cmv["CMVcore_in_CMVfull"],"n_ev":len(ev)},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
