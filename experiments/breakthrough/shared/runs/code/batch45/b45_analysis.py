# -*- coding: utf-8 -*-
"""batch45 phase A (freeze) + aggregation. ZERO fit. Reads only syn.txt.gz + cached evidence."""
import json, math, collections, zlib, sys, hashlib, pickle
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT = C.OUT
def wj(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
raw = C.SRC.read_bytes()
d = zlib.decompressobj(16+zlib.MAX_WBITS); dec = d.decompress(raw)
freeze = {
  "batch":"batch45","version":"v6.32.0-synthetic-design-utility.rc1","phase":"A_freeze",
  "source_file":"data/breakthrough_20261002/batch44/syn.txt.gz",
  "official_filename":"GSE301246_Synthetic_Promoter_Libraries_Small_and_Large_Data.txt.gz",
  "compressed_bytes": len(raw), "decompressed_bytes": len(dec),
  "gzip_eof": bool(d.eof), "gzip_unused_tail_bytes": len(d.unused_data),
  "gzip_crc_ok": bool(d.eof) and len(d.unused_data)==0,
  "sha256_compressed": C.sha256_file(C.SRC),
  "sha256_decompressed": hashlib.sha256(dec).hexdigest(),
  "provenance": {
     "GEO_suppl_listing_entry":"GSE301246_Synthetic_Promoter_Libraries_Small_and_Large_Data.txt.gz  2025-06-27 09:11  376K",
     "accession":"GSE301246",
     "declared_by_PARM_primary_fulltext":"All data used in this paper is available through GEO under accession number GSE301246.",
     "primary_fulltext_cached":"data/breakthrough_20261002/batch44/parm_fulltext.xml (Europe PMC PMC13017510 OA)"},
  "schema":{"n_columns":19,"columns":C.COLS},
  "rules_frozen_before_activity": {
     "unique_observation_unit":"unique query DNA (query.seq) within one library/construct context",
     "info_vs_query":"info id <-> query.seq is 1:1; distinct info ids sharing DNA would NOT be independent",
     "barcode_rows_are_not_independent":"4 barcode rows per fragment are technical replicates of the same DNA, never n",
     "replicate_label_semantics":"'replicate00..09' in evolutionary info ids are the 10 independent GENETIC-ALGORITHM SEARCH LINEAGES, NOT biological replicates",
     "rna_replicate_semantics":"expression.RNA.replicate1/replicate2 are the two independent (biological) transfection replicates per primary Methods",
     "K562_is_design_objective":"synthetic promoters optimized with the PARM model trained on K562 data (primary Methods); HepG2 is cross-cell",
     "primary_matched_rule":"gen0 selection=max vs largest generation COMMON to ALL unmodified lineages, same library/length, metadata-only (activity-blind)"}}
wj("freeze_integrity.json", freeze)
print("freeze ok", freeze["sha256_compressed"])

rows = C.parse()
n = len(rows)
uniq_q = {r[2] for r in rows}; uniq_info = {r[1] for r in rows}
uniq_bc = {r[3] for r in rows}; uniq_oligo = {r[0] for r in rows}
bc_per = collections.Counter(); qlen = collections.Counter(); olen = collections.Counter()
off = collections.Counter(); libs = collections.Counter(); libq = collections.defaultdict(set)
q2info = collections.defaultdict(set); pred_k = collections.defaultdict(set); pred_h = collections.defaultdict(set)
for r in rows:
    libs[r[5]] += 1; libq[r[5]].add(r[2]); bc_per[(r[5], r[2])] += 1
    qlen[len(r[2])] += 1; olen[len(r[0])] += 1
    if r[2] in r[0]: off[r[0].index(r[2])] += 1
    q2info[r[2]].add(r[1]); pred_k[r[2]].add(r[18]); pred_h[r[2]].add(r[17])
frag_len = collections.defaultdict(set)
for r in rows: frag_len[(C.libtag(r[5]), C.category(r[1]))].add(len(r[2]))
o0 = off.most_common(1)[0][0]
identity = {
  "n_data_rows": n, "n_unique_query_seq": len(uniq_q), "n_unique_info": len(uniq_info),
  "n_unique_barcode": len(uniq_bc), "n_unique_full_oligo": len(uniq_oligo),
  "query_info_is_1to1": len(uniq_q)==len(uniq_info) and all(len(v)==1 for v in q2info.values()),
  "barcodes_per_fragment_dist": dict(collections.Counter(bc_per.values())),
  "query_length_dist": {str(k):v for k,v in sorted(qlen.items())},
  "full_oligo_length_dist": {str(k):v for k,v in sorted(olen.items())},
  "query_is_exact_substring_of_full_oligo_all_rows": len(off)==1 and sum(off.values())==n,
  "query_start_offset_in_full_oligo": o0,
  "flank_lengths_bp": {"5prime_flank": o0, "3prime_flank": sorted(olen)[-1]-o0-sorted(qlen)[-1]},
  "library_row_counts": dict(libs),
  "library_unique_query": {k:len(v) for k,v in libq.items()},
  "length_by_library_category": {k[0]+":"+k[1]: sorted(v) for k,v in sorted(frag_len.items())},
  "length_230_vs_232_note": ("library named 'long.232bp' (nominal GA individual length 232 bp per Methods). "
      "Measured SYNTHETIC long inserts (design/best/MUTATED) are 230 bp; the 42 NATURAL control fragments in the long library are 232 bp. "
      "No two distinct info ids share the same DNA. The 2 bp difference is retained explicitly, not padded/clipped. "
      "Which of the two lengths the library label refers to is NOT uniquely determined from cached evidence."),
  "query_DNA_identity":"query.seq is an exact substring (offset 18) of full.oligo in every row; full.oligo = 18 bp 5' flank + query.seq + 50 bp 3' flank (flanks NOT interpreted as adapter/reference/RC).",
  "prediction_consistency": {"K562_queries_with_conflicting_prediction": sum(1 for v in pred_k.values() if len(v)>1),
                             "HepG2_queries_with_conflicting_prediction": sum(1 for v in pred_h.values() if len(v)>1)}}
wj("identity_metadata.json", identity)
print("identity ok", identity["n_unique_query_seq"])

nonfin = collections.Counter(); dup = 0; seen = set()
for r in rows:
    key = tuple(r)
    if key in seen: dup += 1
    seen.add(key)
    for idx,nm in [(11,"K562.r1"),(12,"K562.r2"),(13,"HepG2.r1"),(14,"HepG2.r2"),(15,"K562.avg"),(16,"HepG2.avg")]:
        v = C.to_float(r[idx])
        if v is None or not math.isfinite(v): nonfin[nm] += 1
wj("data_quality.json", {
  "exact_duplicate_source_rows": dup,
  "nonfinite_normalized_values_per_column": dict(nonfin),
  "explanation": "one barcode row (info=K562_fitness_232bp_replicate01_generation0_individual192_median, barcode TTGGATGCGACGA) has 0 raw RNA counts in all four RNA columns -> -Inf normalized. Retained in raw table; excluded from the finite arithmetic mean and counted."})

per = {}
for r in rows:
    q = r[2]; k = q
    if k not in per:
        per[k] = {"query.seq":q,"info":r[1],"library":r[5],"lib":C.libtag(r[5]),"category":C.category(r[1]),
                  "q_len":len(q),"full_oligo_len":len(r[0]),"GC":C.gc_fraction(q),
                  "pred_K562":C.to_float(r[18]),"pred_HepG2":C.to_float(r[17]),
                  "K562_r1":[],"K562_r2":[],"HepG2_r1":[],"HepG2_r2":[],
                  "pDNA":[], "barcodes":[], "n_nonfinite_cells":0}
    p = per[k]
    p["barcodes"].append(r[3]); p["pDNA"].append(C.to_float(r[6]))
    for col,key in [(11,"K562_r1"),(12,"K562_r2"),(13,"HepG2_r1"),(14,"HepG2_r2")]:
        v = C.to_float(r[col])
        if v is None or not math.isfinite(v): p["n_nonfinite_cells"] += 1
        else: p[key].append(v)
for k,p in per.items():
    p["n_barcodes"] = len(p["barcodes"])
    for kk in ["K562_r1","K562_r2","HepG2_r1","HepG2_r2"]:
        p[kk+"_mean"] = C.mean_finite(p[kk])
    p["K562_meas"] = C.mean_finite([p["K562_r1_mean"], p["K562_r2_mean"]])
    p["HepG2_meas"] = C.mean_finite([p["HepG2_r1_mean"], p["HepG2_r2_mean"]])
    allv = p["K562_r1"]+p["K562_r2"]
    p["K562_barcode_spread_sd"] = (lambda v: (sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5 if len(v)>1 else None)(allv)
    for kk in ["K562_r1","K562_r2","HepG2_r1","HepG2_r2"]: del p[kk]
(OUT/"_per.pkl").write_bytes(pickle.dumps(per))
print("agg done, fragments =", len(per))

wj("aggregation.json", {
  "fixed_definition":"per unique query DNA within a library: per author RNA replicate = arithmetic mean of author normalized.log2rpm over finite barcode rows; assay summary = mean of the two replicate means.",
  "author_contract_from_methods":"'Fragment activity was calculated by taking the mean log2[RNA/DNA] ratio of four barcodes.' (PARM Methods); two independent (biological) transfection replicates per cell line.",
  "contract_verdict":"IDENTICAL in form (equal-weight mean over the 4 barcodes per replicate, then mean over the 2 replicates). No different author formula found; the fixed summary is not an alternative estimator.",
  "verified_author_column_identity":"expression.normalized.log2rpm.*.averaged == mean(replicate1,replicate2) for every row.",
  "n_unique_fragments": len(per),
  "n_fragments_with_barcode_dropout": sum(1 for p in per.values() if p["n_nonfinite_cells"]>0),
  "exclusions": {"nonfinite_barcode_cells_excluded": sum(p["n_nonfinite_cells"] for p in per.values())}})
hdr = ["info","library","category","q_len","GC","n_barcodes",
       "pred_K562","pred_HepG2","K562_r1_mean","K562_r2_mean","HepG2_r1_mean","HepG2_r2_mean",
       "K562_meas","HepG2_meas","K562_barcode_spread_sd","n_nonfinite_cells"]
with open(OUT/"per_sequence.tsv","w",encoding="utf-8") as f:
    f.write("\t".join(hdr)+"\n")
    for k,p in sorted(per.items()):
        f.write("\t".join(str(p[c]) for c in hdr)+"\n")
print("wrote aggregation.json + per_sequence.tsv")
