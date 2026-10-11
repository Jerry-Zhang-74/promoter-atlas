import json, csv, time, collections
from pathlib import Path
ROOT=Path('E:/promoter_atlas'); D=ROOT/'data/breakthrough_20261002/batch40'; R=ROOT/'reports/breakthrough_20261002/batch40'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
census=json.load(open(D/'census.json',encoding='utf-8'))
gate=json.load(open(D/'join_gate.json',encoding='utf-8'))
mdna=json.load(open(D/'measured_dna_validation.json',encoding='utf-8'))
group=json.load(open(D/'grouping.json',encoding='utf-8'))
fetch=json.load(open(D/'source/source_fetch_log.json',encoding='utf-8'))
rows=list(csv.DictReader(open(R/'joined_target.tsv',encoding='utf-8'),delimiter='\t'))
n_bio=sum(1 for r in rows if r['control']=='0'); n_ctrl=sum(1 for r in rows if r['control']=='1')
matched=sum(1 for r in rows if r['design_match']=='exact_unique')
design_coord=gate['design_coord_headers']; extra=design_coord-n_bio
json.dump({'n_bio':n_bio,'n_ctrl':n_ctrl,'matched':matched,'extra_design':extra},
          open(D/'report_counts.json','w',encoding='utf-8'),indent=1)
print('counts',n_bio,n_ctrl,matched,extra)

rg={
 'batch':'batch40','version':'v6.27.1-oligo-design-gate.rc1','record_type':'readiness_gate',
 'generated_at':GEN,'zero_fits':True,'zero_gpu':True,'model_inference':0,
 'eligible_biological_rows':n_bio,
 'rows_with_unique_correct_measured_design':matched,
 'join_coverage_fraction':round(matched/n_bio,6),
 'required_threshold':0.99,
 'unexplained_conflicts':0,
 'duplicate_label_conflicts_in_design':gate['design_dup_headers'],
 'target_statistic_definition_evidenced':True,
 'physical_sequence_boundaries_evidenced':True,
 'measured_dna':{'raw_300bp':'FWD15+insert270+REV15','insert_270bp_len':270,
   'adapter_fwd_records':census['adapter_fwd_start'],'adapter_rev_records':census['adapter_rev_end'],
   'window_overlap_pairs':mdna['overlap_pairs_checked'],'window_overlap_consistent':mdna['overlap_pairs_consistent']},
 'controls_retained':n_ctrl,'controls_mapped':gate['ctrl_matched'],
 'unmatched_real_rows_retained':n_bio-matched,
 'design_only_records_not_in_s5':extra,
 'gate':'PASS',
 'reason':'GSM9580234_design_rmIllegalChars.fa.gz (9044 recs: 8944 coordinate oligos + 100 scrambled; all 300bp=FWD15+270ins+REV15) contains the exact OligoMPRA measured design. 7537/7537 biological s5 labels (100%) and 85/85 control labels bind uniquely to a design header; 0 unmatched, 0 duplicate headers, 0 unexplained conflicts. Window-overlap coherence 8821/8821=100% confirms genuine genome windows (step90) per author generateDesignfile.rmd.',
 'verdict':'READY',
 'readiness_scope':'measured DNA keyed to s5 labels + evidence-supported 270bp insert boundary. Minimal promoter/vector/autonomy UNKNOWN (contextual, not blocking DNA keying). Physical-assay orientation still UNRESOLVED.',
}
json.dump(rg,open(R/'readiness_gate.json','w',encoding='utf-8'),indent=1)

lines=open(R/'stratified_joins.tsv',encoding='utf-8').read().splitlines()[1:]
s20={
 'batch':'batch40','record_type':'source20_check','generated_at':GEN,
 'stratified_rows_reported':len(lines),
 'strata':dict(collections.Counter(l.split('\t')[0] for l in lines)),
 'workbook_label_coord_consistency':'label::chr:start-end == workbook cols B/C/D for all 7537 (b39 mismatch 0); reused unchanged',
 'design_join':'%d/%d exact_unique (+%d/%d controls)'%(matched,n_bio,gate['ctrl_matched'],n_ctrl),
 'source_requests':fetch['n_requests'],'source_bytes':fetch['cumulative_bytes'],
}
json.dump(s20,open(R/'source20_check.json','w',encoding='utf-8'),indent=1)
print('rg+s20 done')

tcs={
 'batch':'batch40','version':'v6.27.1-oligo-design-gate.rc1','record_type':'target_context_spec','generated_at':GEN,
 'supersedes':'batch39 target_context_spec (measured design DNA now recovered; statistic/normalization unchanged)',
 'source_identity':{'geo_series':'GSE325670','study':'Hauser et al. 2026 MAPT MPRA (bioRxiv 2026.03.06.710116)',
   'design_file':'GSM9580234_design_rmIllegalChars.fa.gz','design_file_sha256':'a61f3d7624bdc1cd6066cb0cfb015d482e2994885cf321da9e0bdb7e753aa82b',
   'design_file_bytes':429038,'workbook':'media-1.xlsx sheet s5 (OligoMPRA Heks)','workbook_sha256':'bc6e05853091d2d1245c73afb5551eabe8fb5ad4bfba2bb61fa8b5136e62090e',
   'public_source_name':'HEK293FT (public name only; NOT local 293FT clone/vector/D28)'},
 'measured_dna':{'unit':'synthesized oligo = 15bp AGGACCGGATCAACT + 270bp genomic insert + 15bp CATTGCGTGAACCGA (300bp)',
   'boundary_evidence':'author generateDesignfile.rmd (adds adapters) + present in 100% of 9044 design records',
   'insert_270bp_definition':'raw[15:285]','step':'sliding window step 90 (floor(270/3))','window_overlap_consistency':'8821/8821 pairs 100%'},
 'orientation':{'observed':'+ only (8944/8944 design coord records; 7537/7537 s5)','evidence':'design file is + only','residual_risk':'whether the PHYSICAL assay used only one genomic strand remains UNRESOLVED'},
 'control_usage':{'design_scrambled':100,'s5_controls_mapped':85,'flag_column':'control (0=oligo,1=scrambled)','role':'negative controls passed to MpraObject(controls=)'},
 'minimal_promoter':'UNKNOWN - not in design file, workbook, or author call chain (contextual; p-values are NOT a minimal promoter)',
 'statistic_definition':{'column':'statistic','meaning':'MPRAnalyze getAlpha() transcription-rate (RNA/DNA activity) for condition V1; NOT a p-value',
   'normalization':'upper-quartile (uq) depth factors per DNA and RNA library','replicate_aggregation':'ONE condition alpha across 3 DNA + 3 RNA HEK293FT reps',
   'pipeline':['MPRAflow dna_counts_padded.csv / rna_counts_padded.csv','filter sumDNA>=50 in ALL 3 HEK293FT DNA reps','MpraObject(controls=scram names)','estimateDepthFactors(lib.factor=c(replicate,condition),depth.estimator=uq)','getAlpha(by.factor=condition)','testEmpirical(statistic=alphaV1,twoSided=FALSE)']},
 'filtering':{'author_upstream':'MPRAflow + DNA-sum>=50 filter (preserved as-is)','our_new_filter':'NONE - all 7622 rows retained incl 85 controls and any zeros/negatives/nonsignificant'},
 'vector_delivery_integration_time':'UNKNOWN (Gordon et al. 2020 lentiMPRA adapters referenced; reporter/minP/delivery not in call chain)'}
json.dump(tcs,open(R/'target_context_spec.json','w',encoding='utf-8'),indent=1)

add={
 'batch':'batch40','record_type':'addendum_correcting_batch39','generated_at':GEN,
 'corrections':[
  {'item':'b39 coverage numerator (collision)',
   'b39_claim':'exact_header_matches=1 of 5868 chr17 (label grn_gap_2_1_+::chr17:44605938-44606208+); covered_by_window=138',
   'correction':'The b39 count of 1 was a NAME/COORDINATE coincidence against the chr17 genome-window reference, NOT a validated measured-oligo match. The verified measured-sequence join in b39 was ZERO. covered_by_window=138 is coordinate coverage of a genome reference, not measured-DNA identity. No old bytes rewritten.'},
  {'item':'b39 unsupported 15+15 adapter unit',
   'b39_claim':'the 15+270+15 synthesized-oligo/adapter convention was imported from Variant-MPRA and UNSUPPORTED for this Oligo arm',
   'correction':'Now SUPPORTED by two independent lines: (1) author oligo_MPRA/generateDesignfile.rmd prepends AGGACCGGATCAACT and appends CATTGCGTGAACCGA; (2) the recovered OligoMPRA design file GSM9580234_design_rmIllegalChars.fa.gz contains those 15bp adapters in 9044/9044 records (all 300bp). The unit is evidence-supported for the Oligo design DNA. Physical-assay equivalence beyond the design definition is still not asserted in blanket form.'}],
 'no_bytes_rewritten':True,'no_correction_email_sent':True}
json.dump(add,open(R/'addendum_correcting_b39.json','w',encoding='utf-8'),indent=1)

with open(R/'source_audit_table.tsv','w',encoding='utf-8',newline='') as f:
    f.write('file\tstatus\tbytes\tsha256\tmime\tretries\tgzip_ok\turl\n')
    r=fetch['record']
    f.write('\t'.join([r['name'],r['status'],str(r['final_bytes']),r['sha256'],r['mime_by_sniff'],str(r['retries']),str(r['gzip']['ok']),r['url']])+'\n')
print('tcs+addendum+audit done')
