import json, time
from pathlib import Path
ROOT=Path('E:/promoter_atlas')
D=ROOT/'data/breakthrough_20261002/batch39'
R=ROOT/'reports/breakthrough_20261002/batch39'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
spec={
 'batch':'batch39','version':'v6.27.0-hekft-fragment-readiness.rc1','record_type':'target_context_spec',
 'generated_at':GEN,
 'source_identity':{'geo_series':'GSE325670','study':'Hauser et al. 2026 MAPT MPRA (bioRxiv 2026.03.06.710116)',
   'workbook':'media-1.xlsx sheet s5 (OligoMPRA Heks)',
   'workbook_sha256':'bc6e05853091d2d1245c73afb5551eabe8fb5ad4bfba2bb61fa8b5136e62090e',
   'public_source_name':'HEK293FT (keep public name; NOT local 293FT clone/vector/D28)'},
 'unit':'270 bp genomic insert per oligo (0-based half-open end=start+270); synthesized oligo = 15bp AGGACCGGATCAACT + 270bp + 15bp CATTGCGTGAACCGA',
 'paper270_vs_span271':'RESOLVED from author code: design BED end=start+elementsize (elementsize=270) -> bedtools getfasta extracts 270 bp. Naive inclusive span (end-start+1=271) is a counting artifact; all 7537 s5 rows have end-start==270; no 271bp sequence exists.',
 'orientation':{'observed':'+ only (7537/7537)','risk':'plus-only may be a pipeline default/aggregate, NOT evidence that only one strand was assayed; author addon block does emit +/- so a - arm may exist elsewhere; UNRESOLVED'},
 'control_usage':{'n_scrambled':85,'flag_column':'control (0=oligo,1=scrambled)','role':'negative controls passed to MpraObject(controls=); testEmpirical empirical null'},
 'minimal_promoter':'UNKNOWN - no minimal promoter, vector architecture or autonomy stated in the fetched call chain. Parent correction stands: p-value columns are NOT a minimal promoter.',
 'statistic_definition':{'column':'statistic','meaning':'MPRAnalyze getAlpha() transcription-rate estimate (RNA/DNA activity) for condition V1; NOT a p-value',
   'pipeline':['MPRAflow dna_counts_padded.csv / rna_counts_padded.csv','filter rows with sumDNA>=50 in ALL 3 HEK293FT DNA reps','MpraObject(controls=scram...)',
     'estimateDepthFactors(lib.factor=c(replicate,condition), depth.estimator=uq) for DNA and RNA',
     'alpha<-getAlpha(by.factor=condition)','testEmpirical(statistic=alpha$V1, twoSided=FALSE) -> zscore/mad.score/pval.empirical/pval.mad/pval.zscore']},
 'normalization':'upper-quartile (uq) library depth factors per library (DNA and RNA) by replicate+condition',
 'replicate_aggregation':'aggregated into ONE condition alpha across the 3 DNA + 3 RNA HEK293FT replicates (no per-replicate value in the sheet)',
 'vector_delivery_integration_time':'UNKNOWN from fetched sources (Gordon et al. 2020 lentiMPRA adapters referenced; reporter/minP/delivery/time not in call chain)',
 'context_distinction':'HEK293FT plasmid-transfection RNA/DNA reporter sampling is DISTINCT from neuronal lentiviral transduction; no HEK integration inferred from neuron methods',
 'labels_are':'per-fragment reporter ACTIVITY (alpha); p/z/mad columns are control-referenced statistics, not the target',
 'filtering':{'author_upstream':'MPRAflow + DNA-sum>=50 filter (author selection, preserved as-is)','our_new_filter':'NONE - all 7622 rows retained incl 85 controls and any zeros/negatives/nonsignificant'}}
json.dump(spec,open(R/'target_context_spec.json','w',encoding='utf-8'),indent=1)
src=json.load(open(D/'source/source_fetch_log.json',encoding='utf-8'))
with open(R/'source_audit_table.tsv','w',encoding='utf-8',newline='') as f:
    f.write('file\tstatus\tbytes\tsha256\tmime\tretries\tgzip_crc_ok\turl\n')
    for r in src['records']:
        g=r.get('gzip') or {}
        f.write('\t'.join(map(str,[r['name'],r['status'],r.get('final_bytes'),r.get('sha256'),
            r.get('mime_by_sniff'),r['retries'],g.get('crc_verified',''),r['url']]))+'\n')
bundle={'generated_at':GEN,'grouping':json.load(open(D/'grouping.json',encoding='utf-8')),
        'join':{'chr17_s5':5868,'exact_header_matches':1,'covered_by_window':138,'design_records':3946070,
                'design_unique_windows':2717466,'design_adapter_records':0,'design_chroms':['chr17']},
        's5':json.load(open(D/'s5_audit.json',encoding='utf-8')),
        'source':{'n_requests':src['n_requests'],'cumulative_bytes':src['cumulative_bytes']}}
json.dump(bundle,open(D/'analysis_bundle.json','w',encoding='utf-8'),indent=1)
print('B done')
