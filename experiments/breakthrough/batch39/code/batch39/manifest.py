import json,hashlib,time
from pathlib import Path
ROOT=Path('E:/promoter_atlas');R=ROOT/'reports/breakthrough_20261002/batch39'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inc=[
 'code/batch39/fetch_sources.py','code/batch39/parse_s5_join.py','code/batch39/analyze_s5.py',
 'code/batch39/join_test.py','code/batch39/offset_test.py','code/batch39/span_check.py',
 'code/batch39/finalize_a.py','code/batch39/finalize_b.py','code/batch39/figures.py','code/batch39/rep.py','code/batch39/jart.py',
 'data/breakthrough_20261002/batch39/source/source_fetch_log.json',
 'data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz',
 'data/breakthrough_20261002/batch39/source/MPRAnalyze_oligoMPRA_heks.R',
 'data/breakthrough_20261002/batch39/source/generateDesignfile.rmd',
 'data/breakthrough_20261002/batch39/source/mpranalyze_padding.py',
 'data/breakthrough_20261002/batch39/source/MPRAnalyze_oligoMPRA_neurons.R',
 'data/breakthrough_20261002/batch39/s5_rows_raw.json','data/breakthrough_20261002/batch39/s5_audit.json',
 'data/breakthrough_20261002/batch39/s5_summary.json','data/breakthrough_20261002/batch39/grouping.json',
 'data/breakthrough_20261002/batch39/join_design.json','data/breakthrough_20261002/batch39/analysis_bundle.json',
 'reports/breakthrough_20261002/batch39/data_readiness_report.md','reports/breakthrough_20261002/batch39/target_context_spec.json',
 'reports/breakthrough_20261002/batch39/joined_target.tsv','reports/breakthrough_20261002/batch39/source_audit_table.tsv',
 'reports/breakthrough_20261002/batch39/readiness_gate.json','reports/breakthrough_20261002/batch39/version_registry_entry.json',
 'figures/batch39/fig1_tiles_by_chrom.png','figures/batch39/fig2_group_sizes.png']
man={'batch':'batch39','generated_at':GEN,'excludes':['sha_manifest.json','status.json','worker_selfcheck.json','remaining_work.json','next_batch_proposal.json','independent_review.json','mail artifacts'],
 'files':{}}
for f in inc:
    p=ROOT/f
    if p.exists(): man['files'][f]={'sha256':sha(p),'bytes':p.stat().st_size}
    else: man['files'][f]={'sha256':None,'reason':'missing'}
json.dump(man,open(R/'sha_manifest.json','w',encoding='utf-8'),indent=1)
print('manifest files',len(man['files']),'missing',sum(1 for v in man['files'].values() if v['sha256'] is None))
