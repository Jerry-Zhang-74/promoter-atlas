import json,time
from pathlib import Path
ROOT=Path('E:/promoter_atlas');R=ROOT/'reports/breakthrough_20261002/batch39';D=ROOT/'data/breakthrough_20261002/batch39'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
g=json.load(open(D/'grouping.json',encoding='utf-8'));s5=json.load(open(D/'s5_audit.json',encoding='utf-8'))['summary']
T=[
"# Batch39 v6.27.0-hekft-fragment-readiness.rc1 - data readiness report",
"",f"Generated {GEN} | ZERO fits/optimizer/gradient/model-scoring/neural-inference/GPU.","",
"## Verdict: NOT READY (LIMITED) - measured OligoMPRA DNA not in the authorized source",
"Pre-declared gate: >99% of eligible biological rows must uniquely match the correct measured design, with target statistic AND physical boundaries evidenced.",
f"- eligible rows = {s5['n_real']} ; rows with a unique correct measured design = 1 -> coverage 0.013% (need 99%) -> GATE FAIL; boundary UNRESOLVED.",
f"- controls ({s5['n_scrambled']} scrambled) and all UNRESOLVED rows retained; no filtering.","",
"## A. Source recovery (5 requests, 25,959,333 B, 0 retries)",
"gzip CRC+ISIZE OK; sizes match the frozen GEO filelist. Files: GSM9609016_designfile.fa.gz (25,939,760 B) + author oligo_MPRA call chain (MPRAnalyze_oligoMPRA_heks.R, generateDesignfile.rmd, mpranalyze_padding.py, MPRAnalyze_oligoMPRA_neurons.R).",
"KEY: GSM9609016 title = 'Barcode Association Sequencing'. Its designfile is a chr17-only barcode-association reference: 3,946,070 records (2,717,466 unique 0-based half-open windows, spans 146-350bp), soft-masked genome, 0 records contain the plasmid adapter AGGACCGGATCAACT, and it contains none of the s5 labels (exact join 1/5868 chr17, 0/1669 non-chr17). NOT the 270bp OligoMPRA design.","",
"## B. Frozen sequence/target rules (exact author call chain)",
"- `statistic` = MPRAnalyze getAlpha() transcription-rate (RNA/DNA ACTIVITY), NOT a p-value. z/mad/pval.* come from testEmpirical(statistic=alpha$V1, twoSided=FALSE).",
"- Normalization: depth.estimator='uq' (upper-quartile) per DNA and RNA library, lib.factor=c(replicate,condition). Replicates: ONE condition alpha over 3 DNA + 3 RNA HEK293FT reps.",
"- Controls: names containing 'scram' -> MpraObject(controls=).",
"- paper270 vs span271 RESOLVED: author generateDesignfile.rmd element size 270, BED end=start+270, bedtools 0-based half-open -> 270 bp insert. end-start+1=271 is a counting artifact; all 7537 rows end-start==270.",
"- Orientation: 7537/7537 are '+'. Plus-only = pipeline default/aggregate, NOT proof only one strand was assayed. UNRESOLVED.",
"- minimal promoter = UNKNOWN (no minP/vector/autonomy in call chain). PARENT CORRECTION UPHELD: p-value columns are NOT a minimal promoter; b38 readiness_C 'minP=p-values' was wrong.",
"- vector/delivery/integration/time = UNKNOWN. HEK293FT plasmid RNA/DNA sampling != neuronal transduction; no HEK integration inferred from neuron methods. Public name HEK293FT kept (NOT local 293FT/D28).","",
"## C. Label-free grouping (no DNA)",
f"Coordinates only: {g['n_groups']} connected components of overlapping 270bp windows (step90); largest {g['largest'][0][0]} tiles. chr17 tiles sit in one broader GRN-MAPT-KANSL1 locus; group count is NOT independent CRE families. No split forced.","",
"## D. Preserved failures / honesty",
"- Parent minP correction recorded (b38 semantic error; old bytes untouched; no extra email).",
"- Historical batch37 preflight FAIL (KV valid 78.47%) preserved; no retroactive PASS.",
"- local 293FT D28 FAIL and PARM600 limit preserved.",
"- NO predictive gain claimed (readiness only; no model scored). Old pointers v6.3/v6.4 unchanged.","",
"## One next bounded decision",
"Missing: the actual OligoMPRA measured design DNA keyed to s5 labels (companion GEO file GSM9580234_design_rmIllegalChars.fa.gz, 429,038 B, same 'Barcode Association Sequencing' title, NOT authorized in b39).",
"batch40 candidate (ZERO fit): fetch ONLY that 429KB file (1 request) and apply the SAME frozen join gate; if >99% of 7537 labels bind uniquely, freeze the seq->alpha target + groups; else STOP and escalate. Different, smaller source with a falsifiable gate - not a repeat of the GSM9609016 fetch.",
]
(R/'data_readiness_report.md').write_text("\n".join(T)+"\n",encoding='utf-8')
print('report written')
