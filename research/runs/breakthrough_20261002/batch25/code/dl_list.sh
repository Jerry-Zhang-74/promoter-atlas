cd /e/promoter_atlas
S=runs/breakthrough_20261002/batch25/source
L=runs/breakthrough_20261002/batch25/source_attempts_ledger.jsonl
P=http://127.0.0.1:7897
PY=.venv/Scripts/python.exe
fetch(){ $PY -X utf8 runs/breakthrough_20261002/batch25/code/fetch.py "$1" "$2" --ledger $L --timeout 175 --proxy $P; }
B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE306nnn/GSE306816/suppl
D=data/breakthrough_20261002/batch25/source
fetch $B/GSE306816_hSTR1_association_merged_BC_STRs.tsv.gz $S/GSE306816_hSTR1_association_merged_BC_STRs.tsv.gz
fetch $B/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz $S/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz
fetch $B/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep2.csv.gz $S/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep2.csv.gz
fetch $B/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep3.csv.gz $S/GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep3.csv.gz
fetch $B/GSE306816_hSTR1_HEK293_linear_regression.csv.gz $S/GSE306816_hSTR1_HEK293_linear_regression.csv.gz
fetch $B/GSE306816_hSTR2_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz $S/GSE306816_hSTR2_HEK293_BC_cDNA_gDNA_ratio_rep1.csv.gz
fetch https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9209nnn/GSM9209852/suppl/GSM9209852_dpSTR_association_Illumina_BC_STRs.tsv.gz $S/GSM9209852_dpSTR_association_Illumina_BC_STRs.tsv.gz
# GitHub design (raw via proxy)
G=https://raw.githubusercontent.com/gymreklab/str_mpra_design/main/design
fetch $G/tss_str_pairs.tsv $S/str_mpra_design_tss_str_pairs.tsv
fetch $G/array_probes.tsv $S/str_mpra_design_array_probes.tsv
GH=https://raw.githubusercontent.com/gymreklab/str-mpra/main/design-uber
fetch $GH/uber_str_set.bed $S/str-mpra_uber_str_set.bed
fetch $GH/uber-v3.oligos.split.tab $S/str-mpra_uber-v3.oligos.split.tab
fetch $GH/round1_probe_seqs.tab $S/str-mpra_round1_probe_seqs.tab
echo "ALL_DOWNLOADS_DONE"
