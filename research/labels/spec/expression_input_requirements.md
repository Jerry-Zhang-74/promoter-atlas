# Expression Input Requirements

## Required Upstream Assets

- GTEx-derived expression matrix
- gene identifier mapping compatible with GENCODE v49 gene_id
- tissue metadata table
- sample-level or tissue-aggregated expression summary

## Minimum Columns Needed

- gene_id
- tissue_id or tissue_name
- expression_value
- expression_unit

## Preferred Expression Semantics

- use normalized expression values that are biologically interpretable
- keep a clear distinction between gene x sample and gene x tissue matrices
- retain tissue coverage information for breadth and stability calculations
- do not bury filtering thresholds inside undocumented preprocessing

## Requirements For E / S / B

### For Score_E
- need a robust expression strength summary across tissues or across samples within tissues
- summary rule must be documented explicitly

### For Score_S
- need a stability metric that does not simply mirror expression magnitude
- any variance normalization or residualization rule must be documented explicitly

### For Score_B
- need a detectable-expression rule per tissue
- threshold choice must be explicit and easy to audit

## Required Join Key

- join expression labels to promoter records by gene_id first
- transcript-level promoter records may share the same gene-level label in the first baseline
- later transcript-aware refinements can be added separately
