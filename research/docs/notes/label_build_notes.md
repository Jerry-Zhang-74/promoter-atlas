# Label Build Notes

- current label system: candidate_E_long_table_B_gated_detectable_tissue_MAD
- gene expression input is long-format gene x tissue mean_tpm
- repeated gene_id+tissue rows use nonzero-preferred collapse when zero and nonzero coexist
- Score_S remains empty when S_raw is undefined
- Score_HKG remains empty when Score_S is undefined