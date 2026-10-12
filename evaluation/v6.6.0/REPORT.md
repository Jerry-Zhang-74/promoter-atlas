# v6.6 原留出分割结果

模型从具有历史同源监督的v6.0权重初始化；本报告使用项目原留出分割。

单位: ESB=original Score_E/Score_S/Score_B scales; profile=log2(1+TPM); fragment=Log2RPM_HEK293。

检查点基于 DEV 上 E 头基因级 Spearman 选择，随后在 TEST 上评估；bootstrap 来自冻结的预测缓存。

## 直接 ESB 头
- E: valid_genes=5603, spearman=0.6611136738146578, rmse=0.24622159606544483, bootstrap_ci=[0.6452552105164991, 0.6775654594412627] valid_replicates=1000
- S: valid_genes=1855, spearman=0.3758097852749756, rmse=0.08667428229740565, bootstrap_ci=[0.33835694718584597, 0.41469385017107924] valid_replicates=1000
- B: valid_genes=5603, spearman=0.6645609156178874, rmse=0.24315463779549742, bootstrap_ci=[0.6478711904983051, 0.6805551932786404] valid_replicates=1000

## 组织表达谱 (log2(1+TPM))
- mean_spearman=0.6047043356243096, median_spearman=0.6048813362557894, macro_rmse=1.572067085802311, n_tissue_with_spearman=54

## 片段活性 (Log2RPM_HEK293)
- valid_rows=117155, spearman=0.5523131000010539, rmse=1.0419960937599653
- valid_groups=794, group_spearman=0.7554064745875672, group_rmse=0.5173368087612409
- group_bootstrap_ci=[0.71705977759205, 0.7890305791745545] valid_replicates=1000

## 派生谱统计
- n_complete_genes=5594
- E_raw: valid_genes=5594, spearman=0.6412414639749526, rmse=1.4416270453475974
- B_raw: valid_genes=5594, spearman=0.5819164330989852, rmse=0.5247967422921156
- S_raw: valid_genes=1717, spearman=-0.020246186250323473, rmse=0.45738284344353747

## DEV 选择
- selected=last
- best: spearman_E=0.6519002295741468, sha256=d19b0bc5c42964bf500497b18fe40f07c17053d0cb829a4fd5b428f65d0a8555
- last: spearman_E=0.655400919875656, sha256=b0a90dddc2325efd3dd5a09822047362d51968865a4bc3ee6ab647ecd626d548

