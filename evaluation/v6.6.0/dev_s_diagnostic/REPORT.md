# DEV 缓存诊断报告

- 范围: DEV_cache_only; 未改动模型选择 (model_selection_changed=false)
- 冻结选择: selected=last, checkpoint_sha256=b0a90dddc2325efd3dd5a09822047362d51968865a4bc3ee6ab647ecd626d548
- 判据: max gene-level E Spearman on human DEV; tie -> last
- 目标单位: 直接 ESB 原始 Score 尺度; profile 为 log2(1+TPM); 未做逆变换
- 门控定义: log2(1+TPM)>=1; S_raw=-median(|det-median(det)|) 当 #det>=3 且 B_raw>=0.3
- bootstrap: n=1000, seed=20261012

## best
- 全直接 S 基因数: 1856, Spearman=0.3836376214165516
- 完整目标 profile 数: 1855
- 真值 profile 派生 S 定义数: 1855
- 预测 profile 派生 S 定义数: 1716
- 配对基因数: 1716, 覆盖率=0.9245689655172413
- 配对 bootstrap: delta=-0.3631477172228126, CI=[-0.42003876277167046, -0.3049736476764211], 有效重复=1000

## last
- 全直接 S 基因数: 1856, Spearman=0.3611538977549096
- 完整目标 profile 数: 1855
- 真值 profile 派生 S 定义数: 1855
- 预测 profile 派生 S 定义数: 1699
- 配对基因数: 1699, 覆盖率=0.9154094827586207
- 配对 bootstrap: delta=-0.3795341360764748, CI=[-0.44460401691534834, -0.32120593487301585], 有效重复=1000

## 跨检查点共同队列
- 共同基因数: 1683
- 直接 S (last-best): {'n_genes': 1683, 'n_boot': 1000, 'valid_replicates': 1000, 'delta': -0.023884755530282853, 'ci_low': -0.03967736316581355, 'ci_high': -0.009238355144136423, 'rho_direct': 0.3845960363306078, 'rho_derived': 0.36071128080032494}
- 派生 S (last-best): {'n_genes': 1683, 'n_boot': 1000, 'valid_replicates': 1000, 'delta': -0.03031223594355205, 'ci_low': -0.06248118218007746, 'ci_high': -4.717173526233015e-06, 'rho_direct': 0.009776986826056303, 'rho_derived': -0.020535249117495744}

## 公式一致性
- best: Spearman=1.0, 相邻逆序=0, 未定义 raw 但有 Score_S=1
- last: Spearman=1.0, 相邻逆序=0, 未定义 raw 但有 Score_S=1

## 歧义归因
- preparation 已知: True

## 完整性说明
- 检查点身份通过 selection<->candidate<->DEV 缓存元数据链验证; 未打开 .pt 文件
- 未读取任何 TEST 输入
