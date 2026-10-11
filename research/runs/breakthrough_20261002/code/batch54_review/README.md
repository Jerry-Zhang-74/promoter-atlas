# batch54 独立评估 · batch54 independent evaluation

[中文](#中文) · [English](#english)

## 中文

版本 `v6.38.0-first-model-independent-acceptance.rc1`，使用 1 CPU 核、≤2 GB RAM，代码和报告分别写入 batch54_review 与 reports/.../batch54。

`indep_metrics.py` 独立计算 pooled Spearman、组等权 MAE、原单位 RMSE、按稳定 ID 解平局的 top-k、配对组 MAE、macro 严格符号准确率（真零分列，预测平局计错）及保留抽样重复次数的分组 bootstrap（至少 30 组）。`indep_cohort.py` 读取 cohort TSV。

`test_fixtures.py` 检查重复 bootstrap 组、常数预测、NA、真零/平局、固定 ID 和穷举一致性。`model_checks.py` 核对 ready、checkpoint SHA、config SHA、datahash、目标、normalizer、全量 TRAIN 覆盖、DEV 选择、初始字节、执行台账和合成前向。

`run_acceptance.py` 提供 fixtures/r0/metrics/checks/accept；`assemble_acceptance.py` 输出 acceptance.json 和 report.md；`wait_for_modelA.py` 每至多 30 秒检查 Nabs_ready，历史等待截止 12:20 Asia/Shanghai。

GPU 权重回放在独占短槽位（≤60 秒）取得 PID+创建时间锁后执行。TEST 标签物理上位于 endpoints.tsv，隔离通过流程管理。序列为 HG02601 的参考代理，测定来源为 HEK293，293FT 迁移待评估。记录包括 fixture_evidence、provenance_evidence、modelA_wait.log、modelA_wait_event 和可选 code_selfcheck。

---

## English

Version `v6.38.0-first-model-independent-acceptance.rc1` uses one CPU core and ≤2 GB RAM, writing code and reports under batch54_review and reports/.../batch54.

`indep_metrics.py` independently calculates pooled Spearman, group-equal MAE, original-unit RMSE, stable-ID-tied top-k, pair group MAE, macro strict sign accuracy (true zeros separate, predicted ties incorrect), and grouped bootstrap retaining sampling multiplicity (at least 30 groups). `indep_cohort.py` reads cohort TSVs.

`test_fixtures.py` covers repeated bootstrap groups, constant predictions, NA, true zeros/ties, fixed IDs, and brute-force agreement. `model_checks.py` checks ready markers, checkpoint/config SHAs, datahash, target, normalizer, full TRAIN coverage, DEV selection, initial bytes, execution ledger, and synthetic forward.

`run_acceptance.py` exposes fixtures/r0/metrics/checks/accept; `assemble_acceptance.py` writes acceptance.json and report.md; `wait_for_modelA.py` checks Nabs_ready at intervals up to 30 seconds, with the historical deadline 12:20 Asia/Shanghai.

GPU replay uses an exclusive short slot (≤60 seconds) and PID-plus-creation-time lock. TEST labels are physically present in endpoints.tsv and separated procedurally. Sequences are reference proxies for HG02601; measurements are HEK293, and 293FT transfer awaits evaluation. Records include fixture_evidence, provenance_evidence, modelA_wait.log, modelA_wait_event, and optional code_selfcheck.
