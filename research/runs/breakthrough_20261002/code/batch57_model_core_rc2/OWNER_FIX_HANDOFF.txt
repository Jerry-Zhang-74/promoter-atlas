# batch57 核心修复与交接 · batch57 core repairs and handoff

[中文](#中文) · [English](#english)

## 中文

### 修订与兼容性

`v6.38.1-model-core-critical-repair.rc2` 对 batch53 审查 D1–D12 进行修复，模块/API 和 config SHA `12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157` 保持兼容，可读取在训 checkpoint。交接时原台账已用 4/4（batch52-R0、R0、Nabs、Njoint），运行状态按实际台账读取。

- D1：四个独立 FILETIME 缓冲区读取 PID/OS 创建时间，存活进程持锁，释放时验证身份。
- D2：续训恢复尚待实现，`--resume-*` 在预约前抛出 ResumeUnsupported；重启执行占用新槽位。
- D3：固定方向基线选择从 min 改为 max。
- D4：SOURCE_SHA256 使用真实 64 位哈希 `1bbe5db7...d96`，SOURCE_SHA1 单列。
- D5：要求并核对 `--expected-acceptance-sha`。
- D6：actual_status 动态读取台账与 RUN_POLICY 的 12:00/13:00/20:00 时间要求。
- D7：Njoint 首梯度前核对 Nabs init_sha。
- D8：top10 使用冻结坐标/ID 解平局。
- D9：记录真实 opt_steps，样本数另列。
- D10：逐 epoch 确定性配对排列及重复次数。
- D11：选中 checkpoint 附 `.sha256.json`。
- D12：拒绝列数不齐，解码 NPZ byte ID 并按声明长度切片；歧义碱基默认拒绝，allow_ambiguous=True 时按有效碱基长度处理。

### 应用方式与检查

锁、训练、台账及 checkpoint 修复在导入新代码的下一进程生效；D3/D8/D12 的 CPU 评估可单独使用。交接时在训任务走 fresh-fit 路径，按原执行继续完成。模型权重与当前进程独立保留，TEST 结果由评估流程管理。

将 HELPER 指向 batch57_model_core_rc2；协调器新增 `--expected-acceptance-sha <sha>`，其他入口保持 data_adapter、train_neural、r0_baseline、eval_replay、pipeline_coordinator、checkpoint、epoch_cap、gpu_lock、exec_gate。

`python selfcheck_rc2.py` 输出 reports/.../batch57/code_selfcheck.json：test_nofit_rc2 为 21/21，test_pipeline_rc2 为 6/6，含 checkpoint ABI 读取和人工前向。batch53 的 r0_ridge_one_solve_and_delta 测试实际调用过 np.linalg.solve；本修复测试使用纯软件检查。CODE_READY.json 记录文件哈希，续训恢复是后续工作。

---

## English

### Repairs and compatibility

`v6.38.1-model-core-critical-repair.rc2` addresses batch53 review findings D1–D12. Module/API and config SHA `12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157` remain compatible with live checkpoints. The ledger snapshot at handoff was 4/4 consumed (batch52-R0, R0, Nabs, Njoint); read actual ledger state for current status.

- D1: four distinct FILETIME buffers establish PID/OS-creation identity; live holders retain the lock and release checks ownership.
- D2: restore remains pending; `--resume-*` raises ResumeUnsupported before reservation, and restarted executions consume a new slot.
- D3: fixed-direction baseline selection changes min to max.
- D4: SOURCE_SHA256 uses the actual 64-hex hash `1bbe5db7...d96`, with SOURCE_SHA1 separate.
- D5: require and verify `--expected-acceptance-sha`.
- D6: actual_status reads the live ledger and RUN_POLICY 12:00/13:00/20:00 deadlines.
- D7: Njoint verifies Nabs init_sha before its first gradient.
- D8: top10 uses frozen coordinate/ID tie-breaking.
- D9: record actual opt_steps separately from sample count.
- D10: deterministic per-epoch pair ordering and repeat counts.
- D11: selected checkpoints receive `.sha256.json` sidecars.
- D12: reject ragged rows, decode NPZ byte IDs, slice declared lengths, and handle ambiguous bases consistently; default rejection, or valid-base length with allow_ambiguous=True.

### Integration and checks

Lock, training, ledger, and checkpoint repairs activate in the next process importing the new code. D3/D8/D12 CPU evaluation changes can be used independently. The active run at handoff followed the fresh-fit path and continued its existing execution. Weights and the current process were preserved; TEST results stayed under evaluator control.

Point HELPER to batch57_model_core_rc2 and add `--expected-acceptance-sha <sha>` to the coordinator. Other entry points remain data_adapter, train_neural, r0_baseline, eval_replay, pipeline_coordinator, checkpoint, epoch_cap, gpu_lock, and exec_gate.

`python selfcheck_rc2.py` writes reports/.../batch57/code_selfcheck.json: test_nofit_rc2 21/21 and test_pipeline_rc2 6/6, including checkpoint ABI loading and artificial forward. The batch53 r0_ridge_one_solve_and_delta test called np.linalg.solve; this repaired suite uses software-only checks. CODE_READY.json records file hashes, and checkpoint restoration remains follow-up work.
