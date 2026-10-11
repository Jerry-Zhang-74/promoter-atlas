# batch58 吞吐与续训变体 · batch58 throughput and resume variant

[中文](#中文) · [English](#english)

## 中文

版本 `v6.39.0-throughput-checkpoint.rc1` 改变有效优化器 batch，研究资源配置与续训。数据为修正队列全部 531860 TRAIN 行，目标 Log2RPM_HEK293；沿用 batch57 架构、fp32、学习率、损失、seed 和逐 epoch 顺序。batch 变化同时改变每 epoch 优化器步数。

### 实现与执行

b58_gate 管理独立上限 3 的台账；_exec_gate_vendored 固定已审核 gate；b58_core 实现 v2 checkpoint、两代保留、停止信号和 OOM 恢复；b58_probe 测试 batch32..2048 的前向/反向吞吐；b58_run 等待 GPU 并执行 profile/train/resume；selfcheck_b58 检查单元行为；smoke_b58 检查人工 stop/resume；dryrun_b58 检查真实数据小切片调用；freeze_code_ready 固定文件哈希；finalize_b58 记录结果；b58_manifest 管理产物清单。

三次执行均计入台账：b58-profile（梯度吞吐检查）、b58-Nabs-throughput-variant（epoch0 内受控停止）、b58-resume-continuation（新进程恢复并完成 2 epochs）。batch52 的 batch32 比较单独保留。

### 续训语义

`batch58.checkpoint.v2` 保存权重、AdamW、Python/NumPy/CPU/CUDA RNG、normalizer、loglen/GC 状态、epoch、next_pos、顺序 seed、配对指针、历史、best、优化器步数、config/data 哈希和设备。next_pos 指向已处理批次之后；旧 b57 的 pos 含义不同，加载时校验 schema。

恢复时重建顺序并核对前后批次 ID 与覆盖前缀哈希；先写 SHA/sidecar 再原子更新 latest.json，保留最近两代。SIGINT/SIGTERM/stopfile 在完整优化器保存边界退出；CUDA OOM 恢复步前状态/RNG，缩小 microbatch 重试并记录。

### GPU 条件与结果

等待原训练 PID/创建时间结束、Nabs_ready 和 Njoint_ready、batch52/GPU_EVALUATION_DONE 齐备且计算设备空闲，取得 batch52 共享锁后执行。

2026-10-04 10:12 修订以 samples/second（batch/median_s）选 batch，要求有效显存测量并加入评估交接条件。单元例子 batch32@0.010s 与 batch128@0.020s 选 128。

实测选择 batch64，显存峰值约 2.8 GiB；原目标为 6.7–7.1 GiB。batch2048 使用 7.61 GiB、仅剩 0.015 GiB，低于 0.7 GiB 余量，单步约慢 22 倍。结果记录于 reports/.../batch58/throughput_analysis.json。

---

## English

Version `v6.39.0-throughput-checkpoint.rc1` changes effective optimizer batch to study resource configuration and resume. It uses all 531860 corrected-cohort TRAIN rows, targeting Log2RPM_HEK293, with the batch57 architecture, fp32, learning rate, loss, seed, and epoch ordering. Changing batch also changes optimizer steps per epoch.

### Implementation and executions

b58_gate manages a separate three-slot ledger; _exec_gate_vendored pins the reviewed gate; b58_core implements v2 checkpoints, two-generation retention, stop signals, and OOM restore; b58_probe measures forward/backward throughput at batch32..2048; b58_run waits for GPU and drives profile/train/resume; selfcheck_b58 tests units; smoke_b58 tests artificial stop/resume; dryrun_b58 checks a small real-data slice; freeze_code_ready pins hashes; finalize_b58 records results; b58_manifest manages artifacts.

All three executions consumed slots: b58-profile (gradient throughput), b58-Nabs-throughput-variant (controlled stop inside epoch0), and b58-resume-continuation (new-process restore completing two epochs). The batch52 batch32 comparison is retained separately.

### Resume semantics

`batch58.checkpoint.v2` stores weights, AdamW, Python/NumPy/CPU/CUDA RNG, normalizer, loglength/GC state, epoch, next_pos, order seed, pair pointer, history, best, optimizer steps, config/data hashes, and device. next_pos points after the processed batch; the older b57 pos has different semantics, so loading validates the schema.

Resume rebuilds ordering and checks next/last batch IDs and a coverage-prefix hash. Write SHA/sidecar before atomically updating latest.json, retaining two valid generations. SIGINT/SIGTERM/stopfile exit at fully saved optimizer boundaries. CUDA OOM restores pre-step state/RNG, reduces microbatch, and records retries.

### GPU conditions and results

Wait for the original trainer PID/creation identity to end, Nabs_ready, Njoint_ready, batch52/GPU_EVALUATION_DONE, and an available compute device, then acquire the batch52 shared lock.

The 2026-10-04 10:12 repair selects by samples/second (batch/median_s), requires valid memory measurements, and adds evaluation-handoff checks. The unit example selects batch128@0.020s over batch32@0.010s.

Measured selection was batch64 at approximately 2.8 GiB peak, compared with the requested 6.7–7.1 GiB. Batch2048 used 7.61 GiB with 0.015 GiB free, below the 0.7 GiB margin, and was about 22 times slower per step. Results are in reports/.../batch58/throughput_analysis.json.
