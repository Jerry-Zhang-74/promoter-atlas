# batch44 · 执行器模型配置错误记录 / Executor model-configuration errors

[中文](#中文) · [English](#english)

## 中文

这里保存 batch44 的执行端与审查端调度错误。日志返回 `unrecognized_model`，所请求的模型为 `deepseek-flash`；该信息用于定位当时执行器的模型配置问题。

另有 operator 启动记录，时间为 2026-10-04 03:22。后续合成启动子文库的实际分析见 [batch45](../batch45/README.md)。

- [reviewer.stderr.log](runtime/control_44/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_44/worker.stderr.log)

---

## English

This directory retains batch44 worker and reviewer dispatch errors. The logs return `unrecognized_model` for `deepseek-flash`, identifying the executor model-configuration issue at that time.

An operator launch is recorded at 03:22 on 2026-10-04. The subsequent synthetic-promoter library analysis is in [batch45](../batch45/README.md).

- [reviewer.stderr.log](runtime/control_44/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_44/worker.stderr.log)
