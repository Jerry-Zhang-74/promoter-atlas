# batch87 · 执行器模型配置错误记录 / Executor model-configuration errors

[中文](#中文) · [English](#english)

## 中文

这里保存 batch87 的执行端与审查端调度错误。日志返回 `unrecognized_model`，所请求的模型为 `deepseek-flash`；该信息用于定位当时执行器的模型配置问题。

执行端的错误来源为 `compact`，审查端为 `sdk`，两个原始日志分别保留。

- [reviewer.stderr.log](runtime/control_87/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_87/worker.stderr.log)

---

## English

This directory retains batch87 worker and reviewer dispatch errors. The logs return `unrecognized_model` for `deepseek-flash`, identifying the executor model-configuration issue at that time.

The worker error is attributed to `compact` and the reviewer error to `sdk`; both original logs are retained.

- [reviewer.stderr.log](runtime/control_87/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_87/worker.stderr.log)
