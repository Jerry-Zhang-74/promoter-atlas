# batch01 · 执行器模型配置错误记录 / Executor model-configuration errors

[中文](#中文) · [English](#english)

## 中文

这里保存 batch01 的执行端与审查端调度错误。日志返回 `unrecognized_model`，所请求的模型为 `deepseek-v4-pro`；该信息用于定位当时执行器的模型配置问题。

- [reviewer.stderr.log](runtime/control_01/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_01/worker.stderr.log)

---

## English

This directory retains batch01 worker and reviewer dispatch errors. The logs return `unrecognized_model` for `deepseek-v4-pro`, identifying the executor model-configuration issue at that time.

- [reviewer.stderr.log](runtime/control_01/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_01/worker.stderr.log)
