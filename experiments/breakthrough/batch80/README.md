# batch80 · 执行器模型配置错误记录 / Executor model-configuration errors

[中文](#中文) · [English](#english)

## 中文

这里保存 batch80 的执行端与审查端调度错误。日志返回 `unrecognized_model`，所请求的模型为 `deepseek-flash`；该信息用于定位当时执行器的模型配置问题。

- [reviewer.stderr.log](runtime/control_80/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_80/worker.stderr.log)

---

## English

This directory retains batch80 worker and reviewer dispatch errors. The logs return `unrecognized_model` for `deepseek-flash`, identifying the executor model-configuration issue at that time.

- [reviewer.stderr.log](runtime/control_80/reviewer.stderr.log)
- [worker.stderr.log](runtime/control_80/worker.stderr.log)
