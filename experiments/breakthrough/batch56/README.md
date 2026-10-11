# batch56 · 训练前关键条件检查 / Critical training preflight checks

[中文](#中文) · [English](#english)

## 中文

把训练前容易影响正确性的条件集中为可执行检查，针对模型核心、数据和运行约束进行预检。代码入口为 `check_critical_preflight.py`，相邻 `selfcheck.json` 保存该实现的检查结果。

此目录用于理解后续核心修复的检查依据；训练实现的修订与兼容性说明见 batch57。

- [关键条件检查](code/batch56_checks/check_critical_preflight.py)
- [检查结果](code/batch56_checks/selfcheck.json)
- [核心修复](../batch57/README.md)

---

## English

Collect correctness-sensitive conditions into executable preflight checks for the model core, data, and runtime constraints. `check_critical_preflight.py` is the entry point; the adjacent `selfcheck.json` retains its check results.

These records explain the checks leading into the subsequent core repairs. Batch57 describes the implementation changes and compatibility.

- [Critical preflight](code/batch56_checks/check_critical_preflight.py)
- [Check results](code/batch56_checks/selfcheck.json)
- [Core repairs](../batch57/README.md)
