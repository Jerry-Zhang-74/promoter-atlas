# batch12 · 稳定度预测接口冻结 / Frozen stability-prediction interface

[中文](#中文) · [English](#english)

## 中文

将原始稳定度读出整理成固定推理接口：输入 1,069 维组成与位置特征，经保存的变换后由 HGB 输出 S_raw，并提供训练参考分布下的排名解释。

`freeze_manifest.json` 记录模型参数、特征顺序、S 标签适用条件和逆变换；`replay_evidence.json` 记录序列化模型的回放。该版本用于后续工具调用时保持输入顺序和输出单位一致。

- [接口和尺度清单](models/freeze_manifest.json)
- [模型回放](models/replay_evidence.json)

---

## English

Freeze an inference interface for raw stability: 1,069 composition and positional features pass through saved transforms into an HGB predictor of S_raw, with ranking relative to the training reference distribution.

`freeze_manifest.json` records model parameters, feature order, S eligibility, and inverse transformation. `replay_evidence.json` records serialized-model replay. This version provides consistent input ordering and output units for downstream tools.

- [Interface and scale manifest](models/freeze_manifest.json)
- [Model replay](models/replay_evidence.json)
