# batch35 · MAPT 队列准备与片段推理测试 / MAPT cohort preparation and fragment-inference tests

[中文](#中文) · [English](#english)

## 中文

跟进 MAPT MPRA 的设计序列和数据可用性，整理后续匹配实测标签所需的文件、标识和长度信息。同时冻结片段推理的代码、权重引用和输入检查资产。

`mapt_readiness.py` 实现队列准备核对，`build_frozen_assets.py` 汇总推理资产，测试清单和独立哈希检查保留执行结果。下一批从补充表恢复对应的实测变体效应。

- [MAPT 数据准备](code/batch35/mapt_readiness.py)
- [推理测试清单](reports/test_manifest.json)
- [资产校验](reports/verification/sha_verify_result.json)

---

## English

Follow the MAPT MPRA design sequences and data availability, organizing files, identifiers, and lengths needed to join measured labels. Freeze fragment-inference code, weight references, and input-validation assets in parallel.

`mapt_readiness.py` checks cohort preparation; `build_frozen_assets.py` assembles inference assets. Test manifests and independent checksum records retain execution results. The next batch recovers measured variant effects from supplementary tables.

- [MAPT data preparation](code/batch35/mapt_readiness.py)
- [Inference test manifest](reports/test_manifest.json)
- [Asset verification](reports/verification/sha_verify_result.json)
