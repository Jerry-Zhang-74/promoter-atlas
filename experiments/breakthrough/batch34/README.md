# batch34 · 限定用途的片段推理与 293FT 来源核查 / Scoped fragment inference and 293FT source review

[中文](#中文) · [English](#english)

## 中文

为 FASTA 片段推理补充输入检查和来源登记，并检索 293FT 启动子活性实测证据。核对的两篇研究确实使用 293FT 测量启动子活性，但公开材料以成像/Western blot 等读出为主；可用于序列配对定量分析的数据线索转向 MAPT MPRA。

`primary_source_evidence.json` 记录论文、具体测量与后续来源线索；`parm_fasta_predict.py` 是冻结 PARM 推理入口，`test_manifest.json` 索引其输入和输出检查。

- [293FT 原始研究核查](reports/primary_source_evidence.json)
- [片段推理入口](code/batch34/parm_fasta_predict.py)
- [接口检查清单](reports/test_manifest.json)

---

## English

Add input checks and source registration for FASTA fragment inference, and review primary evidence for promoter-activity assays in 293FT. The two inspected studies used 293FT for activity measurement, primarily with imaging/Western-blot readouts. The next sequence-paired quantitative-data lead was MAPT MPRA.

`primary_source_evidence.json` records papers, assays, and the next source lead. `parm_fasta_predict.py` is the frozen PARM entry point; `test_manifest.json` indexes input/output checks.

- [293FT primary-study review](reports/primary_source_evidence.json)
- [Fragment inference](code/batch34/parm_fasta_predict.py)
- [Interface checks](reports/test_manifest.json)
