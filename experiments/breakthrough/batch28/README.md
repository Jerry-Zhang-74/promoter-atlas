# batch28 · SuRE 全表计数检查与无标签抽样 / SuRE count census and label-free sampling

[中文](#中文) · [English](#english)

## 中文

运行日志保存了 SuRE 原始计数表的扫描、列检查和计数相关性诊断。`colprobe.log` 记录扫描 153,562,178 行；`passA.log` 保存文库总量以及按坐标哈希选择 25,000 条合格记录的规则。

`correl.log` 检查 iPCR、质粒、cDNA 与片段长度的关系，为下一批固定 RNA/DNA 代理目标提供依据。调度器错误另存于 control_28，分析日志可独立阅读。

- [扫描与抽样记录](runtime/batch28/passA.log)
- [计数相关性](runtime/batch28/correl.log)

---

## English

Runtime logs retain the SuRE raw-count census, column checks, and count-correlation diagnostics. `colprobe.log` records scanning 153,562,178 rows; `passA.log` stores library totals and the coordinate-hash rule selecting 25,000 eligible records.

`correl.log` examines iPCR, plasmid, cDNA, and fragment-length relationships, informing the fixed RNA/DNA proxy target in the following batch. Controller errors are stored separately under control_28.

- [Census and sampling](runtime/batch28/passA.log)
- [Count correlations](runtime/batch28/correl.log)
