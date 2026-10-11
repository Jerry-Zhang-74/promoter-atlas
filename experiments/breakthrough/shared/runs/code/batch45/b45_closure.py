# -*- coding: utf-8 -*-
"""batch45 phase D: b44 source-closure note (preserve, no re-audit) + remaining_work."""
import json, datetime
from pathlib import Path
OUT=Path("E:/promoter_atlas/reports/breakthrough_20261002/batch45")
now=datetime.datetime.now().astimezone().isoformat()
note=f"""# batch44 来源路线收尾记录（batch45 写入，保留原失败，不回溯改旧）

生成时间: {now}  (batch45 v6.32.0-synthetic-design-utility.rc1)

本记录**只**为遵守 PARENT_ACCEPTANCE 的要求补齐 b44 的 source-closure，**不改写** batch44 任何旧产物，也不重新审计百万行或旧计数。

## 1. b44 合规失败（保留 FAIL）
- 失败性质是**执行/选择失败**，不是“公开数据整体不存在”。
- b44 在未知 schema 前就选择了较小的 **v1_pHY3** 分片，实际是 **HAP1/AGS** 标签，不是目标 **HEK293**。
- 已产生的字节/请求/尝试按真实日志记录；最终 gzip CRC/解析有效**不**等于请求/尝试预算全部合规。
- 结论：**compliance FAIL 保留**；不得以“最终文件能解析”豁免失败尝试。

## 2. 仍缺失的证据（UNKNOWN，未通过）
- **实际供体 DNA**：GSE301246 未分发供体 HG02601 的真实插入 DNA；用参考基因组重建的序列只是**坐标重建**，不是**实测供体 DNA**。
- **union 分组**：b44 的 (chrom,strand,end) 3′端同键**不是** union-parent 分组；100/20/30 门**仍为 UNKNOWN**，未被 331837 同端键“满足”。
- 独立父本端点未确立；依赖端点不是独立父本。

## 3. 当前预算算术（不重置、不充值）
- 基准：原始 **142.67** CNY + 已观测 **100.00** CNY topup = **242.67** CNY 记账基准。
- 最新 manifest（control_53/status.json）：prior_spent=163.66，spent≈163.83，balance≈78.84，cumulative_topup=100.00。
- 不充值、不重置；低余额<=20 才触发一次提醒（本批未触发）。

## 4. 本批边界
- 未新增下载/审计；仅使用 **已存在** 的 `data/breakthrough_20261002/batch44/syn.txt.gz` 与缓存的 primary fulltext/listing。
- 本批把“是否在测量中实现设计增益”的**不同**问题在**同一已下载数据**上回答完毕（见 report.md）。
"""
(OUT/"source_closure_b44.md").write_text(note,encoding="utf-8")
rem={"batch":"batch45","generated_at":now,"remaining_work":[
 {"id":"293FT_traceable_pairing","status":"open","desc":"用户自有 293FT 构建的 sequence->construct->activity 可追溯配对；决定 293FT 可预测性能否被检验"},
 {"id":"b44_union_grouping","status":"UNKNOWN_closed","desc":"真实供体 DNA 与 union-parent 分组仍缺；100/20/30 门 UNKNOWN；该 branch 已关闭，不重启同源获取"},
 {"id":"230_vs_232_label","status":"UNKNOWN","desc":"长库标签指 230/232 哪一长度从缓存证据不能唯一判定；已显式保留"},
 {"id":"293FT_not_validated","status":"unchanged","desc":"D28FAIL / HEK 弱迁移 / 2000bp 不支持 未被本批改变"}],
 "next_bounded_action":"batch46 v6.33.0-293FT-traceable-scoring.rc1（见 next_batch_proposal.json）"}
(OUT/"remaining_work.json").write_text(json.dumps(rem,ensure_ascii=False,indent=2),encoding="utf-8")
print("closure + remaining_work written")
