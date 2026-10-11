# -*- coding: utf-8 -*-
"""batch45: build metric_definitions, report.md (ZH), status, selfcheck, next_proposal."""
import json, sys, pickle, statistics, hashlib, math
from pathlib import Path
sys.path.insert(0, str(Path(r"E:/promoter_atlas/runs/breakthrough_20261002/code/batch45")))
import b45_common as C
OUT=C.OUT
def rj(n): return json.loads((OUT/n).read_text(encoding="utf-8"))
def wj(n,o): (OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
fr=rj("freeze_integrity.json"); idm=rj("identity_metadata.json"); agg=rj("aggregation.json")
prim=rj("primary_matched_lineages.json"); sec=rj("secondary_reliability.json")
ver=rj("verification.json"); mut=rj("table_mutated_vs_strongest.json"); dq=rj("data_quality.json")
cat=rj("table_category_summary.json")
defs={
 "source":"data/breakthrough_20261002/batch44/syn.txt.gz  sha256="+fr["sha256_compressed"],
 "denominators":{"source_rows":idm["n_data_rows"],"unique_query_DNA":idm["n_unique_query_seq"],
   "unique_barcode":idm["n_unique_barcode"],"barcodes_per_fragment":idm["barcodes_per_fragment_dist"],
   "library_unique_query":idm["library_unique_query"],"category_summary":cat["summary"]},
 "aggregation":agg["fixed_definition"],
 "metrics":{
   "delta_K562_predicted_score":"published PARM K562 prediction at generation G minus at generation 0 (source score units)",
   "delta_K562_measured_summary":"fixed measured activity summary (mean of the 2 replicate means of author normalized.log2rpm) at G minus at 0 (log2 RNA/DNA units)",
   "spearman_rho":"tie-aware rank corr between source prediction and measured summary",
   "gc_baseline_rho":"rank corr between query-DNA GC fraction and measured summary",
   "top10_precision_enrichment":"k=ceil(0.1n); precision=overlap/k; enrichment=precision/(k/n); lexical-id tie rule; only n>=30",
   "replicate_spearman":"rank corr between replicate-1 and replicate-2 measured summaries (measurement diagnostic)"},
 "unit_warning":"prediction and experiment have different units; NO MAE/R2/calibration computed; negative prediction not log-transformed; no sign flip.",
 "independence_warning":"889 DNA are NOT 889 independent families (shared genetic-optimization ancestry + model-selected candidates); 3556 barcodes and 2 RNA columns are not independent n."}
wj("metric_definitions_denominators.json",defs)
print("defs ok")

def g(d,*ks):
    for k in ks: d=d[k]
    return d
sp=prim["per_library"]["short"]; lp=prim["per_library"]["long"]
def fmt(d): return "median %+.2f (min %+.2f, max %+.2f; 正 %d/负 %d)"%(d["median"],d["min"],d["max"],d["n_positive"],d["n_negative"])
L=[]
L.append("# batch45 v6.32.0-synthetic-design-utility.rc1 — 已发表合成启动子设计研究的回顾性再分析\n")
L.append("> 性质：对作者**已发表**的预测列与真实测量活性做**固定聚合/元数据配对**的回顾性分析；不是我们模型的新预测，也不是独立验证。ZERO 拟合/GPU/新预测。\n")
L.append("## A. 数据冻结（活动统计之前完成）")
L.append("- 来源文件：`data/breakthrough_20261002/batch44/syn.txt.gz`（官方名 `%s`，GSE301246 suppl 目录 376K）。"%fr["official_filename"])
L.append("- 完整性：compressed=%d B, decompressed=%d B, gzip EOF/CRC ok=%s, sha256=%s。"%(fr["compressed_bytes"],fr["decompressed_bytes"],fr["gzip_crc_ok"],fr["sha256_compressed"]))
L.append("- 结构：%d 行 × 19 列；%d 个唯一 query DNA，%d 个唯一 info，1:1；%d 个唯一 barcode，每 DNA 恰好 4 个 barcode。"%(idm["n_data_rows"],idm["n_unique_query_seq"],idm["n_unique_info"],idm["n_unique_barcode"]))
L.append("- 序列同一性：query.seq 在每行都是 full.oligo 的精确子串（偏移 18），即 full.oligo = 18bp 5' 侧翼 + query.seq + 50bp 3' 侧翼；侧翼不作 adapter/参考DNA/RC 猜测。")
L.append("- 预测一致性：每个 query 的 K562/HepG2 PARM 预测在 4 个 barcode 行间完全一致（冲突 0）——无冲突需降级。")
L.append("- 230 vs 232：长库名为 `long.232bp`（Methods 中 GA 个体名义长度 232bp）；但实测**合成**长插入（design/best/MUTATED）为 **230bp**，长库中的 **42 个天然对照**为 **232bp**。差异显式保留，未 padding/clip；标签究竟指哪一长度从缓存证据不能唯一判定。")
L.append("- RNA 重复语义：列中 replicate1/2 为两次独立（生物学）转染重复（Methods）；info 里的 `replicate00..09` 是 10 条**遗传算法搜索谱系**，**不是**生物学重复。K562 为设计目标（Methods 证实）。")
L.append("- 数据质量：无精确重复行；仅 1 条 barcode（0 原始 RNA 计数）产生 -Inf，已在原始表保留、在有限算术均值中排除并计数。")
L.append("## B. 主分析：原生 K562 设计轨迹（metadata-only 冻结，activity-blind）")
L.append("- 规则：gen0 selection=max 对 **所有未修饰谱系共同存在的最大 generation** 的 max；同库同长度；MUTATED 排除。")
L.append("- 短库 G=48，长库 G=44；每库 10 条谱系全部配对（含负效应/缺失partner已如实计数）。")
L.append("- 短库 K562：Δ预测 %s；Δ实测 %s。"%(fmt(sp["delta_K562_predicted_score"]),fmt(sp["delta_K562_measured_summary"])))
L.append("- 长库 K562：Δ预测 %s；Δ实测 %s。"%(fmt(lp["delta_K562_predicted_score"]),fmt(lp["delta_K562_measured_summary"])))
L.append("- 判读(i)：在固定的原生 K562 设计轨迹上，**评分增益与实测表达增益同向**（两库均 10/10 谱系为正）；这是对**作者已选研究**的回顾，含模型选择偏倚，**不构成我们模型的新进展**。")
L.append("## C. 次要分析：可靠性/效应（全部合格唯一 DNA）")
for k,v in sec["by_library_cell_subgroup"].items():
    t=v.get("top10",{})
    L.append("- %s：n=%d，预测-实测 rho=%.3f，GC 基线 rho=%.3f，top10 precision=%s，enrichment=%s。"%(
        k,v["n_unique_DNA"],v["pred_vs_measured"]["rho"],v["gc_vs_measured_baseline"]["rho"],
        ("%.3f"%t["precision"]) if "precision" in t else "n/a", ("%.2f"%t["enrichment"]) if "enrichment" in t else "n/a"))
L.append("- 测量重复一致性（K562/HepG2，短/长）："+"; ".join("%s rho=%.3f"%(
    k,sec["replicate_diagnostics"][k]["spearman_rep1_vs_rep2"]) for k in sec["replicate_diagnostics"])+"（诊断量，非预测上限/生物学CI）。")
L.append("- MUTATED vs 同谱系最强（描述性、非主分析）：21 对，Δ实测 中位数 %+.2f，负向 %d/正向 %d。"%(mut["median_delta_K562_meas"],mut["n_delta_meas_negative"],mut["n_delta_meas_positive"]))
L.append("- 判读(ii)：HepG2（跨细胞）在同一库同样显示正向增益（合成 rho 低于 K562），**支持方向可迁移**，但不作因果/等强声明。")
L.append("## D. 证据范围与局限")
L.append("- 889 DNA **不是** 889 个独立家族（共享遗传优化祖先+模型筛出的候选）；3556 barcode、2 RNA 列不是独立 n。")
L.append("- 合成集的相关性被设计选择（gen0 随机→gen48 优化）抬高；对照与合成分开报告，未把对照并入合成性能。")
L.append("- 预测列是**分析输入**而非我们的新模型输出；本批不改变 D28FAIL / HEK 迁移偏弱 / 2000bp 不支持 的既有结论。")
L.append("- 判读(iii)：可面向**用户工具**所用的证据是“(i) 设计轨迹中评分与实测同向 + (ii) HepG2 方向可迁移”；而对**293FT** 仍为**未验证**。")
L.append("## E. 下一步（唯一有界动作）")
L.append("- 动作：把本批的**固定度量与分母规范**用于**用户自有 293FT 构建记录**的可追溯打分（输入=已存在记录，非新下载/非新预测）。")
L.append("- 增益判据：若能在 ≥10 个独立 293FT 构建上得到预测-实测 Spearman 及其 GC 基线对照（predeclared），则记为 293FT 可预测性证据。")
L.append("- 停止条件：若 293FT 记录缺序列-构建-活性可追溯配对，则如实记为 UNKNOWN 并停止该子路线。")
(OUT/"report.md").write_text("\n\n".join(L),encoding="utf-8")
print("report.md bytes", (OUT/"report.md").stat().st_size)
