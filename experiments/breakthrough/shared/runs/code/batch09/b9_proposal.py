#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 next-batch proposal: BATCH10_DRAFT.txt + next_batch_proposal.json.

One bounded, distinct, evidence-supported question. batch09 diagnosed that target
representation is not the lever; the residual S signal gap is most plausibly in
the feature representation (global composition/count only, no regulatory grammar)
or in target noise. The discriminating, CPU-bounded next step is a deterministic
richer sequence-feature representation on the same frozen split/eligibility.
"""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import REPORT_OUT, CANDIDATE_VERSION, atomic_write_json, atomic_write_text
from datetime import datetime, timezone, timedelta

CN = timezone(timedelta(hours=8))


def now_cn():
    return datetime.now(CN).strftime("%Y-%m-%dT%H:%M:%S%z")


def main():
    proposal = {
        "batch_id": "batch10",
        "parent_batch": "batch09",
        "parent_result": ("v6.10.0-starget.rc1: integrity PASS; NO scoped target-representation "
                          "improvement (rank-vs-raw deltas +0.014 Ridge / +0.012 HGB, both <0.03)"),
        "proposed_candidate_id": "v6.11.0-sfeatgap.rc1",
        "question_type": "S feature-gap diagnosis (bounded CPU; distinct from batch09 target-representation diagnosis)",
        "question": ("在冻结的 batch05 修正组件切分与 18769 个合格 S 定义基因上，加入确定性的、"
                     "更丰富的序列特征表示（位置感知 k-mer/基序计数，基因级平均，CPU、无神经训练），"
                     "是否能在同一测试行上把 S 预测的测试 Spearman / 目标单位 R2 提升到 batch09 的 "
                     "45 维组成+计数基线之上？即：剩余 S 信号是“特征表示缺口”还是“目标噪声不可约上限”。"),
        "evidence_backing": {
            "from_batch09": {
                "target_representation_not_lever": ("rank 目标只比 raw 目标高 +0.014(Ridge)/+0.012(HGB)，"
                                                    "均 <0.03 阈值；HGB 校正区间含 0"),
                "modest_but_nonzero_signal": ("组成/计数 45 维取得测试 Spearman 0.370–0.421、R2 0.112–0.174，"
                                              "显著高于 train 均值常数基线(R2≈0)，说明存在可学信号"),
                "remaining_gap": ("全局组成/计数不含调控语法与位置信息；Spearman≈0.4 意味着大部分 S 方差"
                                  "仍未被解释——缺口最可能在特征表示或目标噪声"),
            },
        },
        "frozen_design": {
            "population": "18769 个合格 S 定义基因（batch05 切分不变，train/val/test = 15004/1898/1867）",
            "baseline_reference": "batch09 已冻结的 45 维组成/计数特征与四臂预测（同一测试行，读-only 复用）",
            "new_features": ("确定性、CPU-only 的丰富序列特征：位置感知 k-mer 频率(k=3,4，必要时 k=5) 与"
                             "位置特异的二核苷酸频率（对 400bp 启动子按位置对齐，2000bp 上游按分箱位置），"
                             "基因级均值聚合；仅用序列，不含任何 E/S/B 标签"),
            "models": "复用 batch09 冻结设计：Ridge(alpha=0.1) 与 HGB(同超参)，单种子 20261011，",
            "targets": "连续 raw S 与训练参考经验秩 F（与 batch09 完全相同的标准化/逆变换），",
            "evaluation": ("主判据 = 测试 Spearman 对共同全精度 raw S；次判据 = 目标单位 R2；"
                           "新特征 vs batch09 基线的差值用组件配对 bootstrap（97.5%% 双侧百分位区间）"),
        },
        "resource_budget": {
            "cpu": "<= 4 cores, <= 45 minutes",
            "gpu": "0 minutes",
            "no_search": "无超参/架构/阈值搜索；无神经训练；无外部数据库或网络",
            "notes": "不 topup；保留全部既有工件；不改变主 v6.3 / 细胞系 v6.4 指针",
        },
        "deliverables": [
            "中文报告 HTML+text（候选版本 + 任务旁注指标）",
            "新特征构建代码与 SHA、新特征矩阵 SHA",
            "同测试行上的新特征 vs batch09 基线比较表与 bootstrap 区间",
            "冻结 config + SHA registry + 保存模型/重放示例",
            "next_batch_proposal.json + BATCH11_DRAFT.txt",
        ],
        "unsupported_scope": [
            "不训练任何深度/神经网络，不使用 GPU",
            "不声称更好的 S 预测器或生物学验证",
            "不改变主模型 v6.3 / 细胞系 v6.4 指针",
            "不重复等价重标定或机械调参",
            "不做哺乳动物通用启动子强度 / 合成启动子活性 / HEK293T 构建 / 时间稳定性声明",
        ],
        "why_distinct_from_batch09": ("batch09 诊断目标表示（raw vs rank）不是杠杆；batch10 诊断特征表示"
                                      "缺口 vs 目标噪声上限，是全新的自变量方向，非等价重标定、非机械调参。"),
    }
    atomic_write_json(REPORT_OUT / "next_batch_proposal.json", proposal)

    draft = """==============================================================================
BATCH10 DRAFT（v6.11.0-sfeatgap.rc1）：S 特征缺口诊断（丰富序列特征 vs 组成/计数基线）
==============================================================================
状态：草案（待父级审批/冻结）
来源：batch09 v6.10.0-starget.rc1 完整性 PASS；无目标表示改进（rank-vs-raw 差值
+0.014/+0.012 均 <0.03）。目标表示不是杠杆，剩余 S 信号缺口最可能在特征表示。

------------------------------------------------------------------------------
1. 问题（ONE question，与 batch09 不同）
------------------------------------------------------------------------------
batch09 证明：连续 raw S 与训练参考秩 F 两种目标在固定组成/计数 45 维基线下，测试
Spearman 仅差 +0.014(Ridge)/+0.012(HGB)，均低于 0.03 接受阈值（HGB 校正区间含 0）。
但组成/计数仍取得测试 Spearman 0.370–0.421、R2 0.112–0.174，显著高于常数基线，
说明存在可学信号而大部分 S 方差未被解释——缺口最可能在“特征表示”（全局组成/计数
不含调控语法与位置信息）或“目标噪声”。

问题：在冻结 batch05 修正组件切分与 18769 个合格 S 定义基因上，加入确定性的、更
丰富的序列特征表示（位置感知 k-mer/基序计数，CPU、无神经训练、仅用序列），能否
在同一测试行上把 S 测试 Spearman / 目标单位 R2 提升到 batch09 的 45 维组成+计数
基线之上——即判断剩余 S 信号是特征表示缺口还是目标噪声不可约上限。

------------------------------------------------------------------------------
2. 证据支撑（只读，无 GPU）
------------------------------------------------------------------------------
来自 batch09（reports/breakthrough_20261002/batch09/metrics.json 已冻结）：
  - rank-vs-raw 测试 Spearman 差值：Ridge +0.0136（97.5%% 区间 [0.0020, 0.0261]，
    幅度 <0.03），HGB +0.0118（区间 [-0.0103, 0.0340] 含 0）→ 目标表示非杠杆。
  - 组成/计数 45 维：测试 Spearman 0.370(Ridge-raw)/0.421(HGB-rank)，R2 0.112–0.174；
    train 均值常数基线 R2≈0 → 存在可学但远未饱和的信号。
  - 结论：缺口最可能在特征表示或目标噪声，需一个判别性、有界的新自变量实验。

------------------------------------------------------------------------------
3. 冻结设计（草案要点）
------------------------------------------------------------------------------
- 群体/切分：18769 个合格 S 定义基因，batch05 修正组件切分不变（train/val/test
  = 15004/1898/1867，回顾性、非前瞻）。
- 基线参考：batch09 已冻结的 45 维组成/计数特征与四臂预测（同一测试行，只读复用）。
- 新特征：确定性、CPU-only、仅用序列——位置感知 k-mer 频率(k=3,4，必要时 k=5) 与
  位置特异二核苷酸频率（400bp 启动子按位置对齐，2000bp 上游按分箱位置），基因级
  均值聚合；登记构建代码与特征矩阵 SHA。
- 模型/目标：复用 batch09 冻结设计——Ridge(alpha=0.1) 与 HGB(同超参)、单种子
  20261011、连续 raw S 与训练参考秩 F 的同一标准化/逆变换。
- 判据：主判据 = 测试 Spearman 对共同全精度 raw S；次判据 = 目标单位 R2；新特征
  vs batch09 基线的差值用组件配对 bootstrap（97.5%% 双侧百分位区间）。
- 接受：新特征测试 Spearman 相对 batch09 的校正下界>0 且幅度有意义，才视为“特征
  缺口”证据；否则记录为“目标噪声不可约上限”证据（负结果同样有效）。

------------------------------------------------------------------------------
4. 资源与恢复
------------------------------------------------------------------------------
- GPU：0 分钟；CPU <= 4 核、<= 45 分钟；单种子、无超参/阈值/架构扫描、无神经训练。
- 不 topup/不浪费；保留全部既有工件；不改变主 v6.3 / 细胞系 pilot v6.4 指针。

------------------------------------------------------------------------------
5. 交付物
------------------------------------------------------------------------------
中文报告 HTML+text（候选版本 + 任务旁注指标）；新特征构建代码与 SHA；同测试行上的
新特征 vs batch09 基线比较表与 bootstrap 区间；冻结 config + SHA 登记 + 保存模型/
重放示例；next_batch_proposal.json + BATCH11_DRAFT.txt。

------------------------------------------------------------------------------
6. unsupported_scope
------------------------------------------------------------------------------
- 不得训练任何深度/神经网络，不得使用 GPU；
- 不得声称更好的 S 预测器或新生物学验证；
- 不得改变主模型 v6.3 或细胞系 pilot v6.4 指针；
- 不得重复等价重标定或机械调参；
- 不得做哺乳动物通用启动子强度 / 合成启动子活性 / HEK293T 构建 / 时间稳定性声明。
==============================================================================
"""
    atomic_write_text(REPORT_OUT / "BATCH10_DRAFT.txt", draft)
    print("wrote next_batch_proposal.json and BATCH10_DRAFT.txt")


if __name__ == "__main__":
    main()
