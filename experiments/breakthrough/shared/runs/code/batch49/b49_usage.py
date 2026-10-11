# -*- coding: utf-8 -*-
"""batch49 Protocol D: concise Chinese USAGE (PowerShell-first). Writes reports/batch49/USAGE_zh.md.
Immutable doc -> written BEFORE the manifest is collected. No GUI, no zip, no new visuals.
Paths use forward slashes so the PowerShell examples are copy-paste safe on Windows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b49_common as C  # noqa: E402

USAGE = u"""# batch49 v6.35.0-catalog-tradeoff-selection.rc1 — 中文使用说明

> 本批**新增一个任务 `gene-rank`**（目录权衡选择 / catalog tradeoff selection），在**同一个新版 CLI** 上；
> `fragment` 与 `gene-observed` 的行为**逐字节保持**与 batch48 相同（复用同一加载器）。
> **不是**新预测器、**不是** GUI、**不拟合/不训练/不下载/不使用 GPU/不使用神经网络权重**。

## 三个任务（必须显式选择；CLI 从不猜测）

### 任务 1 `fragment`：原始 FASTA → 冻结 PARM HEK293 模型分数
- 复用（只读、不改动）batch35 PARM 引擎与外部权重；官方口径：中心零填充到 L_max=600，无 RC / 无改 DNA / 无隐藏窗口。
- 支持长度 1..600；>600（**含 2000**）→ 分数 `null`，状态 `UNSUPPORTED_LENGTH`。
- 分数标签固定 `PARM_HEK293_model_score`；**不是 293FT 活性 / 不是 GFP / 不是概率 / 不是 E/S/B / 无可靠区间**。
- 模型细胞 HEK293（公开）；用户目标细胞 293FT：**UNVALIDATED_FOR_293FT**（`--target-cell` 仅元数据，不选权重）。

### 任务 2 `gene-observed`：基因符号 / Ensembl ID → 目录观测标签
- 用已有可追溯标签表 + 符号映射（无网络）；无 GPU / 无 torch / 无学习权重（CUDA 隐藏也能跑）。
- 返回规范 ID/符号、`observedScore_E`、`S_raw`、`B_raw`，以及可选 `observedScore_S/B`。
- **来源目录标签**（跨组织 E/S/B），**不是**单个启动子实测、**不是** 293FT/GFP 活性、**不是**模型预测；`S` 是**跨组织**稳定性（非时间）。
- `S_raw` 缺失 → `null`（不用 0 / 不用预测填补），掩码 `S_defined_main==1`。
- `observedScore_S` 语义（源自 `runs/scripts/build_candidate_e_labels.py`）：对**全部 58988 行**做升序排名、除以 `n_total-1=58987`；
  `S_raw` 未定义的行排在**最底部**（`-inf`），再存为 `null`；已定义基因只落在**顶部区间** `[0.6788, 1.0]`；仅作信息参考。

### 任务 3 `gene-rank`（本批新增）：目录权衡选择
在**冻结的标签表**上，用两条**设计轴** `E_raw` 与 `S_raw` 做**确定性选择**；`B_raw` 仅作**门控**。

- **两个轴（都越大越好）**
  - `E_raw` = 跨全部组织 `log2(1+mean_tpm)` 的**均值**（表达强度）。
  - `S_raw` = 在**可检出组织**上 `-MAD(log2(1+mean_tpm))`（**跨组织稳定性**，不是时间；**越接近 0/越不负 = 跨组织越稳定**）。
- **B 门控**：`--min-b`（默认 `0.30`，闭区间）。`B_raw` = 可检出组织占比（TPM>=1）。**B 只作门控**，绝不与 E/S 相乘或合成单一分数。
- **可用性（eligible）**：`E_raw`/`S_raw`/`B_raw` 均有限，且 `S_defined_main==1`。未定义的 S **绝不填 0、绝不重算/用 B 覆盖**。
- **支配（dominance）**：p 支配 q 当且仅当 `E_raw(p)>=E_raw(q)` 且 `S_raw(p)>=S_raw(q)` 且至少一个**严格**大。
- **Pareto 前沿**：没有任何其它 **eligible 基因** 支配的基因。**完全相等的 (E,S) 对全部并列保留**，不合并、不按基因去重。
- **排序 `--sort`**：`expression`→E 降、S 降、ID 升；`stability`→S 降、E 降、ID 升；`pareto`（默认）→前沿，E 降、S 降、ID 升。
- **`--min-e`**：可选 `E_raw` 下界（闭区间）；默认**无**截断。
- **`--limit`**：仅影响**显示**；**先算完整前沿**，再截断返回视图。`--all-eligible-tsv` 可选导出全部 eligible 行并带 `is_frontier` 标志。
- **不做**任意加权 E+S、不做乘积分数；`E x S` 是设计思想，不是数值相乘。
- **为什么 B 门控与 S 定义重要**：`S_defined_main==1` 本身已要求 `is_in_main_subset==1`（即 `B_raw>=0.30`）且可检出组织>=3；
  因此 `--min-b` **小于 0.30 不会扩大**可用集合（S 的定义仍然限制可用范围）。
- **输出**：全原始字段 + 信息性 `observedScore_E/S/B`（全目录排名）+ `B_raw` + 掩码 + 规范 `gene_id`/`symbol` + `biotype`
  （符号映射有 `gene_type` 则输出，否则 `unknown`）+ 状态。目录可含未注释/假基因，**不假装**是蛋白编码/管家基因筛选。
- **合规**：目录数为**观测派生**，无预测字段 / 无置信区间 / 不宣称 293FT 活性；**跨组织**上下文，**不是 293FT**；
  自然基因候选**不是**保证的启动子，**不是**湿实验推荐，**不是**自主启动子证明。
"""

USAGE += u"""
## 可直接运行的命令示例（PowerShell — 用户当前默认 shell）

```powershell
$PY  = "E:/promoter_atlas/.venv/Scripts/python.exe"
$CLI = "E:/promoter_atlas/runs/breakthrough_20261002/code/batch49/b49_cli.py"
$OUT = "E:/promoter_atlas/data/breakthrough_20261002/batch49/out"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 任务 3（默认）：E x S 的 Pareto 前沿（B 门控 0.30，无 E 截断，显示上限 50），并冻结契约
& $PY $CLI gene-rank --out-json "$OUT/gene_rank_default.json" --out-tsv "$OUT/gene_rank_default.tsv" --all-eligible-tsv "$OUT/all_eligible.tsv"
# 冻结契约写在 --out-json 同名的 .contract.json（先于任何候选值写出）

# 变体：按表达排序 / 按稳定度排序 / 加 E 下界
& $PY $CLI gene-rank --sort expression --limit 20 --out-json "$OUT/rank_expr.json" --out-tsv "$OUT/rank_expr.tsv"
& $PY $CLI gene-rank --sort stability  --limit 20 --out-json "$OUT/rank_stab.json" --out-tsv "$OUT/rank_stab.tsv"
& $PY $CLI gene-rank --min-e 5.0      --min-b 0.8   --out-json "$OUT/rank_mine.json" --out-tsv "$OUT/rank_mine.tsv"

# 任务 2：基因符号 / Ensembl ID -> 目录观测标签（无需 GPU）
& $PY $CLI gene-observed --query ACTB GAPDH RPLP0 --out-json "$OUT/demo_gene.json" --out-tsv "$OUT/demo_gene.tsv"
```

### 可选：cmd.exe（非默认 shell，此处仅供参考）

```bat
:: 可选：cmd.exe
set "PY=E:/promoter_atlas/.venv/Scripts/python.exe"
set "CLI=E:/promoter_atlas/runs/breakthrough_20261002/code/batch49/b49_cli.py"
"%PY%" "%CLI%" gene-rank --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch49/out/demo.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch49/out/demo.tsv"
```
"""

USAGE += u"""
## 明确区分与不支持项
- **公开 HEK293 模型**（fragment）：PARM 官方 HEK293 启动子活性模型（固定）。
- **用户 293FT**：仅为调用方声明的目标细胞元数据，**未验证**（`UNVALIDATED_FOR_293FT`）。
- **gene-rank / gene-observed**：均为**跨组织目录标签**，与 293FT/GFP 实测无关。
- 本入口明确**不支持**：`mpra-fasta`、`gene`、`gene-bag-fasta`、`constructs`，以及旧 164bp K562 神经路径、新基因 E/S/B 推断。
- 外部权重/许可/指针**未变**（仍为 batch31/_external，SHA 校验未改）。生产指针**未变**（v6.3 主 / v6.4 cellline pilot）。
- 本批**不结束付费工作、不宣称项目完成**；剩余的首要是**用户 293FT 机箱的性能差距**（需新的独立监督数据，而非重复清点）。
"""


def write_usage():
    (C.B49_REPORT / "USAGE_zh.md").write_text(USAGE, encoding="utf-8")
    return C.B49_REPORT / "USAGE_zh.md"


if __name__ == "__main__":
    print(write_usage())
