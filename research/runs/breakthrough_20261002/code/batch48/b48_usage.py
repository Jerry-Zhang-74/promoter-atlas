# -*- coding: utf-8 -*-
"""batch48 Protocol D: concise Chinese USAGE (PowerShell-first). Writes reports/batch48/USAGE_zh.md.
Immutable doc -> written BEFORE the manifest is collected. No GUI, no zip, no new visuals.
Paths use forward slashes so the PowerShell/cmd examples are copy-paste safe on Windows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b48_common as C  # noqa: E402

USAGE = u"""# batch48 v6.34.1-scoped-input-routing.rc2 — 中文使用说明

> 本批是对 batch47 (rc1) 的**两处正确性修复 + 使用说明修订**；**能力与 batch47 相同**，
> **不是**新预测器、**不是** GUI、**不宣称任何性能提升**。
> 修复 1（R1）：产物**定稿顺序** —— 先写全部不可变文件，再收集 manifest，最后两次只读校验。
> 修复 2（R2）：`gene-observed` 的 `observedScore_S` 字段定义改为与**原始标签生成代码**一致。

## 两个任务（必须显式选择；CLI 从不猜测）

### 任务 1 `fragment`：原始 FASTA → 冻结 PARM HEK293 模型分数
- 复用（只读、不改动）batch35 PARM 引擎与外部权重；官方口径：中心零填充到 L_max=600，无 RC / 无改 DNA / 无隐藏窗口。
- 支持长度 1..600；>600（**含 2000**）→ 分数 `null`，状态 `UNSUPPORTED_LENGTH`。
- 分数标签固定 `PARM_HEK293_model_score`（五折等权均值）；**不是 293FT 活性 / 不是 GFP / 不是概率 / 不是 E/S/B / 无可靠区间**。
- `--target-cell`（默认 `293FT`）**仅为元数据**，**绝不**选择模型权重（模型细胞固定 HEK293）。
- 模型细胞：HEK293（公开）；用户目标细胞 293FT：**UNVALIDATED_FOR_293FT**。

### 任务 2 `gene-observed`：基因符号 / Ensembl ID → 目录观测标签
- 用已有可追溯标签表 + 符号映射（无网络）；无 GPU / 无 torch / 无学习权重（CUDA 隐藏也能跑）。
- 返回规范 ID/符号、`observedScore_E`、`S_raw`、`B_raw`，以及可选 `observedScore_S/B`。
- 这些是**来源目录标签**（跨组织 E/S/B），**不是**单个启动子实测、**不是** 293FT/GFP 活性、**不是**模型预测；`S` 是**跨组织**稳定性（非时间）。
- `S_raw` 缺失 → `null`（不用 0 / 不用预测填补），掩码 `S_defined_main==1`。
- `observedScore_S` 语义（修正后，源自 `runs/scripts/build_candidate_e_labels.py`）：
  对**全部 58988 个目录基因行**做升序排名、除以 `n_total-1=58987`；`S_raw` 未定义的行排在**最底部**（排序键 `-inf`），随后再存为 `null`。
  故已定义基因的 `observedScore_S = (n_undefined + 定义内升序名次)/(n_total-1)`，只落在**顶部区间** `[0.6788, 1.0]`；
  **并非**只在已定义基因内排名；仅作信息参考，单位不同于 `S_raw`。
- 符号歧义 → `AMBIGUOUS`+candidates；未知 → `ABSENT`+reason；Ensembl 仅按精确基 ID 归一（如 `ENSG00000075624.18` → `ENSG00000075624`）。
"""

USAGE += u"""
## 可直接运行的命令示例（PowerShell — 用户当前默认 shell）

```powershell
$PY  = "E:/promoter_atlas/.venv/Scripts/python.exe"
$CLI = "E:/promoter_atlas/runs/breakthrough_20261002/code/batch48/b48_cli.py"

# 任务 1：原始 FASTA -> PARM HEK293 模型分数（需要 CUDA；本批不重跑前向）
& $PY $CLI fragment --fasta "E:/promoter_atlas/data/breakthrough_20261002/batch35/testdata/known_control.fasta" --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo.tsv" --target-cell 293FT

# 任务 2：基因符号 / Ensembl ID -> 目录观测标签（无需 GPU）
& $PY $CLI gene-observed --query ACTB GAPDH RPLP0 --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo_gene.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo_gene.tsv"
```

### 可选：cmd.exe（非默认 shell，此处仅供参考）

```bat
:: 可选：cmd.exe
set "PY=E:/promoter_atlas/.venv/Scripts/python.exe"
set "CLI=E:/promoter_atlas/runs/breakthrough_20261002/code/batch48/b48_cli.py"
"%PY%" "%CLI%" gene-observed --query ACTB GAPDH RPLP0 --out-json "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo_gene.json" --out-tsv "E:/promoter_atlas/data/breakthrough_20261002/batch48/demo_gene.tsv"
```
"""

USAGE += u"""
## 示例输出（摘要）
- `gene-observed`：`ACTB` → `observedScore_E=0.999644`、`S_raw=-1.079803`、`B_raw=1.0`、`S_defined_mask=1`；
  未定义基因如 `ENSG00000000005` → `S_raw=null`、`observedScore_S=null`（不填 0）。
- `fragment`：控制序列分数与 b31/b34/b35 一致（`3.1036716`，容差 1e-4）；>600（含 2000）为 `null`。

## 明确区分
- **公开 HEK293 模型**：PARM 官方 HEK293 启动子活性模型（固定）。
- **用户 293FT**：仅为调用方声明的目标细胞元数据，**未验证**（`UNVALIDATED_FOR_293FT`）。
- **不做任何“自主启动子/GFP 预测”声明**；模型分数仅为跨细胞排序描述子。
- 外部权重/许可/指针**未变**（仍为 batch31/_external，SHA 校验未改）。

## 本入口明确**不支持**的任务（不静默切换）
下列命令在当前 CLI **未实现**：`mpra-fasta`、`gene`、`gene-bag-fasta`、`constructs`；
以及旧 164bp K562 神经路径、新基因 E/S/B 推断。它们各需不同输入：
- `gene`（新基因 E/S/B 推断）：需序列 + 对应训练模型权重，本批不含。
- `mpra-fasta`：需 MPRA 设计/参照输入，本批不含。
- `constructs`：旧构建输出含已作废的 HEK293T 措辞，本批禁止暴露。
"""


def write_usage():
    (C.B48_REPORT / "USAGE_zh.md").write_text(USAGE, encoding="utf-8")
    return C.B48_REPORT / "USAGE_zh.md"


if __name__ == "__main__":
    print(write_usage())
