# 启动子活性预测模型 使用指南（{VERSION} · 扁平布局终版）

本模型包用于对 **HEK293 启动子片段活性**进行预测。目标量为作者公开的
`Log2RPM_HEK293`（log2 RPKM/CPM 形式的活性值）。**注意：目标不是 `Log2Norm_HEK293`。**

> 科学范围：训练序列来自参考基因组代理（GRCh37/hg19 代理），不是供体克隆自身序列；
> 细胞背景为 HEK293，**未在本地 293FT 细胞上验证**，不做 293FT 迁移性能声明；
> 模型仅用作者 fold0 数据从零训练。支持长度 **100–600 bp**，**>600 bp 不支持**（不做滑窗）。

模型包为**扁平布局**：`predict.py` 位于解压根目录，和其同级的 `pinned_core/` 一起使用。

---

## 1. 安装依赖

需要 Python 3.12、PyTorch 2.11、NumPy 2.5。真实神经推理需要可用的 CUDA 显卡（无 CPU 回退）。
在 PowerShell 中（把 `<解压目录>` 换成你的实际解压目录）：

```powershell
# 用任意 Python 3.12 解释器（此处以 PATH 中的 python 为例，也可换成自建环境的完整路径）
python -X utf8 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

# 安装本包依赖（示例）：
python -m pip install -r "<解压目录>\requirements.txt"
```

## 2. 运行预测

设 `$BUNDLE` 为解压根目录，内含 `MODEL_MANIFEST.json`、`predict.py`、`pinned_core\`、`model\` 等：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
$BUNDLE = "<解压目录>"                                  # 路径可含空格与中文
$INPUT  = "D:\我的数据\候选启动子 序列.fa"
$OUT    = "D:\我的数据\预测结果.tsv"
python -X utf8 "$BUNDLE\predict.py" --bundle "$BUNDLE" --fasta "$INPUT" --out "$OUT"
```

`predict.py` 依据自身位置寻找同目录的 `pinned_core/`，**不依赖当前工作目录**，可从任意目录运行。
`python` 可替换为任何 Python 3.12 解释器的完整路径。

## 3. 输入格式（严格校验）

FASTA 文本，每条记录：

```
>candidate_1
ACGT...ACGT
>candidate_2
ACGT...ACGT
```

会被**拒绝**的输入（不会静默截断）：
- 空文件、无记录；
- 空序列（null）、或全为 `N` 的歧义序列；
- 重复的序列 ID；
- 非法碱基字符（只允许 A/C/G/T/N，大小写均可）；
- 任一记录长度 **> 600 bp**（模型不支持 >600，不做滑窗）。

## 4. 输出说明

`$OUT`（TSV，UTF-8）：

```
seq_id    length    activity_pred_Log2RPM    model_id    version
candidate_1  480    3.214567    <model_id>  <version>
```

同时生成 `$OUT.meta.json`，含输入/输出的 SHA256、设备、模型版本与 checkpoint 哈希。

## 5. 结果字段说明（来自冻结的真实评估 JSON）

本包 `results\` 直接包含真实评测与独立复核文件（数值逐字复制，未编造）：
`{ARM}_TEST_parent_metrics.json`（真实留出 TEST 指标）、`batch54_independent_review.json`、
`batch59_release_evidence.json`。区间/对比口径见 `MODEL_CARD_zh.md` 与这些 JSON 原文。

## 6. 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `CUDA unavailable ... no CPU fallback` | 真实模型推理需要显卡；请在带 CUDA 的机器上运行 |
| `record ... length > 600` | 请先自行截取 ≤600 bp 片段 |
| `duplicate sequence id` | 保证每条 FASTA 记录 ID 唯一 |
| `INPUT ERROR: ... invalid base characters` | 移除 A/C/G/T/N 之外字符 |
| `checkpoint SHA256 mismatch` | 模型包被改动，请重新获取 |
