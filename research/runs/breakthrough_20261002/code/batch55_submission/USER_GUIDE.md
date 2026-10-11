# 启动子活性预测模型 使用指南（v6.38.0 第一版模型包）

本模型包用于对 **HEK293 启动子片段活性**进行预测。目标量（单位）为作者公开的
`Log2RPM_HEK293`（log2 RPKM/CPM 形式的活性值）。**注意：目标不是 `Log2Norm_HEK293`。**

> 科学范围声明：训练使用的序列来自参考基因组代理（reference proxy），不是供体克隆自身
> 序列；细胞背景为 HEK293，**未在本地 293FT 细胞上验证**，因此不做 293FT 迁移性能声明。
> 模型仅使用作者 fold0 数据从零训练。

---

## 1. 安装依赖

需要 Python 3.12、PyTorch 2.11、NumPy 2.5。真实神经推理需要可用的 CUDA 显卡
（不提供 CPU 回退）。在 PowerShell 中：

```powershell
# 使用项目自带虚拟环境（已含依赖）：
E:\promoter_atlas\.venv\Scripts\python.exe -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

# 或在新环境安装（示例）：
pip install -r "E:\promoter_atlas\releases\batch52_first_model\<版本号>\requirements.txt"
```

## 2. 运行预测

模型包目录（下称 `$BUNDLE`）内含 `MODEL_MANIFEST.json`、`model\`、`code\`。
假设已发布到 `E:\promoter_atlas\releases\batch52_first_model\v6.38.0-first-model\`：

```powershell
$env:PYTHONUTF8=1
$env:PYTHONIOENCODING="utf-8"

$BUNDLE = "E:\promoter_atlas\releases\batch52_first_model\v6.38.0-first-model"
$INPUT  = "D:\我的数据\候选启动子 序列.fa"      # 路径可含空格与中文
$OUT    = "D:\我的数据\预测结果.tsv"

E:\promoter_atlas\.venv\Scripts\python.exe -X utf8 "$BUNDLE\code\predict.py" `
    --bundle "$BUNDLE" `
    --fasta  "$INPUT" `
    --out    "$OUT"
```

`predict.py` 不依赖当前工作目录，可从任意目录运行；也可直接 `python "$BUNDLE\code\predict.py" ...`。

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

若模型包内 `results\` 含评测结果，其字段含义如下（数值直接复制自实际评测文件，绝不编造）。
评测文件可能是单模型形式，也可能是 Nabs/Njoint 对比形式：

| 字段 | 含义 |
| --- | --- |
| `split` | 评估划分（TEST 为最终留出集） |
| `scope` | 评估范围/数据集说明（存在时） |
| `Nabs` / `Njoint` | 两个模型的评测块（对比结果时） |
| `*.activity.rho` | 活性预测与真实值的 Spearman 相关 |
| `*.activity.group_mae` | 按组等权的平均绝对误差 |
| `*.activity.rmse` | 同单位均方根误差 |
| `*.activity.top10_precision` | 头部 top10% 命中精度 |
| `*.pair.*` | 配对差异任务指标（group_mean_mae / macro_sign_accuracy 等，若适用） |
| `fixed_baselines` | 冻结的对照基线（如 zero_delta / always_positive / always_negative） |
| `bootstrap` | 分组自助法置信区间（组数足够时） |
| `verdict` | 预注册目标的达成判定（TARGET_MET / LIMITED） |

单模型评测则直接使用 `activity.rho`、`activity.group_mae` 等扁平字段。

## 6. 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `CUDA unavailable ... no CPU fallback` | 真实模型推理需要显卡；请在带 CUDA 的机器上运行 |
| `record ... length > 600` | 请先自行截取 ≤600bp 片段 |
| `duplicate sequence id` | 保证每条 FASTA 记录 ID 唯一 |
| `INPUT ERROR: ... invalid base characters` | 移除 A/C/G/T/N 之外字符 |
| `checkpoint SHA256 mismatch` | 模型包被改动，请重新获取 |

