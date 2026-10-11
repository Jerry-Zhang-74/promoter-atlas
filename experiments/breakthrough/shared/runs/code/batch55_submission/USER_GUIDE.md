# HEK293 片段预测使用指南 · HEK293 fragment prediction guide

[中文](#中文) · [English](#english)

## 中文

### 用途和依赖

预测 HEK293 启动子片段的 `Log2RPM_HEK293`。模型使用作者 fold0、参考基因组代理序列从随机初始化训练；293FT 迁移待评估。经验训练长度为 100–600 bp。

需要 Python 3.12、PyTorch 2.11、NumPy 2.5 和 CUDA GPU。将 `<bundle-directory>`、输入和输出路径替换为实际位置，路径支持中文和空格。预测入口为 `code\predict.py`，按自身位置寻找模型代码。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$BUNDLE = '<bundle-directory>'
$INPUT = 'D:\data\promoters.fa'
$OUT = 'D:\data\predictions.tsv'
python -m pip install -r "$BUNDLE\requirements.txt"
python -X utf8 "$BUNDLE\code\predict.py" --bundle "$BUNDLE" --fasta "$INPUT" --out "$OUT"
```

### 输入与输出

输入是 FASTA，每条记录有唯一 `>id`，碱基为 A/C/G/T/N，大小写均可。空文件、空序列、全 N、重复 ID、非法碱基或长度超过 600 bp 会触发输入错误（退出码 2）。长度问题应通过准备符合实验设计的有效片段解决。

UTF-8 TSV 输出字段为 `seq_id`、`length`、`activity_pred_Log2RPM`、`model_id`、`version`。同时生成 `<out>.meta.json`，记录输入/输出 SHA-256、设备、版本和 checkpoint 哈希。

### 评估文件

`results/` 保存评估与复核记录，模型卡说明统计口径。`split` 标记数据划分，`scope` 描述评估范围，Nabs/Njoint 是对比模型块；单模型直接使用 `activity.*`。

`activity.rho` 是 Spearman；`group_mae` 是组等权 MAE；`rmse` 为原单位 RMSE；`top10_precision` 为前 10% 命中率；`pair.*` 为配对 group_mean_mae / macro_sign_accuracy；`fixed_baselines` 为 zero_delta / always_positive / always_negative 对照；`bootstrap` 为分组区间；`verdict` 记录 TARGET_MET / LIMITED 等预设结论。

### 常见问题

CUDA 不可用时换到有 CUDA 的环境；ID 重复时给每条记录唯一名称；碱基错误时检查 A/C/G/T/N 字符；checkpoint SHA-256 不符时重新获取对应模型包。

---

## English

### Purpose and dependencies

Predict HEK293 promoter-fragment `Log2RPM_HEK293`. Training used author fold0, reference-genome proxy sequences, and random initialization. Transfer to 293FT awaits evaluation. The empirical training domain is 100–600 bp.

Use Python 3.12, PyTorch 2.11, NumPy 2.5, and a CUDA GPU. Replace `<bundle-directory>`, input, and output with actual paths; spaces and non-ASCII characters are supported. The predictor at `code\predict.py` resolves its model code relative to its own location.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$BUNDLE = '<bundle-directory>'
$INPUT = 'D:\data\promoters.fa'
$OUT = 'D:\data\predictions.tsv'
python -m pip install -r "$BUNDLE\requirements.txt"
python -X utf8 "$BUNDLE\code\predict.py" --bundle "$BUNDLE" --fasta "$INPUT" --out "$OUT"
```

### Input and output

Input is FASTA with a unique `>id` per record and case-insensitive A/C/G/T/N bases. Empty files, empty/all-N sequences, duplicate IDs, invalid bases, and sequences above 600 bp raise input errors (exit code 2). Resolve length issues by preparing valid fragments appropriate to the experimental design.

UTF-8 TSV output fields are `seq_id`, `length`, `activity_pred_Log2RPM`, `model_id`, and `version`. `<out>.meta.json` records input/output SHA-256, device, version, and checkpoint hash.

### Evaluation files

`results/` contains evaluation and review records; the model card explains the statistical scope. `split` identifies the partition, `scope` describes the evaluation, and Nabs/Njoint identify comparison blocks. Single-model results use `activity.*` directly.

`activity.rho` is Spearman; `group_mae` is group-equal MAE; `rmse` uses original units; `top10_precision` measures top-10% precision; `pair.*` includes group_mean_mae / macro_sign_accuracy; `fixed_baselines` contains zero_delta / always_positive / always_negative; `bootstrap` contains grouped intervals; and `verdict` records prespecified outcomes such as TARGET_MET / LIMITED.

### Troubleshooting

Use a CUDA environment when CUDA is unavailable, assign unique IDs, check A/C/G/T/N characters, and obtain the corresponding model bundle again if checkpoint SHA-256 verification fails.
