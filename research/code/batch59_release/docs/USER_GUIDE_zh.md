# 启动子活性预测模型 使用指南（中文速查 · v6.38.0 终版）

本包预测 **HEK293 启动子片段活性**，目标量为作者公开的 `Log2RPM_HEK293`。
科学范围：序列来自参考基因组代理（GRCh37/hg19 代理），**不是**供体克隆自身序列；
细胞背景 HEK293，**未在本地 293FT 验证**；仅用作者 fold0 从零训练。支持长度 **100–600 bp**；
**>600 bp 不支持**（不做滑窗、不截取）。

## 一、依赖
Python 3.12 + PyTorch 2.11 + NumPy 2.5。**真实模型推理必须使用 CUDA 显卡（无 CPU 回退）**。

## 二、一条命令跑通（PowerShell，可在任意工作目录运行）
把下面整段复制进 PowerShell（假设模型包解压在 `D:\模型\v6.38.0-nabs-reviewed`，
输入 FASTA 在 `D:\我的数据\候选启动子 序列.fa`）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
$BUNDLE='D:\模型\v6.38.0-nabs-reviewed'; $INPUT='D:\我的数据\候选启动子 序列.fa'; $OUT='D:\我的数据\预测结果.tsv'
& "E:\promoter_atlas\.venv\Scripts\python.exe" -X utf8 "$BUNDLE\predict.py" --bundle "$BUNDLE" --fasta "$INPUT" --out "$OUT"
```

`predict.py` 依据自身位置寻找同目录的 `pinned_core/`，**不依赖当前工作目录**，路径可含空格与中文。

## 三、输入格式
FASTA，每条记录 `>id` 换行后接 ACGT（可含 N，大小写均可）。以下输入会被**明确拒绝**（退出码 2，不静默修复）：
空文件 / 无记录、空序列、重复 id、非法碱基、**任一记录 > 600 bp**。

## 四、输出
TSV（UTF-8）：`seq_id  length  activity_pred_Log2RPM  model_id  version`，并生成 `$OUT.meta.json`
（含输入/输出 SHA256、设备、checkpoint 哈希）。

## 五、真实性能（公开 HEK293 TEST，直接复制自评测文件，未编造）
见同目录 `MODEL_CARD_zh.md` 与 `results/`。**不做 293FT 迁移声明。**
