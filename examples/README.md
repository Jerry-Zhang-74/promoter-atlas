# 输入示例与参考预测 · Input examples and reference predictions

## 中文

这里提供两组可以直接运行的输入，以及对应发布权重的参考预测。先在仓库根目录安装项目并运行 `python scripts/download_weights.py`。

### 双窗口启动子预测

[`demo_input.tsv`](demo_input.tsv) 包含 8 个候选。每一行提供：

| 字段 | 内容 |
|---|---|
| `candidate_id` | 文件内唯一的候选名称 |
| `sequence_sym400` | 围绕 TSS 的 400 bp 局部序列 |
| `sequence_up2000` | 按转录方向排列的 2000 bp 上游上下文 |
| `species` | 支持物种的标准名称，如 `homo_sapiens` |
| `gene_id`、`record_id` | 可选的基因与源记录标识 |

两段序列采用 A/C/G/T/N 字符。用自己的候选时保持窗口含义和长度，并从相应参考序列取得两个窗口。输入检查规则见 [`spec_contract.py`](../src/promoter_atlas/spec_contract.py)。复现历史结果时，沿用冻结序列和行索引；构建器的坐标约定如下。

以 `t` 表示 TSS 碱基的基因组 0-based 坐标。在窗口完整、无需边界补齐时：

- [人类局部窗口构建器](../scripts/research/data/build_human_sym400_promoters.py)及[多物种构建器](../scripts/research/data/build_multispecies_promoters.py)均截取基因组区间 `[t−200,t+200)`，负链再取反向互补。因此正链 TSS 在输出索引 200，负链在 199。
- 多物种构建器的 up2000 正链区间为 `[t−2000,t)`；负链区间为 `[t,t+2000)`，反向互补后 TSS 位于末位。
- [人类 up2000 编码器](../scripts/research/data/build_promoter_sequence_memmap_up2000.py)读取已有 FASTA，将序列转为 memmap，并保留其坐标来源。人类与其他物种的上游窗口各自沿用对应的冻结序列。

边界截取及补齐逻辑见对应构建器。当前预测入口校验序列长度和字符，坐标提取由输入准备流程负责。

```bash
python -m promoter_atlas.main --input examples/demo_input.tsv --outdir results/demo
```

输出 `results/demo/results.csv` 按 B 门槛、E、S 排序，因此可能改变输入行序。`results.csv.run.json` 保存运行参数和文件指纹。参考 E/S/B 值位于 [`expected/esb.json`](expected/esb.json)，按 `candidate_id` 对应。

### 片段活性预测

[`activity_demo.tsv`](activity_demo.tsv) 含 3 条 `seq_id` 与 `sequence` 记录；每条序列在 100–600 bp 范围内。

```bash
python -m promoter_atlas.diagnostic_activity --input examples/activity_demo.tsv --out activity.tsv
```

结果按输入行序写入 `activity.tsv`，列出长度、标准化活性和 HEK293 Log2RPM 刻度上的预测。表格示例见 [`activity_results.tsv`](activity_results.tsv)，程序比较使用 [`expected/activity.json`](expected/activity.json)。

### 自动比较

```bash
python scripts/verify_release.py
```

检查程序在临时目录重新生成两组结果，以 ID 核对样本，以 `5e-4` 绝对容差比较 E/S/B、片段长度与活性预测，并输出最大差值。这些文件是发布模型的回归检查样例；模型的测试集指标见相应[模型版本](../models/README.md)。

## English

This directory provides two runnable input sets and reference predictions from the released weights. Install the project and run `python scripts/download_weights.py` from the repository root first.

### Dual-window promoter prediction

[`demo_input.tsv`](demo_input.tsv) contains eight candidates. Each row supplies:

| Field | Contents |
|---|---|
| `candidate_id` | A candidate name unique within the file |
| `sequence_sym400` | A 400 bp local sequence around the TSS |
| `sequence_up2000` | A 2,000 bp upstream context oriented in the transcription direction |
| `species` | A supported species name, such as `homo_sapiens` |
| `gene_id`, `record_id` | Optional gene and source-record identifiers |

Both sequences use A/C/G/T/N characters. For custom candidates, preserve these window definitions and lengths and obtain both windows from the corresponding reference sequence. Validation rules are in [`spec_contract.py`](../src/promoter_atlas/spec_contract.py). To reproduce historical results, use the frozen sequences and row indexes. The builders use the following coordinate conventions.

Let `t` be the zero-based genomic coordinate of the TSS base. For complete windows requiring no boundary padding:

- Both the [human local-window builder](../scripts/research/data/build_human_sym400_promoters.py) and the [multispecies builder](../scripts/research/data/build_multispecies_promoters.py) extract genomic interval `[t−200,t+200)` and reverse-complement negative-strand sequences. The TSS therefore appears at output index 200 on the positive strand and 199 on the negative strand.
- For up2000, the multispecies builder extracts `[t−2000,t)` on the positive strand and `[t,t+2000)` on the negative strand. After reverse complementation, the negative-strand TSS is the final base.
- The [human up2000 encoder](../scripts/research/data/build_promoter_sequence_memmap_up2000.py) reads an existing FASTA and converts its sequences to a memmap, retaining their coordinate provenance. Human and other-species upstream windows each follow their corresponding frozen sequences.

See the builders for boundary clipping and padding. The current prediction interface checks sequence lengths and characters; input preparation handles coordinate extraction.

```bash
python -m promoter_atlas.main --input examples/demo_input.tsv --outdir results/demo
```

`results/demo/results.csv` is sorted by B-gate passage, E and S, so its row order may differ from the input. `results.csv.run.json` stores runtime parameters and file fingerprints. Reference E/S/B values in [`expected/esb.json`](expected/esb.json) are matched by `candidate_id`.

### Fragment-activity prediction

[`activity_demo.tsv`](activity_demo.tsv) contains three `seq_id` and `sequence` records, each 100–600 bp long.

```bash
python -m promoter_atlas.diagnostic_activity --input examples/activity_demo.tsv --out activity.tsv
```

The output `activity.tsv` preserves input order and reports length, standardized activity and the prediction on the HEK293 Log2RPM scale. See [`activity_results.tsv`](activity_results.tsv) for the table format; the checker uses [`expected/activity.json`](expected/activity.json).

### Automated comparison

```bash
python scripts/verify_release.py
```

The checker regenerates both sets in a temporary directory, matches sample IDs, compares E/S/B values, fragment lengths and activity predictions with an absolute tolerance of `5e-4`, and reports the largest differences. These files are regression examples for the released model. Test-set metrics are recorded with the relevant [model version](../models/README.md#english).
