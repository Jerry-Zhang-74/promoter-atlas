# AI驱动的启动子核酸元件模块化设计与Bxb1定点整合系统验证平台

当前版本 **v6.6.0**，运行标识 `v6.6.0-measured-20261012`。模型以 v6.0.0 为初始化，接收 400 bp 局部窗口、2000 bp 上游窗口和物种，预测 E/S/B；另提供片段活性与 54 组织表达谱预测。

E/S/B 与组织表达谱分别输出，S 不由预测剖面自动聚合。

参数量 4,869,741；本轮实际完成 23,209 个优化步、2 个完整 epoch，另有 1001 步。停止原因：`user_requested`。选模来源：`last`。
按用户要求结束训练，保留最佳检查点。实际优化步数 23,209，已保存 resume 步数 22,817，未保存尾部步数 392。

| 指标 | 基因数 | Spearman | 95% CI |
| --- | ---: | ---: | --- |
| E | 5603 | 0.6611 | [0.6453, 0.6776] |
| S | 1855 | 0.3758 | [0.3384, 0.4147] |
| B | 5603 | 0.6646 | [0.6479, 0.6806] |

初始模型训练时，同源基因监督可能覆盖原划分中的留出基因；本次沿用该划分的结果用于版本对照。

## 安装与运行

在 Windows / Python 3.12 的项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_weights.py
.\.venv\Scripts\python.exe run_demo.py --outdir demo_run
```

`demo_run` 使用新建或空目录。演示写出三项任务的 CSV、E/S 筛选表和运行记录。权重由[发布资产](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/download/v6.6.0/promoter_atlas_v6.6.0.pt)下载，SHA-256：`b0a90dddc2325efd3dd5a09822047362d51968865a4bc3ee6ab647ecd626d548`。

```powershell
.\.venv\Scripts\python.exe predict.py --task esb --input examples/esb_demo.tsv --output esb_predictions.csv
.\.venv\Scripts\python.exe predict.py --task activity --input examples/fragment_demo.tsv --output fragment_activity.csv
.\.venv\Scripts\python.exe predict.py --task profile --input examples/esb_demo.tsv --output tissue_predictions.csv
```

双窗口输入为 TSV，列为 `candidate_id`、`sequence_sym400`（400 bp）、`sequence_up2000`（2000 bp）；`species` 可省略，默认 `homo_sapiens`。片段输入列为 `candidate_id`、`sequence`，长度 100–600 bp。

E/S/B 输出字段为 `Score_E`、`Score_S`、`Score_B`；片段活性为 `predicted_Log2RPM_HEK293`。组织 CSV 使用检查点中的 54 个真实组织列名，数值为预测的 `log2(1+mean TPM)`；加 `--profile-unit tpm` 可输出反变换值。

## 数据与历史

监督来源包括 GTEx v10、FANTOM5/ENCODE 实测信号与 TF 特征、跨物种同源映射。文件来源和校验值见[数据清单](data/DATA_SOURCE_MANIFEST.json)与[训练数据发布页](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/training-data-20261012)。训练输入的文件清单、校验值和可用下载位置见上述清单。

27 个历史版本的模型卡、日志和评估位于 `models/versions/`，已有共用与冻结源码保留原有入口。当前 `run_demo.py` 使用 v6.6.0。复训步骤见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，本轮冻结记录见 `training/v6.6.0/`，评估见 `evaluation/v6.6.0/`。

完整的检查点候选与归档位于本地提交归档中；公共 Git 包含源码、日志与结果，选定权重通过发布资产下载。

## English

Version v6.6.0 predicts promoter E/S/B from paired sequence windows, fragment activity, and a 54-tissue expression profile. Install the listed dependencies, download the verified release weight, then run the CPU demo above. Training provenance and measured evaluation results accompany this release.
