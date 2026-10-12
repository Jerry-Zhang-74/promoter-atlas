# v6.6.0 运行与复训

使用 Python 3.12，在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_weights.py
.\.venv\Scripts\python.exe run_demo.py --outdir demo_run
```

`demo_run` 须为新建或空目录。模型由实际发布资产下载并核对字节数和 SHA-256；三项 CPU 演示使用包内示例。

本轮原始配置为 `configs/v6.6.json`，冻结源码为 `training/v6.6.0/source/`，训练日志与步骤记录为 `training/v6.6.0/`，选模和评估文件为 `evaluation/v6.6.0/`。文件清单见 `RUN_MANIFEST.json`。

完整复训需先按 `data/DATA_SOURCE_MANIFEST.json` 准备各项输入，并将 v6.0.0 初始化权重放到该版本配置指定的位置。数据资产见 `artifacts/manifests/TRAINING_ASSETS.json`；这些大文件不存入 Git。

原始配置保留实际截止时间。复训先生成配置副本，指定独立输出目录，再用未来截止时间覆盖：

```powershell
.\.venv\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('configs/v6.6.json'); c=json.loads(p.read_text(encoding='utf-8')); c['outdir']='experiments/retrain/v66_reproduction'; pathlib.Path('configs/v6.6.reproduction.json').write_text(json.dumps(c,indent=2),encoding='utf-8')"
$env:PYTHONPATH="src"
$deadline=(Get-Date).AddHours(2).ToString("yyyy-MM-ddTHH:mm:sszzz")
.\.venv\Scripts\python.exe -m promoter_atlas.measured_train --config configs/v6.6.reproduction.json --deadline $deadline --device cuda --train-only
```

无 CUDA 环境时可改用 `--device cpu`。版本与运行标识沿用冻结源码，复训结果以独立输出目录区分。原始配置与归档源码保持原样。
