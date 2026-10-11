# PromoterAtlas v6.15.1-release.rc1

[中文](#中文) · [English](#english)

## 中文

该实验候选预测人类基因级自然表达，使用三个独立 CNN 和 batch16 权重。模型与评估见 [模型卡](MODEL_CARD.md)。

E 为 [0,1] 的 `Score_E`，S 为可检出组织的 `S_raw=-MAD(log2(1+TPM))`，B 为 `detectable_tissue_count/tissue_count`。主 seed 为 20261016；共享臂 B/C 验证增益 -0.0061/-0.0049，选择门槛 +0.02。

### 解压完整历史包后运行

此文件是打包文档源稿。下列命令对应完整包内的 `code/`、权重与 transforms；本目录用于维护文档内容。

```bash
python code/run_example.py
python code/promoter_atlas_predict.py --input code/example_input.npz --out preds.json
```

```python
import numpy as np, sys
sys.path.insert(0, "code")
from promoter_atlas_predict import PromoterAtlasGenePredictor, PromoterAtlasInputError
p = PromoterAtlasGenePredictor()
z = np.load("code/example_input.npz", allow_pickle=True)
res = p.predict_batch(z["seqs"], z["features"])
one = p.predict_gene(z["seqs"][0], z["features"][0])
```

### 输入输出

`seqs` 为 uint8 `(N,4,400)` 或 `(4,400)`，A/C/G/T/N 编码为 0/1/2/3/4。`features` 为有限浮点 `(N,1069)` 或 `(1069,)`，按训练顺序提供原始值，包内 transforms 完成标准化。格式错误抛出 `PromoterAtlasInputError`。

`predict_gene` 返回原单位 E/S/B，以及 `is_model_prediction: true`、`is_catalog_observed: false`。观察标签的来源见 [标签说明](labels_README.md)。

`models/` 存 E/S/B 权重，`transforms/scalers.npz` 存 TRAIN 拟合的特征和目标尺度，`code/` 存推理与示例。顶层包括 README、MODEL_CARD、requirements、environment、schema、VERSION_MANIFEST 及 negative_ablation_results。

---

## English

This experimental candidate predicts human gene-level natural expression with three independent CNNs and batch16 weights. See the [model card](MODEL_CARD.md) for model and evaluation details.

E is `Score_E` in [0,1], S is detectable-tissue `S_raw=-MAD(log2(1+TPM))`, and B is `detectable_tissue_count/tissue_count`. The reference seed is 20261016; shared-arm B/C validation gains were -0.0061/-0.0049 against a +0.02 selection threshold.

### Run after extracting the complete historical package

This file is packaging-document source. The commands below refer to the complete package’s `code/`, weights, and transforms; this directory maintains the documentation.

```bash
python code/run_example.py
python code/promoter_atlas_predict.py --input code/example_input.npz --out preds.json
```

```python
import numpy as np, sys
sys.path.insert(0, "code")
from promoter_atlas_predict import PromoterAtlasGenePredictor, PromoterAtlasInputError
p = PromoterAtlasGenePredictor()
z = np.load("code/example_input.npz", allow_pickle=True)
res = p.predict_batch(z["seqs"], z["features"])
one = p.predict_gene(z["seqs"][0], z["features"][0])
```

### Input and output

`seqs` is uint8 `(N,4,400)` or `(4,400)`, with A/C/G/T/N encoded as 0/1/2/3/4. `features` is finite float `(N,1069)` or `(1069,)` in training order, using raw values; bundled transforms handle standardization. Invalid inputs raise `PromoterAtlasInputError`.

`predict_gene` returns original-unit E/S/B with `is_model_prediction: true` and `is_catalog_observed: false`. See [label documentation](labels_README.md) for observed-label sources.

`models/` stores E/S/B weights, `transforms/scalers.npz` stores TRAIN-fitted feature and target scaling, and `code/` contains prediction and examples. Top-level files include README, MODEL_CARD, requirements, environment, schema, VERSION_MANIFEST, and negative_ablation_results.
