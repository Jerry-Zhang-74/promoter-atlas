# Promoter Atlas v6.15.1-release.rc1 (EXPERIMENTAL CANDIDATE)

**Status: experimental candidate, independent-arm composite. NOT a validated shared network,
NOT a HEK293T / fragment-activity model.** The version increment is packaging/provenance repair
only - the weights are the unchanged batch16 (v6.15.0-multitask.rc1) fits.

## What this is
A human **gene-level natural-expression** predictor for three targets, each from a gene bag of
4 promoter sequences + one frozen 1069-dim composition/position feature vector:

| head | target | definition |
|---|---|---|
| E | `Score_E` | rank-scaled expression strength, [0,1] |
| S | `S_raw` | `-MAD(log2(1+detectable TPM))` over detectable tissues (S_defined only) |
| B | `B_raw` | `detectable_tissue_count / tissue_count`, [0,1] |

The shipped architecture is **independent** (3 separate single-task CNNs, reference seed 20261016).
The shared-backbone arms B (E/S/B heads) and C (+epigenetic auxiliary supervision) were tested and
**rejected by the frozen validation rule** (mean per-target val rho gain -0.0061 and -0.0049, both
< +0.02). See `negative_ablation_results.json`.

## Quick start (no training assets needed)
```
python code/run_example.py
python code/promoter_atlas_predict.py --input code/example_input.npz --out preds.json
```

## Python API
```python
import numpy as np, sys
sys.path.insert(0, "code")
from promoter_atlas_predict import PromoterAtlasGenePredictor, PromoterAtlasInputError
p = PromoterAtlasGenePredictor()
z = np.load("code/example_input.npz", allow_pickle=True)
res = p.predict_batch(z["seqs"], z["features"])     # dict E/S/B -> np.ndarray
one = p.predict_gene(z["seqs"][0], z["features"][0])  # structured dict + scope flags
```

## Input contract (strict)
* `seqs`: shape `(N,4,400)` (or `(4,400)`), integer `uint8`, alphabet `{A,C,G,T,N} -> {0,1,2,3,4}`.
* `features`: shape `(N,1069)` (or `(1069,)`), finite floats, **RAW** (unstandardized) values in the
  fixed training order. Do **not** standardize manually - the shipped transforms do it.
* Violations raise `PromoterAtlasInputError` (shape / dtype / alphabet / non-finite).

## Output contract
`predict_gene` returns raw-unit E/S/B values plus `is_model_prediction: true` and
`is_catalog_observed: false`. Catalog **observed** labels are a separate artifact - see `labels/README.md`.

## Not supported
Arbitrary promoter-fragment activity; HEK293T/synthetic-construct activity; multi-species; any
temporal-stability reading of S. Raw S is a **negative cross-tissue MAD**, not temporal stability.

## Contents
```
README.md  MODEL_CARD.md  requirements.txt  environment.json  schema.json
VERSION_MANIFEST.json  negative_ablation_results.json
models/       model_E.pt model_S.pt model_B.pt   (independent arm, reference seed 20261016)
transforms/   scalers.npz                        (train-fitted: X mean/std + per-target mean/std)
code/         promoter_atlas_predict.py run_example.py example_input.npz example_expected_output.json
labels/       README.md                          (observed-label reference pointer)
```
