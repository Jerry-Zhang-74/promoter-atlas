# License, provenance and redistribution notes

## What is included
- New project model architecture, training configuration and predictor code written for this
  project (batch52/batch53/batch55). These are project-authored artifacts.
- A model trained from scratch on author-published, publicly redistributable HEK293
  MPRA-style measurements (the GSE301246 PARM training library, fold 0). Only the *trained
  parameters* of the new model are packaged - never the upstream source data or reference.
- The pinned model-core modules (`spec.py`, `model_core.py`, `normalizer.py`, `checkpoint.py`,
  `io_utils.py`) are frozen copies of the project's own batch53 code with recorded SHA256.

## Checkpoint state (accurate statement)
- The released checkpoint is a *complete* training checkpoint, so in addition to the model
  parameters it also contains optimizer, scheduler, sampler and RNG state entries. This is an
  artifact of how the training code saves checkpoints; the file is shipped **unchanged**
  (bytes identical to the reviewed release, SHA256 `{SHA}`).
- Their presence is a **packaging fact only**. This bundle does **not** claim, validate or
  support exact-resume training, and no resume workflow is documented or provided here.
- Prediction depends only on the model parameters together with the SHA256-pinned config and
  normalizer; the predictor reads exactly one sidecar field from `model/*.sha256.json`
  (`checkpoint_sha256`) and never reads its recorded provenance path at runtime.

## What is deliberately NOT included
- No official PARM / external pretrained weights and no source-data redistribution.
- No reference genome, no raw measurement tables, no large caches.
- No API keys, email keys, credentials or private tokens of any kind.
- No old failed wrong-target (Log2Norm) R0 artifacts.

## Third-party components
- PyTorch (BSD-3-Clause) and NumPy (BSD-3-Clause) are runtime dependencies, not redistributed.
- No third-party model weights are bundled.

## Scientific scope and limits (no over-claiming)
- Target: author-documented **Log2RPM_HEK293**. This is *not* `Log2Norm_HEK293`.
- The encoded sequences are a **reference-genome proxy** for the donor library, not the donor
  clone's own sequence.
- Cell identity is HEK293 (the measurement source). It is **not validated on local 293FT**
  cells; no local 293FT transfer claim is made.
- The model was trained on author fold 0 only; this is not the entire multi-fold study.
- Fresh random initialization was used; no external pretrained exposure is claimed beyond what
  the project protocol records.

## Honest quality statement
Any performance numbers shipped with a model are copied verbatim from the frozen, actual
held-out results JSON produced by the independent evaluator. If that file is absent, the
bundle says so; no invented weights or metrics are ever emitted.
