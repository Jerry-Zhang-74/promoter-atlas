# Model card - Promoter Atlas v6.15.1-release.rc1

## Overview
* Package version: `v6.15.1-release.rc1` (packaging/provenance repair of `v6.15.0-multitask.rc1` weights).
* Task: human gene-level natural-expression regression, targets E / S / B.
* Architecture: independent arm A = three single-task CNNs (1D CNN trunk over a 4-promoter bag +
  frozen 1069-feature composition/position MLP branch; 217,857 params each).
* Reference seed: 20261016 (the split-key primary seed). Second seed 20261017 is a replication only.
* Selection: validation-only frozen rule (mean per-target val rho gain >= 0.02 AND no target drop
  > 0.02). The rule did not promote the shared arms, so the independent composite ships.

## Intended use
Exploratory ranking of human genes by natural expression strength / cross-tissue stability / breadth,
given a fixed 4-promoter bag and a fixed 1069-dim composition/position feature vector.

## Out of scope
Promoter-fragment or synthetic-construct activity; HEK293T reporter activity; non-human species;
isoform/tissue-specific expression; any temporal/dynamic stability claim from S.

## Metrics (from saved predictions, recomputed)
Test split = retrospective heldout (the split predates batch16; its genes were development-visible).

Shipped single seed (A_s20261016), raw units, test:
| head | Spearman | R2 | MAE | n |
|---|---|---|---|---|
| E | 0.7428 | 0.5618 | 0.1513 | 5851 |
| S | 0.4341 | 0.1698 | 0.2066 | 1880 |
| B | 0.7262 | 0.6030 | 0.1786 | 5851 |

Mean-of-seed (2 seeds) test: E 0.7411 / S 0.4273 / B 0.7215.
Ensemble-prediction (mean over seeds) test: E 0.7459 / S 0.4335 / B 0.7252.
Matched HGB baseline test: E 0.7335 / S 0.4334 / B 0.7191 (S is a statistical tie with the neural model).

## Known limitations (read before use)
1. **Trainer/init/data-order confound.** The A reference fit and A seed-2 E/S came from the original
   trainer (global RNG shuffle); the B/C arms and A seed-2 B came from the recovery trainer
   (deterministic per-epoch sampler seed) after a process loss. Same algorithm/optimizer/loss/epochs,
   but different initialization RNG stream and data-order schedule. This confounds every
   shared-vs-independent contrast and blocks a strictly controlled causal task-sharing conclusion.
2. **Auxiliary missingness unresolved.** Epigenetic auxiliary targets encode uncovered assays as 0.0
   with no NaN flag; zeros are treated as unresolved absence-vs-missing. No benefit or detriment of
   auxiliary supervision is claimed.
3. **Family generalization.** Discovered regulation families cover only 14 constructs; generalization
   to unseen families is not demonstrated.
4. **Split non-comparability.** The corrected component-union split is not comparable with earlier
   random-split results; old scores must not be compared directly to these.
5. **Inputs required.** Without the true 4-promoter bag + features the model cannot be meaningfully
   applied; do not fabricate inputs.
