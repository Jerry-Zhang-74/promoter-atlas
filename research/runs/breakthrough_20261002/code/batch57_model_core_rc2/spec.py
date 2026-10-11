# -*- coding: utf-8 -*-
"""batch53 frozen specification (single source of truth for BOTH neural arms).

Every value here is transcribed ONLY from batch52/PROTOCOL.txt (v6.38.0, top amendment),
batch52/PARENT_DATA_ACCEPTANCE.txt and batch52/PARALLEL_PIPELINE.txt. No new model choice
or hyperparameter sweep is introduced by this module. Both Nabs and Njoint MUST build
against the SAME constants so their initial weights, architecture, optimizer, seed and
config hash are identical; the nested-difference loss is the only intentional difference.

Target correction (PARENT_DATA_ACCEPTANCE): the author-documented model target is
Log2RPM_HEK293. Log2Norm_HEK293 is a DIFFERENT column and is forbidden as a fit target.
"""
from __future__ import annotations
import hashlib, json

# ---- data contract ---------------------------------------------------------
TARGET_COLUMN = "Log2RPM_HEK293"          # corrected author target; NOT Log2Norm_HEK293
# REPAIR b52-integration-01 (2026-10-04): the parent-stated source identity is the SHA256
# "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96" (64 hex). The batch53
# helper had transcribed it with the leading '1' lost (63 hex) as SOURCE_SHA256 and had
# duplicated it into SOURCE_SHA1. Corrected to the actual FILE hashes:
#   sha256sum GSE301246_..._fold0.bed.gz = 1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96
#   sha1sum   GSE301246_..._fold0.bed.gz = 667d17055dd642ca445fa4956d12240d5e221166
# See runs/breakthrough_20261002/batch52/INTEGRATION_SPEC_FIX.txt (config re-freeze).
SOURCE_SHA256 = "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96"  # b52 data owner
SOURCE_SHA1   = "667d17055dd642ca445fa4956d12240d5e221166"
MAX_LEN = 600                              # >600 UNSUPPORTED; 2000 is NOT windowed
MIN_ELIGIBLE_LEN = 100
MAX_ELIGIBLE_LEN = 600
PAD_CODE = 4                               # row of all-zero one-hot (padding / ambiguous)

# ---- frozen architecture (identical bytes for Nabs and Njoint) -------------
SEED = 20261004
IN_CH = 4
STEM_CH = 64
STEM_K = 9
BLOCK_CH = 64
BLOCK_K = 5
BLOCK_DILATIONS = (1, 2, 4)                # exactly THREE residual blocks
N_RES_BLOCKS = 3
DROPOUT = 0.1
HEAD_HIDDEN = 64
POOL_DIM = STEM_CH * 2 + 2                 # masked mean + masked max + loglen + GC = 130
UNSUPPORTED = ">600/null explicitly rejected"

# ---- optimizer / training loop (identical for both arms) -------------------
OPTIMIZER = "AdamW"
LR = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
PAIR_MICROBATCH = 16
MAX_EPOCHS = 6
EPOCH_CAP_CANDIDATES = (6, 4, 2)           # choose ONE cap, apply to BOTH arms
NJOINT_PAIR_WEIGHT = 0.5                   # Njoint = activity + 0.5 * pair-difference

# ---- loss weights ----------------------------------------------------------
WEIGHT_DEF = "inverse TRAIN group size, normalized mean 1"
PAIR_WEIGHT_DEF = "inverse TRAIN pair-group size, normalized mean 1"

# ---- R0 composition ridge --------------------------------------------------
R0_ALPHA = 100.0
R0_SCALE = 1.0
R0_FEATURES = "mono(4)+di(16)+tri(64)+tetra(256)+GCfrac+loglength = 342"  # 340 kmers + 2


def config_dict() -> dict:
    """Canonical frozen config. Persisted into every checkpoint and hashed by the gate."""
    return {
        "batch": "batch53_model_core",
        "spec_version": "v6.38.0-model-pipeline-code.rc1",
        "parent_protocol": "batch52/PROTOCOL.txt",
        "target_column": TARGET_COLUMN,
        "source_sha256": SOURCE_SHA256,
        "max_len": MAX_LEN,
        "min_eligible_len": MIN_ELIGIBLE_LEN,
        "max_eligible_len": MAX_ELIGIBLE_LEN,
        "seed": SEED,
        "in_ch": IN_CH,
        "stem_ch": STEM_CH,
        "stem_k": STEM_K,
        "block_ch": BLOCK_CH,
        "block_k": BLOCK_K,
        "block_dilations": list(BLOCK_DILATIONS),
        "n_res_blocks": N_RES_BLOCKS,
        "dropout": DROPOUT,
        "head_hidden": HEAD_HIDDEN,
        "pool_dim": POOL_DIM,
        "optimizer": OPTIMIZER,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "batch_size": BATCH_SIZE,
        "pair_microbatch": PAIR_MICROBATCH,
        "max_epochs": MAX_EPOCHS,
        "njoint_pair_weight": NJOINT_PAIR_WEIGHT,
        "weight_def": WEIGHT_DEF,
        "pair_weight_def": PAIR_WEIGHT_DEF,
        "r0_alpha": R0_ALPHA,
    }


def config_sha256() -> str:
    blob = json.dumps(config_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_target(column: str) -> None:
    if column != TARGET_COLUMN:
        raise ValueError(
            "target column must be %r (author-documented Log2RPM), got %r; "
            "Log2Norm_HEK293 is forbidden as a fit target" % (TARGET_COLUMN, column))


def assert_source_sha(sha: str) -> None:
    if sha.lower() != SOURCE_SHA256:
        raise ValueError("source SHA mismatch: %r != %r" % (sha, SOURCE_SHA256))
