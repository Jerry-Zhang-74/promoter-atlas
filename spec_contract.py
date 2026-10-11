# -*- coding: utf-8 -*-
"""双窗口输入、物种编码与 E/S/B 定义。"""
from __future__ import annotations

import numpy as np

LOCAL_LEN = 400
CONTEXT_LEN = 2000
PAD_CODE = 4
ESB_COLS = ("Score_E", "Score_S", "Score_B")

BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

SPECIES = ["homo_sapiens", "pan_troglodytes", "macaca_mulatta", "callithrix_jacchus",
           "mus_musculus", "rattus_norvegicus", "cavia_porcellus",
           "oryctolagus_cuniculus", "bos_taurus", "sus_scrofa", "ovis_aries",
           "capra_hircus", "equus_caballus", "canis_lupus_familiaris",
           "felis_catus", "pteropus_vampyrus", "loxodonta_africana",
           "monodelphis_domestica", "ornithorhynchus_anatinus"]
SPECIES_ID = {s: i for i, s in enumerate(SPECIES)}

# 三个端点均在同一套 GTEx 基因表达量上派生，按来源基因求值后做全目录秩归一化。
# 标签公式见 labels/spec/label_definition.md 与 labels/build 下的构建脚本：
#   E_raw = mean_tissues( log2(1 + mean_tpm) )
#   B_raw = #{tissue: mean_tpm >= 1} / #tissue
#   S_raw = -MAD_detectable_tissues( log2(1 + mean_tpm) )，仅在该基因 B_raw >= 0.30 且可检出组织数 >= 3 时定义
#   Score_X = rank_scale(X_raw)，升序秩 / (n - 1)，取自完整目录
ESB_DEFINITIONS = {
    "Score_E": "E 轴：表达强度。取自基因层面 GTEx 表达量，E_raw = 各组织 log2(1+mean_tpm) 的均值；"
               "目标是该量的全目录秩归一化分数 Score_E。",
    "Score_S": "S 轴：跨组织稳定性。S_raw = 可检出组织上 log2(1+mean_tpm) 的负 MAD，仅在 B_raw >= 0.30 "
               "且可检出组织数 >= 3 时定义；目标是该量的全目录秩归一化分数 Score_S，未定义处为空。"
               "筛选时分别比较 E 与 S。",
    "Score_B": "B 轴：表达广度。B_raw = 可检出组织占比（mean_tpm >= 1 的组织数 / 组织总数）；"
               "目标是该量的全目录秩归一化分数 Score_B，在选型中用作门槛/分层。",
}
# 标签本身住在 [0,1] 的秩归一化刻度上；模型头是线性层，输出保留连续值。
LABEL_SCALE = (0.0, 1.0)

INPUT_COLUMNS = ("candidate_id", "sequence_sym400", "sequence_up2000", "species")
OPTIONAL_INPUT_COLUMNS = ("gene_id", "record_id", "parent_id", "edit")


def encode_sequence(seq: str) -> np.ndarray:
    """A/C/G/T -> 0..3; any other character -> 4 (padding / ambiguous)."""
    return np.array([BASE_TO_CODE.get(b, PAD_CODE) for b in seq.upper()], dtype=np.uint8)


def validate_row(row: dict) -> dict:
    """Return the row with sequences upper-cased and validated. Raise ValueError if invalid."""
    cid = (row.get("candidate_id") or "").strip()
    if not cid:
        raise ValueError("candidate_id must be a non-empty string")
    out = dict(row)
    out["candidate_id"] = cid
    for col, length in (("sequence_sym400", LOCAL_LEN), ("sequence_up2000", CONTEXT_LEN)):
        seq = (row.get(col) or "").upper()
        if len(seq) != length:
            raise ValueError("%s: %s must contain exactly %d bases (got %d)"
                             % (cid, col, length, len(seq)))
        bad = set(seq) - set("ACGTN")
        if bad:
            raise ValueError("%s: %s contains unsupported characters %s"
                             % (cid, col, "".join(sorted(bad))))
        out[col] = seq
    species = (row.get("species") or "homo_sapiens").strip()
    if species not in SPECIES_ID:
        raise ValueError("%s: unknown species %r (see spec_contract.SPECIES)" % (cid, species))
    out["species"] = species
    return out
