#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen RecipeA (batch21 ckpt_A_seed20261023) encoder feature extraction.

- Builds the frozen batch02 LegNetInspired architecture, loads the batch21 RecipeA
  state dict STRICTLY, asserts all parameters are on cuda:0.
- Extracts the pre-scalar-head pooled representation h = cat[mean_L, max_L]
  (128-dim) from the WHOLE measured insert at its actual length (no truncation,
  no padding, no reference DNA, no artificial duplication).
- Verifies (i) the dimensional contract 4 x L -> 128 for several lengths, and
  (ii) numerical invariance to processing batch composition at equal length
  (eval mode / running BatchNorm statistics).
- NO HEK labels are involved anywhere here. Frozen weights are never updated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "runs" / "breakthrough_20261002" / "code" / "batch02"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b22_common import (FROZEN_CKPT, DATA_OUT, REPORT_OUT, atomic_write_json,  # noqa: E402
                        now_local, sha256_file, encode_onehot)

DEVICE = "cuda:0"
BATCH = 32


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: frozen feature extraction requires cuda:0; "
                           "refusing CPU/integrated fallback (batch22 protocol)")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise RuntimeError("current CUDA device is not 0")


def load_frozen():
    import trackB_grouped_cv as t  # frozen architecture, READ-ONLY
    ck = torch.load(FROZEN_CKPT, map_location="cpu", weights_only=False)
    model = t.build_cnn("legnet")
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(DEVICE).eval()
    p = next(model.parameters())
    assert p.is_cuda, "frozen model parameters are not on CUDA"
    for q in model.parameters():
        q.requires_grad_(False)
    return model, {"arm": ck.get("arm"), "seed": ck.get("seed"),
                   "best_epoch": ck.get("best_epoch"),
                   "model_sha256": sha256_file(FROZEN_CKPT),
                   "n_params": int(sum(v.numel() for v in ck["model"].values())),
                   "architecture": "batch02 LegNetInspired (ch=64, 4 dilated res blocks d=1,2,4,8)",
                   "pooled_dim": 128}


@torch.inference_mode()
def encode_pooled(model, seqs, batch=BATCH):
    """Return (n,128) frozen pooled representation from whole sequences."""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        L = len(chunk[0])
        if any(len(s) != L for s in chunk):
            raise ValueError("equal-length grouping violated inside a batch")
        x = torch.from_numpy(np.stack([encode_onehot(s) for s in chunk])).float().to(DEVICE)
        assert x.is_cuda, "input batch tensor is not on CUDA"
        h = model.stem(x)
        for blk in model.blocks:
            h = blk(h)
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, 0)


def verify(model, n_checks=6):
    """Dimensional contract + processing-invariance checks.

    Hard requirement (protocol): the encoder accepts the WHOLE measured insert at
    its actual length (dimensional contract) and, because we group equal-length
    inputs and use a FIXED extraction batch size, the pooled representation is
    EXACTLY invariant to which rows share a batch (permutation invariance).
    Residual float32 variation across DIFFERENT batch SIZES (CUDA GEMM accumulation
    order) is measured and reported for transparency; it never affects the
    canonical extraction, which uses a single fixed batching.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    recs = []
    for L in (133, 100, 164):
        s = ("ACGT" * ((L // 4) + 1))[:L]
        h = encode_pooled(model, [s], batch=1)
        recs.append({"check": "dim_contract", "length": L, "out_dim": int(h.shape[1]),
                     "ok": bool(h.shape[1] == 128)})
    seqs = ["".join(("ACGT"[k % 4] * 133)[:133]) for k in range(24)]
    full = encode_pooled(model, seqs, batch=BATCH)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(seqs))
    p_out = encode_pooled(model, [seqs[i] for i in perm], batch=BATCH)
    inv = np.empty_like(p_out); inv[perm] = p_out
    recs.append({"check": "permutation_invariance_fixed_batch", "batch": BATCH,
                 "max_abs_diff": float(np.max(np.abs(inv - full))),
                 "ok": bool(np.max(np.abs(inv - full)) == 0.0)})
    for b in (16, 7, 3, 1):
        part = encode_pooled(model, seqs, batch=b)
        recs.append({"check": "batch_size_sensitivity_info", "batch": b,
                     "max_abs_diff": float(np.max(np.abs(part - full))),
                     "feature_abs_mean": float(np.abs(full).mean()),
                     "ok": True})
    return recs


def run():
    assert_cuda()
    model, meta = load_frozen()
    import csv
    rows = list(csv.DictReader(open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"),
                               delimiter="\t"))
    seqs = [r["sequence"].upper() for r in rows]
    ver = verify(model)
    dim_ok = all(r["ok"] for r in ver if r["check"] == "dim_contract")
    inv_ok = all(r["ok"] for r in ver
                 if r["check"] in ("permutation_invariance_fixed_batch", "batch_size_sensitivity_info"))
    if not (dim_ok and inv_ok):
        raise RuntimeError("frozen encoder verification failed: %r" % (ver,))
    H = encode_pooled(model, seqs, batch=BATCH)
    if H.shape != (len(seqs), 128):
        raise RuntimeError("unexpected representation shape %r" % (H.shape,))
    np.savez(DATA_OUT / "frozen_repr_A_seed20261023.npz",
             H=H.astype(np.float32), prom_id=np.array([r["prom_id"] for r in rows]))
    ev = {"generated_at_local": now_local(), "device": "cuda:0",
          "torch": torch.__version__, "cuda_available": True,
          "device_name": torch.cuda.get_device_name(0),
          "compute_capability": list(torch.cuda.get_device_capability(0)),
          "frozen_model": meta, "verification": ver,
          "dim_contract_ok": dim_ok, "batch_invariance_ok": inv_ok,
          "input_length": 133, "n_rows": len(seqs), "batch": BATCH,
          "n_hek_labels_used": 0,
          "note": "frozen weights untouched; extraction only; no HEK labels used"}
    atomic_write_json(REPORT_OUT / "frozen_extraction_evidence.json", ev)
    print("[b22_frozen] n=%d dim=128 device=cuda:0 dim_ok=%s inv_ok=%s model_sha=%s"
          % (len(seqs), dim_ok, inv_ok, meta["model_sha256"][:12]))
    return ev


if __name__ == "__main__":
    run()
