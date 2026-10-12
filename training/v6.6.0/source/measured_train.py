# -*- coding: utf-8 -*-
"""Multi-task promoter training with measured ESB, omics, TF, tissue profiles and fragment activity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import JointDualESBModel, ESB_COLS
from .paths import ROOT
from .layout import project_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMSTART_REL = "models/versions/v6.0.0/weights/promoter_atlas_v6_round1.pt"
WARMSTART_SHA256 = (
    "109fc8919c570531f142390c703f956d4300cd63ccf825c65e7e3bd0748f1096"
)
OLD_TRAINER_REL = "models/versions/v6.0.0/source/trainer_v6_round1.py"

OMICS_COLS = [
    "cage_tss", "procap_tss", "h3k4me3", "h3k27ac", "h3k27me3",
    "h3k4me1", "h3k36me3", "dnase", "atac",
]
OLD_OMICS_COLS = OMICS_COLS[:8]

N_SPECIES = 19
N_TF = 400
N_OMICS = 9
N_DIST = 0

ARCH = {
    "local_ch": 160,
    "ctx_ch": 192,
    "local_blocks": 6,
    "ctx_blocks": 8,
    "tf_layers": 0,
    "ctx10k": False,
}

ACTIVITY_HIDDEN = 256
ACTIVITY_DROPOUT = 0.1

TASK_WEIGHTS = {
    "esb": 1.0,
    "ortho": 0.5,
    "omics": 0.5,
    "tf": 0.3,
    "species": 0.2,
    "mask": 0.3,
    "activity": 0.2,
}

MLM_RATE = 0.15
MLM_MASK_CODE = 4

DEFAULT_DEADLINE = "2026-10-12T20:28:00+08:00"
DEFAULT_OUTDIR = "experiments/retrain/v6.6_measured_20261012"
DEFAULT_TABLE = "data/processed/labels/build/promoter_atlas_v6_6_input.tsv"

FORMAT = "promoter_atlas_joint_esb.v1"
MODEL_VERSION = "v6.6.0"
RUN_ID = "v6.6.0-measured-20261012"

RESUME_FORMAT = "promoter_atlas_measured_v6.6.resume.v1"

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_deadline(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError("deadline must be timezone-aware: %r" % s)
    return dt


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def rng_state_all():
    import random as _random
    st = {
        "python": list(_random.getstate()[1]),
        "python_version": _random.getstate()[0],
        "python_gauss": _random.getstate()[2],
        "numpy": np.random.get_state()[1].tolist(),
        "numpy_pos": int(np.random.get_state()[2]),
        "numpy_has_gauss": float(np.random.get_state()[3]),
        "numpy_cached": float(np.random.get_state()[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_restore_all(st):
    import random as _random
    _random.setstate((st["python_version"], tuple(st["python"]), st.get("python_gauss")))
    np.random.set_state(("MT19937", np.array(st["numpy"], dtype=np.uint32),
                         int(st["numpy_pos"]), float(st["numpy_has_gauss"]),
                         float(st["numpy_cached"])))
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def deterministic_perm(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    return rng.permutation(n)


# ---------------------------------------------------------------------------
# Warmstart
# ---------------------------------------------------------------------------


def _check_warmstart_hash(path) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError("warmstart checkpoint not found: %s" % path)
    got = sha256_file(path)
    if got != WARMSTART_SHA256:
        raise ValueError(
            "warmstart SHA256 mismatch: got %s expected %s" % (got, WARMSTART_SHA256)
        )
    return got


def _rotate_fuse_input(old_w: torch.Tensor) -> torch.Tensor:
    """Old fuse.0.weight input order [a(local320), b(context384), s(species32)]
    -> current [s, a, b]. Rotate last 32 columns to front."""
    if old_w.shape[1] < 32:
        raise ValueError("fuse weight has too few input columns: %s" % (old_w.shape,))
    a_b = old_w[:, :-32]
    s = old_w[:, -32:]
    return torch.cat([s, a_b], dim=1)


def _reparam_linear_row(old_w, old_b, old_mu, old_sd, new_mu, new_sd):
    """Reparameterize a linear head row from old standardized scale to new.

    new W = oldW * (old_sd / new_sd)
    new b = (old_b * old_sd + old_mu - new_mu) / new_sd
    """
    old_sd_t = torch.as_tensor(old_sd, dtype=old_w.dtype)
    old_mu_t = torch.as_tensor(old_mu, dtype=old_w.dtype)
    new_sd_t = torch.as_tensor(new_sd, dtype=old_w.dtype)
    new_mu_t = torch.as_tensor(new_mu, dtype=old_w.dtype)
    new_w = old_w * (old_sd_t / new_sd_t).unsqueeze(1)
    new_b = (old_b * old_sd_t + old_mu_t - new_mu_t) / new_sd_t
    return new_w, new_b


def _load_old_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError("warmstart checkpoint missing state_dict")
    sd = obj["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("warmstart state_dict is not a dict")
    return obj, sd


def warmstart_model(cfg, normalizers):
    """Build a JointDualESBModel and warmstart it from the clean v6.0.0 checkpoint.

    Returns (model, report). The report records transfer counts and all transformations.
    """
    root = cfg.get("root") or str(ROOT)
    ws_rel = cfg.get("warmstart", WARMSTART_REL)
    ws_path = project_path(root, ws_rel) if not os.path.isabs(ws_rel) else ws_rel
    ws_sha = _check_warmstart_hash(ws_path)

    ck, old_sd = _load_old_state_dict(ws_path)
    if int(ck.get("n_tf", -1)) != N_TF:
        raise ValueError("warmstart n_tf mismatch: %r" % ck.get("n_tf"))
    if int(ck.get("n_omics", -1)) != 8:
        raise ValueError("warmstart n_omics mismatch: %r" % ck.get("n_omics"))
    s_stats = ck.get("s_stats")
    if not s_stats or len(s_stats) != 2:
        raise ValueError("warmstart missing s_stats")
    s_mean, s_sd = float(s_stats[0]), float(s_stats[1])

    n_profile = int(len(normalizers["profile"]["columns"]))
    model = JointDualESBModel(
        arch=dict(ARCH), n_dist=N_DIST, n_omics=N_OMICS, n_tf=N_TF,
        activity_hidden=ACTIVITY_HIDDEN, activity_dropout=ACTIVITY_DROPOUT,
        n_profile=n_profile,
    )

    report = {
        "warmstart_path": ws_path,
        "warmstart_sha256": ws_sha,
        "warmstart_name": ck.get("name"),
        "s_stats": [s_mean, s_sd],
        "transferred": [],
        "reinitialized": [],
        "reparameterized": [],
        "rotated": [],
        "unexpected": [],
    }

    new_sd = model.state_dict()
    new_sd = {k: v.clone() for k, v in new_sd.items()}

    # Map old unprefixed keys to base.*
    old_to_new = {}
    for k in old_sd:
        old_to_new[k] = "base." + k

    # Keys that must be reinitialized (not transferred)
    reinit_prefixes = (
        "base.head_species.",
        "base.head_mpra.",
        "base.head_profile.",
        "head_activity.",
    )

    # Old omics order -> new omics order mapping
    old_omics_idx = {name: i for i, name in enumerate(OLD_OMICS_COLS)}
    new_omics_idx = {name: i for i, name in enumerate(OMICS_COLS)}

    # Normalizers
    esb_norm = normalizers["esb"]
    esb_mean = np.asarray(esb_norm["mean"], dtype=np.float64)
    esb_std = np.asarray(esb_norm["std"], dtype=np.float64)
    ortho_norm = normalizers["ortholog"]
    ortho_mean = float(np.asarray(ortho_norm["mean"], dtype=np.float64)[0])
    ortho_std = float(np.asarray(ortho_norm["std"], dtype=np.float64)[0])
    omics_norm = normalizers["omics"]
    omics_mean = np.asarray(omics_norm["mean"], dtype=np.float64)
    omics_std = np.asarray(omics_norm["std"], dtype=np.float64)

    # Old ESB mu/sd: E and B original scale, S standardized by ck.s_stats
    old_esb_mu = np.array([0.0, s_mean, 0.0], dtype=np.float64)
    old_esb_sd = np.array([1.0, s_sd, 1.0], dtype=np.float64)

    for old_key, new_key in old_to_new.items():
        if any(new_key.startswith(p) for p in reinit_prefixes):
            report["reinitialized"].append(new_key)
            continue
        if new_key not in new_sd:
            report["unexpected"].append(old_key)
            continue
        old_t = old_sd[old_key]
        new_t = new_sd[new_key]

        # fuse.0.weight rotation
        if old_key == "fuse.0.weight":
            if old_t.shape[1] != new_t.shape[1]:
                raise ValueError(
                    "fuse.0.weight input dim mismatch: old %s new %s"
                    % (old_t.shape, new_t.shape)
                )
            new_t = _rotate_fuse_input(old_t)
            report["rotated"].append(new_key)
            new_sd[new_key] = new_t
            report["transferred"].append(new_key)
            continue

        # head_esb reparameterization
        if old_key == "head_esb.weight":
            new_w, _ = _reparam_linear_row(
                old_t, torch.zeros(old_t.shape[0], dtype=old_t.dtype),
                old_esb_mu, old_esb_sd, esb_mean, esb_std,
            )
            new_sd[new_key] = new_w
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue
        if old_key == "head_esb.bias":
            _, new_b = _reparam_linear_row(
                torch.zeros(1, old_t.shape[0], dtype=old_t.dtype), old_t,
                old_esb_mu, old_esb_sd, esb_mean, esb_std,
            )
            new_sd[new_key] = new_b
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue

        # head_ortho reparameterization (old raw E -> new ortholog z)
        if old_key == "head_ortho.weight":
            new_w, _ = _reparam_linear_row(
                old_t, torch.zeros(old_t.shape[0], dtype=old_t.dtype),
                [0.0], [1.0], [ortho_mean], [ortho_std],
            )
            new_sd[new_key] = new_w
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue
        if old_key == "head_ortho.bias":
            _, new_b = _reparam_linear_row(
                torch.zeros(1, old_t.shape[0], dtype=old_t.dtype), old_t,
                [0.0], [1.0], [ortho_mean], [ortho_std],
            )
            new_sd[new_key] = new_b
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue

        # head_omics: map old 8 columns to new 9 by name
        if old_key == "head_omics.weight":
            if old_t.shape[0] != len(OLD_OMICS_COLS):
                raise ValueError("old head_omics rows mismatch: %s" % (old_t.shape,))
            new_w = new_sd[new_key].clone()
            for name, oi in old_omics_idx.items():
                ni = new_omics_idx[name]
                w_row, _ = _reparam_linear_row(
                    old_t[oi:oi + 1], torch.zeros(1, dtype=old_t.dtype),
                    [0.0], [1.0], [omics_mean[ni]], [omics_std[ni]],
                )
                new_w[ni] = w_row[0]
            new_sd[new_key] = new_w
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue
        if old_key == "head_omics.bias":
            if old_t.shape[0] != len(OLD_OMICS_COLS):
                raise ValueError("old head_omics bias mismatch: %s" % (old_t.shape,))
            new_b = new_sd[new_key].clone()
            for name, oi in old_omics_idx.items():
                ni = new_omics_idx[name]
                _, b_val = _reparam_linear_row(
                    torch.zeros(1, 1, dtype=old_t.dtype), old_t[oi:oi + 1],
                    [0.0], [1.0], [omics_mean[ni]], [omics_std[ni]],
                )
                new_b[ni] = b_val[0]
            new_sd[new_key] = new_b
            report["reparameterized"].append(new_key)
            report["transferred"].append(new_key)
            continue

        # head_tf: MANDATORY reset (old TF head learned misaligned labels)
        if old_key.startswith("head_tf."):
            report["reinitialized"].append(new_key)
            continue

        # Generic transfer with shape check
        if tuple(old_t.shape) != tuple(new_t.shape):
            raise ValueError(
                "shape mismatch for %s -> %s: old %s new %s"
                % (old_key, new_key, tuple(old_t.shape), tuple(new_t.shape))
            )
        new_sd[new_key] = old_t.clone()
        report["transferred"].append(new_key)

    # Load the assembled state dict
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        raise ValueError("unexpected keys after warmstart: %s" % (unexpected,))
    # Missing keys are the reinitialized ones; verify they are only allowed ones
    allowed_missing = set()
    for k in model.state_dict():
        if any(k.startswith(p) for p in reinit_prefixes):
            allowed_missing.add(k)
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise ValueError("unexpected missing keys after warmstart: %s" % (bad_missing,))

    # Freeze unused head_mpra
    for p in model.base.head_mpra.parameters():
        p.requires_grad_(False)

    report["n_transferred"] = len(report["transferred"])
    report["n_reinitialized"] = len(report["reinitialized"])
    report["n_reparameterized"] = len(report["reparameterized"])
    report["n_rotated"] = len(report["rotated"])
    report["n_unexpected"] = len(report["unexpected"])
    return model, report


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _masked_mse(pred, target, mask):
    """Return (numerator_sum, denominator_count)."""
    m = mask.to(pred.dtype)
    diff = (pred - target) * m
    num = (diff * diff).sum()
    den = m.sum()
    return num, den


def compute_multitask_loss(out, batch, weights=None):
    """Compute the multi-task loss.

    Returns (total_loss, stats) where stats maps task -> dict(num, den, loss).
    Tasks with zero valid mask count are skipped (no NaN).
    """
    if weights is None:
        weights = TASK_WEIGHTS
    stats = {}
    total = None

    def add(task, num, den):
        nonlocal total
        den_f = float(den.detach().cpu()) if torch.is_tensor(den) else float(den)
        if den_f <= 0.0:
            stats[task] = {"num": 0.0, "den": 0.0, "loss": None}
            return
        loss = num / den
        stats[task] = {
            "num": float(num.detach().cpu()),
            "den": den_f,
            "loss": float(loss.detach().cpu()),
        }
        w = float(weights.get(task, 0.0))
        term = w * loss
        total = term if total is None else total + term

    # ESB
    if "esb" in out and "targets_ESB_z" in batch:
        num, den = _masked_mse(out["esb"], batch["targets_ESB_z"], batch["finite_mask"])
        add("esb", num, den)

    # Ortholog
    if "ortho" in out and "ortholog_E_z" in batch:
        num, den = _masked_mse(out["ortho"], batch["ortholog_E_z"], batch["ortholog_mask"])
        add("ortho", num, den)

    # Omics
    if "omics" in out and "targets_omics_z" in batch:
        num, den = _masked_mse(out["omics"], batch["targets_omics_z"], batch["omics_mask"])
        add("omics", num, den)

    # Profile
    if "profile" in out and "targets_profile_z" in batch:
        num, den = _masked_mse(out["profile"], batch["targets_profile_z"],
                               batch["profile_mask"])
        add("profile", num, den)

    # TF (BCE with logits)
    if "tf" in out and "targets_tf" in batch:
        m = batch["tf_mask"].to(out["tf"].dtype)
        bce = F.binary_cross_entropy_with_logits(
            out["tf"], batch["targets_tf"], reduction="none"
        )
        num = (bce * m).sum()
        den = m.sum()
        add("tf", num, den)

    # Species (cross entropy on TRAIN rows only)
    if "species" in out and "species" in batch:
        sp_mask = batch.get("species_loss_mask")
        if sp_mask is not None:
            sel = sp_mask > 0
            if sel.any():
                logits = out["species"][sel]
                tgt = batch["species"][sel].long()
                num = F.cross_entropy(logits, tgt, reduction="sum")
                den = sel.sum().to(num.dtype)
                add("species", num, den)
            else:
                stats["species"] = {"num": 0.0, "den": 0.0, "loss": None}

    # MLM mask head
    if "mask" in out and "mlm_labels" in batch:
        labels = batch["mlm_labels"]
        sel = labels >= 0
        if sel.any():
            logits = out["mask"][sel]
            tgt = labels[sel].long()
            num = F.cross_entropy(logits, tgt, reduction="sum")
            den = sel.sum().to(num.dtype)
            add("mask", num, den)
        else:
            stats["mask"] = {"num": 0.0, "den": 0.0, "loss": None}

    # Activity (fragment)
    if "activity_pred" in batch and "activity_target" in batch:
        w = batch["activity_weight"]
        diff = batch["activity_pred"] - batch["activity_target"]
        num = (w * diff * diff).sum()
        den = w.sum()
        add("activity", num, den)


    if total is None:
        total = torch.zeros((), dtype=torch.float32)
    return total, stats


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_streams(cfg):
    from .measured_data import MeasuredESBStream, FragmentStream
    root = cfg.get("root") or str(ROOT)
    table = cfg.get("table_path", DEFAULT_TABLE)
    table_path = project_path(root, table) if not os.path.isabs(table) else table
    esb = MeasuredESBStream(root, table_path=table_path)
    frag = FragmentStream(root)
    return esb, frag


def _collate_esb(records, device):
    """Collate ESB records into a batch dict."""
    B = len(records)
    x400 = torch.tensor(
        np.stack([r["local_codes400"] for r in records]), dtype=torch.long, device=device
    )
    x2000 = torch.tensor(
        np.stack([r["context_codes2000"] for r in records]), dtype=torch.long, device=device
    )
    species = torch.tensor(
        [int(r["species"]) for r in records], dtype=torch.long, device=device
    )
    targets_ESB_z = torch.tensor(
        np.stack([r["targets_ESB_z"] for r in records]), dtype=torch.float32, device=device
    )
    finite_mask = torch.tensor(
        np.stack([r["finite_mask"] for r in records]), dtype=torch.float32, device=device
    )
    ortholog_E_z = torch.tensor(
        [float(r["ortholog_E_z"]) for r in records], dtype=torch.float32, device=device
    )
    ortholog_mask = torch.tensor(
        [float(r["ortholog_mask"]) for r in records], dtype=torch.float32, device=device
    )
    targets_omics_z = torch.tensor(
        np.stack([r["targets_omics_z"] for r in records]), dtype=torch.float32, device=device
    )
    omics_mask = torch.tensor(
        np.stack([r["omics_mask"] for r in records]), dtype=torch.float32, device=device
    )
    targets_tf = torch.tensor(
        np.stack([r["targets_tf"] for r in records]), dtype=torch.float32, device=device
    )
    tf_mask = torch.tensor(
        np.stack([r["tf_mask"] for r in records]), dtype=torch.float32, device=device
    )
    species_loss_mask = torch.tensor(
        [float(r.get("species_loss_mask", 0.0)) for r in records],
        dtype=torch.float32, device=device,
    )
    mlm_loss_mask = torch.tensor(
        [float(r.get("mlm_loss_mask", 0.0)) for r in records],
        dtype=torch.float32, device=device,
    )
    targets_profile_z = torch.tensor(
        np.stack([r["targets_profile_z"] for r in records]),
        dtype=torch.float32, device=device,
    )
    profile_mask = torch.tensor(
        np.stack([r["profile_mask"] for r in records]),
        dtype=torch.float32, device=device,
    )
    return {
        "x400": x400,
        "x2000": x2000,
        "species": species,
        "targets_ESB_z": targets_ESB_z,
        "finite_mask": finite_mask,
        "ortholog_E_z": ortholog_E_z,
        "ortholog_mask": ortholog_mask,
        "targets_omics_z": targets_omics_z,
        "omics_mask": omics_mask,
        "targets_tf": targets_tf,
        "tf_mask": tf_mask,
        "species_loss_mask": species_loss_mask,
        "mlm_loss_mask": mlm_loss_mask,
        "targets_profile_z": targets_profile_z,
        "profile_mask": profile_mask,
    }


def _apply_mlm(x400, mlm_loss_mask, generator):
    """Independently select 15% positions where original codes<4, replace with 4.

    Returns (x400_masked, mlm_labels). Labels are -100 except selected positions.
    """
    B, L = x400.shape
    x400_masked = x400.clone()
    mlm_labels = torch.full((B, L), -100, dtype=torch.long, device=x400.device)
    eligible = (x400 < 4) & (mlm_loss_mask.unsqueeze(1) > 0)
    rand = torch.rand((B, L), generator=generator, device=x400.device)
    selected = eligible & (rand < MLM_RATE)
    if selected.any():
        mlm_labels[selected] = x400[selected]
        x400_masked[selected] = MLM_MASK_CODE
    return x400_masked, mlm_labels


def _collate_frag(records, device):
    """Pad fragment batches on RIGHT with N=4 to a fixed length of 600.

    All actual records have length 100..600; padding to a constant 600
    prevents GroupNorm statistics from depending on companion lengths.
    """
    max_len = 600
    B = len(records)
    codes = np.full((B, max_len), 4, dtype=np.uint8)
    for i, r in enumerate(records):
        c = np.asarray(r["codes"], dtype=np.uint8)
        if len(c) != int(r["length"]):
            raise ValueError(
                f"record {i}: len(codes)={len(c)} != length={r['length']}"
            )
        if not (1 <= len(c) <= 600):
            raise ValueError(f"record {i}: length {len(c)} not in [1, 600]")
        codes[i, :len(c)] = c
    codes_t = torch.tensor(codes, dtype=torch.long, device=device)
    y = torch.tensor(
        [float(r["y_Log2RPM"]) for r in records], dtype=torch.float32, device=device
    )
    w = torch.tensor(
        [float(r["groupweight"]) for r in records], dtype=torch.float32, device=device
    )
    return {"codes": codes_t, "y": y, "weight": w}


# ---------------------------------------------------------------------------
# Stream schedule
# ---------------------------------------------------------------------------


def _stream_schedule(n_rows, batch_size, steps):
    """Return dict step -> list[(start, end)] covering each row exactly once."""
    if n_rows <= 0 or batch_size <= 0 or steps <= 0:
        return {}
    n_batches = int(math.ceil(n_rows / batch_size))
    out = {}
    for step in range(steps):
        before = (step * n_batches) // steps
        after = ((step + 1) * n_batches) // steps
        for b in range(before, after):
            start = b * batch_size
            end = min(start + batch_size, n_rows)
            out.setdefault(step, []).append((start, end))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _spearman(x, y):
    """Spearman via pandas average rank / numpy correlation."""
    import pandas as pd
    if len(x) < 2:
        return None
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _gene_aggregate(gene_ids, values, masks):
    """Aggregate per-gene mean of finite values. Returns (gene_ids, means, counts)."""
    order = {}
    sums = {}
    counts = {}
    for g, v, m in zip(gene_ids, values, masks):
        if not m:
            continue
        if g not in order:
            order[g] = len(order)
            sums[g] = 0.0
            counts[g] = 0
        sums[g] += float(v)
        counts[g] += 1
    genes = list(order.keys())
    means = np.array([sums[g] / counts[g] for g in genes], dtype=np.float64)
    cnts = np.array([counts[g] for g in genes], dtype=np.int64)
    return genes, means, cnts


def evaluate_esb(model, esb, indices, device, normalizers, batch_size=16, max_rows=None):
    """Evaluate ESB on given indices. Returns metrics dict."""
    if max_rows is not None and len(indices) > max_rows:
        indices = indices[:max_rows]
    model.eval()
    preds = []
    truths = []
    masks = []
    genes = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            records = [esb.record(int(i)) for i in chunk]
            batch = _collate_esb(records, device)
            out = model.forward_esb(batch["x400"], batch["x2000"], batch["species"])
            preds.append(out["esb"].float().cpu().numpy())
            truths.append(batch["targets_ESB_z"].float().cpu().numpy())
            masks.append(batch["finite_mask"].float().cpu().numpy())
            genes.extend([r["gene_id"] for r in records])
    if not preds:
        return {"n": 0, "axes": {}}
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    masks = np.concatenate(masks, axis=0)
    esb_mean = np.asarray(normalizers["esb"]["mean"], dtype=np.float64)
    esb_std = np.asarray(normalizers["esb"]["std"], dtype=np.float64)
    preds_orig = preds * esb_std + esb_mean
    truths_orig = truths * esb_std + esb_mean
    axes = {}
    for j, name in enumerate(ESB_COLS):
        m = masks[:, j] > 0
        n = int(m.sum())
        if n < 2:
            axes[name] = {"n": n, "spearman": None}
            continue
        g, means, cnts = _gene_aggregate(
            [genes[i] for i in range(len(genes)) if m[i]],
            preds_orig[m, j],
            [True] * n,
        )
        # gene-level truth
        g2, tmeans, _ = _gene_aggregate(
            [genes[i] for i in range(len(genes)) if m[i]],
            truths_orig[m, j],
            [True] * n,
        )
        if g != g2:
            raise ValueError("gene aggregation mismatch")
        sp = _spearman(means, tmeans)
        axes[name] = {"n": n, "n_genes": len(g), "spearman": sp}
    return {"n": int(len(indices)), "axes": axes}


def evaluate_fragment(model, frag, indices, device, batch_size=16):
    """Evaluate fragment activity on given indices. Returns metrics dict."""
    model.eval()
    preds = []
    truths = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            records = [frag.record(int(i)) for i in chunk]
            batch = _collate_frag(records, device)
            z = model.forward_fragment(batch["codes"])
            preds.append(z.float().cpu().numpy())
            truths.append(batch["y"].float().cpu().numpy())
    if not preds:
        return {"n": 0, "spearman": None}
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    sp = _spearman(preds, truths)
    return {"n": int(len(indices)), "spearman": sp}


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _synthetic_equivalence_check(cfg, normalizers, report):
    """Prove old-vs-new inverse-scaled E/S/B agree <=2e-5 on synthetic inputs.

    Loads old frozen architecture classes from the v6.0.0 trainer source without
    running main. CPU only. Synthetic inputs are explicitly synthetic.
    """
    root = cfg.get("root") or str(ROOT)
    old_src = project_path(root, OLD_TRAINER_REL)
    if not os.path.isfile(old_src):
        raise FileNotFoundError("old trainer source not found: %s" % old_src)

    import importlib.util
    spec = importlib.util.spec_from_file_location("old_trainer_v6_round1", old_src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("old trainer import failed: %s" % e)

    old_cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.lower().startswith("promoteratlas"):
            old_cls = obj
            break
    if old_cls is None:
        raise RuntimeError("old architecture class not found")

    ws_rel = cfg.get("warmstart", WARMSTART_REL)
    ws_path = project_path(root, ws_rel) if not os.path.isabs(ws_rel) else ws_rel
    ck, old_sd = _load_old_state_dict(ws_path)

    old_model = old_cls(n_tf=N_TF, n_omics=8)
    old_model.load_state_dict(old_sd, strict=True)
    old_model.eval()

    model, _ = warmstart_model(cfg, normalizers)
    model.eval()

    torch.manual_seed(0)
    B = 2
    x400 = torch.randint(0, 4, (B, 400), dtype=torch.long)
    x2000 = torch.randint(0, 4, (B, 2000), dtype=torch.long)
    species = torch.tensor([0, 1], dtype=torch.long)

    with torch.inference_mode():
        old_out = old_model(x400, x2000, species)["esb"].float().numpy()
        new_out = model.forward_esb(x400, x2000, species)["esb"].float().numpy()

    esb_mean = np.asarray(normalizers["esb"]["mean"], dtype=np.float64)
    esb_std = np.asarray(normalizers["esb"]["std"], dtype=np.float64)
    s_mean, s_sd = float(ck["s_stats"][0]), float(ck["s_stats"][1])
    old_mu = np.array([0.0, s_mean, 0.0])
    old_sd = np.array([1.0, s_sd, 1.0])

    old_orig = old_out * old_sd + old_mu
    new_orig = new_out * esb_std + esb_mean
    diff = np.abs(old_orig - new_orig)
    max_diff = float(diff.max())
    ok = max_diff <= 2e-5
    return {
        "status": "ok" if ok else "failed",
        "max_abs_diff": max_diff,
        "tolerance": 2e-5,
        "synthetic": True,
        "note": "synthetic inputs; not dataset or performance claims",
    }


def run_prepare(cfg, outdir):
    """Load data, hash audit, warmstart, prove equivalence. No backward/GPU allocation."""
    torch.set_num_threads(4)
    seed = int(cfg.get("seed", 20261012))
    np.random.seed(seed)
    torch.manual_seed(seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    esb, frag = _load_streams(cfg)
    normalizers = esb.normalizers

    audit = esb.audit(hash_sources=True)
    model, report = warmstart_model(cfg, normalizers)

    equiv = _synthetic_equivalence_check(cfg, normalizers, report)

    prep = {
        "format": "promoter_atlas_measured_v6.6.prepare.v1",
        "model_version": MODEL_VERSION,
        "run_id": RUN_ID,
        "created_utc": now_utc().isoformat(),
        "config": cfg,
        "data_audit": audit,
        "source_paths": {
            "esb": getattr(esb, "source_paths", {}),
            "fragment": getattr(frag, "source_paths", {}),
        },
        "normalizers": normalizers,
        "warmstart_report": report,
        "equivalence_check": equiv,
        "n_esb": len(esb),
        "n_fragment": len(frag),
        "splits": {
            "esb": {s: int(len(esb.split_indices(s))) for s in ("TRAIN", "DEV", "TEST")},
            "fragment": {s: int(len(frag.split_indices(s))) for s in ("TRAIN", "DEV", "TEST")},
        },
    }
    atomic_write_json(outdir / "preparation.json", prep)
    return prep


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fixed_dev_subset(esb, seed=1234, max_n=4096):
    """Fixed reproducibly selected human DEV indices subset."""
    dev = esb.split_indices("DEV")
    human = []
    for i in dev:
        r = esb.record(int(i))
        if int(r["species"]) == 0:
            human.append(int(i))
    human = np.array(human, dtype=np.int64)
    if len(human) > max_n:
        rng = np.random.RandomState(seed)
        human = np.sort(rng.choice(human, size=max_n, replace=False))
    return human


def _config_hash(cfg):
    c = {k: v for k, v in cfg.items() if k not in ("root", "outdir", "deadline")}
    return sha256_bytes(json.dumps(c, sort_keys=True, default=str).encode("utf-8"))


def _source_hashes():
    out = {}
    for name in ("measured_train.py", "model.py", "measured_data.py"):
        p = Path(__file__).resolve().parent / name
        out[name] = sha256_file(p) if p.is_file() else None
    return out


def _save_checkpoint(path, model, optimizer, epoch, next_step, global_step, cfg,
                     normalizers, best_score, dev_ids, loss_log, data_hashes,
                     optimization_elapsed, n_profile, tissue_names):
    obj = {
        "format": RESUME_FORMAT, "model_version": MODEL_VERSION, "run_id": RUN_ID,
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": int(epoch), "next_step": int(next_step), "global_step": int(global_step),
        "rng": rng_state_all(), "config": cfg, "normalizers": normalizers,
        "best_score": best_score, "dev_ids": np.asarray(dev_ids, dtype=np.int64).tolist(),
        "loss_log": loss_log[-1:], "data_hashes": data_hashes,
        "optimization_elapsed": float(optimization_elapsed),
        "warmstart_sha256": WARMSTART_SHA256, "arch": dict(ARCH),
        "n_tf": N_TF, "n_omics": N_OMICS, "n_dist": N_DIST,
        "n_profile": int(n_profile), "tissue_names": list(tissue_names),
        "config_hash": _config_hash(cfg), "source_hashes": _source_hashes(),
    }
    atomic_torch_save(path, obj)


def _save_inference(path, model, normalizers, frag, n_profile, tissue_names):
    obj = {
        "format": FORMAT, "state_dict": model.state_dict(), "arch": dict(ARCH),
        "n_tf": N_TF, "n_omics": N_OMICS, "n_dist": N_DIST,
        "n_profile": int(n_profile), "tissue_names": list(tissue_names),
        "profile_normalizers": normalizers["profile"],
        "activity_hidden": ACTIVITY_HIDDEN, "activity_dropout": ACTIVITY_DROPOUT,
        "zstats": {name: [float(normalizers["esb"]["mean"][j]),
                          float(normalizers["esb"]["std"][j])]
                   for j, name in enumerate(ESB_COLS)},
        "activity_normalizer": {"mean": float(frag.train_mean), "std": float(frag.train_std),
                               "target": "Log2RPM_HEK293"},
        "version": MODEL_VERSION, "run_id": RUN_ID, "parent_sha256": WARMSTART_SHA256, "fragment_padding_length": 600,
    }
    atomic_torch_save(path, obj)


def train(cfg, outdir, device="cpu", max_steps=None, deadline=None, resume=None, profile_steps=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "final_test_started.json").exists() or (outdir / "final_test.json").exists():
        raise RuntimeError("final TEST already started or completed; refusing to train")
    root = cfg.get("root") or str(ROOT)
    cfg = dict(cfg)
    cfg["root"] = root
    torch.set_num_threads(4)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    seed = int(cfg.get("seed", 20261012))
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    esb, frag = _load_streams(cfg)
    normalizers = esb.normalizers
    n_profile = int(len(normalizers["profile"]["columns"]))
    tissue_names = list(normalizers["profile"]["columns"])

    audit = esb.audit(hash_sources=True)
    atomic_write_json(outdir / "effective_config.json", cfg)
    atomic_write_json(outdir / "source_hashes.json", _source_hashes())
    data_hashes = {"esb_audit": audit, "fragment_source_paths": {}}
    for k, v in (getattr(frag, "source_paths", {}) or {}).items():
        if v and os.path.isfile(v):
            data_hashes["fragment_source_paths"][k] = sha256_file(v)
        else:
            data_hashes["fragment_source_paths"][k] = None
    data_hashes["warmstart"] = WARMSTART_SHA256

    model, report = warmstart_model(cfg, normalizers)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    esb_bs = int(cfg.get("esb_batch_size", 16))
    frag_bs = int(cfg.get("fragment_batch_size", 16))
    epochs = min(int(cfg.get("epochs", 8)), 8)
    weights = dict(TASK_WEIGHTS)
    weights.update(cfg.get("task_weights", {}) or {})

    train_esb = esb.split_indices("TRAIN")
    train_frag = frag.split_indices("TRAIN")
    train_esb = train_esb[deterministic_perm(len(train_esb), seed)]
    train_frag = train_frag[deterministic_perm(len(train_frag), seed)]
    dev_ids = _fixed_dev_subset(esb)
    test_esb = esb.split_indices("TEST")
    test_frag = frag.split_indices("TEST")

    start_epoch = 0
    start_step = 0
    global_step = 0
    best_score = None
    optimization_elapsed = 0.0
    loss_log = []

    if resume:
        obj = torch.load(resume, map_location="cpu", weights_only=True)
        if obj.get("format") != RESUME_FORMAT:
            raise ValueError("resume format mismatch: %r" % obj.get("format"))
        if obj.get("warmstart_sha256") != WARMSTART_SHA256:
            raise ValueError("resume warmstart lineage mismatch")
        if obj.get("arch") != dict(ARCH):
            raise ValueError("resume architecture mismatch")
        if obj.get("n_profile") != n_profile:
            raise ValueError("resume n_profile mismatch")
        if obj.get("tissue_names") != tissue_names:
            raise ValueError("resume tissue_names mismatch")
        if obj.get("normalizers") != normalizers:
            raise ValueError("resume normalizers mismatch")
        if obj.get("data_hashes") != data_hashes:
            raise ValueError("resume data_hashes mismatch")
        if obj.get("run_id") != RUN_ID:
            raise ValueError("resume run_id mismatch")
        if obj.get("config_hash") != _config_hash(cfg):
            raise ValueError("resume config_hash mismatch")
        if obj.get("source_hashes") != _source_hashes():
            raise ValueError("resume source_hashes mismatch")
        model.load_state_dict(obj["state_dict"], strict=True)
        optimizer.load_state_dict(obj["optimizer"])
        start_epoch = int(obj["epoch"])
        start_step = int(obj["next_step"])
        global_step = int(obj["global_step"])
        best_score = obj.get("best_score")
        optimization_elapsed = float(obj.get("optimization_elapsed", 0.0))
        loss_log = obj.get("loss_log", [])
        rng_restore_all(obj["rng"])

    deadline_dt = parse_deadline(deadline) if deadline else None
    stop_reason = "completed"
    completed_epochs = start_epoch
    partial_epoch_steps = 0
    cursor_epoch = start_epoch
    cursor_step = start_step
    ckpt_interval = float(cfg.get("checkpoint_interval_sec", 120))
    dev_interval = float(cfg.get("fast_eval_interval_sec", 900))
    last_ckpt = time.perf_counter()
    last_dev = time.perf_counter()
    smoke = max_steps is not None

    def should_stop():
        if deadline_dt is not None and now_utc() >= deadline_dt:
            return "deadline"
        if max_steps is not None and global_step >= max_steps:
            return "max_steps"
        return None

    def save_resume(epoch, next_step):
        _save_checkpoint(
            outdir / "checkpoints" / "resume.pt", model, optimizer, epoch,
            next_step, global_step, cfg, normalizers, best_score, dev_ids,
            loss_log, data_hashes, optimization_elapsed, n_profile, tissue_names,
        )

    def save_cursor():
        save_resume(cursor_epoch, cursor_step)

    def run_dev():
        nonlocal best_score
        model.eval()
        metrics = evaluate_esb(model, esb, dev_ids, device, normalizers)
        model.train()
        print(json.dumps({"dev": metrics}, sort_keys=True), flush=True)
        score = metrics["axes"].get("Score_E", {}).get("spearman")
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            _save_inference(outdir / "best.pt", model, normalizers, frag, n_profile, tissue_names)
        _save_inference(outdir / "last.pt", model, normalizers, frag, n_profile, tissue_names)
        snap_path = outdir / "snapshots" / f"step_{global_step:08d}.pt"
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        _save_inference(snap_path, model, normalizers, frag, n_profile, tissue_names)
        snap = torch.load(snap_path, map_location="cpu", weights_only=True)
        snap["global_step"] = global_step
        snap["optimization_elapsed"] = optimization_elapsed
        snap["epoch"] = cursor_epoch
        snap["next_step"] = cursor_step
        atomic_torch_save(snap_path, snap)
        return metrics

    for epoch in range(start_epoch, epochs):
        perm_esb = deterministic_perm(len(train_esb), seed + epoch)
        perm_frag = deterministic_perm(len(train_frag), seed + epoch + 100000)
        esb_order = train_esb[perm_esb]
        frag_order = train_frag[perm_frag]
        steps = max(
            int(math.ceil(len(esb_order) / esb_bs)) if len(esb_order) else 0,
            int(math.ceil(len(frag_order) / frag_bs)) if len(frag_order) else 0,
        )
        if steps == 0:
            break
        esb_map = _stream_schedule(len(esb_order), esb_bs, steps)
        frag_map = _stream_schedule(len(frag_order), frag_bs, steps)
        model.train()
        step = start_step if epoch == start_epoch else 0
        while step < steps:
            reason = should_stop()
            if reason:
                stop_reason = reason
                break
            t_opt0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            step_stats = {}
            if step in esb_map:
                idxs = []
                for (s, e) in esb_map[step]:
                    idxs.extend(esb_order[s:e].tolist())
                records = [esb.record(int(i)) for i in idxs]
                batch = _collate_esb(records, device)
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + epoch * 1000 + step)
                x400_masked, mlm_labels = _apply_mlm(batch["x400"], batch["mlm_loss_mask"], gen)
                batch["mlm_labels"] = mlm_labels
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=str(device).startswith("cuda")):
                    out = model.forward_esb(batch["x400"], batch["x2000"], batch["species"],
                                            x400_masked=x400_masked)
                out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
                weights["profile"] = 0.3 * min(1.0, optimization_elapsed / 1800.0)
                loss, stats = compute_multitask_loss(out, batch, weights)
                if loss is not None and loss.requires_grad:
                    total_loss = loss if total_loss is None else total_loss + loss
                step_stats.update(stats)
            if step in frag_map:
                idxs = []
                for (s, e) in frag_map[step]:
                    idxs.extend(frag_order[s:e].tolist())
                records = [frag.record(int(i)) for i in idxs]
                fb = _collate_frag(records, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=str(device).startswith("cuda")):
                    z = model.forward_fragment(fb["codes"])
                z = z.float()
                mean = float(frag.train_mean)
                std = float(frag.train_std)
                fb["activity_pred"] = z
                fb["activity_target"] = (fb["y"] - mean) / std
                fb["activity_weight"] = fb["weight"]
                loss, stats = compute_multitask_loss({}, fb, weights)
                if loss is not None and loss.requires_grad:
                    total_loss = loss if total_loss is None else total_loss + loss
                step_stats.update(stats)
            if total_loss is None:
                cursor_epoch = epoch
                cursor_step = step + 1
                step += 1
                continue
            if not torch.isfinite(total_loss):
                save_cursor()
                _save_inference(outdir / "last.pt", model, normalizers, frag,
                                n_profile, tissue_names)
                raise FloatingPointError("non-finite loss at epoch %d step %d" % (epoch, step))
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                    error_if_nonfinite=True)
            except Exception:
                save_cursor()
                _save_inference(outdir / "last.pt", model, normalizers, frag,
                                n_profile, tissue_names)
                raise
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            global_step += 1
            cursor_epoch = epoch
            cursor_step = step + 1
            optimization_elapsed += time.perf_counter() - t_opt0
            loss_log.append({
                "global_step": global_step, "epoch": epoch, "step": step,
                "loss": float(total_loss.detach().cpu()), "tasks": step_stats,
                "optimization_elapsed": optimization_elapsed,
                "profile_weight": float(weights.get("profile", 0.0)),
            })
            del loss_log[:-1]
            with open(outdir / "losses.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(loss_log[-1], sort_keys=True) + "\n")
            now = time.perf_counter()
            if now - last_ckpt >= ckpt_interval:
                save_cursor()
                print(json.dumps({"global_step": global_step,
                                  "optimization_elapsed": optimization_elapsed,
                                  "profile_weight": float(weights.get("profile", 0.0))},
                                 sort_keys=True), flush=True)
                last_ckpt = now
            if not smoke and now - last_dev >= dev_interval:
                run_dev()
                last_dev = now
            step += 1
        if stop_reason in ("deadline", "max_steps"):
            partial_epoch_steps = step
            save_cursor()
            break
        completed_epochs = epoch + 1
        start_step = 0
        cursor_epoch = epoch + 1
        cursor_step = 0
        if not smoke:
            run_dev()
            last_dev = time.perf_counter()
        save_cursor()

    if stop_reason not in ("deadline", "max_steps") or global_step == 0:
        save_cursor()
    _save_inference(outdir / "last.pt", model, normalizers, frag, n_profile, tissue_names)

    result = {
        "format": "promoter_atlas_measured_v6.6.final.v1",
        "model_version": MODEL_VERSION, "run_id": RUN_ID,
        "completed_epochs": completed_epochs, "partial_epoch_steps": partial_epoch_steps,
        "stop_reason": stop_reason, "global_step": global_step,
        "best_score": best_score, "optimization_elapsed": optimization_elapsed,
        "warmstart_report": report, "data_hashes": data_hashes,
    }
    if smoke or cfg.get('train_only', False):
        result["smoke"] = smoke
        result["train_only"] = cfg.get('train_only', False)
        atomic_write_json(outdir / "final_report.json", result)
        return result

    started = outdir / "final_test_started.json"
    if started.exists() or (outdir / "final_test.json").exists():
        raise RuntimeError("final TEST already started or completed; refusing to repeat")

    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    best_metrics = None
    last_metrics = None
    full_dev = [i for i in esb.split_indices('DEV') if esb.species[i] == 0]
    test_esb = [i for i in test_esb if esb.species[i] == 0]
    if best_path.exists():
        obj = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(obj["state_dict"], strict=True)
        model.to(device)
        best_metrics = evaluate_esb(model, esb, full_dev, device, normalizers)
    obj = torch.load(last_path, map_location="cpu", weights_only=True)
    model.load_state_dict(obj["state_dict"], strict=True)
    model.to(device)
    last_metrics = evaluate_esb(model, esb, full_dev, device, normalizers)
    bE = (best_metrics or {}).get("axes", {}).get("Score_E", {}).get("spearman")
    lE = last_metrics["axes"].get("Score_E", {}).get("spearman")
    if bE is not None and (lE is None or bE > lE):
        selected = "best"
        sel_path = best_path
    else:
        selected = "last"
        sel_path = last_path
    obj = torch.load(sel_path, map_location="cpu", weights_only=True)
    model.load_state_dict(obj["state_dict"], strict=True)
    model.to(device)
    model.eval()
    _save_inference(outdir / "selected.pt", model, normalizers, frag, n_profile, tissue_names)
    atomic_write_json(started, {"started_utc": now_utc().isoformat(), "run_id": RUN_ID})
    test_esb_metrics = evaluate_esb(model, esb, test_esb, device, normalizers)
    test_frag_metrics = evaluate_fragment(model, frag, test_frag, device)
    result["selected"] = selected
    result["selected_checkpoint"] = str(sel_path)
    result["dev_best"] = best_metrics
    result["dev_last"] = last_metrics
    result["test_esb"] = test_esb_metrics
    result["test_fragment"] = test_frag_metrics
    atomic_write_json(outdir / "final_test.json", result)
    atomic_write_json(outdir / "final_report.json", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/v6.6.json")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args(argv)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(str(ROOT), cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if args.train_only:
        cfg["train_only"] = True

    root = cfg.get("root") or str(ROOT)
    cfg["root"] = root
    outdir = cfg.get("outdir", DEFAULT_OUTDIR)
    if not os.path.isabs(outdir):
        outdir = os.path.join(root, outdir)

    if args.prepare_only:
        prep = run_prepare(cfg, outdir)
        print(json.dumps({"prepare": prep.get("equivalence_check")}, indent=2))
        return 0

    if args.profile_steps is not None:
        prof = profile_training(cfg, outdir, device=args.device, steps=args.profile_steps)
        print(json.dumps(prof, indent=2))
        return 0

    deadline = args.deadline or cfg.get("deadline") or DEFAULT_DEADLINE
    report = train(
        cfg, outdir, device=args.device, max_steps=args.max_steps,
        deadline=deadline, resume=args.resume,
    )
    if report.get("smoke"):
        print(json.dumps({"smoke": True, "stop_reason": report.get("stop_reason")}, indent=2))
        return 0
    print(json.dumps(report, indent=2))
    return 0


def profile_training(cfg, outdir, device="cpu", steps=5):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = cfg.get("root") or str(ROOT)
    cfg = dict(cfg)
    cfg["root"] = root
    torch.set_num_threads(4)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    seed = int(cfg.get("seed", 20261012))
    np.random.seed(seed)
    torch.manual_seed(seed)
    esb, frag = _load_streams(cfg)
    normalizers = esb.normalizers
    n_profile = int(len(normalizers["profile"]["columns"]))
    model, _ = warmstart_model(cfg, normalizers)
    model.to(device)
    model.train()
    esb_bs = int(cfg.get("esb_batch_size", 16))
    frag_bs = int(cfg.get("fragment_batch_size", 16))
    train_esb = esb.split_indices("TRAIN")
    train_frag = frag.split_indices("TRAIN")
    weights = dict(TASK_WEIGHTS)
    weights.update(cfg.get("task_weights", {}) or {})
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times = []
    rows = 0
    grad_sums = {}
    task_valid = {}
    for step in range(int(steps)):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idxs = train_esb[(step * esb_bs) % max(1, len(train_esb)):
                        (step * esb_bs) % max(1, len(train_esb)) + esb_bs]
        if len(idxs) == 0:
            idxs = train_esb[:esb_bs]
        records = [esb.record(int(i)) for i in idxs]
        batch = _collate_esb(records, device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + step)
        x400_masked, mlm_labels = _apply_mlm(batch["x400"], batch["mlm_loss_mask"], gen)
        batch["mlm_labels"] = mlm_labels
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            out = model.forward_esb(batch["x400"], batch["x2000"], batch["species"],
                                    x400_masked=x400_masked)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        weights["profile"] = 0.3
        loss, _stats = compute_multitask_loss(out, batch, weights)
        for _k, _v in _stats.items():
            if _v.get("loss") is not None:
                task_valid[_k] = task_valid.get(_k, 0) + 1
        fidx = train_frag[(step * frag_bs) % max(1, len(train_frag)):
                          (step * frag_bs) % max(1, len(train_frag)) + frag_bs]
        if len(fidx) == 0:
            fidx = train_frag[:frag_bs]
        frecords = [frag.record(int(i)) for i in fidx]
        fb = _collate_frag(frecords, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            z = model.forward_fragment(fb["codes"])
        z = z.float()
        fb["activity_pred"] = z
        fb["activity_target"] = (fb["y"] - float(frag.train_mean)) / float(frag.train_std)
        fb["activity_weight"] = fb["weight"]
        floss, _fstats = compute_multitask_loss({}, fb, weights)
        for _k, _v in _fstats.items():
            if _v.get("loss") is not None:
                task_valid[_k] = task_valid.get(_k, 0) + 1
        total = loss + floss
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite profile loss at step %d" % step)
        total.backward()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = float(p.grad.detach().abs().sum().cpu())
                grad_sums[name] = grad_sums.get(name, 0.0) + g
        model.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t0)
        rows += len(idxs) + len(fidx)
    peak = 0.0
    if str(device).startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    step_seconds = float(np.mean(times)) if times else None
    throughput = (rows / sum(times)) if times and sum(times) > 0 else None
    deadline_dt = parse_deadline(cfg.get("deadline") or DEFAULT_DEADLINE)
    remaining = max(0.0, (deadline_dt - now_utc()).total_seconds())
    est_rows = int(throughput * remaining) if throughput else None
    prof = {
        "format": "promoter_atlas_measured_v6.6.profile.v1",
        "device": str(device), "steps": int(steps),
        "step_seconds": step_seconds, "throughput_rows_per_s": throughput,
        "peak_gpu_allocated_mb": peak, "rows_measured": rows,
        "estimated_rows_until_deadline": est_rows,
        "n_profile": n_profile, "note": "timed forward/backward only; no optimizer step; weights discarded",
        "grad_sum_by_param": grad_sums,
        "task_valid": task_valid,
    }
    atomic_write_json(outdir / "profile.json", prof)
    return prof

if __name__ == "__main__":
    sys.exit(main())
