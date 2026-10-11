# -*- coding: utf-8 -*-
"""Atomic checkpoints carrying exact optimizer / RNG / sampler / config / datahash / gate id.

batch52 PROTOCOL: save atomically after each epoch and <=5 min intervals, recording
actual model + optimizer + (scheduler if any) + Python/NumPy/Torch CPU/CUDA RNG + epoch +
global step + batch sampler position + normalizers + config + dataSHA + exec_gate attemptID.
Strict loading refuses a config/datahash mismatch (no silent wrong-target resume).
"""
from __future__ import annotations
import json, os, random

import numpy as np
import torch

from io_utils import sha256_file, atomic_write_bytes
from spec import config_dict, config_sha256


def capture_rng() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path, model, optimizer, scheduler=None, epoch=0, global_step=0,
                    sampler_state=None, normalizer_state=None, datahash=None,
                    attempt_id=None, extra=None) -> dict:
    payload = {
        "format": "batch53.checkpoint.v1",
        "config": config_dict(),
        "config_sha256": config_sha256(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "sampler_state": sampler_state or {},
        "normalizer": normalizer_state or {},
        "datahash": datahash,
        "gate_attempt_id": attempt_id,
        "rng": capture_rng(),
        "extra": extra or {},
    }
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    side = {
        "checkpoint": os.path.basename(path),
        "path": path,
        "checkpoint_sha256": sha256_file(path),
        "config_sha256": payload["config_sha256"],
        "datahash": datahash,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "gate_attempt_id": attempt_id,
    }
    atomic_write_bytes(path + ".sha256.json", json.dumps(side, ensure_ascii=False, indent=2).encode("utf-8"))
    return side


def write_sidecar(path, *, config_sha=None, datahash=None, epoch=None, global_step=None,
                  attempt_id=None, extra=None) -> dict:
    """D11: write the byte-integrity sidecar for an EXISTING checkpoint file.

    The batch53 selected-checkpoint copy skipped this, so strict load could not verify its
    bytes. Every selected checkpoint now carries a .sha256.json sidecar."""
    if config_sha is None:
        config_sha = config_sha256()
    side = {
        "checkpoint": os.path.basename(path),
        "path": path,
        "checkpoint_sha256": sha256_file(path),
        "config_sha256": config_sha,
        "datahash": datahash,
        "epoch": None if epoch is None else int(epoch),
        "global_step": None if global_step is None else int(global_step),
        "gate_attempt_id": attempt_id,
        "selected": True,
    }
    if extra:
        side.update(extra)
    atomic_write_bytes(path + ".sha256.json",
                       json.dumps(side, ensure_ascii=False, indent=2).encode("utf-8"))
    return side


def select_checkpoint(src_path, dst_path, *, epoch=None, global_step=None, attempt_id=None,
                      datahash=None) -> dict:
    """D11: copy a checkpoint to its SELECTED path atomically and write its sidecar."""
    obj = _load_raw(src_path)
    tmp = dst_path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst_path)
    return write_sidecar(dst_path, datahash=datahash or obj.get("datahash"),
                         epoch=epoch if epoch is not None else obj.get("epoch"),
                         global_step=global_step if global_step is not None else obj.get("global_step"),
                         attempt_id=attempt_id)


def _load_raw(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


class CheckpointMismatch(Exception):
    pass


def load_checkpoint(path, expect_config_sha=None, expect_datahash=None, strict=True) -> dict:
    """Strict load. Verifies the sidecar sha256 and (optionally) config/datahash match."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    side_path = path + ".sha256.json"
    obj = _load_raw(path)
    if obj.get("format") != "batch53.checkpoint.v1":
        raise CheckpointMismatch("unexpected checkpoint format %r" % obj.get("format"))
    if strict and os.path.exists(side_path):
        with open(side_path, "r", encoding="utf-8") as f:
            side = json.load(f)
        actual = sha256_file(path)
        if side.get("checkpoint_sha256") and side["checkpoint_sha256"] != actual:
            raise CheckpointMismatch("checkpoint bytes changed: %s != %s" % (actual, side["checkpoint_sha256"]))
    if strict and expect_config_sha and obj.get("config_sha256") != expect_config_sha:
        raise CheckpointMismatch("config sha mismatch %s != %s" % (obj.get("config_sha256"), expect_config_sha))
    if strict and expect_datahash and obj.get("datahash") != expect_datahash:
        raise CheckpointMismatch("datahash mismatch %s != %s" % (obj.get("datahash"), expect_datahash))
    if obj.get("datahash") is None:
        raise CheckpointMismatch("checkpoint has no datahash")
    return obj


def load_model_into(model, path, expect_config_sha=None, expect_datahash=None) -> dict:
    obj = load_checkpoint(path, expect_config_sha=expect_config_sha, expect_datahash=expect_datahash)
    model.load_state_dict(obj["model"])
    return obj
