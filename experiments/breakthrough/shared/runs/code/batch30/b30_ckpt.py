# -*- coding: utf-8 -*-
"""batch30 durable checkpoint / resume logic (STRUCTURAL only in this batch: NO gradients run).

Design goals (protocol REPAIR-2):
  * latest.pt  : matching model + deep-copied optimizer/scheduler + Python/NumPy/TorchCPU/
                 all-CUDA RNG + epoch + next_batch_cursor + global_step + earlystop/best state
                 + config/data/split/code hashes + gate exec_id.  Written atomically after EVERY
                 completed epoch; early stop and crashes preserve the previous latest.pt.
  * best.pt    : exact deep-copied snapshot at the moment of improvement; no reference is ever
                 mutated afterwards.
  * resume     : restore ALL states and continue at the next unconsumed batch/epoch; verify
                 immutable identity (config/data/split/code hashes + gate exec_id) and refuse on mismatch.
"""
from __future__ import annotations
import copy
import os
import random
import numpy as np
import torch


def _deep_state_dict(sd):
    """Deep copy a model state_dict so later in-place mutation cannot alter the snapshot."""
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in sd.items()}


def _deep_optimizer(opt):
    sd = opt.state_dict()
    out = {"state": {}, "param_groups": copy.deepcopy(sd["param_groups"])}
    for pid, st in sd["state"].items():
        out["state"][pid] = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                             for k, v in st.items()}
    return out


def capture_state(model, opt, sched, *, epoch, next_batch_cursor, global_step, early, cfg, hashes, exec_id):
    """Capture a fully self-contained, mutation-proof training snapshot (CPU tensors)."""
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return {
        "model": _deep_state_dict(model.state_dict()),
        "optimizer": _deep_optimizer(opt),
        "scheduler": copy.deepcopy(sched.state_dict()),
        "rng": rng,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "next_batch_cursor": int(next_batch_cursor),
        "global_step": int(global_step),
        "early": copy.deepcopy(early),
        "config": copy.deepcopy(cfg),
        "hashes": copy.deepcopy(hashes),
        "exec_id": exec_id,
    }


def restore_state(state, model, opt, sched):
    """Restore model/optimizer/scheduler/RNG into live objects and return (epoch, cursor, step, early)."""
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    if torch.cuda.is_available() and state["rng"]["torch_cuda_all"]:
        torch.cuda.set_rng_state_all(state["rng"]["torch_cuda_all"])
    return state["epoch"], state["next_batch_cursor"], state["global_step"], state["early"]


def verify_identity(state, hashes, exec_id):
    """Refuse to resume if the immutable identity does not match the current run/gate reservation."""
    mism = {}
    for k, v in hashes.items():
        if state.get("hashes", {}).get(k) != v:
            mism[k] = {"snapshot": state.get("hashes", {}).get(k), "current": v}
    if exec_id is not None and state.get("exec_id") != exec_id:
        mism["exec_id"] = {"snapshot": state.get("exec_id"), "current": exec_id}
    if mism:
        raise ValueError("resume identity mismatch, refusing: %r" % mism)
    return True


def atomic_torch_save(obj, path):
    path = str(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def state_digest(state):
    """Stable structural digest of a snapshot (model param sum + RNG bytes + counters)."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state["model"]):
        h.update(k.encode())
        h.update(np.asarray(state["model"][k], dtype=np.float64).tobytes())
    h.update(repr(state["rng"]["python"]).encode())
    h.update(np.asarray(state["rng"]["numpy"][1], dtype=np.int64).tobytes())
    h.update(np.asarray(state["rng"]["torch_cpu"], dtype=np.uint8).tobytes())
    h.update(("%d|%d|%d" % (state["epoch"], state["next_batch_cursor"], state["global_step"])).encode())
    return h.hexdigest()
