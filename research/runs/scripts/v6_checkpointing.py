"""Atomic checkpoints and resumable shuffled batches for single-worker training."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


class StatefulBatchSampler:
    """Drop-last batches; cursor always points to the next unconsumed batch.

    Use with num_workers=0: prefetch would advance the cursor before training.
    Length is the full cycle length so scheduler horizons stay fixed on resume.
    """

    def __init__(self, dataset_size, batch_size, seed):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        if self.batch_size < 1 or self.dataset_size < self.batch_size:
            raise ValueError("Training dataset must contain at least one full batch")
        self.generator = torch.Generator().manual_seed(self.seed)
        self.order = torch.empty(0, dtype=torch.int64)
        self.cursor = 0

    def __len__(self):
        return self.dataset_size // self.batch_size

    def start_cycle(self):
        self.order = torch.randperm(self.dataset_size, generator=self.generator)
        self.cursor = 0

    def __iter__(self):
        if self.order.numel() != self.dataset_size:
            raise RuntimeError("Call start_cycle before iterating a fresh sampler")
        while self.cursor < len(self):
            begin = self.cursor * self.batch_size
            batch = self.order[begin:begin + self.batch_size].tolist()
            self.cursor += 1
            yield batch

    def state_dict(self):
        return {"dataset_size": self.dataset_size, "batch_size": self.batch_size,
                "seed": self.seed, "cursor": self.cursor, "order": self.order.clone(),
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        for key in ("dataset_size", "batch_size", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume sampler {key} mismatch")
        order = state["order"].cpu()
        cursor = int(state["cursor"])
        if (order.dtype != torch.int64 or order.numel() != self.dataset_size
                or not 0 <= cursor <= len(self)
                or not torch.equal(torch.sort(order).values, torch.arange(self.dataset_size))):
            raise ValueError("Invalid saved permutation or cursor")
        self.order = order.clone()
        self.cursor = cursor
        self.generator.set_state(state["generator"].cpu())


def capture_rng_state():
    # NumPy ndarrays are not accepted by torch.load(weights_only=True).
    # Encode MT19937 state using plain scalars and a tensor instead.
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": algorithm,
                      "keys": torch.tensor(keys.astype(np.int64)),
                      "position": int(position), "has_gauss": int(has_gauss),
                      "cached_gaussian": float(cached_gaussian)},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].cpu().numpy().astype(np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(obj, path):
    """Flush a complete same-directory temporary file before replacing the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".",
                                         suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(obj, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
