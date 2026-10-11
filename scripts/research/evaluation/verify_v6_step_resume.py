#!/usr/bin/env python3
"""CPU-only regression checks for v6 step checkpointing.

Uses tiny shuffled loaders, independent dataset RNGs, dropout, AdamW,
GradScaler and OneCycleLR. No atlas input files or GPU are accessed.
"""
import copy
import hashlib
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from v6_checkpointing import (
    StatefulBatchSampler,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


class ToyDataset(Dataset):
    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        augmentation = self.rng.random(3).astype(np.float32)
        features = torch.tensor([index / self.size, *augmentation])
        return index, features


def assert_same(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


class ToyTrainer:
    epochs = 2

    def __init__(self):
        random.seed(20261001)
        np.random.seed(20261001)
        torch.manual_seed(20261001)
        self.datasets = {
            "human": ToyDataset(20, 21),
            "species": ToyDataset(12, 22),
            "mpra": ToyDataset(8, 23),
        }
        self.samplers = {
            key: StatefulBatchSampler(len(ds), 4, 30 + index)
            for index, (key, ds) in enumerate(self.datasets.items())
        }
        self.loaders = {
            key: DataLoader(
                ds, batch_sampler=self.samplers[key], num_workers=0,
                generator=torch.Generator().manual_seed(40 + index),
            )
            for index, (key, ds) in enumerate(self.datasets.items())
        }
        self.model = nn.Sequential(nn.Linear(4, 9), nn.Dropout(0.3), nn.Linear(9, 1))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.005, total_steps=10, pct_start=0.3,
        )
        self.scaler = torch.amp.GradScaler("cpu", init_scale=64.0, growth_interval=2)
        self.epoch = 0
        self.nb = 0
        self.total = 0.0
        self.trace = []
        self.epoch_totals = []
        for sampler in self.samplers.values():
            sampler.start_cycle()
        self.iterators = {key: iter(loader) for key, loader in self.loaders.items()}

    def _end_epoch(self):
        assert self.nb == len(self.loaders["human"])
        # Atlas validation loaders consume a Torch iterator base seed. Include
        # this boundary effect to catch early/late RNG restoration errors.
        next(iter(DataLoader(TensorDataset(torch.arange(3)), batch_size=3)))
        self.epoch_totals.append((self.epoch, self.total, self.nb))
        self.epoch += 1
        self.total = 0.0
        self.nb = 0

    def _next_batch(self, key):
        try:
            return next(self.iterators[key])
        except StopIteration:
            if key == "human":
                self._end_epoch()
                assert self.epoch < self.epochs
                self.samplers["human"].start_cycle()
                self.iterators["human"] = iter(self.loaders["human"])
            else:
                self.samplers[key].start_cycle()
                self.iterators[key] = iter(self.loaders[key])
            return next(self.iterators[key])

    def run_steps(self, count):
        for _ in range(count):
            batches = {key: self._next_batch(key) for key in self.datasets}
            draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
            predictions = {key: self.model(batch[1]) for key, batch in batches.items()}
            loss = sum((pred - 0.25).square().mean() for pred in predictions.values())
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.nb += 1
            self.total += float(loss.detach())
            self.trace.append({
                "epoch": self.epoch,
                "batch": self.nb,
                "indices": {key: batch[0].clone() for key, batch in batches.items()},
                "augmentation": {key: batch[1].clone() for key, batch in batches.items()},
                "predictions": {key: pred.detach().clone() for key, pred in predictions.items()},
                "global_draws": draws,
                "loss": float(loss.detach()),
                "lr": self.optimizer.param_groups[0]["lr"],
                "scale": self.scaler.get_scale(),
            })

    def snapshot(self):
        return copy.deepcopy({
            "format_version": 2,
            "phase": "epoch_complete" if self.epoch > 0 and self.nb == 0 else "train",
            "epoch": self.epoch - 1,
            "resume_epoch": self.epoch,
            "next_batch": self.nb,
            "epoch_total": self.total,
            "epoch_nb": self.nb,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "samplers": {key: sampler.state_dict() for key, sampler in self.samplers.items()},
            "dataset_rng": {key: ds.rng.bit_generator.state for key, ds in self.datasets.items()},
            "rng": capture_rng_state(),
            "trace": self.trace,
            "epoch_totals": self.epoch_totals,
        })

    def restore(self, state):
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["sched"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = state["resume_epoch"]
        self.nb = state["epoch_nb"]
        self.total = state["epoch_total"]
        self.trace = copy.deepcopy(state["trace"])
        self.epoch_totals = copy.deepcopy(state["epoch_totals"])
        for key, sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][key])
            self.datasets[key].rng.bit_generator.state = state["dataset_rng"][key]
        if state["phase"] == "epoch_complete" and self.epoch < self.epochs:
            self.samplers["human"].start_cycle()
        self.iterators = {key: iter(loader) for key, loader in self.loaders.items()}
        restore_rng_state(state["rng"])


def verify_resume(directory):
    baseline = ToyTrainer()
    baseline.run_steps(10)
    baseline._end_epoch()
    expected = baseline.snapshot()
    for cut in (1, 3, 5, 7, 10):
        interrupted = ToyTrainer()
        interrupted.run_steps(cut)
        checkpoint = directory / f"step_{cut}.pt"
        state = interrupted.snapshot()
        atomic_torch_save(state, checkpoint)
        # Intentionally use the default safe torch.load configuration.
        persisted = torch.load(checkpoint, map_location="cpu")
        assert_same(state, persisted, f"roundtrip_{cut}")
        restored = ToyTrainer()
        restored.restore(persisted)
        assert len(restored.samplers["human"]) == 5
        restored.run_steps(10 - cut)
        restored._end_epoch()
        assert_same(expected, restored.snapshot(), f"resume_at_{cut}")
        print(f"PASS exact CPU continuation at step {cut}/10", flush=True)

    # A completed epoch checkpoint must start the next human permutation while
    # leaving the auxiliary cursors and cycles untouched.
    completed = ToyTrainer()
    completed.run_steps(5)
    completed._end_epoch()
    checkpoint = directory / "completed_epoch.pt"
    atomic_torch_save(completed.snapshot(), checkpoint)
    restored = ToyTrainer()
    restored.restore(torch.load(checkpoint, map_location="cpu"))
    restored.run_steps(5)
    restored._end_epoch()
    assert_same(expected, restored.snapshot(), "resume_after_validation")
    print("PASS completed-epoch continuation and persistent auxiliary cursors", flush=True)


def verify_atomic_failure(directory):
    checkpoint = directory / "atomic.pt"
    original = {"marker": "previous-good-checkpoint", "tensor": torch.arange(7)}
    atomic_torch_save(original, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def failing_save(obj, destination, *args, **kwargs):
        if hasattr(destination, "write"):
            destination.write(b"incomplete checkpoint")
        else:
            Path(destination).write_bytes(b"incomplete checkpoint")
        raise OSError("injected checkpoint write failure")

    with patch("torch.save", side_effect=failing_save):
        try:
            atomic_torch_save({"marker": "replacement"}, checkpoint)
        except OSError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("injected failure did not propagate")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    assert_same(original, torch.load(checkpoint, map_location="cpu"), "atomic_original")
    print("PASS interrupted atomic save preserves previous readable checkpoint", flush=True)


def main():
    torch.set_num_threads(1)
    # Guard GPU APIs to keep this verification strictly CPU-only, including
    # checkpoint RNG capture on machines that have a CUDA device.
    with patch("torch.cuda.is_available", return_value=False):
        with tempfile.TemporaryDirectory(prefix="v6_resume_verify_") as temporary:
            directory = Path(temporary)
            verify_resume(directory)
            verify_atomic_failure(directory)
    print("All v6 step-resume checks passed.", flush=True)


if __name__ == "__main__":
    main()
