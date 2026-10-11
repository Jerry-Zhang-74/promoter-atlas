# -*- coding: utf-8 -*-
"""batch58 artificial CPU integration smoke: full run() -> graceful stop -> resume -> coverage.

ARTIFICIAL FIXTURE ONLY (tiny random cohort, CPU, CUDA hidden, throwaway ledger). This is NOT
real data and NOT a fit; it exercises the real training/resume code path so the GPU slots are
not wasted on an avoidable bug.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys, tempfile

import torch
torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.abspath(os.path.join(BASE, "..", "batch57_model_core_rc2")))

from b58_core import B58Trainer  # noqa: E402
from spec import SOURCE_SHA256, TARGET_COLUMN  # noqa: E402


class TinyCohort:
    def __init__(self, n_train=40, n_dev=12, length=120):
        self.datahash = "f" * 64
        self.source_sha256 = SOURCE_SHA256
        self.target_column = TARGET_COLUMN
        g = torch.Generator().manual_seed(1)
        n = n_train + n_dev
        self.coord_ids = ["x%d" % i for i in range(n)]
        self.group_ids = ["G%d" % (i // 4) for i in range(n)]
        self.splits = ["TRAIN"] * n_train + ["DEV"] * n_dev
        codes = torch.randint(0, 4, (n, 600), generator=g)
        codes[:, length:] = 4
        self.codes = codes
        self.lengths = torch.full((n,), float(length))
        self.targets = torch.randn(n, generator=g) * 0.5 + 1.0
        self.pairs = {s: torch.zeros((0, 2), dtype=torch.long) for s in ("TRAIN", "DEV", "TEST")}

    def split_indices(self, split):
        return torch.tensor([i for i, s in enumerate(self.splits) if s == split], dtype=torch.long)

    def activity_weights(self, split="TRAIN"):
        sel = self.split_indices(split)
        w = torch.ones(len(sel), dtype=torch.float64)
        return sel, w / w.mean()

    def pair_weights(self, split="TRAIN"):
        return self.pairs.get(split), torch.zeros(0, dtype=torch.float64)


def main():
    tmp = tempfile.mkdtemp(prefix="b58smoke_")
    led = os.path.join(tmp, "ledger.jsonl")
    out = os.path.join(tmp, "models")
    coh = TinyCohort()
    ef = 8
    t1 = B58Trainer(coh, out, led, device="cpu", effective_batch=ef, epoch_cap=2, opt_in=True,
                    accepted_data_sha=SOURCE_SHA256, run_id="Smoke", stop_after_opt_steps=1)
    r1 = t1.run(purpose="b58-Nabs-throughput-variant")
    assert r1["stopped"] and r1["stop_marker"]["saved"], r1
    latest = os.path.join(out, "latest.json")
    assert os.path.exists(latest)
    t2 = B58Trainer(coh, out, led, device="cpu", effective_batch=ef, epoch_cap=2, opt_in=True,
                    accepted_data_sha=SOURCE_SHA256, run_id="Smoke", resume_latest=latest)
    r2 = t2.run(purpose="b58-resume-continuation")
    n = 40
    truths = {
        "resumed": r2["resumed"],
        "carried_steps_match": (r2["resumed_from"]["carried_opt_steps"] == r1["opt_steps"]),
        "epochs_completed_2": r2["epochs_completed"] == 2,
        "global_samples_2n": r2["global_samples"] == 2 * n,
        "coverage_all_ok": all(h["coverage_ok"] for h in r2["history"]),
        "gate_consumed_2": True,
    }
    from b58_gate import actual_status
    truths["gate_consumed_2"] = actual_status(led)["consumed"] == 2
    ok = all(truths.values())
    print(json.dumps({"ok": ok, "truths": truths, "r1_opt_steps": r1["opt_steps"],
                      "r2_opt_steps": r2["opt_steps"], "r1_next_pos": r1["next_pos"],
                      "r2_resumed_from": r2["resumed_from"], "history": r2["history"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
