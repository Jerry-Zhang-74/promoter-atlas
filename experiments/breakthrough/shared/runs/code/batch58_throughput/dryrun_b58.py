# -*- coding: utf-8 -*-
"""batch58 CPU DRY-RUN validator: exercises the EXACT real-data call path the GPU probe and
trainer use, on a small REAL slice, with CUDA hidden. ZERO gate reservation, ZERO GPU,
ZERO real optimizer semantics (it does one tiny backward on CPU only to prove the loss/model
signature works on real codes). Purpose: catch interface errors BEFORE a consuming gate slot.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONUTF8"] = "1"
import json, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.abspath(os.path.join(BASE, "..", "batch57_model_core_rc2")))
sys.path.insert(0, os.path.abspath(os.path.join(BASE, "..", "..", "..", "..", "code", "batch52_data_v2")))

import torch  # noqa: E402
from cohort_v2 import load_cohort_v2  # noqa: E402
from model_core import build_model, param_sha256  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from spec import SEED, SOURCE_SHA256  # noqa: E402
from train_neural import epoch_order, compute_gc  # noqa: E402
import losses  # noqa: E402
import b58_core  # noqa: E402

COHORT = os.path.abspath(os.path.join(BASE, "..", "..", "..", "..", "data",
                                      "breakthrough_20261002", "batch52", "corrected", "cohort_v2"))
Nabs = "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f"


def main():
    t0 = time.time()
    c = load_cohort_v2(COHORT, source_sha256=SOURCE_SHA256, mmap_codes=True)
    counts = {s: sum(1 for x in c.splits if x == s) for s in ("TRAIN", "DEV", "TEST")}
    tr, wtr = c.activity_weights("TRAIN")
    sub = tr[:64]
    codes = c.codes[sub].long()
    y = c.targets[sub].float()
    w = wtr[:64].float()
    gc = compute_gc(c.codes[sub])
    model = build_model(SEED)
    init_sha = param_sha256(model)
    model.pool_stats.fit(c.lengths[sub], gc)
    norm = TargetNormalizer.fit(c.targets[sub], wtr[:64])
    yz = norm.forward(y)
    out = model(codes)
    loss, parts = losses.total_loss(out, yz, w)
    loss.backward()
    perm0 = epoch_order(64, 0).tolist()
    perm1 = epoch_order(64, 1).tolist()
    trn = b58_core.B58Trainer(cohort=c, out_dir=os.path.abspath(os.path.join(BASE, "..", "..", "..", "..",
                            "data", "breakthrough_20261002", "batch58", "_dryrun_tmp")),
                            ledger_path=os.path.join(BASE, "_dryrun_ledger.jsonl"), effective_batch=8,
                            device="cpu", run_id="dryrun", epoch_cap=1, accepted_data_sha=SOURCE_SHA256,
                            snapshot_interval_s=1e9, stopfile=None)
    print(json.dumps({"ok": True, "counts": counts, "n_train": int(tr.numel()),
                      "init_sha": init_sha, "init_sha_matches_frozen_Nabs": init_sha == Nabs,
                      "loss": float(loss), "loss_parts": [str(p) for p in (parts or [])],
                      "perm0_head": perm0[:5], "perm0_ne_perm1": perm0 != perm1,
                      "datahash": c.datahash, "elapsed_s": round(time.time() - t0, 2)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
