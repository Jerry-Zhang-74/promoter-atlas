# -*- coding: utf-8 -*-
"""Pre-fit smoke test (NO solve, NO backward, NO optimizer.step, NO gate slot): verifies the
real cohort loads, both arms share init bytes, CUDA forward works for every parameter on
cuda:0, masked max ignores padding, and checkpoint save/load round-trips strictly."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "breakthrough_20261002", "code", "batch53_model_core"))
sys.path.insert(0, HERE); sys.path.insert(0, HELPER)
from spec import SEED, MAX_LEN, SOURCE_SHA256, config_sha256
from model_core import build_model, param_sha256, assert_cuda_all
from normalizer import TargetNormalizer
from checkpoint import save_checkpoint, load_checkpoint
import cohort_v2

COH = os.path.abspath(os.path.join(HERE, "..", "..", "data", "breakthrough_20261002", "batch52", "corrected", "cohort_v2"))
R = {"checks": {}}


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda:0"
    R["cuda"] = {"available": True, "name": torch.cuda.get_device_name(0)}
    c = cohort_v2.load_cohort_v2(COH, source_sha256=SOURCE_SHA256, mmap_codes=True)
    R["checks"]["cohort"] = {"n": len(c.coord_ids), "target": c.target_column, "datahash": c.datahash}

    a, b = build_model(SEED), build_model(SEED)
    R["checks"]["same_init_bytes"] = {"sha_a": param_sha256(a), "sha_b": param_sha256(b),
                                      "equal": param_sha256(a) == param_sha256(b)}
    assert param_sha256(a) == param_sha256(b)

    tr = c.split_indices("TRAIN")[:32]
    codes = c.codes[tr].to(dev)
    a = a.to(dev)
    with torch.no_grad():
        out = a(codes)
    assert out.shape == (32,)
    assert_cuda_all(a, codes)
    R["checks"]["cuda_forward"] = {"out_shape": list(out.shape), "finite": bool(torch.isfinite(out).all())}

    # masked max ignores padding: keep the SAME valid mask, change only the invalid (pad)
    # content 4 -> 5 (still >=4 => invalid). Output must be identical if padding is masked.
    x1 = codes.clone(); x2 = codes.clone()
    padmask = (x1 == 4)
    x2[padmask] = 5
    assert bool((x1 < 4).equal(x2 < 4)), "valid mask changed"
    a.eval()   # dropout off for a deterministic invariance check
    with torch.no_grad():
        o1, o2 = a(x1), a(x2)
    diff = float((o1 - o2).abs().max())
    R["checks"]["padding_invariant"] = {"max_abs_diff": diff, "n_pad_positions": int(padmask.sum())}
    assert diff < 1e-5, "padding content leaked into the pooled features"

    # checkpoint round-trip (no step taken; optimizer state is the fresh AdamW init)
    opt = torch.optim.AdamW(a.parameters(), lr=0.001, weight_decay=0.0001)
    norm = TargetNormalizer(0.0, 1.0)
    p = os.path.join(HERE, "_smoke_ckpt.pt")
    side = save_checkpoint(p, a, opt, epoch=0, global_step=0, datahash=c.datahash,
                           attempt_id="smoke", normalizer_state=norm.state(), extra={"smoke": True})
    obj = load_checkpoint(p, expect_config_sha=config_sha256(), expect_datahash=c.datahash)
    a2 = build_model(SEED)
    a2.load_state_dict(obj["model"])
    assert param_sha256(a2) == param_sha256(a)
    R["checks"]["checkpoint_roundtrip"] = {"side": side["checkpoint_sha256"],
                                           "config_sha256": obj["config_sha256"],
                                           "datahash_ok": obj["datahash"] == c.datahash}
    os.remove(p); os.remove(p + ".sha256.json")
    R["ok"] = True
    R["config_sha256"] = config_sha256()
    out = os.path.join(HERE, "smoke_v2.json")
    json.dump(R, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(R, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
