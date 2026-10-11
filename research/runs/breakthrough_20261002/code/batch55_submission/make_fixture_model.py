#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an ARTIFICIAL CPU no-fit SOFTWAREONLY fixture bundle for batch55 testing.

This produces a bundle whose weights are a FRESH deterministic initialization (no training,
no gradient, no GPU, no data). It exists ONLY to exercise the packaging + predictor plumbing
without a real model. It is hard-labelled SOFTWAREONLY_FIXTURE everywhere and must never be
presented as, or staged as, a trained model.

    python make_fixture_model.py --out <bundle_dir>
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PINNED = os.path.join(_HERE, "pinned_core")
if _PINNED not in sys.path:
    sys.path.insert(0, _PINNED)

import torch  # noqa: E402
import spec  # noqa: E402
from model_core import build_model, param_sha256  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from checkpoint import save_checkpoint  # noqa: E402
from io_utils import sha256_file  # noqa: E402

BANNER = ("SOFTWAREONLY FIXTURE - artificial no-fit weights (fresh deterministic init).\n"
          "This is NOT a trained model and carries NO performance information.\n"
          "It exists only to test packaging/predictor plumbing (batch55).\n")


def build_fixture(out_dir: str, seed: int = spec.SEED) -> dict:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)

    model = build_model(seed)                     # FRESH init, never trained
    init_sha = param_sha256(model)
    norm = TargetNormalizer(mean=0.0, std=1.0)    # identity; arbitrary for a fixture
    datahash = "FIXTURE-" + hashlib.sha256(b"batch55-softwareonly-fixture").hexdigest()

    ck_rel = os.path.join("model", "FIXTURE_selected.ckpt")
    ck_abs = os.path.join(out_dir, ck_rel)
    save_checkpoint(ck_abs, model, optimizer=None, epoch=0, global_step=0,
                    normalizer_state=norm.state(), datahash=datahash, attempt_id=None,
                    extra={"fixture": True, "software_only": True, "init_sha": init_sha,
                           "note": "NOT TRAINED - artificial fixture"})
    ck_sha = sha256_file(ck_abs)

    manifest = {
        "format": "batch55.model_bundle.v1",
        "model_id": "SOFTWAREONLY_FIXTURE",
        "version": "fixture-" + init_sha[:12],
        "kind": "SOFTWAREONLY_FIXTURE",
        "target_column": spec.TARGET_COLUMN,
        "units": "artificial; NOT author Log2RPM_HEK293 predictions of a trained model",
        "source_sha256": spec.SOURCE_SHA256,
        "source_sha1": spec.SOURCE_SHA1,
        "config_sha256": spec.config_sha256(),
        "datahash": datahash,
        "checkpoint": {"path": ck_rel, "sha256": ck_sha, "epoch": 0, "global_step": 0,
                       "gate_attempt_id": None, "untrained": True},
        "normalizer": {"mean": 0.0, "std": 1.0},
        "reference": {"assembly": "hg19", "proxy_not_donor_clone": True,
                      "local_293ft_validated": False},
        "created_at": None,
        "banner": BANNER,
        "notes": ["SOFTWAREONLY artificial fixture; no training occurred."],
    }
    with open(os.path.join(out_dir, "MODEL_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "SOFTWAREONLY_BANNER.txt"), "w", encoding="utf-8") as f:
        f.write(BANNER)
    return {"bundle": out_dir, "checkpoint": ck_abs, "checkpoint_sha256": ck_sha,
            "init_sha": init_sha, "kind": "SOFTWAREONLY_FIXTURE"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="build SOFTWAREONLY fixture bundle (no fit)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=spec.SEED)
    a = ap.parse_args(argv)
    res = build_fixture(a.out, a.seed)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
