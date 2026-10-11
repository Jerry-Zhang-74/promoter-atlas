# -*- coding: utf-8 -*-
"""Create NEW release provenance for a *_selected.ckpt WITHOUT altering any training bytes.

batch53's save_checkpoint wrote a .sha256.json sidecar for the per-epoch and _mid checkpoints
but NOT for *_selected.ckpt (canonical review b53 defect D11). We do NOT rewrite or touch the
old training bytes; we (a) prove the selected file's model state is byte-identical to its
immutable source epoch checkpoint, then (b) write a new sidecar + a provenance record.

Refuses to overwrite. Zero fit / zero GPU / zero network.
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib, datetime as dt

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "breakthrough_20261002",
                                      "code", "batch53_model_core"))
sys.path.insert(0, HELPER)
from spec import config_sha256, TARGET_COLUMN  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        h.update(k.encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(selected, src_epoch_ckpt, arm, out_sidecar, out_prov):
    for p in (out_sidecar, out_prov):
        if os.path.exists(p):
            raise RuntimeError("refusing to overwrite existing provenance: %s" % p)
    sel_sha_before = sha256_file(selected)
    obj = torch.load(selected, map_location="cpu", weights_only=False)
    src = torch.load(src_epoch_ckpt, map_location="cpu", weights_only=False)
    assert obj["format"] == "batch53.checkpoint.v1", obj["format"]
    assert obj["config"]["target_column"] == TARGET_COLUMN
    assert obj["config_sha256"] == config_sha256(), "config sha drift"
    assert obj["datahash"] == src["datahash"], "datahash differs from source epoch ckpt"
    ms, msrc = state_sha(obj["model"]), state_sha(src["model"])
    sel_sha_after = sha256_file(selected)
    assert sel_sha_before == sel_sha_after, "selected bytes changed during verification"
    side = {
        "checkpoint": os.path.basename(selected),
        "path": selected,
        "checkpoint_sha256": sel_sha_before,
        "config_sha256": obj["config_sha256"],
        "datahash": obj["datahash"],
        "epoch": int(obj["epoch"]),
        "global_step": int(obj["global_step"]),
        "gate_attempt_id": obj.get("gate_attempt_id"),
    }
    with open(out_sidecar, "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    prov = {
        "kind": "batch52_new_release_provenance",
        "note": ("sidecar ABSENT for *_selected.ckpt because of batch53 defect D11; old training "
                 "bytes preserved unchanged; this file is new provenance only"),
        "arm": arm,
        "selected": selected,
        "selected_sha256": sel_sha_before,
        "selected_sha256_after_verify": sel_sha_after,
        "source_epoch_checkpoint": src_epoch_ckpt,
        "source_epoch_checkpoint_sha256": sha256_file(src_epoch_ckpt),
        "model_state_sha256_selected": ms,
        "model_state_sha256_source": msrc,
        "model_state_identical": bool(ms == msrc),
        "selected_epoch": int(obj["epoch"]),
        "optimizer_state_present": obj.get("optimizer") is not None,
        "rng_present": "rng" in obj,
        "config_sha256": obj["config_sha256"],
        "datahash": obj["datahash"],
        "target_column": obj["config"]["target_column"],
        "gate_attempt_id": obj.get("gate_attempt_id"),
        "sidecar_written": out_sidecar,
        "sidecar_sha256": sha256_file(out_sidecar),
        "created_at": dt.datetime.now().astimezone().isoformat(),
    }
    with open(out_prov, "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--source-epoch-ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-sidecar", required=True)
    ap.add_argument("--out-prov", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.selected, a.source_epoch_ckpt, a.arm, a.out_sidecar, a.out_prov),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
