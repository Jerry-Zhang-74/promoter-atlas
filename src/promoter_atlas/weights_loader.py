#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""联合模型预测、筛选及文件处理。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True
from .paths import MODELS, MANIFESTS
SHARD_FORMAT = "pa.state_shard.v1"
MANIFEST_FORMAT = "pa.weight_manifest.v1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(t) -> str:
    h = hashlib.sha256()
    b = t.detach().cpu().contiguous()
    h.update(str(tuple(b.shape)).encode("ascii"))
    h.update(b.numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(sd) -> str:
    h = hashlib.sha256()
    for name, t in sorted(sd.items()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assemble(out_path=None, base=None, write=True):
    """Merge shards into the full checkpoint dict; verify every hash."""
    base = Path(base or MODELS)
    mpath = base / "WEIGHTS_MANIFEST.json"
    if not mpath.exists():
        mpath = MANIFESTS / "WEIGHTS_MANIFEST.json"
    if not mpath.exists():
        raise FileNotFoundError("missing WEIGHTS_MANIFEST.json in %s" % base)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    if man.get("format") != MANIFEST_FORMAT:
        raise ValueError("unexpected manifest format: %r" % man.get("format"))

    merged = {}
    owner = {}
    for part in man["parts"]:
        fp = base / part["file"]
        if not fp.exists():
            raise FileNotFoundError("missing shard: %s" % fp)
        got = sha256_file(fp)
        if got != part["sha256"]:
            raise ValueError("shard file SHA-256 mismatch: %s" % part["file"])
        obj = torch.load(str(fp), map_location="cpu", weights_only=False)
        if obj.get("format") != SHARD_FORMAT:
            raise ValueError("unexpected shard format in %s: %r" % (part["file"], obj.get("format")))
        for name, t in obj["tensors"].items():
            if name in owner:
                raise ValueError("tensor appears in two shards: %s" % name)
            owner[name] = part["part"]
            merged[name] = t

    expected = list(man["tensor_names"])
    if sorted(merged) != sorted(expected):
        missing = sorted(set(expected) - set(merged))
        extra = sorted(set(merged) - set(expected))
        raise ValueError("tensor key set mismatch; missing=%s extra=%s" % (missing, extra))

    for name, want in man["tensor_sha256"].items():
        if tensor_sha256(merged[name]) != want:
            raise ValueError("tensor SHA-256 mismatch: %s" % name)

    whole = state_dict_sha256(merged)
    if whole != man["state_dict_sha256"]:
        raise ValueError("assembled state_dict SHA-256 does not match the unsharded export")

    ckpt = dict(man["checkpoint_meta"])
    ckpt["state_dict"] = merged
    report = {"n_parts": man["n_parts"], "n_tensors": len(merged), "param_count": man["param_count"],
              "state_dict_sha256": whole, "verified": True}
    if write:
        out = Path(out_path or (base / man["checkpoint_file"]))
        torch.save(ckpt, str(out))
        report["written"] = str(out)
        report["written_sha256"] = sha256_file(out)
        report["written_bytes"] = out.stat().st_size
    return ckpt, report


def main(argv=None):
    ck, rep = assemble()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
