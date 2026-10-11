# -*- coding: utf-8 -*-
"""Atomic filesystem helpers (tmp + fsync + os.replace) and ready markers.

Every batch53 artifact (checkpoint, prediction, sealed TEST output, CODE_READY) is written
atomically. A ready marker is created ONLY after the real artifact bytes are on disk, so a
consumer can poll for the marker and never read a partial file.
"""
from __future__ import annotations
import hashlib, json, os, time


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> str:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: str, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, obj) -> str:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def write_ready(ready_path: str, payload: dict) -> str:
    """Create the ready marker AFTER the artifact is complete. payload documents hashes."""
    payload = dict(payload)
    payload.setdefault("ready_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return atomic_write_json(ready_path, payload)


def read_ready(ready_path: str, required_keys=("artifact_sha256",)) -> dict:
    if not os.path.exists(ready_path):
        raise FileNotFoundError("ready marker absent: %s" % ready_path)
    with open(ready_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for k in required_keys:
        if k not in obj:
            raise ValueError("ready marker missing %r" % k)
    return obj


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
