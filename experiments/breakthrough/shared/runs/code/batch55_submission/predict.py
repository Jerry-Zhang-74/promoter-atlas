#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""batch55 minimal usable predictor for the NEW batch52/batch53 first model.

Standalone: works from ANY current working directory (including paths with spaces or
Chinese characters). It locates its OWN pinned model-core copy via __file__, so it never
depends on the repository cwd and never imports the mutable batch53 working tree.

It consumes a MODEL BUNDLE produced by stage_submission.py:
    <bundle>/MODEL_MANIFEST.json      (kind TRAINED or SOFTWAREONLY_FIXTURE)
    <bundle>/<model/xxx.ckpt>         real trained weights (kind=TRAINED)

Device policy (batch55 PROTOCOL):
  * kind == "TRAINED"              -> CUDA is REQUIRED. There is NO CPU fallback for real
                                      neural inference; the run refuses if CUDA is absent.
  * kind == "SOFTWAREONLY_FIXTURE" -> CPU allowed; the artifact is an artificial no-fit
                                      software fixture and is NEVER a trained model.

Input: a FASTA file of promoter/DNA sequences (raw ACGT). STRICT validation: empty file,
empty record, duplicate id, invalid character, all-ambiguous ("null") record, or length > 600
are all rejected with an actionable error (no silent truncation, no windowing).

Output: TSV  seq_id <tab> length <tab> activity_pred_Log2RPM <tab> model_id <tab> version
plus a .meta.json sidecar with the input/output SHA256 and model provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

# ---- path independence: pin the model core relative to THIS file -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PINNED = os.path.join(_HERE, "pinned_core")
if _PINNED not in sys.path:
    sys.path.insert(0, _PINNED)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import spec as _spec  # noqa: E402
from model_core import build_model  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402

MAX_LEN = _spec.MAX_LEN                     # 600, hard; >600 unsupported
VALID_CHARS = set("ACGTN")
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


class InputError(Exception):
    """Strict user-input validation failure (never silently repaired)."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fasta_strict(path: str, max_len: int = MAX_LEN):
    """Return list of (seq_id, seq). Reject empty/duplicate/null/invalid/oversized records."""
    if not os.path.exists(path):
        raise InputError("FASTA not found: %s" % path)
    if os.path.getsize(path) == 0:
        raise InputError("FASTA is empty (0 bytes): %s" % path)
    records = []
    seen = set()
    cur_id, buf = None, []
    started = False
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(buf)))
                cur_id, buf = line[1:].strip(), []
                if not cur_id:
                    raise InputError("empty FASTA header at line %d" % lineno)
                if cur_id in seen:
                    raise InputError("duplicate sequence id %r (line %d)" % (cur_id, lineno))
                seen.add(cur_id)
                started = True
            elif line.strip():
                if not started:
                    raise InputError("FASTA must start with a '>' header (line %d)" % lineno)
                buf.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, "".join(buf)))
    if not records:
        raise InputError("FASTA has no records: %s" % path)
    for sid, seq in records:
        s = seq.upper()
        if not s:
            raise InputError("null/empty sequence for record %r" % sid)
        bad = sorted(set(s) - VALID_CHARS)
        if bad:
            raise InputError("invalid base characters %s in record %r" % (bad, sid))
        if len(s) > max_len:
            raise InputError("record %r length %d > %d (unsupported, no windowing)"
                             % (sid, len(s), max_len))
        if set(s) <= {"N"}:
            raise InputError("null/all-ambiguous sequence for record %r" % sid)
    return records


def encode(seq: str, width: int) -> np.ndarray:
    arr = np.full(width, 4, dtype=np.int64)     # 4 = pad/ambiguous (masked in the model)
    for i, b in enumerate(seq.upper()):
        arr[i] = _BASE.get(b, 4)
    return arr


def load_bundle(bundle_dir: str):
    man_path = os.path.join(bundle_dir, "MODEL_MANIFEST.json")
    if not os.path.exists(man_path):
        raise InputError("MODEL_MANIFEST.json not found in bundle: %s" % bundle_dir)
    with open(man_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    if man.get("target_column") != _spec.TARGET_COLUMN:
        raise InputError("bundle target_column %r != required %r"
                         % (man.get("target_column"), _spec.TARGET_COLUMN))
    ck = man.get("checkpoint") or {}
    ck_path = ck.get("path", "")
    ck_abs = ck_path if os.path.isabs(ck_path) else os.path.join(bundle_dir, ck_path)
    if not os.path.exists(ck_abs):
        raise InputError("checkpoint missing in bundle: %s" % ck_abs)
    if ck.get("sha256"):
        actual = sha256_file(ck_abs)
        if actual != ck["sha256"]:
            raise InputError("checkpoint SHA256 mismatch: %s != %s" % (actual, ck["sha256"]))
    obj = load_checkpoint(ck_abs, expect_config_sha=man.get("config_sha256"), expect_datahash=None)
    if obj["config"].get("target_column") != _spec.TARGET_COLUMN:
        raise InputError("checkpoint target %r != %r"
                         % (obj["config"].get("target_column"), _spec.TARGET_COLUMN))
    return man, ck_abs, obj


def resolve_device(kind: str, requested: str) -> str:
    cuda = torch.cuda.is_available()
    if kind == "TRAINED":
        if requested in ("cpu",):
            raise InputError("real neural inference has NO CPU fallback (batch55 policy)")
        if not cuda:
            raise InputError("CUDA unavailable: real neural inference refuses to fall back to CPU")
        return "cuda:0" if requested in ("auto", "cuda", "cuda:0") else requested
    if kind == "SOFTWAREONLY_FIXTURE":
        # Fixtures are software-only: default to CPU so they never contend for the GPU that
        # the real training/eval owner may be using. GPU only if explicitly requested.
        if requested == "auto":
            return "cpu"
        return requested
    raise InputError("unknown bundle kind %r (expected TRAINED or SOFTWAREONLY_FIXTURE)" % kind)


def main(argv=None):
    ap = argparse.ArgumentParser(description="batch55 standalone predictor (new first model)")
    ap.add_argument("--bundle", required=True, help="model bundle dir with MODEL_MANIFEST.json")
    ap.add_argument("--fasta", required=True, help="input FASTA (ACGT, <=600bp per record)")
    ap.add_argument("--out", required=True, help="output TSV path")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args(argv)

    bundle_dir = os.path.abspath(a.bundle)
    man, ck_abs, obj = load_bundle(bundle_dir)
    kind = man.get("kind")
    device = resolve_device(kind, a.device)

    if kind == "SOFTWAREONLY_FIXTURE":
        sys.stderr.write("*** SOFTWAREONLY FIXTURE: artificial no-fit weights - NOT a trained model ***\n")

    records = parse_fasta_strict(a.fasta)
    model = build_model()
    model.load_state_dict(obj["model"])
    model.to(device).eval()
    norm = TargetNormalizer(**{k: v for k, v in obj.get("normalizer", {}).items()
                               if k in ("mean", "std")})

    preds = []
    with torch.no_grad():
        for s in range(0, len(records), a.batch_size):
            chunk = records[s:s + a.batch_size]
            width = max(len(seq) for _, seq in chunk)
            codes = torch.from_numpy(np.stack([encode(seq, width) for _, seq in chunk]))
            z = model(codes.to(device))
            preds.extend(float(x) for x in norm.inverse(z).cpu().tolist())

    out_dir = os.path.dirname(os.path.abspath(a.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    lines = ["seq_id\tlength\tactivity_pred_Log2RPM\tmodel_id\tversion"]
    for (sid, seq), v in zip(records, preds):
        lines.append("%s\t%d\t%.6f\t%s\t%s" % (sid, len(seq), v,
                    man.get("model_id", "unknown"), man.get("version", "unknown")))
    tmp = a.out + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, a.out)

    meta = {"input_fasta": os.path.abspath(a.fasta), "input_sha256": sha256_file(a.fasta),
            "output_tsv": os.path.abspath(a.out), "output_sha256": sha256_file(a.out),
            "n_records": len(records), "device": device, "kind": kind,
            "model_id": man.get("model_id"), "version": man.get("version"),
            "target_column": man.get("target_column"),
            "units": man.get("units"), "checkpoint_sha256": man.get("checkpoint", {}).get("sha256"),
            "software_only_fixture": kind == "SOFTWAREONLY_FIXTURE",
            "source_sha256": man.get("source_sha256")}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({k: meta[k] for k in ("n_records", "device", "output_tsv",
                                           "output_sha256", "kind")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as e:
        sys.stderr.write("INPUT ERROR: %s\n" % e)
        raise SystemExit(2)
