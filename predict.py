#!/usr/bin/env python3
"""Anonymous CPU inference CLI for the promoter-atlas joint ESB model.

Tasks
-----
* ``esb``      : 400 bp + 2000 bp context -> rank-normalized ESB scores.
* ``activity`` : 100-600 bp fragments -> predicted Log2RPM_HEK293.
* ``profile``  : 400 bp + 2000 bp context -> 54 GTEx tissue profiles.

Plain Python 3.12 with torch/numpy.  CPU by default.  No network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time

sys.dont_write_bytecode = True

import numpy as np
import torch

from _bootstrap import project_root, resolve_path

_ROOT = project_root()
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from promoter_atlas.model import JointDualESBModel  # noqa: E402
from promoter_atlas.spec_contract import ESB_COLS, SPECIES_ID  # noqa: E402

FORMAT = "promoter_atlas_joint_esb.v1"
VERSION = "v6.6.0"
DEFAULT_WEIGHTS = os.path.join("models", "v6.6.0.pt")
DEFAULT_SPECIES = "homo_sapiens"
DNA_CHARS = frozenset("ACGTN")
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
ESB_AXES = ("Score_E", "Score_S", "Score_B")

# Activity fragment padding contract for the v6.6 model.
#
# The trained v6.6 checkpoint stores ``fragment_padding_length = 600`` and
# training/eval always right-pad fragments with N (code 4) to a fixed width of
# 600 bp.  Inference must reproduce that exact contract: every activity batch
# is right-padded to width 600 with code 4.
FRAGMENT_PADDING_LENGTH = 600
FRAGMENT_PADDING_STRATEGY = "right_to_600"

# Metadata columns that must never collide with profile tissue columns.
_METADATA_COLUMNS = frozenset(
    ["candidate_id", "species", "model_version", "run_id", "length_bp",
     "predicted_Log2RPM_HEK293"]
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _fail(msg: str) -> "NoReturn":
    raise SystemExit("error: " + msg)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_tsv(path: str) -> tuple[list[str], list[dict]]:
    if not os.path.isfile(path):
        _fail("input TSV not found: %s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            _fail("input TSV has no header: %s" % path)
        fields = [f.strip() for f in reader.fieldnames]
        rows = []
        for raw in reader:
            rows.append({(k.strip() if k else k): v for k, v in raw.items()})
    if not rows:
        _fail("input TSV has no data rows: %s" % path)
    return fields, rows


def _require_columns(fields: list[str], needed: list[str], path: str) -> None:
    missing = [c for c in needed if c not in fields]
    if missing:
        _fail("input TSV %s is missing column(s): %s" % (path, ", ".join(missing)))


def _check_candidate_ids(rows: list[dict]) -> None:
    seen = set()
    for i, row in enumerate(rows):
        cid = (row.get("candidate_id") or "").strip()
        if not cid:
            _fail("row %d has an empty candidate_id" % (i + 2))
        if cid in seen:
            _fail("duplicate candidate_id: %s" % cid)
        seen.add(cid)
        row["candidate_id"] = cid


def _encode_dna(seq: str, expected_len: int | None, label: str) -> list[int]:
    if seq is None:
        _fail("%s is missing" % label)
    seq = seq.strip()
    if not seq:
        _fail("%s is empty" % label)
    if seq != seq.upper():
        _fail("%s must be uppercase ACGTN" % label)
    bad = sorted(set(seq) - DNA_CHARS)
    if bad:
        _fail("%s contains invalid character(s): %s" % (label, "".join(bad)))
    if expected_len is not None and len(seq) != expected_len:
        _fail("%s must be exactly %d bp, got %d" % (label, expected_len, len(seq)))
    return [BASE_CODE[c] for c in seq]


def _resolve_species(value: str | None) -> tuple[str, int]:
    name = (value or "").strip() or DEFAULT_SPECIES
    if name not in SPECIES_ID:
        _fail("unknown species %r; known: %s" % (name, ", ".join(sorted(SPECIES_ID))))
    return name, int(SPECIES_ID[name])


def _pick_device(requested: str) -> torch.device:
    req = (requested or "cpu").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            _fail("device %r requested but CUDA is not available" % requested)
        return torch.device(req)
    _fail("unsupported device %r (use cpu or cuda)" % requested)


def _atomic_write(path: str, text: str) -> None:
    outdir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(outdir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=outdir, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    _atomic_write(path, buf.getvalue())


def _write_json(path: str, obj: dict) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _guard_output(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        _fail("output already exists (use --overwrite): %s" % path)


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #

def _load_checkpoint(weights: str, device: torch.device):
    if not os.path.isfile(weights):
        _fail("checkpoint not found: %s" % weights)
    try:
        obj = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        _fail("failed to load checkpoint %s: %s" % (weights, exc))
    if not isinstance(obj, dict):
        _fail("checkpoint is not a dict")
    if obj.get("format") != FORMAT:
        _fail("unexpected checkpoint format %r (expected %r)" % (obj.get("format"), FORMAT))
    if obj.get("version") != VERSION:
        _fail("unexpected checkpoint version %r (expected %r)" % (obj.get("version"), VERSION))
    run_id = obj.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        _fail("checkpoint run_id is missing or empty")
    if obj.get("n_dist") != 0:
        _fail("checkpoint n_dist must be 0, got %r" % (obj.get("n_dist"),))
    if not isinstance(obj.get("state_dict"), dict):
        _fail("checkpoint state_dict is missing")
    return obj


def _build_model(obj: dict, device: torch.device) -> JointDualESBModel:
    try:
        model = JointDualESBModel.from_checkpoint(obj)
    except Exception as exc:  # noqa: BLE001
        _fail("failed to build model from checkpoint: %s" % exc)
    try:
        model.load_state_dict(obj["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        _fail("strict state_dict load failed: %s" % exc)
    model.to(device)
    model.eval()
    return model


def _checkpoint_identity(obj: dict) -> tuple[str, str]:
    """Return (version, run_id) taken from the checkpoint object itself."""
    version = obj.get("version")
    run_id = obj.get("run_id")
    if not isinstance(version, str) or not version.strip():
        _fail("checkpoint version is missing or empty")
    if not isinstance(run_id, str) or not run_id.strip():
        _fail("checkpoint run_id is missing or empty")
    return version, run_id


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #

def _zstats(obj: dict) -> dict:
    zs = obj.get("zstats")
    if not isinstance(zs, dict):
        _fail("checkpoint zstats missing")
    out = {}
    for axis in ESB_AXES:
        pair = zs.get(axis)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            _fail("zstats[%s] must be [mu, sd]" % axis)
        mu, sd = float(pair[0]), float(pair[1])
        if not (math.isfinite(mu) and math.isfinite(sd)) or sd <= 0:
            _fail("zstats[%s] must be finite with positive sd" % axis)
        out[axis] = (mu, sd)
    return out


def _activity_normalizer(obj: dict) -> tuple[float, float]:
    an = obj.get("activity_normalizer")
    if not isinstance(an, dict):
        _fail("checkpoint activity_normalizer missing")
    target = an.get("target")
    if target != "Log2RPM_HEK293":
        _fail("activity_normalizer.target must be 'Log2RPM_HEK293', got %r" % (target,))
    mean, std = float(an.get("mean")), float(an.get("std"))
    if not (math.isfinite(mean) and math.isfinite(std)) or std <= 0:
        _fail("activity_normalizer mean/std must be finite with positive std")
    return mean, std


def _profile_normalizers(obj: dict) -> dict:
    pn = obj.get("profile_normalizers")
    if not isinstance(pn, dict):
        _fail("checkpoint profile_normalizers missing")
    cols = pn.get("columns")
    mean = pn.get("mean")
    std = pn.get("std")
    count = pn.get("count")
    if not isinstance(cols, list) or not cols:
        _fail("profile_normalizers.columns must be a non-empty list")
    for i, c in enumerate(cols):
        if not isinstance(c, str) or not c.strip():
            _fail("profile_normalizers.columns[%d] must be a non-empty string" % i)
    if len(set(cols)) != len(cols):
        _fail("profile_normalizers.columns contains duplicates")
    collisions = sorted(set(cols) & _METADATA_COLUMNS)
    if collisions:
        _fail("profile_normalizers.columns collide with metadata columns: %s" % ", ".join(collisions))
    n = len(cols)
    if not (isinstance(mean, list) and isinstance(std, list) and isinstance(count, list)):
        _fail("profile_normalizers mean/std/count must be lists")
    if not (len(mean) == len(std) == len(count) == n):
        _fail("profile_normalizers mean/std/count length mismatch")
    mean = [float(v) for v in mean]
    std = [float(v) for v in std]
    count = [int(v) for v in count]
    for i in range(n):
        if not math.isfinite(mean[i]):
            _fail("profile_normalizers.mean[%d] is not finite" % i)
        if not math.isfinite(std[i]) or std[i] <= 0:
            _fail("profile_normalizers.std[%d] must be finite and positive" % i)
    n_profile = obj.get("n_profile")
    if n_profile != n:
        _fail("n_profile (%r) does not match profile_normalizers length (%d)" % (n_profile, n))
    tissue_names = obj.get("tissue_names")
    if not isinstance(tissue_names, list) or list(tissue_names) != list(cols):
        _fail("tissue_names must equal profile_normalizers.columns in order")
    for i, t in enumerate(tissue_names):
        if not isinstance(t, str) or not t.strip():
            _fail("tissue_names[%d] must be a non-empty string" % i)
    return {"columns": list(cols), "mean": mean, "std": std, "count": count}


def _fragment_padding(obj: dict) -> str:
    """Resolve the activity padding strategy for the v6.6 model.

    The trained v6.6 checkpoint stores ``fragment_padding_length = 600`` and
    training/eval always right-pad with N (code 4) to width 600.  This function
    requires that exact integer contract and returns the fixed strategy name
    ``right_to_600``.  If an optional string ``fragment_padding`` field is also
    present it must agree with the fixed-width contract.
    """
    length = obj.get("fragment_padding_length")
    if isinstance(length, bool) or not isinstance(length, int):
        _fail(
            "checkpoint fragment_padding_length must be the integer %d, got %r"
            % (FRAGMENT_PADDING_LENGTH, length)
        )
    if length != FRAGMENT_PADDING_LENGTH:
        _fail(
            "checkpoint fragment_padding_length must be exactly %d, got %d"
            % (FRAGMENT_PADDING_LENGTH, length)
        )
    value = obj.get("fragment_padding")
    if value is not None:
        if not isinstance(value, str) or value != FRAGMENT_PADDING_STRATEGY:
            _fail(
                "checkpoint fragment_padding %r disagrees with fragment_padding_length=%d "
                "(expected %r)"
                % (value, FRAGMENT_PADDING_LENGTH, FRAGMENT_PADDING_STRATEGY)
            )
    return FRAGMENT_PADDING_STRATEGY


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #

def _run_esb(model, obj, rows, device, batch_size, version, run_id):
    zs = _zstats(obj)
    out_rows = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        x400, x2000, species_ids, species_names = [], [], [], []
        for row in chunk:
            name, sid = _resolve_species(row.get("species"))
            x400.append(_encode_dna(row.get("sequence_sym400"), 400, "sequence_sym400"))
            x2000.append(_encode_dna(row.get("sequence_up2000"), 2000, "sequence_up2000"))
            species_ids.append(sid)
            species_names.append(name)
        t400 = torch.tensor(x400, dtype=torch.long, device=device)
        t2000 = torch.tensor(x2000, dtype=torch.long, device=device)
        tsid = torch.tensor(species_ids, dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.forward_esb(t400, t2000, tsid)
        esb = out["esb"].detach().to("cpu").to(torch.float64).numpy()
        if esb.shape != (len(chunk), 3):
            _fail("model returned ESB shape %s, expected (%d, 3)" % (esb.shape, len(chunk)))
        if not np.isfinite(esb).all():
            _fail("model produced non-finite ESB predictions")
        for i, row in enumerate(chunk):
            vals = []
            for j, axis in enumerate(ESB_AXES):
                mu, sd = zs[axis]
                vals.append(esb[i, j] * sd + mu)
            if not all(math.isfinite(v) for v in vals):
                _fail("non-finite ESB value after inverse z for %s" % row["candidate_id"])
            out_rows.append([
                row["candidate_id"], species_names[i],
                repr(float(vals[0])), repr(float(vals[1])), repr(float(vals[2])),
                version, run_id,
            ])
    header = ["candidate_id", "species"] + list(ESB_COLS) + ["model_version", "run_id"]
    return header, out_rows


def _run_profile(model, obj, rows, device, batch_size, unit, version, run_id):
    pn = _profile_normalizers(obj)
    cols = pn["columns"]
    mean = np.asarray(pn["mean"], dtype=np.float64)
    std = np.asarray(pn["std"], dtype=np.float64)
    out_rows = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        x400, x2000, species_ids, species_names = [], [], [], []
        for row in chunk:
            name, sid = _resolve_species(row.get("species"))
            x400.append(_encode_dna(row.get("sequence_sym400"), 400, "sequence_sym400"))
            x2000.append(_encode_dna(row.get("sequence_up2000"), 2000, "sequence_up2000"))
            species_ids.append(sid)
            species_names.append(name)
        t400 = torch.tensor(x400, dtype=torch.long, device=device)
        t2000 = torch.tensor(x2000, dtype=torch.long, device=device)
        tsid = torch.tensor(species_ids, dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.forward_esb(t400, t2000, tsid)
        prof = out["profile"].detach().to("cpu").to(torch.float64).numpy()
        if prof.shape != (len(chunk), len(cols)):
            _fail("model returned profile shape %s, expected (%d, %d)" % (prof.shape, len(chunk), len(cols)))
        if not np.isfinite(prof).all():
            _fail("model produced non-finite profile predictions")
        logvals = prof * std[None, :] + mean[None, :]
        if not np.isfinite(logvals).all():
            _fail("non-finite profile value after inverse z")
        if unit == "log2p1":
            values = logvals
        else:  # tpm
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.expm1(logvals * math.log(2.0))
            if not np.isfinite(values).all():
                _fail("non-finite TPM value after inverse transform (overflow?)")
            values = np.maximum(0.0, values)
        for i, row in enumerate(chunk):
            out_rows.append(
                [row["candidate_id"], species_names[i]]
                + [repr(float(v)) for v in values[i]]
                + [version, run_id]
            )
    header = ["candidate_id", "species"] + list(cols) + ["model_version", "run_id"]
    return header, out_rows, cols


def _run_activity(model, obj, rows, device, batch_size, version, run_id):
    mean, std = _activity_normalizer(obj)
    padding = _fragment_padding(obj)
    out_rows = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        encoded, lengths = [], []
        for row in chunk:
            seq = (row.get("sequence") or "").strip()
            codes = _encode_dna(seq, None, "sequence")
            if not (100 <= len(codes) <= 600):
                _fail("sequence for %s must be 100-600 bp, got %d" % (row["candidate_id"], len(codes)))
            if all(c == 4 for c in codes):
                _fail("sequence for %s is all-N" % row["candidate_id"])
            encoded.append(codes)
            lengths.append(len(codes))
        # v6.6 contract: always right-pad with N (code 4) to width 600.
        width = FRAGMENT_PADDING_LENGTH
        batch = np.full((len(chunk), width), 4, dtype=np.int64)
        for i, codes in enumerate(encoded):
            batch[i, :len(codes)] = codes
        t = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.inference_mode():
            act = model.forward_fragment(t)
        act = act.detach().to("cpu").to(torch.float64).numpy().reshape(-1)
        if act.shape[0] != len(chunk):
            _fail("model returned activity shape %s, expected (%d,)" % (act.shape, len(chunk)))
        if not np.isfinite(act).all():
            _fail("model produced non-finite activity predictions")
        for i, row in enumerate(chunk):
            value = float(act[i]) * std + mean
            if not math.isfinite(value):
                _fail("non-finite activity value after inverse z for %s" % row["candidate_id"])
            out_rows.append([
                row["candidate_id"], lengths[i], repr(value), version, run_id,
            ])
    header = ["candidate_id", "length_bp", "predicted_Log2RPM_HEK293", "model_version", "run_id"]
    return header, out_rows, padding


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Promoter-atlas CPU inference (anonymous wrapper).")
    parser.add_argument("--task", required=True, choices=["esb", "activity", "profile"])
    parser.add_argument("--input", required=True, help="input TSV path")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--weights", default=None, help="checkpoint path (default models/v6.6.0.pt)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--profile-unit", choices=["log2p1", "tpm"], default="log2p1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.batch <= 0:
        _fail("--batch must be > 0")
    if args.threads <= 0:
        _fail("--threads must be > 0")

    torch.set_num_threads(args.threads)
    device = _pick_device(args.device)

    cwd = os.getcwd()
    # Explicit --weights resolve against the caller CWD; only an omitted
    # --weights falls back to the project-root default.
    if args.weights:
        weights = resolve_path(args.weights, cwd)
    else:
        weights = resolve_path(DEFAULT_WEIGHTS, _ROOT)
    input_path = resolve_path(args.input, cwd)
    output_path = resolve_path(args.output, cwd)

    # Preflight: guard every file we may write before writing anything.
    _guard_output(output_path, args.overwrite)
    _guard_output(output_path + ".run.json", args.overwrite)
    if args.task == "profile":
        _guard_output(output_path + ".sidecar.json", args.overwrite)

    started = time.time()
    obj = _load_checkpoint(weights, device)
    version, run_id = _checkpoint_identity(obj)
    model = _build_model(obj, device)
    checkpoint_sha = _sha256_file(weights)

    fields, rows = _read_tsv(input_path)
    _check_candidate_ids(rows)

    run_meta = {
        "task": args.task,
        "rows": len(rows),
        "version": version,
        "run_id": run_id,
        "checkpoint_basename": os.path.basename(weights),
        "checkpoint_sha256": checkpoint_sha,
        "parent_sha256": obj.get("parent_sha256"),
        "device": str(device),
        "batch_size": args.batch,
    }

    if args.task == "esb":
        _require_columns(fields, ["candidate_id", "sequence_sym400", "sequence_up2000"], input_path)
        header, out_rows = _run_esb(model, obj, rows, device, args.batch, version, run_id)
        run_meta["units"] = {
            axis: "rank_normalized_score" for axis in ESB_AXES
        }
        run_meta["units_note"] = (
            "Score_E/S/B are rank-normalized predictions after undoing the "
            "training z-score; no clipping is applied."
        )
    elif args.task == "profile":
        _require_columns(fields, ["candidate_id", "sequence_sym400", "sequence_up2000"], input_path)
        header, out_rows, cols = _run_profile(
            model, obj, rows, device, args.batch, args.profile_unit, version, run_id
        )
        unit = "log2(1+mean_tpm)" if args.profile_unit == "log2p1" else "mean_tpm"
        run_meta["units"] = {"profile": unit}
        run_meta["n_profile"] = len(cols)
        run_meta["tissue_names"] = list(cols)
        run_meta["profile_unit"] = args.profile_unit
        run_meta["prediction_not_measurement"] = True
        sidecar = {
            "task": "profile",
            "profile_unit": args.profile_unit,
            "profile_units": unit,
            "n_profile": len(cols),
            "tissue_names": list(cols),
            "prediction_not_measurement": True,
            "version": version,
            "run_id": run_id,
            "checkpoint_basename": os.path.basename(weights),
            "checkpoint_sha256": checkpoint_sha,
        }
        _write_json(output_path + ".sidecar.json", sidecar)
    else:  # activity
        _require_columns(fields, ["candidate_id", "sequence"], input_path)
        header, out_rows, padding = _run_activity(
            model, obj, rows, device, args.batch, version, run_id
        )
        run_meta["units"] = {"predicted_Log2RPM_HEK293": "log2 RPM in HEK293 (predicted)"}
        run_meta["fragment_padding"] = padding
        run_meta["fragment_padding_length"] = FRAGMENT_PADDING_LENGTH

    _write_csv(output_path, header, out_rows)
    run_meta["elapsed_seconds"] = round(time.time() - started, 3)
    _write_json(output_path + ".run.json", run_meta)
    # Successful public log: filename only, no absolute machine paths.
    print("wrote %s (%d rows)" % (os.path.basename(output_path), len(out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
