import argparse
import hashlib
import json
import math
import os
import sys
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from .model import JointDualESBModel
from .measured_data import MeasuredESBStream
from .joint_data import FragmentStream
from .paths import ROOT
from .layout import project_path


FORMAT = "promoter_atlas_measured_eval.v1"
CKPT_FORMAT = "promoter_atlas_joint_esb.v1"
FRAG_PAD = 600
N_PROFILE = 54


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path, obj):
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    _atomic_write_bytes(path, data)


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            converted = {}
            for k, v in arrays.items():
                if isinstance(v, np.ndarray) and v.dtype == object:
                    v = v.astype(str)
                converted[k] = v
            np.savez_compressed(f, **converted)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _exclusive_marker(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        f.flush()
        os.fsync(f.fileno())


def _hash_sorted_map(m):
    items = sorted((str(k), str(v)) for k, v in m.items())
    h = hashlib.sha256()
    for k, v in items:
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(v.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def _fingerprint_paths(source_paths):
    out = {}
    for name, p in source_paths.items():
        out[str(name)] = _sha256_file(p)
    return out


def _nonfinite_to_none(x):
    if isinstance(x, dict):
        return {k: _nonfinite_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nonfinite_to_none(v) for v in x]
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return x
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_nonfinite_to_none(v) for v in x.tolist()]
    return x


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("spearman shape mismatch")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = x.size
    if n < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= 0.0:
        return None
    return float((rx * ry).sum() / denom)


from scipy.stats import rankdata as _scipy_rankdata

def _rankdata(a):
    return _scipy_rankdata(a, method="average")


def bootstrap_spearman(x, y, n_boot, seed):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = x.size
    if n < 2:
        return {"point": None, "ci_low": None, "ci_high": None,
                "valid_replicates": 0, "n_boot": int(n_boot)}
    point = spearman(x, y)
    rng = np.random.default_rng(int(seed))
    reps = np.empty(int(n_boot), dtype=np.float64)
    reps.fill(np.nan)
    valid = 0
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        s = spearman(x[idx], y[idx])
        if s is not None and math.isfinite(s):
            reps[b] = s
            valid += 1
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        lo = hi = None
    else:
        lo = float(np.quantile(finite, 0.025))
        hi = float(np.quantile(finite, 0.975))
    return {"point": point, "ci_low": lo, "ci_high": hi,
            "valid_replicates": int(valid), "n_boot": int(n_boot)}


def aggregate_genes(ids, pred, target, mask):
    ids = np.asarray(ids)
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2 or mask.ndim != 2:
        raise ValueError("aggregate_genes expects 2D arrays")
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("aggregate_genes shape mismatch")
    if ids.shape[0] != pred.shape[0]:
        raise ValueError("aggregate_genes ids length mismatch")
    D = pred.shape[1]
    uniq, inverse = np.unique(ids, return_inverse=True)
    G = uniq.size
    out_pred = np.zeros((G, D), dtype=np.float64)
    out_target = np.zeros((G, D), dtype=np.float64)
    out_count = np.zeros((G, D), dtype=np.int64)
    valid = (mask > 0) & np.isfinite(pred) & np.isfinite(target)
    for c in range(D):
        col_valid = valid[:, c]
        if not col_valid.any():
            continue
        idx = inverse[col_valid]
        p = pred[col_valid, c]
        t = target[col_valid, c]
        np.add.at(out_pred[:, c], idx, p)
        np.add.at(out_target[:, c], idx, t)
        np.add.at(out_count[:, c], idx, 1)
    nz = out_count > 0
    out_pred[nz] = out_pred[nz] / out_count[nz]
    out_target[nz] = out_target[nz] / out_count[nz]
    out_pred[~nz] = np.nan
    out_target[~nz] = np.nan
    return uniq, out_pred, out_target, out_count


def derive_profile(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("derive_profile expects [G,T]")
    G, T = values.shape
    out = np.full((G, 3), np.nan, dtype=np.float64)
    for g in range(G):
        row = values[g]
        finite = np.isfinite(row)
        if finite.all() and T > 0:
            out[g, 0] = float(row.mean())
            out[g, 1] = float((row >= 1.0).mean())
        det = row[finite & (row >= 1.0)]
        if det.size >= 3 and out[g, 1] >= 0.3:
            med = float(np.median(det))
            out[g, 2] = -float(np.median(np.abs(det - med)))
    return out


def _load_checkpoint(path):
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or obj.get("format") != CKPT_FORMAT:
        raise RuntimeError(f"checkpoint format mismatch at {path}")
    return obj


def _validate_checkpoint(obj, tissue_names):
    z = obj.get("zstats")
    if not isinstance(z, dict):
        raise RuntimeError("checkpoint missing zstats")
    for k in ("Score_E", "Score_S", "Score_B"):
        v = z.get(k)
        if v is None or len(v) != 2:
            raise RuntimeError(f"zstats[{k}] invalid")
        m, s = float(v[0]), float(v[1])
        if not (math.isfinite(m) and math.isfinite(s) and s > 0):
            raise RuntimeError(f"zstats[{k}] non-finite or std<=0")
    pn = obj.get("profile_normalizers")
    if not isinstance(pn, dict):
        raise RuntimeError("checkpoint missing profile_normalizers")
    mean = np.asarray(pn.get("mean"), dtype=np.float64)
    std = np.asarray(pn.get("std"), dtype=np.float64)
    cnt = np.asarray(pn.get("count"), dtype=np.float64)
    cols = list(pn.get("columns", []))
    tnames = list(pn.get("tissue_names", []))
    if mean.shape != (N_PROFILE,) or std.shape != (N_PROFILE,) or cnt.shape != (N_PROFILE,):
        raise RuntimeError("profile_normalizers dimension mismatch")
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(cnt).all()):
        raise RuntimeError("profile_normalizers non-finite")
    if not (std > 0).all():
        raise RuntimeError("profile_normalizers std<=0")
    if len(cols) != N_PROFILE or len(tnames) != N_PROFILE:
        raise RuntimeError("profile_normalizers names length mismatch")
    if int(obj.get("n_profile", -1)) != N_PROFILE:
        raise RuntimeError("checkpoint n_profile mismatch")
    if list(cols) != list(tnames):
        raise RuntimeError("profile_normalizers columns must match tissue_names")
    if list(tissue_names) != tnames:
        raise RuntimeError("checkpoint tissue_names order mismatch with stream")
    if len(set(tnames)) != N_PROFILE:
        raise RuntimeError("profile_normalizers tissue_names not unique")
    if not (cnt > 0).all():
        raise RuntimeError("profile_normalizers count must be positive")
    ck_tissue = list(obj.get("tissue_names", []))
    if ck_tissue != list(tissue_names):
        raise RuntimeError("checkpoint tissue_names mismatch")
    an = obj.get("activity_normalizer")
    if not isinstance(an, dict):
        raise RuntimeError("checkpoint missing activity_normalizer")
    am, asd = float(an.get("mean")), float(an.get("std"))
    if not (math.isfinite(am) and math.isfinite(asd) and asd > 0):
        raise RuntimeError("activity_normalizer invalid")
    if an.get("target") != "Log2RPM_HEK293":
        raise RuntimeError("activity_normalizer target mismatch")
    if int(obj.get("fragment_padding_length", -1)) != FRAG_PAD:
        raise RuntimeError("fragment_padding_length mismatch")


def _build_model(obj, device):
    model = JointDualESBModel.from_checkpoint(obj)
    model.load_state_dict(obj["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _human_dev_indices(esb):
    sp = np.asarray(esb.species_norm)
    dev_idx = np.asarray(esb.split_indices("DEV"), dtype=np.int64)
    human_mask = sp == "human"
    idx = dev_idx[human_mask[dev_idx]]
    return idx.astype(np.int64)


def _extract_esb(model, esb, indices, device, batch_size, checkpoint_obj):
    n = int(indices.size)
    if n == 0:
        raise RuntimeError("empty ESB index set")
    gene_id = np.empty(n, dtype=object)
    record_id = np.empty(n, dtype=object)
    source_row = np.empty(n, dtype=np.int64)
    esb_pred = np.empty((n, 3), dtype=np.float64)
    esb_target = np.empty((n, 3), dtype=np.float64)
    esb_mask = np.empty((n, 3), dtype=np.float64)
    profile_pred = np.empty((n, N_PROFILE), dtype=np.float64)
    profile_target = np.empty((n, N_PROFILE), dtype=np.float64)
    profile_mask = np.empty((n, N_PROFILE), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            rows = indices[start:end]
            B = rows.size
            x400 = np.empty((B, 400), dtype=np.int64)
            x2000 = np.empty((B, 2000), dtype=np.int64)
            sp = np.empty((B,), dtype=np.int64)
            for j, r in enumerate(rows.tolist()):
                rec = esb.record(int(r))
                c400_raw = np.asarray(rec["local_codes400"])
                c2000_raw = np.asarray(rec["context_codes2000"])
                if c400_raw.dtype.kind not in ('i', 'u') or c2000_raw.dtype.kind not in ('i', 'u'):
                    raise RuntimeError("ESB codes must be integer arrays")
                if np.any(c400_raw < 0) or np.any(c400_raw > 4) or np.any(c2000_raw < 0) or np.any(c2000_raw > 4):
                    raise RuntimeError("ESB codes out of range 0..4")
                c400 = c400_raw.astype(np.int64)
                c2000 = c2000_raw.astype(np.int64)
                if c400.shape != (400,) or c2000.shape != (2000,):
                    raise RuntimeError("ESB code shape mismatch")
                x400[j] = c400
                x2000[j] = c2000
                sp[j] = int(rec["species"])
                gene_id[start + j] = str(rec["gene_id"])
                record_id[start + j] = str(rec["stableID"])
                source_row[start + j] = int(esb.source_rows[int(r)])
                esb_target[start + j] = np.asarray(rec["targets_ESB"], dtype=np.float64)
                esb_mask[start + j] = np.asarray(rec["finite_mask"], dtype=np.float64)
                profile_target[start + j] = np.asarray(rec["targets_profile"], dtype=np.float64)
                profile_mask[start + j] = np.asarray(rec["profile_mask"], dtype=np.float64)
                if np.any((esb_mask[start + j] > 0) & ~np.isfinite(esb_target[start + j])):
                    raise RuntimeError("non-finite ESB target under positive mask")
                if np.any((profile_mask[start + j] > 0) & ~np.isfinite(profile_target[start + j])):
                    raise RuntimeError("non-finite profile target under positive mask")
                if np.any((esb_mask[start + j] != 0) & (esb_mask[start + j] != 1)):
                    raise RuntimeError("ESB mask must be binary 0/1")
                if np.any((profile_mask[start + j] != 0) & (profile_mask[start + j] != 1)):
                    raise RuntimeError("profile mask must be binary 0/1")
            t400 = torch.from_numpy(x400).to(device)
            t2000 = torch.from_numpy(x2000).to(device)
            tsp = torch.from_numpy(sp).to(device)
            out = model.forward_esb(t400, t2000, tsp)
            z = out["esb"].detach().to("cpu").numpy().astype(np.float64)
            prof = out["profile"].detach().to("cpu").numpy().astype(np.float64)
            if z.shape != (B, 3) or prof.shape != (B, N_PROFILE):
                raise RuntimeError("model output shape mismatch")
            zstats = checkpoint_obj["zstats"]
            z = (z * np.array([zstats["Score_E"][1], zstats["Score_S"][1], zstats["Score_B"][1]], dtype=np.float64)
                 + np.array([zstats["Score_E"][0], zstats["Score_S"][0], zstats["Score_B"][0]], dtype=np.float64))
            pn = checkpoint_obj["profile_normalizers"]
            prof = prof * np.asarray(pn["std"], dtype=np.float64) + np.asarray(pn["mean"], dtype=np.float64)
            if not (np.isfinite(z).all() and np.isfinite(prof).all()):
                raise RuntimeError("non-finite model output")
            esb_pred[start:end] = z
            profile_pred[start:end] = prof
    return {
        "gene_id": gene_id,
        "record_id": record_id,
        "source_row": source_row,
        "esb_pred": esb_pred,
        "esb_target": esb_target,
        "esb_mask": esb_mask,
        "profile_pred": profile_pred,
        "profile_target": profile_target,
        "profile_mask": profile_mask,
    }


def _extract_fragment(model, frag, indices, device, batch_size, checkpoint_obj):
    n = int(indices.size)
    if n == 0:
        raise RuntimeError("empty fragment index set")
    record_id = np.empty(n, dtype=object)
    group_id = np.empty(n, dtype=object)
    source_row = np.empty(n, dtype=np.int64)
    pred = np.empty(n, dtype=np.float64)
    target = np.empty(n, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            rows = indices[start:end]
            B = rows.size
            codes = np.full((B, FRAG_PAD), 4, dtype=np.int64)
            for j, r in enumerate(rows.tolist()):
                rec = frag.record(int(r))
                c = np.asarray(rec["codes"])
                if c.ndim != 1 or c.dtype.kind not in ("i", "u"):
                    raise RuntimeError("fragment codes must be 1D integer")
                c = c.astype(np.int64)
                L = int(rec["length"])
                if not (100 <= L <= FRAG_PAD):
                    raise RuntimeError("fragment length out of range")
                if c.shape[0] != L:
                    raise RuntimeError("fragment codes length mismatch")
                if c.size and (c.min() < 0 or c.max() > 4):
                    raise RuntimeError("fragment codes out of range")
                codes[j, :L] = c
                record_id[start + j] = str(rec["stableID"])
                group_id[start + j] = str(frag.group_id[int(r)])
                source_row[start + j] = int(frag.source_rows[int(r)])
                _y = float(rec["y_Log2RPM"])
                if not math.isfinite(_y):
                    raise RuntimeError("non-finite fragment target")
                target[start + j] = _y
            tc = torch.from_numpy(codes).to(device)
            z = model.forward_fragment(tc).detach().to("cpu").numpy().astype(np.float64)
            an = checkpoint_obj["activity_normalizer"]
            z = z * float(an["std"]) + float(an["mean"])
            if z.shape != (B,):
                raise RuntimeError("fragment output shape mismatch")
            if not np.isfinite(z).all():
                raise RuntimeError("non-finite fragment output")
            pred[start:end] = z
    return {
        "record_id": record_id,
        "group_id": group_id,
        "pred": pred,
        "target": target,
        "source_row": source_row,
    }


def _save_esb_cache(path, meta_path, split, ckpt_sha, arrays, tissue_names):
    payload = dict(arrays)
    payload["tissue_names"] = np.asarray(list(tissue_names), dtype=object)
    _atomic_savez(path, **payload)
    npz_sha = _sha256_file(path)
    meta = {
        "format": FORMAT,
        "kind": "esb",
        "split": split,
        "checkpoint_sha256": ckpt_sha,
        "count": int(arrays["esb_pred"].shape[0]),
        "npz_sha256": npz_sha,
        "tissue_names": list(tissue_names),
    }
    _atomic_write_json(meta_path, meta)
    return meta


def _save_fragment_cache(path, meta_path, split, ckpt_sha, arrays):
    _atomic_savez(path, **arrays)
    npz_sha = _sha256_file(path)
    meta = {
        "format": FORMAT,
        "kind": "fragment",
        "split": split,
        "checkpoint_sha256": ckpt_sha,
        "count": int(arrays["pred"].shape[0]),
        "npz_sha256": npz_sha,
    }
    _atomic_write_json(meta_path, meta)
    return meta


def _load_npz(path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _verify_cache(path, meta_path, expected_sha, kind, split, ckptsha, tissue_names=None):
    path = Path(path)
    meta_path = Path(meta_path)
    if not path.exists():
        raise RuntimeError(f"cache missing: {path}")
    if not meta_path.exists():
        raise RuntimeError(f"cache meta missing: {meta_path}")
    meta = _load_json(meta_path)
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"cache format mismatch: {meta_path}")
    if meta.get("kind") != kind:
        raise RuntimeError(f"cache kind mismatch: {meta_path}")
    if meta.get("split") != split:
        raise RuntimeError(f"cache split mismatch: {meta_path}")
    if meta.get("checkpoint_sha256") != ckptsha:
        raise RuntimeError(f"cache checkpoint mismatch: {meta_path}")
    actual_sha = _sha256_file(path)
    if meta.get("npz_sha256") != actual_sha:
        raise RuntimeError(f"cache npz sha mismatch: {path}")
    if expected_sha is None:
        raise RuntimeError("expected_sha must be provided")
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError("expected_sha must be 64 hex characters")
    if actual_sha != expected_sha:
        raise RuntimeError(f"cache expected sha mismatch: {path}")
    arrays = _load_npz(path)
    if kind == "esb":
        if "esb_pred" not in arrays:
            raise RuntimeError(f"cache missing esb_pred: {path}")
        count = int(arrays["esb_pred"].shape[0])
    else:
        if "pred" not in arrays:
            raise RuntimeError(f"cache missing pred: {path}")
        count = int(arrays["pred"].shape[0])
    if int(meta.get("count", -1)) != count:
        raise RuntimeError(f"cache count mismatch: {path}")
    if tissue_names is not None:
        if "tissue_names" not in arrays:
            raise RuntimeError(f"cache missing tissue_names: {path}")
        cached = [str(x) for x in np.asarray(arrays["tissue_names"]).tolist()]
        if cached != list(tissue_names):
            raise RuntimeError(f"cache tissue_names order mismatch: {path}")
        if list(meta.get("tissue_names", [])) != list(tissue_names):
            raise RuntimeError(f"cache meta tissue_names mismatch: {path}")
    return arrays


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    root = cfg.get("root") or str(ROOT)
    root = str(Path(root).resolve())
    table_rel = cfg.get("table_path")
    if table_rel is None:
        raise RuntimeError("config missing table_path")
    table_path = Path(table_rel)
    if not table_path.is_absolute():
        table_path = project_path(root, table_rel)
    else:
        table_path = str(table_path.resolve())
    outdir = cfg.get("outdir")
    if outdir is not None:
        outdir_path = Path(outdir)
        if not outdir_path.is_absolute():
            outdir_path = Path(root) / outdir_path
        cfg["outdir"] = str(outdir_path.resolve())
    return cfg, root, table_path


def _evaldir(args, cfg):
    if args.outdir:
        return Path(args.outdir).resolve()
    train_out = cfg.get("outdir")
    if not train_out:
        raise RuntimeError("config missing outdir and --outdir not given")
    return (Path(train_out) / "final_evaluation").resolve()


def _check_legacy_markers(train_dir):
    for name in ("final_test_started.json", "final_test.json"):
        p = Path(train_dir) / name
        if p.exists():
            raise RuntimeError(f"legacy training marker present: {p}")


def _esb_metrics(arrays, tissue_names, n_boot=1000, seed=20261012):
    gene_id = arrays["gene_id"]
    pred = arrays["esb_pred"]
    target = arrays["esb_target"]
    mask = arrays["esb_mask"]
    heads = ["E", "S", "B"]
    gids, gpred, gtarget, gcount = aggregate_genes(gene_id, pred, target, mask)
    out = {"n_rows": int(pred.shape[0]), "n_genes": int(gids.size), "heads": {}}
    out["raw_label_rows"] = int(np.any(mask > 0, axis=1).sum())
    out["raw_label_rows_per_head"] = {
        "E": int(np.any(mask[:, 0:1] > 0, axis=1).sum()),
        "S": int(np.any(mask[:, 1:2] > 0, axis=1).sum()),
        "B": int(np.any(mask[:, 2:3] > 0, axis=1).sum()),
    }
    out["source_index_fingerprint"] = _sha256_bytes(np.ascontiguousarray(arrays["source_row"], dtype="<i8").tobytes())
    for h, name in enumerate(heads):
        p = gpred[:, h]
        t = gtarget[:, h]
        c = gcount[:, h]
        valid = (c > 0) & np.isfinite(p) & np.isfinite(t)
        nv = int(valid.sum())
        sp = spearman(p[valid], t[valid]) if nv >= 2 else None
        rmse = None
        if nv >= 1:
            rmse = float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))
        boot = bootstrap_spearman(p[valid], t[valid], n_boot, seed + h)
        out["heads"][name] = {
            "valid_genes": nv,
            "spearman": sp,
            "rmse": rmse,
            "bootstrap": boot,
        }
    out["gene_table"] = {
        "gene_id": [str(g) for g in gids.tolist()],
        "pred": gpred,
        "target": gtarget,
        "count": gcount,
    }
    return out


def _profile_metrics(arrays, tissue_names):
    gene_id = arrays["gene_id"]
    pred = arrays["profile_pred"]
    target = arrays["profile_target"]
    mask = arrays["profile_mask"]
    gids, gpred, gtarget, gcount = aggregate_genes(gene_id, pred, target, mask)
    T = len(tissue_names)
    per_tissue = []
    sps = []
    rmses = []
    for t in range(T):
        p = gpred[:, t]
        tg = gtarget[:, t]
        c = gcount[:, t]
        valid = (c > 0) & np.isfinite(p) & np.isfinite(tg)
        nv = int(valid.sum())
        sp = spearman(p[valid], tg[valid]) if nv >= 2 else None
        rmse = float(np.sqrt(np.mean((p[valid] - tg[valid]) ** 2))) if nv >= 1 else None
        per_tissue.append({
            "tissue": str(tissue_names[t]),
            "n_gene": nv,
            "spearman": sp,
            "rmse": rmse,
        })
        if sp is not None:
            sps.append(sp)
        if rmse is not None:
            rmses.append(rmse)
    summary = {
        "mean_spearman": float(np.mean(sps)) if sps else None,
        "median_spearman": float(np.median(sps)) if sps else None,
        "macro_rmse": float(np.mean(rmses)) if rmses else None,
        "n_tissue_with_spearman": len(sps),
    }
    return {
        "n_rows": int(pred.shape[0]),
        "n_genes": int(gids.size),
        "per_tissue": per_tissue,
        "summary": summary,
        "gene_table": {
            "gene_id": [str(g) for g in gids.tolist()],
            "pred": gpred,
            "target": gtarget,
            "count": gcount,
        },
        "raw_label_rows": int(np.any(mask > 0, axis=1).sum()),
        "source_index_fingerprint": _sha256_bytes(np.ascontiguousarray(arrays["source_row"], dtype="<i8").tobytes()),
    }


def _derived_metrics(arrays):
    gene_id = arrays["gene_id"]
    pred = arrays["profile_pred"]
    target = arrays["profile_target"]
    mask = arrays["profile_mask"]
    gids, gpred, gtarget, gcount = aggregate_genes(gene_id, pred, target, mask)
    complete = (gcount > 0).all(axis=1)
    if not complete.any():
        return {"n_complete_genes": 0, "heads": {}}
    dp = derive_profile(gpred[complete])
    dt = derive_profile(gtarget[complete])
    names = ["E_raw", "B_raw", "S_raw"]
    heads = {}
    for h, name in enumerate(names):
        p = dp[:, h]
        t = dt[:, h]
        valid = np.isfinite(p) & np.isfinite(t)
        nv = int(valid.sum())
        sp = spearman(p[valid], t[valid]) if nv >= 2 else None
        rmse = float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2))) if nv >= 1 else None
        heads[name] = {"valid_genes": nv, "spearman": sp, "rmse": rmse}
    return {"n_complete_genes": int(complete.sum()), "heads": heads}


def _fragment_metrics(arrays, n_boot=1000, seed=20261012):
    pred = arrays["pred"]
    target = arrays["target"]
    group_id = arrays["group_id"]
    valid = np.isfinite(pred) & np.isfinite(target)
    nv = int(valid.sum())
    sp = spearman(pred[valid], target[valid]) if nv >= 2 else None
    rmse = float(np.sqrt(np.mean((pred[valid] - target[valid]) ** 2))) if nv >= 1 else None
    uniq = np.unique(group_id)
    gp = np.full(uniq.size, np.nan)
    gt = np.full(uniq.size, np.nan)
    gc = np.zeros(uniq.size, dtype=np.int64)
    index = {g: i for i, g in enumerate(uniq.tolist())}
    for r in range(pred.shape[0]):
        if not valid[r]:
            continue
        gi = index[group_id[r]]
        if gc[gi] == 0:
            gp[gi] = pred[r]
            gt[gi] = target[r]
        else:
            gp[gi] += pred[r]
            gt[gi] += target[r]
        gc[gi] += 1
    nz = gc > 0
    gp[nz] = gp[nz] / gc[nz]
    gt[nz] = gt[nz] / gc[nz]
    gvalid = nz & np.isfinite(gp) & np.isfinite(gt)
    ng = int(gvalid.sum())
    gsp = spearman(gp[gvalid], gt[gvalid]) if ng >= 2 else None
    grmse = float(np.sqrt(np.mean((gp[gvalid] - gt[gvalid]) ** 2))) if ng >= 1 else None
    boot = bootstrap_spearman(gp[gvalid], gt[gvalid], n_boot, seed + 7)
    return {
        "n_rows": int(pred.shape[0]),
        "valid_rows": nv,
        "spearman": sp,
        "rmse": rmse,
        "n_groups": int(uniq.size),
        "valid_groups": ng,
        "group_spearman": gsp,
        "group_rmse": grmse,
        "group_bootstrap": boot,
        "group_table": {
            "group_id": [str(g) for g in uniq.tolist()],
            "pred": gp,
            "target": gt,
            "count": gc,
        },
        "raw_label_rows": int(valid.sum()),
        "source_index_fingerprint": _sha256_bytes(np.ascontiguousarray(arrays["source_row"], dtype="<i8").tobytes()),
    }


def _write_esb_csvs(outdir, prefix, metrics, tissue_names):
    import csv
    gene_path = outdir / f"{prefix}_esb_genes.csv"
    with open(gene_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene_id", "E_pred", "E_target", "E_count",
                    "S_pred", "S_target", "S_count",
                    "B_pred", "B_target", "B_count"])
        gt = metrics["gene_table"]
        for i, g in enumerate(gt["gene_id"]):
            row = [g]
            for h in range(3):
                p = gt["pred"][i, h]
                t = gt["target"][i, h]
                c = gt["count"][i, h]
                row.append("" if not math.isfinite(float(p)) else f"{float(p):.6g}")
                row.append("" if not math.isfinite(float(t)) else f"{float(t):.6g}")
                row.append(int(c))
            w.writerow(row)
    prof = metrics["profile"]
    wide_path = outdir / f"{prefix}_profile_genes.csv"
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["gene_id"]
        for t in tissue_names:
            header += [f"{t}_pred", f"{t}_target", f"{t}_count"]
        w.writerow(header)
        gt = prof["gene_table"]
        for i, g in enumerate(gt["gene_id"]):
            row = [g]
            for t in range(len(tissue_names)):
                p = gt["pred"][i, t]
                tg = gt["target"][i, t]
                c = gt["count"][i, t]
                row.append("" if not math.isfinite(float(p)) else f"{float(p):.6g}")
                row.append("" if not math.isfinite(float(tg)) else f"{float(tg):.6g}")
                row.append(int(c))
            w.writerow(row)
    tissue_path = outdir / f"{prefix}_profile_tissues.csv"
    with open(tissue_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tissue", "n_gene", "spearman", "rmse"])
        for row in prof["per_tissue"]:
            sp = "" if row["spearman"] is None else f"{row['spearman']:.6g}"
            rm = "" if row["rmse"] is None else f"{row['rmse']:.6g}"
            w.writerow([row["tissue"], row["n_gene"], sp, rm])


def _write_fragment_csv(outdir, prefix, metrics):
    import csv
    path = outdir / f"{prefix}_fragment_groups.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "pred", "target", "count"])
        gt = metrics["group_table"]
        for i, g in enumerate(gt["group_id"]):
            p = gt["pred"][i]
            t = gt["target"][i]
            c = gt["count"][i]
            w.writerow([
                g,
                "" if not math.isfinite(float(p)) else f"{float(p):.6g}",
                "" if not math.isfinite(float(t)) else f"{float(t):.6g}",
                int(c),
            ])


def _write_report_md(outdir, report):
    lines = []
    lines.append("# v6.6 原留出分割结果")
    lines.append("")
    lines.append("模型从具有历史同源监督的v6.0权重初始化；本报告使用项目原留出分割。")
    lines.append("")
    lines.append("单位: ESB=original Score_E/Score_S/Score_B scales; profile=log2(1+TPM); fragment=Log2RPM_HEK293。")
    lines.append("")
    lines.append("检查点基于 DEV 上 E 头基因级 Spearman 选择，随后在 TEST 上评估；bootstrap 来自冻结的预测缓存。")
    lines.append("")
    lines.append("## 直接 ESB 头")
    for name, m in report["test"]["esb"]["heads"].items():
        sp = m["spearman"]
        rm = m["rmse"]
        b = m["bootstrap"]
        lines.append(f"- {name}: valid_genes={m['valid_genes']}, spearman={sp}, rmse={rm}, "
                     f"bootstrap_ci=[{b['ci_low']}, {b['ci_high']}] valid_replicates={b['valid_replicates']}")
    lines.append("")
    lines.append("## 组织表达谱 (log2(1+TPM))")
    s = report["test"]["profile"]["summary"]
    lines.append(f"- mean_spearman={s['mean_spearman']}, median_spearman={s['median_spearman']}, "
                 f"macro_rmse={s['macro_rmse']}, n_tissue_with_spearman={s['n_tissue_with_spearman']}")
    lines.append("")
    lines.append("## 片段活性 (Log2RPM_HEK293)")
    fm = report["test"]["fragment"]
    lines.append(f"- valid_rows={fm['valid_rows']}, spearman={fm['spearman']}, rmse={fm['rmse']}")
    lines.append(f"- valid_groups={fm['valid_groups']}, group_spearman={fm['group_spearman']}, "
                 f"group_rmse={fm['group_rmse']}")
    gb = fm["group_bootstrap"]
    lines.append(f"- group_bootstrap_ci=[{gb['ci_low']}, {gb['ci_high']}] valid_replicates={gb['valid_replicates']}")
    lines.append("")
    lines.append("## 派生谱统计")
    dm = report["test"]["derived"]
    lines.append(f"- n_complete_genes={dm['n_complete_genes']}")
    for name, m in dm["heads"].items():
        lines.append(f"- {name}: valid_genes={m['valid_genes']}, spearman={m['spearman']}, rmse={m['rmse']}")
    lines.append("")
    lines.append("## DEV 选择")
    sel = report["selection"]
    lines.append(f"- selected={sel['selected']}")
    for name, c in sel["candidates"].items():
        lines.append(f"- {name}: spearman_E={c['spearman_E']}, sha256={c['sha256']}")
    lines.append("")
    _atomic_write_bytes(outdir / "REPORT.md", ("\n".join(lines) + "\n").encode("utf-8"))


def _cmd_select(args, cfg, root, table_path, evaldir, train_dir):
    sel_path = evaldir / "selection.json"
    if sel_path.exists():
        raise RuntimeError(f"selection already exists: {sel_path}")
    if (evaldir / "test_started.json").exists():
        raise RuntimeError("test already started; refusing to reselect")
    _check_legacy_markers(train_dir)
    evaldir.mkdir(parents=True, exist_ok=True)
    select_started = evaldir / "select_started.json"
    try:
        _exclusive_marker(select_started, {"format": FORMAT})
    except FileExistsError:
        raise RuntimeError("selection already in progress")
    esb = MeasuredESBStream(root, table_path=table_path)
    tissue_names = list(esb.tissue_names)
    dev_idx = _human_dev_indices(esb)
    if dev_idx.size == 0:
        raise RuntimeError("no human DEV rows")
    ckpt_paths = {"best": train_dir / "best.pt", "last": train_dir / "last.pt"}
    for name, p in ckpt_paths.items():
        if not p.exists():
            raise RuntimeError(f"missing checkpoint: {p}")
    esb_source_paths = dict(esb.source_paths)
    esb_fp = _fingerprint_paths(esb_source_paths)
    esb_fp_hash = _hash_sorted_map(esb_fp)
    candidates = {}
    for name, p in ckpt_paths.items():
        sha_before = _sha256_file(p)
        obj = _load_checkpoint(p)
        _validate_checkpoint(obj, tissue_names)
        model = _build_model(obj, args.device)
        arrays = _extract_esb(model, esb, dev_idx, args.device, args.batch_size, obj)
        sha_after = _sha256_file(p)
        if sha_before != sha_after:
            raise RuntimeError(f"checkpoint changed during extraction: {p}")
        gids, gpred, gtarget, gcount = aggregate_genes(
            arrays["gene_id"], arrays["esb_pred"], arrays["esb_target"], arrays["esb_mask"])
        valid = (gcount[:, 0] > 0) & np.isfinite(gpred[:, 0]) & np.isfinite(gtarget[:, 0])
        sp = spearman(gpred[valid, 0], gtarget[valid, 0]) if int(valid.sum()) >= 2 else None
        cache_path = evaldir / f"dev_{name}_esb.npz"
        meta_path = evaldir / f"dev_{name}_esb.json"
        meta = _save_esb_cache(cache_path, meta_path, "DEV", sha_before, arrays, tissue_names)
        candidates[name] = {
            "sha256": sha_before,
            "spearman_E": sp,
            "cache_path": str(cache_path),
            "cache_meta_path": str(meta_path),
            "cache_sha256": meta["npz_sha256"],
            "n_genes": int(gids.size),
            "valid_genes_E": int(valid.sum()),
        }
    valid_candidates = {k: v for k, v in candidates.items() if v["spearman_E"] is not None}
    if not valid_candidates:
        raise RuntimeError("both checkpoints invalid on DEV E")
    best_name = None
    best_score = None
    for name in ("best", "last"):
        if name not in valid_candidates:
            continue
        s = valid_candidates[name]["spearman_E"]
        if best_score is None or s >= best_score:
            best_score = s
            best_name = name
    if best_name is None:
        raise RuntimeError("no valid candidate")
    chosen_sha = candidates[best_name]["sha256"]
    chosen_bytes = Path(ckpt_paths[best_name]).read_bytes()
    if _sha256_bytes(chosen_bytes) != chosen_sha:
        raise RuntimeError("chosen checkpoint sha mismatch on read")
    selected_path = evaldir / "selected.pt"
    _atomic_write_bytes(selected_path, chosen_bytes)
    if _sha256_file(selected_path) != chosen_sha:
        raise RuntimeError("selected.pt sha mismatch after copy")
    frag = FragmentStream(root)
    frag_source_paths = dict(frag.source_paths)
    frag_fp = _fingerprint_paths(frag_source_paths)
    frag_fp_hash = _hash_sorted_map(frag_fp)
    obj = _load_checkpoint(selected_path)
    _validate_checkpoint(obj, tissue_names)
    model = _build_model(obj, args.device)
    frag_dev_idx = np.asarray(frag.split_indices("DEV"), dtype=np.int64)
    frag_arrays = _extract_fragment(model, frag, frag_dev_idx, args.device, args.batch_size, obj)
    frag_cache = evaldir / "dev_selected_fragment.npz"
    frag_meta = evaldir / "dev_selected_fragment.json"
    _save_fragment_cache(frag_cache, frag_meta, "DEV", chosen_sha, frag_arrays)
    selection = {
        "format": FORMAT,
        "selected": best_name,
        "criterion": "max gene-level E Spearman on human DEV; tie -> last",
        "checkpoint_sha256": chosen_sha,
        "selected_path": str(selected_path),
        "candidates": candidates,
        "esb_source_fingerprint": esb_fp,
        "esb_source_fingerprint_hash": esb_fp_hash,
        "fragment_source_fingerprint": frag_fp,
        "fragment_source_fingerprint_hash": frag_fp_hash,
        "tissue_names": tissue_names,
        "dev_count": int(dev_idx.size),
        "fragment_dev_count": int(frag_dev_idx.size),
        "dev_fragment_cache": str(frag_cache),
        "dev_fragment_cache_sha256": _sha256_file(frag_cache),
    }
    _atomic_write_json(sel_path, _nonfinite_to_none(selection))
    try:
        os.unlink(select_started)
    except OSError:
        pass
    return 0


def _cmd_test(args, cfg, root, table_path, evaldir, train_dir):
    sel_path = evaldir / "selection.json"
    if not sel_path.exists():
        raise RuntimeError("selection.json missing; run select first")
    sel = _load_json(sel_path)
    if sel.get("format") != FORMAT:
        raise RuntimeError("selection format mismatch")
    if (evaldir / "test_started.json").exists():
        raise RuntimeError("test_started.json exists; no retry. Use summarize on existing complete cache.")
    _check_legacy_markers(train_dir)
    selected_path = Path(sel["selected_path"])
    if not selected_path.exists():
        raise RuntimeError("selected.pt missing")
    if _sha256_file(selected_path) != sel["checkpoint_sha256"]:
        raise RuntimeError("selected.pt sha mismatch")
    esb = MeasuredESBStream(root, table_path=table_path)
    tissue_names = list(esb.tissue_names)
    if tissue_names != list(sel["tissue_names"]):
        raise RuntimeError("tissue_names mismatch with selection")
    esb_fp = _fingerprint_paths(dict(esb.source_paths))
    if _hash_sorted_map(esb_fp) != sel["esb_source_fingerprint_hash"]:
        raise RuntimeError("ESB source fingerprint mismatch")
    frag = FragmentStream(root)
    frag_fp = _fingerprint_paths(dict(frag.source_paths))
    if _hash_sorted_map(frag_fp) != sel["fragment_source_fingerprint_hash"]:
        raise RuntimeError("fragment source fingerprint mismatch")
    _sel_cand = sel["candidates"][sel["selected"]]
    _verify_cache(
        _sel_cand["cache_path"], _sel_cand["cache_meta_path"],
        _sel_cand["cache_sha256"], "esb", "DEV", sel["checkpoint_sha256"],
        tissue_names=tissue_names)
    _verify_cache(
        sel["dev_fragment_cache"], evaldir / "dev_selected_fragment.json",
        sel["dev_fragment_cache_sha256"], "fragment", "DEV", sel["checkpoint_sha256"])
    _exclusive_marker(evaldir / "test_started.json", {
        "format": FORMAT,
        "checkpoint_sha256": sel["checkpoint_sha256"],
        "esb_source_fingerprint_hash": sel["esb_source_fingerprint_hash"],
        "fragment_source_fingerprint_hash": sel["fragment_source_fingerprint_hash"],
    })
    obj = _load_checkpoint(selected_path)
    _validate_checkpoint(obj, tissue_names)
    model = _build_model(obj, args.device)
    test_idx = np.asarray(esb.split_indices("TEST"), dtype=np.int64)
    _sp_test = np.asarray(esb.species_norm)
    test_idx = test_idx[_sp_test[test_idx] == "human"]
    if test_idx.size == 0:
        raise RuntimeError("empty TEST ESB set")
    arrays = _extract_esb(model, esb, test_idx, args.device, args.batch_size, obj)
    if _sha256_file(selected_path) != sel["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during TEST extraction")
    _save_esb_cache(evaldir / "test_esb.npz", evaldir / "test_esb.json",
                    "TEST", sel["checkpoint_sha256"], arrays, tissue_names)
    frag_test_idx = np.asarray(frag.split_indices("TEST"), dtype=np.int64)
    if frag_test_idx.size == 0:
        raise RuntimeError("empty TEST fragment set")
    frag_arrays = _extract_fragment(model, frag, frag_test_idx, args.device, args.batch_size, obj)
    if _sha256_file(selected_path) != sel["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during TEST fragment extraction")
    _save_fragment_cache(evaldir / "test_fragment.npz", evaldir / "test_fragment.json",
                         "TEST", sel["checkpoint_sha256"], frag_arrays)
    _atomic_write_json(evaldir / "test_complete.json", {
        "format": FORMAT,
        "checkpoint_sha256": sel["checkpoint_sha256"],
        "esb_cache_sha256": _sha256_file(evaldir / "test_esb.npz"),
        "fragment_cache_sha256": _sha256_file(evaldir / "test_fragment.npz"),
    })
    return 0


def _cmd_summarize(args, cfg, root, table_path, evaldir, train_dir):
    sel_path = evaldir / "selection.json"
    if not sel_path.exists():
        raise RuntimeError("selection.json missing")
    sel = _load_json(sel_path)
    if sel.get("format") != FORMAT:
        raise RuntimeError("selection format mismatch")
    if not (evaldir / "test_complete.json").exists():
        raise RuntimeError("test_complete.json missing; cannot summarize")
    tc = _load_json(evaldir / "test_complete.json")
    if tc.get("format") != FORMAT:
        raise RuntimeError("test_complete format mismatch")
    started_path = evaldir / "test_started.json"
    if not started_path.exists():
        raise RuntimeError("test_started.json missing")
    started_data = _load_json(started_path)
    if started_data.get("checkpoint_sha256") != sel["checkpoint_sha256"]:
        raise RuntimeError("test_started.json checkpoint_sha256 mismatch")
    if tc.get("checkpoint_sha256") != sel["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha mismatch between selection and test_complete")
    selected_path = Path(sel["selected_path"])
    if not selected_path.exists():
        raise RuntimeError("selected.pt missing")
    if _sha256_file(selected_path) != sel["checkpoint_sha256"]:
        raise RuntimeError("selected.pt sha mismatch")
    tissue_names = list(sel["tissue_names"])
    _sel_cand = sel["candidates"][sel["selected"]]
    dev_esb = _verify_cache(
        _sel_cand["cache_path"], _sel_cand["cache_meta_path"],
        _sel_cand["cache_sha256"], "esb", "DEV", sel["checkpoint_sha256"],
        tissue_names=tissue_names)
    dev_frag = _verify_cache(
        sel["dev_fragment_cache"], evaldir / "dev_selected_fragment.json",
        sel["dev_fragment_cache_sha256"], "fragment", "DEV", sel["checkpoint_sha256"])
    test_esb = _verify_cache(
        evaldir / "test_esb.npz", evaldir / "test_esb.json",
        tc.get("esb_cache_sha256"), "esb", "TEST", sel["checkpoint_sha256"],
        tissue_names=tissue_names)
    test_frag = _verify_cache(
        evaldir / "test_fragment.npz", evaldir / "test_fragment.json",
        tc.get("fragment_cache_sha256"), "fragment", "TEST", sel["checkpoint_sha256"])
    dev_esb_metrics = _esb_metrics(dev_esb, tissue_names, args.bootstrap, args.seed)
    dev_profile_metrics = _profile_metrics(dev_esb, tissue_names)
    dev_derived = _derived_metrics(dev_esb)
    dev_frag_metrics = _fragment_metrics(dev_frag, args.bootstrap, args.seed)
    test_esb_metrics = _esb_metrics(test_esb, tissue_names, args.bootstrap, args.seed)
    test_profile_metrics = _profile_metrics(test_esb, tissue_names)
    test_derived = _derived_metrics(test_esb)
    test_frag_metrics = _fragment_metrics(test_frag, args.bootstrap, args.seed)
    report = {
        "format": FORMAT,
        "title": "v6.6 原留出分割结果",
        "selection": {
            "selected": sel["selected"],
            "criterion": sel["criterion"],
            "checkpoint_sha256": sel["checkpoint_sha256"],
            "candidates": sel["candidates"],
        },
        "initialization": "model initialized from v6.0 weights with historical homology supervision; original held-out split used",
        "units": {
            "esb": "original Score_E/Score_S/Score_B scales",
            "profile": "log2(1+TPM)",
            "fragment": "Log2RPM_HEK293",
        },
        "dev": {
            "esb": {k: v for k, v in dev_esb_metrics.items() if k != "gene_table"},
            "profile": {k: v for k, v in dev_profile_metrics.items() if k != "gene_table"},
            "derived": dev_derived,
            "fragment": {k: v for k, v in dev_frag_metrics.items() if k != "group_table"},
        },
        "test": {
            "esb": {k: v for k, v in test_esb_metrics.items() if k != "gene_table"},
            "profile": {k: v for k, v in test_profile_metrics.items() if k != "gene_table"},
            "derived": test_derived,
            "fragment": {k: v for k, v in test_frag_metrics.items() if k != "group_table"},
        },
    }
    _atomic_write_json(evaldir / "report.json", _nonfinite_to_none(report))
    _write_esb_csvs(evaldir, "dev", {
        "gene_table": dev_esb_metrics["gene_table"],
        "profile": dev_profile_metrics,
    }, tissue_names)
    _write_esb_csvs(evaldir, "test", {
        "gene_table": test_esb_metrics["gene_table"],
        "profile": test_profile_metrics,
    }, tissue_names)
    _write_fragment_csv(evaldir, "dev", dev_frag_metrics)
    _write_fragment_csv(evaldir, "test", test_frag_metrics)
    _write_report_md(evaldir, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="promoter_atlas.measured_evaluate")
    parser.add_argument("command", choices=["select", "test", "summarize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        raise RuntimeError("--batch-size must be positive")
    if int(args.threads) <= 0:
        raise RuntimeError("--threads must be positive")
    if int(args.bootstrap) <= 0:
        raise RuntimeError("--bootstrap must be positive")
    torch.set_num_threads(int(args.threads))
    cfg, root, table_path = _resolve_config(args.config)
    evaldir = _evaldir(args, cfg)
    train_dir = Path(cfg.get("outdir")).resolve() if cfg.get("outdir") else evaldir.parent
    if evaldir.resolve() == train_dir.resolve():
        raise RuntimeError("evaluation outdir must differ from training outdir")
    evaldir.mkdir(parents=True, exist_ok=True)
    if args.command == "select":
        return _cmd_select(args, cfg, root, table_path, evaldir, train_dir)
    if args.command == "test":
        return _cmd_test(args, cfg, root, table_path, evaldir, train_dir)
    if args.command == "summarize":
        return _cmd_summarize(args, cfg, root, table_path, evaldir, train_dir)
    raise RuntimeError("unknown command")


if __name__ == "__main__":
    sys.exit(main())
