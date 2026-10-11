# -*- coding: utf-8 -*-
"""Portable cohort adapter with HARD validation (batch52 data owner calls this).

The b52 data owner builds the cohort; this module only CONSUMES a documented, frozen
cohort and validates it before training. It does NOT re-derive sequences or labels and
never touches TEST labels for training.

Documented input contract (directory cohort_dir):
  endpoints.tsv : TSV, header MUST contain columns
      coord_id, union_id, split, length, <TARGET_COLUMN>
      (optionally chr/start/end/strand). TARGET_COLUMN is exactly Log2RPM_HEK293.
  groups.tsv    : TSV with columns union_id, split  (one row per union group).
  pairs.tsv     : TSV with columns union_id, split, short_coord_id, long_coord_id
      (optionally three_prime_end / short_len / long_len for cross-checks).
  sequences.fa  : FASTA, one record per coord_id (ACGT; other bases -> ambiguous code 4).
     OR sequences.npz with arrays coord_ids (str) and codes (N x <=600 uint8 base codes).
  reference_provenance.json : per-assembly proxy contract.
  manifest.json (optional)  : cross-checked source_sha256.

Validation performed (all failures raise AdapterError):
  * exact length: seq length == declared length, in [100,600], no nulls, >600 rejected;
  * IDs: FASTA/NPZ ids set-equal to endpoints coord_id set; pair ids exist;
  * group disjoint: each coord_id in exactly one union_id; split per group constant and
    TRAIN/DEV/TEST groups pairwise disjoint; pair endpoints share the pair union_id;
  * pair geometry: short_len < long_len (when columns present);
  * reference proxy: provenance assembly == hg19 and proxy flag present;
  * target column exact-match Log2RPM_HEK293; source sha exact-match.
"""
from __future__ import annotations
import hashlib, json, os

import numpy as np
import torch

from spec import (TARGET_COLUMN, SOURCE_SHA256, MAX_LEN, MIN_ELIGIBLE_LEN, MAX_ELIGIBLE_LEN,
                  assert_target, assert_source_sha)

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
SPLITS = ("TRAIN", "DEV", "TEST")


class AdapterError(Exception):
    pass


def _read_tsv(path):
    """D12: validate that every row has exactly as many fields as the header."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for ln in f:
            if not ln.strip():
                continue
            fields = ln.rstrip("\n").split("\t")
            if len(fields) != len(header):
                raise AdapterError(
                    "short/ragged row in %s: %d fields, expected %d" % (path, len(fields), len(header)))
            rows.append(fields)
    return header, rows


def _seq_to_codes(seq):
    return np.array([_BASE.get(b, 4) for b in seq.upper()], dtype=np.uint8)


def _load_sequences(cohort_dir, expect_ids, expect_lengths=None):
    fa = os.path.join(cohort_dir, "sequences.fa")
    npz = os.path.join(cohort_dir, "sequences.npz")
    len_by_id = None
    if expect_lengths is not None:
        len_by_id = {cid: int(L) for cid, L in zip(expect_ids, expect_lengths)}
    seqs = {}
    if os.path.exists(fa):
        name, buf = None, []
        with open(fa, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if ln.startswith(">"):
                    if name is not None:
                        seqs[name] = "".join(buf)
                    name, buf = ln[1:].strip(), []
                elif ln:
                    buf.append(ln.strip())
        if name is not None:
            seqs[name] = "".join(buf)
    elif os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        # D12: decode byte ids so set-equality with endpoint ids works (str(b'x') would not).
        ids = []
        for s in list(z["coord_ids"]):
            if isinstance(s, (bytes, bytearray, np.bytes_)):
                ids.append(bytes(s).decode("utf-8"))
            else:
                ids.append(str(s))
        codes = np.asarray(z["codes"])
        if codes.ndim != 2 or codes.shape[0] != len(ids):
            raise AdapterError("sequences.npz codes shape %r incompatible with %d ids"
                               % (tuple(codes.shape), len(ids)))
        for i, cid in enumerate(ids):
            row = np.asarray(codes[i], dtype=np.uint8)
            if len_by_id is not None:
                # NPZ rows are rectangular; slice each row to its DECLARED length.
                L = len_by_id[cid]
                if L > row.size:
                    raise AdapterError("sequences.npz row %s width %d < declared %d" % (cid, row.size, L))
                row = row[:L]
            seqs[cid] = row
    else:
        raise AdapterError("no sequences.fa or sequences.npz in %s" % cohort_dir)
    got = set(seqs)
    if got != set(expect_ids):
        missing = sorted(set(expect_ids) - got)[:5]
        extra = sorted(got - set(expect_ids))[:5]
        raise AdapterError("sequence IDs != endpoint IDs (missing=%s extra=%s)" % (missing, extra))
    return seqs


class Cohort:
    """Validated, frozen cohort. Codes are int64 (N, MAX_LEN) with pad/ambiguous code 4."""
    def __init__(self, coord_ids, group_ids, splits, codes, lengths, targets, pairs, datahash,
                 target_column, source_sha256):
        self.coord_ids = coord_ids
        self.group_ids = group_ids
        self.splits = splits
        self.codes = codes
        self.lengths = lengths
        self.targets = targets
        self.pairs = pairs
        self.datahash = datahash
        self.target_column = target_column
        self.source_sha256 = source_sha256
        self._idx = {c: i for i, c in enumerate(coord_ids)}
        self._groups = {}
        for i, g in enumerate(group_ids):
            self._groups.setdefault(g, []).append(i)

    def split_indices(self, split):
        return torch.tensor([i for i, s in enumerate(self.splits) if s == split], dtype=torch.long)

    def activity_weights(self, split="TRAIN"):
        w = torch.ones(len(self.coord_ids), dtype=torch.float64)
        for g, idxs in self._groups.items():
            w[idxs] = 1.0 / len(idxs)
        sel = self.split_indices(split)
        ws = w[sel]
        ws = ws / ws.mean().clamp(min=1e-12)
        return sel, ws

    def pair_weights(self, split="TRAIN"):
        p = self.pairs.get(split)
        if p is None or p.numel() == 0:
            return p, torch.zeros(0, dtype=torch.float64)
        gsz = {}
        for r in p.tolist():
            g = self.group_ids[r[0]]
            gsz[g] = gsz.get(g, 0) + 1
        w = torch.tensor([1.0 / gsz[self.group_ids[r[0]]] for r in p.tolist()], dtype=torch.float64)
        w = w / w.mean().clamp(min=1e-12)
        return p, w


def load_cohort(cohort_dir, source_sha256=SOURCE_SHA256, validate_only=False, require_groups=True,
                allow_ambiguous=False):
    ap = os.path.join(cohort_dir, "endpoints.tsv")
    if not os.path.exists(ap):
        raise AdapterError("missing endpoints.tsv in %s" % cohort_dir)
    header, rows = _read_tsv(ap)
    for col in ("coord_id", "union_id", "split", "length"):
        if col not in header:
            raise AdapterError("endpoints.tsv missing column %r" % col)
    if TARGET_COLUMN not in header:
        raise AdapterError("endpoints.tsv missing target column %r (got %s)" % (TARGET_COLUMN, header))
    assert_target(TARGET_COLUMN)
    if source_sha256 is not None:
        assert_source_sha(source_sha256)
    ci, gi, si, li, ti = (header.index(c) for c in ("coord_id", "union_id", "split", "length", TARGET_COLUMN))

    coord_ids, group_ids, splits, lengths, targets = [], [], [], [], []
    seen = set()
    for r in rows:
        cid = r[ci]
        if cid in seen:
            raise AdapterError("duplicate coord_id %r" % cid)
        seen.add(cid)
        if r[si] not in SPLITS:
            raise AdapterError("bad split %r" % r[si])
        L = int(r[li])
        if L < MIN_ELIGIBLE_LEN or L > MAX_ELIGIBLE_LEN:
            raise AdapterError("length %d out of [%d,%d] for %s" % (L, MIN_ELIGIBLE_LEN, MAX_ELIGIBLE_LEN, cid))
        try:
            y = float(r[ti])
        except ValueError:
            raise AdapterError("null/non-numeric %s for %s" % (TARGET_COLUMN, cid))
        coord_ids.append(cid); group_ids.append(r[gi]); splits.append(r[si]); lengths.append(L); targets.append(y)
    if not coord_ids:
        raise AdapterError("empty endpoints.tsv")

    gsplit = {}
    for g, s in zip(group_ids, splits):
        if g in gsplit and gsplit[g] != s:
            raise AdapterError("group %r spans multiple splits" % g)
        gsplit[g] = s
    split_groups = {s: set() for s in SPLITS}
    for g, s in gsplit.items():
        split_groups[s].add(g)
    if (split_groups["TRAIN"] & split_groups["DEV"]) or (split_groups["TRAIN"] & split_groups["TEST"]) \
       or (split_groups["DEV"] & split_groups["TEST"]):
        raise AdapterError("split groups are not disjoint")

    if require_groups:
        gp = os.path.join(cohort_dir, "groups.tsv")
        if not os.path.exists(gp):
            raise AdapterError("missing groups.tsv")
        gh, gr = _read_tsv(gp)
        gcol = gh.index("union_id")
        listed = {r[gcol] for r in gr}
        if listed != set(gsplit):
            raise AdapterError("groups.tsv union_ids != endpoints union_ids")

    rp = os.path.join(cohort_dir, "reference_provenance.json")
    if not os.path.exists(rp):
        raise AdapterError("missing reference_provenance.json (reference-proxy contract)")
    prov = json.load(open(rp, "r", encoding="utf-8"))
    if str(prov.get("assembly", "")).lower() != "hg19":
        raise AdapterError("reference provenance assembly must be hg19, got %r" % prov.get("assembly"))
    if not prov.get("proxy_not_donor_clone", False):
        raise AdapterError("reference must be declared a proxy, not donor clone")

    mp = os.path.join(cohort_dir, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp, "r", encoding="utf-8"))
        if source_sha256 is not None and "source_sha256" in man and man["source_sha256"].lower() != source_sha256:
            raise AdapterError("manifest source_sha256 mismatch")

    seqs = _load_sequences(cohort_dir, coord_ids, list(lengths))
    codes = np.full((len(coord_ids), MAX_LEN), 4, dtype=np.int64)
    declared_lengths = list(lengths)
    valid_lengths = []
    ambiguous = []
    for i, cid in enumerate(coord_ids):
        s = seqs[cid]
        arr = s.astype(np.int64) if isinstance(s, np.ndarray) else _seq_to_codes(s).astype(np.int64)
        if arr.size > MAX_LEN:
            raise AdapterError("sequence %s length %d > %d (unsupported, no windowing)" % (cid, arr.size, MAX_LEN))
        if arr.size != declared_lengths[i]:
            raise AdapterError("sequence/declared length mismatch for %s: %d != %d"
                               % (cid, arr.size, declared_lengths[i]))
        codes[i, :arr.size] = arr
        # D12: length used for loglen/GC features = count of VALID bases, consistent with the
        # model's mask.sum pooling. Ambiguous (N) bases are excluded consistently or rejected.
        vlen = int((arr < 4).sum())
        if vlen != arr.size:
            ambiguous.append((cid, arr.size - vlen))
        if vlen < MIN_ELIGIBLE_LEN:
            raise AdapterError("sequence %s has only %d valid bases < %d" % (cid, vlen, MIN_ELIGIBLE_LEN))
        valid_lengths.append(vlen)
    if ambiguous and not allow_ambiguous:
        raise AdapterError(
            "ambiguous base(s) in %d sequence(s) (e.g. %r): declared length != valid length. "
            "Pass allow_ambiguous=True to use VALID-base length consistently for loglen/GC."
            % (len(ambiguous), ambiguous[:3]))
    lengths = valid_lengths

    pp = os.path.join(cohort_dir, "pairs.tsv")
    pairs = {s: [] for s in SPLITS}
    if os.path.exists(pp):
        ph, pr = _read_tsv(pp)
        for col in ("union_id", "split", "short_coord_id", "long_coord_id"):
            if col not in ph:
                raise AdapterError("pairs.tsv missing column %r" % col)
        ui, sp, sc, lc = (ph.index(c) for c in ("union_id", "split", "short_coord_id", "long_coord_id"))
        hlen = ph.index("short_len") if "short_len" in ph else None
        llen = ph.index("long_len") if "long_len" in ph else None
        idx = {c: i for i, c in enumerate(coord_ids)}
        for r in pr:
            if r[sc] not in idx or r[lc] not in idx:
                raise AdapterError("pair references unknown coord_id: %s / %s" % (r[sc], r[lc]))
            i, j = idx[r[sc]], idx[r[lc]]
            if group_ids[i] != r[ui] or group_ids[j] != r[ui]:
                raise AdapterError("pair %s/%s not in group %s" % (r[sc], r[lc], r[ui]))
            if splits[i] != r[sp] or splits[j] != r[sp]:
                raise AdapterError("pair split mismatch for %s/%s" % (r[sc], r[lc]))
            if hlen is not None and llen is not None and int(r[hlen]) >= int(r[llen]):
                raise AdapterError("pair short_len >= long_len for %s/%s" % (r[sc], r[lc]))
            pairs[r[sp]].append((i, j))
    pair_t = {s: (torch.tensor(v, dtype=torch.long) if v else torch.zeros((0, 2), dtype=torch.long))
              for s, v in pairs.items()}

    targets_t = torch.tensor(targets, dtype=torch.float32)
    codes_t = torch.from_numpy(codes)
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    datahash = _content_hash(coord_ids, group_ids, splits, codes_t, targets_t, pair_t)

    cohort = Cohort(coord_ids, group_ids, splits, codes_t, lengths_t, targets_t, pair_t, datahash,
                    TARGET_COLUMN, source_sha256)
    if validate_only:
        return {"ok": True, "n_endpoints": len(coord_ids), "n_groups": len(gsplit),
                "counts": {s: sum(1 for x in splits if x == s) for s in SPLITS},
                "pairs": {s: int(pair_t[s].shape[0]) for s in SPLITS},
                "group_counts": {s: len(split_groups[s]) for s in SPLITS}, "datahash": datahash}
    return cohort


def _content_hash(coord_ids, group_ids, splits, codes_t, targets_t, pair_t):
    h = hashlib.sha256()
    h.update(b"b53-cohort-v1|")
    for cid, g, s in zip(coord_ids, group_ids, splits):
        h.update(("%s\t%s\t%s\n" % (cid, g, s)).encode("utf-8"))
    h.update(codes_t.numpy().tobytes())
    h.update(np.asarray(targets_t.numpy(), dtype=np.float32).tobytes())
    for s in SPLITS:
        h.update(pair_t[s].numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort_dir")
    ap.add_argument("--source-sha", default=SOURCE_SHA256)
    a = ap.parse_args()
    print(json.dumps(load_cohort(a.cohort_dir, a.source_sha, validate_only=True), ensure_ascii=False, indent=2))
