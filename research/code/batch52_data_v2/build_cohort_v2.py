# -*- coding: utf-8 -*-
"""batch52 data-owner corrected cohort builder (target = author Log2RPM_HEK293).

Reuses the LABEL-FREE grouping/split already computed in
data/breakthrough_20261002/batch52/cohort/ (members.tsv/groups.tsv/pairs + endpoint_seqs.tsv,
extracted from the official Ensembl GRCh37 primary assembly == hg19). Only the per-endpoint
TARGET is rebuilt from the source BED column Log2RPM_HEK293 (index 7); the old Log2Norm labels
are never copied. Emits the batch53_model_core cohort contract.

ZERO fits. No GPU.
"""
from __future__ import annotations
import gzip, hashlib, json, os, time
import numpy as np

ROOT = r"E:/promoter_atlas"
B51 = os.path.join(ROOT, "data/breakthrough_20261002/batch51")
OLD = os.path.join(ROOT, "data/breakthrough_20261002/batch52")
BED = os.path.join(B51, "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz")
OUT = os.path.join(OLD, "corrected", "cohort_v2")
TARGET_COLUMN = "Log2RPM_HEK293"
SOURCE_SHA256 = "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96"
REFERENCE_SHA256 = "0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865"
MAX_LEN = 600
MINL, MAXL = 100, 600
SPLITS = ("TRAIN", "DEV", "TEST")
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def read_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return header, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    mh, mrows = read_tsv(os.path.join(OLD, "cohort", "members.tsv"))
    ci, gi, si, li = (mh.index(c) for c in ("coord_id", "union_id", "split", "length"))
    coord_ids, group_ids, splits, lengths = [], [], [], []
    for r in mrows:
        coord_ids.append(r[ci]); group_ids.append(r[gi]); splits.append(r[si]); lengths.append(int(r[li]))
    log("members", len(coord_ids))
    assert len(set(coord_ids)) == len(coord_ids), "coord_id not unique in members.tsv"

    sh, srows = read_tsv(os.path.join(OLD, "cohort", "endpoint_seqs.tsv"))
    sci, ssi = sh.index("coord_id"), sh.index("seq")
    seqs = {r[sci]: r[ssi] for r in srows}
    log("seqs", len(seqs))

    tgt = {}
    conflict = set()
    with gzip.open(BED, "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        assert head.index(TARGET_COLUMN) == 7, head
        ti = head.index(TARGET_COLUMN)
        nrow = 0
        for line in f:
            a = line.rstrip("\n").split("\t"); nrow += 1
            L = int(a[3]) - int(a[2])
            if not (MINL <= L <= MAXL):
                continue
            cid = "%s:%s-%s:%s" % (a[1], a[2], a[3], a[4])
            try:
                v = float(a[ti])
            except ValueError:
                continue
            if cid in tgt and tgt[cid] != v:
                conflict.add(cid)
            tgt[cid] = v
    log("source rows", nrow, "target_keyed", len(tgt), "conflict", len(conflict))

    missing_seq = [c for c in coord_ids if c not in seqs]
    missing_tgt = [c for c in coord_ids if c not in tgt or c in conflict]
    join = 1.0 - (len(missing_seq) + len(missing_tgt)) / max(len(coord_ids), 1)
    log("missing_seq", len(missing_seq), "missing_target", len(missing_tgt), "join", round(join, 5))
    assert join >= 0.99, "join coverage below 99%"
    assert not missing_seq and not missing_tgt

    with open(os.path.join(OUT, "endpoints.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("coord_id\tunion_id\tsplit\tlength\t%s\n" % TARGET_COLUMN)
        for i in range(len(coord_ids)):
            f.write("%s\t%s\t%s\t%d\t%.9g\n" % (coord_ids[i], group_ids[i], splits[i], lengths[i], tgt[coord_ids[i]]))
    seen_g = {}
    for i in range(len(coord_ids)):
        g = group_ids[i]
        if g in seen_g:
            assert seen_g[g] == splits[i], "group spans splits: %s" % g
        seen_g[g] = splits[i]
    with open(os.path.join(OUT, "groups.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("union_id\tsplit\n")
        for g, s in seen_g.items():
            f.write("%s\t%s\n" % (g, s))
    log("groups", len(seen_g))

    with open(os.path.join(OUT, "sequences.fa"), "w", encoding="utf-8", newline="\n") as f:
        for i in range(len(coord_ids)):
            f.write(">%s\n%s\n" % (coord_ids[i], seqs[coord_ids[i]]))
    log("sequences.fa written")

    cset = set(coord_ids)
    pair_rows = []
    ph, prows = read_tsv(os.path.join(OLD, "cohort", "pairs_train_dev.tsv"))
    pui, psp, psc, psl, plc, pll = (ph.index(c) for c in
        ("union_id", "split", "short_coord_id", "short_len", "long_coord_id", "long_len"))
    for r in prows:
        pair_rows.append((r[pui], r[psp], r[psc], r[plc], int(r[psl]), int(r[pll])))
    sh2, s2rows = read_tsv(os.path.join(OLD, "SEALED_test_labels.tsv"))
    qsc, qsl, qlc, qll = (sh2.index(c) for c in ("short_coord_id", "short_len", "long_coord_id", "long_len"))
    n_test_pairs = 0
    for r in s2rows:
        pair_rows.append((None, "TEST", r[qsc], r[qlc], int(r[qsl]), int(r[qll])))
        n_test_pairs += 1
    cid2g = {c: g for c, g in zip(coord_ids, group_ids)}
    fixed = []
    for (u, sp, sc, lc, sl, ll) in pair_rows:
        if u is None:
            u = cid2g[sc]
        fixed.append((u, sp, sc, lc, sl, ll))
    pair_rows = fixed
    with open(os.path.join(OUT, "pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("union_id\tsplit\tshort_coord_id\tlong_coord_id\tshort_len\tlong_len\n")
        for (u, sp, sc, lc, sl, ll) in pair_rows:
            assert sc in cset and lc in cset, (sc, lc)
            assert sl < ll, (sc, lc)
            f.write("%s\t%s\t%s\t%s\t%d\t%d\n" % (u, sp, sc, lc, sl, ll))
    log("pairs written", len(pair_rows), "test_pairs", n_test_pairs)

    N = len(coord_ids)
    codes = np.lib.format.open_memmap(os.path.join(OUT, "codes_int64.npy"), mode="w+",
                                      dtype=np.int64, shape=(N, MAX_LEN))
    codes[:] = 4
    for k in range(N):
        s = seqs[coord_ids[k]]
        assert len(s) == lengths[k], (coord_ids[k], len(s), lengths[k])
        arr = np.fromiter((_BASE.get(ch, 4) for ch in s.upper()), dtype=np.int64, count=len(s))
        codes[k, :len(s)] = arr
        if (k + 1) % 200000 == 0:
            log("codes", k + 1)
    codes.flush()

    h = hashlib.sha256()
    h.update(b"b53-cohort-v1|")
    for c, g, s in zip(coord_ids, group_ids, splits):
        h.update(("%s\t%s\t%s\n" % (c, g, s)).encode("utf-8"))
    for a in range(0, N, 50000):
        h.update(np.ascontiguousarray(codes[a:a + 50000]).tobytes())
    tarr = np.array([tgt[c] for c in coord_ids], dtype=np.float32)
    h.update(tarr.tobytes())
    idx = {c: k for k, c in enumerate(coord_ids)}
    for s in SPLITS:
        rows = [(idx[sc], idx[lc]) for (u, sp, sc, lc, sl, ll) in pair_rows if sp == s]
        arr = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 2), dtype=np.int64)
        h.update(arr.tobytes())
    datahash = h.hexdigest()
    with open(os.path.join(OUT, "datahash.txt"), "w", encoding="utf-8") as f:
        f.write(datahash + "\n")
    log("datahash", datahash)

    json.dump({
        "assembly": "hg19",
        "assembly_alias": "GRCh37 (Ensembl primary assembly)",
        "source": "Ensembl FTP GRCh37 primary assembly DNA",
        "url": "https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz",
        "reference_file": "data/breakthrough_20261002/batch52/reference/GRCh37.primary_assembly.fa.gz",
        "reference_sha256": REFERENCE_SHA256,
        "reference_bytes": 869923173,
        "coordinate_contract": "BED 0-based half-open hg19; minus strand reverse-complemented; 3-prime end = end if plus else start",
        "proxy_not_donor_clone": True,
        "proxy_note": "reference-encoded sequence is a stated proxy for donor HG02601, NOT the clone own sequence and NOT local 293FT validation; HEK293 identity preserved",
        "sequence_origin": "reused label-free extraction data/breakthrough_20261002/batch52/cohort/endpoint_seqs.tsv",
    }, open(os.path.join(OUT, "reference_provenance.json"), "w", encoding="utf-8"), indent=2)
    json.dump({"batch": "batch52", "target_column": TARGET_COLUMN,
               "source_bed": BED, "source_bytes": os.path.getsize(BED),
               "source_sha256": SOURCE_SHA256,
               "n_endpoints": N, "n_groups": len(seen_g), "n_pairs": len(pair_rows),
               "salt_split": "b52-full-hek-v1|", "datahash": datahash},
              open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=2)

    cnt = {s: sum(1 for x in splits if x == s) for s in SPLITS}
    pcnt = {s: sum(1 for p in pair_rows if p[1] == s) for s in SPLITS}
    gcnt = {s: sum(1 for g, ss in seen_g.items() if ss == s) for s in SPLITS}
    rep = {"code": "build_cohort_v2.py", "target_column": TARGET_COLUMN,
           "source_sha256": SOURCE_SHA256, "reference_sha256": REFERENCE_SHA256,
           "n_endpoints": N, "endpoint_counts": cnt, "n_pairs": len(pair_rows), "pair_counts": pcnt,
           "group_counts": gcnt, "n_groups": len(seen_g), "join_coverage": round(join, 6),
           "excluded_missing_seq": len(missing_seq), "excluded_target": len(missing_tgt),
           "conflicting_target_labels": len(conflict), "datahash": datahash,
           "n_target_zero": int((tarr == 0).sum()), "runtime_s": round(time.time() - T0, 1)}
    json.dump(rep, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"), indent=2)
    log("BUILD DONE", json.dumps({"endpoints": cnt, "pairs": pcnt, "groups": gcnt}))


if __name__ == "__main__":
    main()
