# -*- coding: utf-8 -*-
"""检查数据文件、标签、切分计数及源文件指纹。构建顺序见 ORIGINAL_SOURCES.md。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HUMAN_KEY = "human"
ORTHOLOG_KEYS = [
    "pan_troglodytes", "macaca_mulatta", "callithrix_jacchus", "mus_musculus",
    "rattus_norvegicus", "cavia_porcellus", "oryctolagus_cuniculus", "bos_taurus",
    "sus_scrofa", "ovis_aries", "capra_hircus", "equus_caballus",
    "canis_lupus_familiaris", "felis_catus", "pteropus_vampyrus",
    "loxodonta_africana", "monodelphis_domestica", "ornithorhynchus_anatinus",
]
COHORT = "data/breakthrough_20261002/batch52/corrected/cohort_v2"

# 每个资产：path、kind（source/reference/derived）、required、可选 sha256/bytes。
# required=True 的资产一旦缺失、尺寸不符或（深度模式下）SHA 不符，校验即失败。
SOURCES = {
    "mpra_bed": {
        "path": "data/breakthrough_20261002/batch51/"
                "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz",
        "sha256": "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96",
        "bytes": 103144961, "kind": "source", "required": True},
    "grch37_ref": {
        "path": "data/breakthrough_20261002/batch52/reference/GRCh37.primary_assembly.fa.gz",
        "sha256": "0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865",
        "bytes": 869923173, "kind": "reference", "required": True},
    "esb_table": {
        "path": "labels/build/promoter_atlas_v6_input.tsv",
        "sha256": None, "bytes": None, "kind": "derived", "required": True},
    "esb_memmap_human_sym400": {
        "path": "promoters/memmap/promoter_sym400_uint8.memmap",
        "sha256": None, "bytes": None, "kind": "derived", "required": True, "row_len": 400},
    "esb_memmap_human_up2000": {
        "path": "promoters/memmap/promoter_up2000_uint8.memmap",
        "sha256": None, "bytes": None, "kind": "derived", "required": True, "row_len": 2000},
    "frag_codes": {
        "path": COHORT + "/codes_int64.npy",
        "sha256": None, "bytes": None, "kind": "derived", "required": True},
    "frag_endpoints": {
        "path": COHORT + "/endpoints.tsv",
        "sha256": None, "bytes": None, "kind": "derived", "required": True},
    "frag_groups": {
        "path": COHORT + "/groups.tsv",
        "sha256": None, "bytes": None, "kind": "derived", "required": True},
    "frag_manifest": {
        "path": COHORT + "/manifest.json",
        "sha256": None, "bytes": None, "kind": "derived", "required": True},
}
for _k in ORTHOLOG_KEYS:
    SOURCES["esb_memmap_%s_sym400" % _k] = {
        "path": "data/multispecies/memmap/%s_sym400_uint8.memmap" % _k,
        "sha256": None, "bytes": None, "kind": "derived", "required": True, "row_len": 400}
    SOURCES["esb_memmap_%s_up2000" % _k] = {
        "path": "data/multispecies/memmap/%s_up2000_uint8.memmap" % _k,
        "sha256": None, "bytes": None, "kind": "derived", "required": True, "row_len": 2000}

EXPECT = {
    "esb_rows": 1746138, "esb_train": 1533642, "esb_dev": 105364, "esb_test": 107132,
    "esb_species": 19,
    "frag_rows": 761655, "frag_train": 531860, "frag_dev": 112640, "frag_test": 117155,
    "frag_groups": 5332, "frag_len": 600,
    "zstats": {"Score_E": [0.7360959, 0.2327551], "Score_S": [0.8307955, 0.0871514],
               "Score_B": [0.7283229, 0.2314467]},
    "activity_normalizer": {"mean": 1.6329246759414673, "std": 1.3893377780914307},
}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _memmap_rows(root, key, row_len):
    """Rows in the sym400/up2000 memmap for a memmap_key, or None if absent."""
    if key == HUMAN_KEY:
        p = os.path.join(root, "promoters", "memmap",
                         "promoter_%s_uint8.memmap" % ("sym400" if row_len == 400 else "up2000"))
    else:
        p = os.path.join(root, "data", "multispecies", "memmap",
                         "%s_%s_uint8.memmap" % (key, "sym400" if row_len == 400 else "up2000"))
    if not os.path.exists(p):
        return None
    sz = os.path.getsize(p)
    return sz // row_len if sz % row_len == 0 else None


def verify(root, deep=False, verbose=True):
    """Read-only check of a local data root. Returns a JSON-able report.

    ok 为 False 当且仅当任一 required 资产缺失、字节数与记录不符、memmap 行长度不对齐，
    或（deep 模式下）记录过的 SHA256 不符；ESB 计数/物种数不符、或 ESB 行索引越出对应
    memmap、或片段 cohort 证据不符，同样置 False。
    """
    rep = {"data_root": os.path.abspath(root), "checks": [], "failures": [], "ok": True}

    def fail(msg):
        rep["ok"] = False
        rep["failures"].append(msg)

    for name, s in SOURCES.items():
        p = os.path.join(root, s["path"])
        c = {"asset": name, "path": s["path"], "kind": s["kind"], "required": s["required"],
             "exists": os.path.exists(p)}
        if c["exists"]:
            c["bytes"] = os.path.getsize(p)
            if s["bytes"] is not None:
                c["bytes_match"] = c["bytes"] == s["bytes"]
                if not c["bytes_match"]:
                    fail("%s bytes %d != recorded %d" % (name, c["bytes"], s["bytes"]))
            if s.get("row_len"):
                c["row_len"] = s["row_len"]
                c["rows"] = c["bytes"] // s["row_len"]
                c["row_aligned"] = c["bytes"] % s["row_len"] == 0
                if not c["row_aligned"]:
                    fail("%s bytes %d not a multiple of row length %d"
                         % (name, c["bytes"], s["row_len"]))
            if deep and s["sha256"]:
                c["sha256"] = sha256_file(p)
                c["sha256_match"] = c["sha256"] == s["sha256"]
                if not c["sha256_match"]:
                    fail("%s sha256 %s != recorded %s" % (name, c["sha256"], s["sha256"]))
        else:
            c["note"] = "not present; obtain per the header, then re-run --data-root --deep"
            if s["required"]:
                fail("%s missing (required)" % name)
        rep["checks"].append(c)

    _check_esb(root, rep, fail)
    _check_fragments(root, rep, fail)
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


def _check_esb(root, rep, fail):
    esb = os.path.join(root, SOURCES["esb_table"]["path"])
    if not os.path.exists(esb):
        return
    import numpy as np
    import pandas as pd
    cols = ["memmap_key", "split", "r400", "r2000", "Score_E", "Score_S", "Score_B"]
    df = pd.read_csv(esb, sep="\t", usecols=cols, low_memory=False)
    vc = df["split"].value_counts().to_dict()
    got = {"esb_rows": int(len(df)), "esb_species": int(df["memmap_key"].nunique()),
           "esb_train": int(vc.get("train", 0)), "esb_dev": int(vc.get("val", 0)),
           "esb_test": int(vc.get("test", 0))}
    tr = df[df.split == "train"]
    zs = {}
    for c in ("Score_E", "Score_S", "Score_B"):
        v = tr[c].dropna()
        zs[c] = [round(float(v.mean()), 7), round(float(v.std(ddof=0)), 7)]
    rep["esb_observed"] = got
    rep["esb_zstats_train"] = zs
    for k in ("esb_rows", "esb_train", "esb_dev", "esb_test", "esb_species"):
        if got[k] != EXPECT[k]:
            fail("ESB %s expected %d observed %d" % (k, EXPECT[k], got[k]))
    # ESB 行索引必须落在对应 memmap 的行范围内。
    bounds = {}
    for key in sorted(df["memmap_key"].astype(str).unique()):
        sub = df[df.memmap_key.astype(str) == key]
        b = {}
        for row_len, col in ((400, "r400"), (2000, "r2000")):
            n = _memmap_rows(root, key, row_len)
            mx = int(sub[col].max()) if len(sub) else -1
            mn = int(sub[col].min()) if len(sub) else 0
            b[col] = {"max_index": mx, "memmap_rows": n,
                      "in_range": (n is not None and 0 <= mn and mx < n)}
            if n is None or mn < 0 or mx >= n:
                fail("ESB %s %s indices %d..%d out of memmap range %r" % (key, col, mn, mx, n))
        bounds[key] = b
    rep["esb_memmap_bounds"] = bounds


def _check_fragments(root, rep, fail):
    import numpy as np
    codes_p = os.path.join(root, SOURCES["frag_codes"]["path"])
    endp_p = os.path.join(root, SOURCES["frag_endpoints"]["path"])
    grp_p = os.path.join(root, SOURCES["frag_groups"]["path"])
    if os.path.exists(codes_p):
        shape = tuple(np.load(codes_p, mmap_mode="r").shape)
        rep["frag_codes_shape"] = list(shape)
        if len(shape) != 2 or shape[1] != EXPECT["frag_len"]:
            fail("frag codes shape %s expected (n, %d)" % (shape, EXPECT["frag_len"]))
        if shape[0] != EXPECT["frag_rows"]:
            fail("frag codes rows %d expected %d" % (shape[0], EXPECT["frag_rows"]))
    if os.path.exists(endp_p):
        vc, n = {}, 0
        with open(endp_p, encoding="utf-8") as f:
            hdr = f.readline().rstrip("\n").split("\t")
            si = hdr.index("split")
            for ln in f:
                if ln.strip():
                    n += 1
                    s = ln.rstrip("\n").split("\t")[si]
                    vc[s] = vc.get(s, 0) + 1
        rep["frag_endpoints_observed"] = {"rows": n, "by_split": vc}
        for split, key in (("TRAIN", "frag_train"), ("DEV", "frag_dev"), ("TEST", "frag_test")):
            if vc.get(split, 0) != EXPECT[key]:
                fail("fragment %s count %d expected %d" % (split, vc.get(split, 0), EXPECT[key]))
        if n != EXPECT["frag_rows"]:
            fail("frag endpoints rows %d expected %d" % (n, EXPECT["frag_rows"]))
    if os.path.exists(grp_p):
        g = set()
        with open(grp_p, encoding="utf-8") as f:
            f.readline()
            for ln in f:
                if ln.strip():
                    g.add(ln.split("\t", 1)[0])
        rep["frag_groups_observed"] = len(g)
        if len(g) != EXPECT["frag_groups"]:
            fail("frag groups %d expected %d" % (len(g), EXPECT["frag_groups"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--deep", action="store_true", help="also hash source files (slow)")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    a = ap.parse_args(argv)
    rep = verify(a.data_root, deep=a.deep, verbose=a.out is None)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print("wrote", a.out, "ok=%s" % rep.get("ok"))
    return 0 if rep.get("ok", True) else 2


if __name__ == "__main__":
    sys.exit(main())
