# -*- coding: utf-8 -*-
"""从项目数据树读取双窗口 E/S/B 记录与片段记录，保留标签掩码、切分与原始行号。"""
from __future__ import annotations

import os

import numpy as np

ESB_COLS = ["Score_E", "Score_S", "Score_B"]
ESB_USECOLS = ["record_id", "species", "species_id", "gene_id", "split", "memmap_key",
               "r2000", "r400", "Score_E", "Score_S", "Score_B"]
CANON_SPLIT = {"train": "TRAIN", "val": "DEV", "test": "TEST"}
FRAG_MIN_LEN = 100
FRAG_MAX_LEN = 600
FRAG_PAD = 4


class ESBStream:
    """Dual-window E/S/B records backed by read-only memmaps."""

    def __init__(self, root):
        import pandas as pd
        self.root = root
        self.table = os.path.join(root, "labels", "build", "promoter_atlas_v6_input.tsv")
        df = pd.read_csv(self.table, sep="\t", low_memory=False, usecols=ESB_USECOLS)
        keep = df["split"].isin(list(CANON_SPLIT)).to_numpy()
        self.excluded_unknown_split = int((~keep).sum())
        self.df = df.iloc[np.nonzero(keep)[0]].reset_index(drop=True)
        self.n = len(self.df)
        self.stable_id = self.df["record_id"].astype(str).to_numpy()
        self.gene_id = self.df["gene_id"].astype(str).to_numpy()
        self.split = np.array([CANON_SPLIT[s] for s in self.df["split"].astype(str)], dtype=object)
        self.keys = self.df["memmap_key"].astype(str).to_numpy()
        self.r2000 = self.df["r2000"].to_numpy(dtype=np.int64)
        self.r400 = self.df["r400"].to_numpy(dtype=np.int64)
        self.species = self.df["species_id"].to_numpy(dtype=np.int64)
        self.y_raw = np.full((self.n, 3), np.nan, dtype=np.float32)
        self.mask = np.zeros((self.n, 3), dtype=np.float32)
        for j, c in enumerate(ESB_COLS):
            v = self.df[c].to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            self.y_raw[ok, j] = v[ok].astype(np.float32)
            self.mask[ok, j] = 1.0
        self.source_paths = {"table": self.table}
        self._mm = {}
        sp_dir = os.path.join(root, "data", "multispecies", "memmap")
        for k in np.unique(self.keys):
            if k == "human":
                p2 = os.path.join(root, "promoters", "memmap", "promoter_up2000_uint8.memmap")
                p4 = os.path.join(root, "promoters", "memmap", "promoter_sym400_uint8.memmap")
            else:
                p2 = os.path.join(sp_dir, "%s_up2000_uint8.memmap" % k)
                p4 = os.path.join(sp_dir, "%s_sym400_uint8.memmap" % k)
            self.source_paths[k + "_2000"] = p2
            self.source_paths[k + "_400"] = p4
            for size, path, rows in ((2000, p2, self.r2000), (400, p4, self.r400)):
                if os.path.getsize(path) % size:
                    raise ValueError("memmap byte length is not row-aligned: " + path)
                selected = rows[self.keys == k]
                if (selected < 0).any() or (selected >= os.path.getsize(path) // size).any():
                    raise ValueError("ESB memmap row out of range: " + k)
            self._mm[k] = {2000: np.memmap(p2, dtype="uint8", mode="r",
                                           shape=(os.path.getsize(p2) // 2000, 2000)),
                           400: np.memmap(p4, dtype="uint8", mode="r",
                                          shape=(os.path.getsize(p4) // 400, 400))}

    def __len__(self):
        return self.n

    def split_indices(self, split):
        return np.nonzero(self.split == split)[0]

    def record(self, i):
        k = self.keys[int(i)]
        x400 = np.asarray(self._mm[k][400][int(self.r400[i])], dtype=np.uint8)
        x2000 = np.asarray(self._mm[k][2000][int(self.r2000[i])], dtype=np.uint8)
        return {"local_codes400": x400, "context_codes2000": x2000,
                "species": int(self.species[i]), "targets_ESB": self.y_raw[i].copy(),
                "finite_mask": self.mask[i].copy(), "stableID": self.stable_id[i],
                "gene_id": self.gene_id[i], "split": self.split[i]}


class FragmentStream:
    """Variable-length measured MPRA fragment records (100..600 bp, no fabricated context)."""

    def __init__(self, root):
        self.root = root
        cohort = os.path.join(root, "data", "breakthrough_20261002", "batch52",
                              "corrected", "cohort_v2")
        self.source_paths = {name: os.path.join(cohort, name) for name in
                             ("codes_int64.npy", "endpoints.tsv", "groups.tsv", "manifest.json")}
        self.codes = np.load(self.source_paths["codes_int64.npy"], mmap_mode="r")
        hdr, rows = self._read_tsv(os.path.join(cohort, "endpoints.tsv"))
        ci, gi, si, li, ti = (hdr.index(c) for c in
                              ("coord_id", "union_id", "split", "length", "Log2RPM_HEK293"))
        coord, group, split, length, yraw, unknown = [], [], [], [], [], 0
        if self.codes.ndim != 2 or self.codes.shape[0] != len(rows) or self.codes.shape[1] < FRAG_MAX_LEN:
            raise ValueError("fragment codes and endpoint rows do not align")
        source_rows = []
        for source_i, r in enumerate(rows):
            if r[si] not in ("TRAIN", "DEV", "TEST"):
                unknown += 1
                continue
            L = int(r[li])
            if not FRAG_MIN_LEN <= L <= FRAG_MAX_LEN:
                continue
            if not np.isfinite(float(r[ti])):
                raise ValueError("non-finite fragment activity: " + r[ci])
            source_rows.append(source_i)
            coord.append(r[ci]); group.append(r[gi]); split.append(r[si])
            length.append(L); yraw.append(float(r[ti]))
        self.excluded_unknown_split = unknown
        self.source_rows = np.asarray(source_rows, dtype=np.int64)
        self.stable_id = np.array(coord, dtype=object)
        self.group_id = np.array(group, dtype=object)
        self.split = np.array(split, dtype=object)
        self.length = np.array(length, dtype=np.int64)
        self.y_raw = np.array(yraw, dtype=np.float32)
        self.n = len(self.stable_id)
        self.groupweight = np.ones(self.n, dtype=np.float32)
        for s in ("TRAIN", "DEV", "TEST"):
            sel = self.split == s
            if not sel.any():
                continue
            gsz = {}
            for g in self.group_id[sel]:
                gsz[g] = gsz.get(g, 0) + 1
            w = np.array([1.0 / gsz[g] for g in self.group_id[sel]], dtype=np.float32)
            self.groupweight[sel] = w / w.mean()
        tr = self.split == "TRAIN"
        self.train_mean = float(self.y_raw[tr].mean()) if tr.any() else None
        self.train_std = float(self.y_raw[tr].std() + 1e-6) if tr.any() else None

    @staticmethod
    def _read_tsv(path):
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        return header, rows

    def __len__(self):
        return self.n

    def split_indices(self, split):
        return np.nonzero(self.split == split)[0]

    def codes_of(self, i):
        L = int(self.length[i])
        return np.asarray(self.codes[int(self.source_rows[i]), :L], dtype=np.uint8)

    def record(self, i):
        return {"codes": self.codes_of(i), "length": int(self.length[i]),
                "y_Log2RPM": float(self.y_raw[i]), "groupweight": float(self.groupweight[i]),
                "stableID": self.stable_id[i], "split": self.split[i]}
