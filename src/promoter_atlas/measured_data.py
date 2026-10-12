from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .layout import project_path
from .joint_data import FragmentStream

__all__ = ["MeasuredESBStream", "FragmentStream"]

OMICS_COLS = [
    "cage_tss", "procap_tss", "h3k4me3", "h3k27ac", "h3k27me3",
    "h3k4me1", "h3k36me3", "dnase", "atac",
]
ESB_COLS = ["Score_E", "Score_S", "Score_B"]
REQUIRED = ["record_id", "species", "species_id", "gene_id", "split", "memmap_key", "r2000", "r400"]
OPTIONAL = ["label_mask"] + ESB_COLS + ["ortholog_E", "tf_row"] + OMICS_COLS

_SPLIT_MAP = {"train": "TRAIN", "val": "DEV", "dev": "DEV", "test": "TEST"}
_HUMAN_ALIASES = {"human", "homo_sapiens"}


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _strip_gene(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.\d+$", "", regex=True)


def _canon_split(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(_SPLIT_MAP)


def _int_strict(s: pd.Series, name: str) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    bad = ~np.isfinite(v) | (v < 0) | (v != np.floor(v))
    if bad.any():
        raise ValueError(f"{name}: non-integral/nonfinite/negative values")
    return v.astype(np.int64)


class MeasuredESBStream:
    def __init__(
        self,
        root: str,
        table_path: Optional[str] = None,
        ortholog_path: Optional[str] = None,
        tf_path: Optional[str] = None,
        profile_path: Optional[str] = None,
    ) -> None:
        self.root = str(root)
        self.table_path = table_path or project_path(root, "labels/build/promoter_atlas_v6_6_input.tsv")
        self.ortholog_path = ortholog_path or project_path(root, "features/multispecies/ortholog_human_to_species.tsv")
        self.tf_path = tf_path or project_path(root, "features/omics/promoter_chip_tf_matrix_v1_uint8.memmap")
        self.profile_path = profile_path or project_path(root, "labels/build/gene_expression_table.tsv")
        self.core_path = project_path(root, "labels/releases/promoter_split_core.tsv")
        self._hash_cache: Dict[str, Dict[str, Any]] = {}
        self._profile_hash: Optional[str] = None
        self._blocked_human_genes: set = set()
        if os.path.exists(self.profile_path):
            self._profile_hash = _sha256(self.profile_path)
        self._build()

    # ---------------- A: read table ----------------
    def _read_table(self) -> pd.DataFrame:
        header = pd.read_csv(self.table_path, sep="\t", nrows=0)
        cols = set(header.columns)
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        use = [c for c in REQUIRED + OPTIONAL if c in cols]
        df = pd.read_csv(self.table_path, sep="\t", usecols=use, low_memory=False)
        df["_source_row"] = np.arange(len(df), dtype=np.int64)
        self._source_counts = df["split"].astype(str).str.strip().str.lower().value_counts().to_dict()
        self._source_table_rows = len(df)

        # check missing required metadata before astype str
        for col in ["record_id", "species", "gene_id", "split", "memmap_key"]:
            if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
                raise ValueError(f"missing required metadata in column {col}")
        sp = df["species"].astype(str)
        sp_norm = sp.str.strip().str.lower().map(lambda x: "human" if x in _HUMAN_ALIASES else x)
        df["species_norm"] = sp_norm
        df["gene_base"] = _strip_gene(df["gene_id"])
        df["split_canon"] = _canon_split(df["split"])

        unknown = df["split_canon"].isna()
        self._excluded_unknown_split = int(unknown.sum())
        df = df.loc[~unknown].copy()

        # species_id strict
        df["species_id"] = _int_strict(df["species_id"], "species_id")
        df["r2000"] = _int_strict(df["r2000"], "r2000")
        df["r400"] = _int_strict(df["r400"], "r400")

        # bijective species_norm <-> species_id
        pairs = df[["species_norm", "species_id"]].drop_duplicates()
        if pairs["species_norm"].duplicated().any() or pairs["species_id"].duplicated().any():
            raise ValueError("species_norm/species_id not bijective")
        human_ids = pairs.loc[pairs["species_norm"] == "human", "species_id"].tolist()
        if human_ids != [0]:
            raise ValueError("human species_id must be 0")
        self._species_to_id = dict(zip(pairs["species_norm"], pairs["species_id"].astype(int)))

        # one split per (species_norm, gene_base)
        g = df.groupby(["species_norm", "gene_base"], sort=False)["split_canon"].nunique()
        if (g > 1).any():
            raise ValueError("gene has multiple splits within species")

        # label_mask
        if "label_mask" in df.columns:
            lm = pd.to_numeric(df["label_mask"], errors="raise").to_numpy(dtype=np.float64)
            df["_label_mask"] = np.where(np.isfinite(lm) & (lm == 1), 1, 0).astype(np.int8)
        else:
            df["_label_mask"] = np.int8(1)
        return df

    # ---------------- B: core + ortholog ----------------
    def _read_core(self) -> pd.DataFrame:
        header = pd.read_csv(self.core_path, sep="\t", nrows=0)
        gcol = "gene_id_base" if "gene_id_base" in header.columns else "gene_id"
        if gcol not in header.columns or "split" not in header.columns:
            raise ValueError("core missing gene_id/split")
        core = pd.read_csv(self.core_path, sep="\t", usecols=[gcol, "split"])
        core = core.rename(columns={gcol: "gene_base"})
        core["gene_base"] = _strip_gene(core["gene_base"])
        core["split_canon"] = _canon_split(core["split"])
        if core["split_canon"].isna().any():
            raise ValueError("core contains unknown split")
        # conflicts
        nun = core.groupby("gene_base")["split_canon"].nunique()
        if (nun > 1).any():
            raise ValueError("core gene has conflicting splits")
        return core.drop_duplicates("gene_base")

    def _build_ortholog(self, df: pd.DataFrame, core: pd.DataFrame) -> None:
        heldout_genes = set(core.loc[core["split_canon"].isin(["DEV", "TEST"]), "gene_base"])
        # table human heldout
        hum = df[df["species_norm"] == "human"]
        heldout_genes |= set(hum.loc[hum["split_canon"].isin(["DEV", "TEST"]), "gene_base"])

        # contradictory human split core vs table
        core_map = dict(zip(core["gene_base"], core["split_canon"]))
        hum_train = set(hum.loc[hum["split_canon"] == "TRAIN", "gene_base"])
        for gb in hum_train:
            cs = core_map.get(gb)
            if cs is not None and cs != "TRAIN":
                raise ValueError(f"contradictory human split for {gb}")

        self._blocked_human_genes.update(heldout_genes)
        donor_train = (hum_train & set(core.loc[core["split_canon"] == "TRAIN", "gene_base"])) - heldout_genes

        ortho = pd.read_csv(self.ortholog_path, sep="\t", low_memory=False)
        if "human_gene_id" not in ortho.columns:
            raise ValueError("ortholog missing human_gene_id")
        ortho["human_gene_id"] = _strip_gene(ortho["human_gene_id"])

        species_cols = [c for c in ortho.columns if c.endswith("_gene") and c != "human_gene_id"]
        self._heldout_spgene: Dict[str, set] = {}
        self._valid_ortholog: Dict[str, set] = {}
        self._ortho_species_cols: Dict[str, str] = {}

        for col in species_cols:
            sp = col[: -len("_gene")]
            if sp not in self._species_to_id:
                continue
            if sp != "human" and sp not in self._ortho_species_cols:
                pass
            sub = ortho[["human_gene_id", col]].dropna()
            sub = sub[sub[col].astype(str).str.len() > 0]
            sub = sub[sub["human_gene_id"].astype(str).str.len() > 0]
            sub = sub.assign(spgene=_strip_gene(sub[col]))
            sub = sub[["human_gene_id", "spgene"]].drop_duplicates()
            self._ortho_species_cols[sp] = col
            self._heldout_spgene[sp] = set(sub.loc[sub["human_gene_id"].isin(heldout_genes), "spgene"])
            # unique donor/receiver for valid ortholog only
            hc = sub["human_gene_id"].value_counts()
            sc = sub["spgene"].value_counts()
            sub_unique = sub[sub["human_gene_id"].isin(hc[hc == 1].index) & sub["spgene"].isin(sc[sc == 1].index)]
            self._valid_ortholog[sp] = set(
                sub_unique.loc[sub_unique["human_gene_id"].isin(donor_train), "spgene"]
            )

        # ensure every nonhuman TRAIN species has an ortholog mapping column
        train_species = set(df.loc[df["split_canon"] == "TRAIN", "species_norm"].unique())
        for sp in train_species:
            if sp != "human" and sp not in self._ortho_species_cols:
                raise ValueError(f"missing ortholog mapping column for TRAIN species {sp}")

        # exclude nonhuman TRAIN rows whose gene in heldout_spgene
        nonhum = df["species_norm"] != "human"
        train = df["split_canon"] == "TRAIN"
        excl = np.zeros(len(df), dtype=bool)
        for sp, hset in self._heldout_spgene.items():
            if not hset:
                continue
            m = nonhum.to_numpy() & train.to_numpy() & (df["species_norm"].to_numpy() == sp) & df["gene_base"].isin(hset).to_numpy()
            excl |= m
        self._leakage_excluded_rows = int(excl.sum())
        self._dropped_leak_per_species = {}
        for sp, hset in self._heldout_spgene.items():
            if not hset:
                continue
            m = nonhum.to_numpy() & train.to_numpy() & (df["species_norm"].to_numpy() == sp) & df["gene_base"].isin(hset).to_numpy()
            self._dropped_leak_per_species[sp] = int(m.sum())
        df.drop(index=df.index[excl], inplace=True)

        # ortholog mask
        ortho_raw = np.full(len(df), np.nan, dtype=np.float64)
        ortho_mask = np.zeros(len(df), dtype=np.float64)
        if "ortholog_E" in df.columns:
            oe = pd.to_numeric(df["ortholog_E"], errors="raise").to_numpy(dtype=np.float64)
            finite = np.isfinite(oe)
            sp_arr = df["species_norm"].to_numpy()
            gb_arr = df["gene_base"].to_numpy()
            for sp, vset in self._valid_ortholog.items():
                if not vset:
                    continue
                m = finite & (sp_arr == sp) & pd.Series(gb_arr).isin(set(vset)).to_numpy()
                ortho_raw[m] = oe[m]
                ortho_mask[m] = 1.0
        df["_ortholog_raw"] = ortho_raw
        df["_ortholog_mask"] = ortho_mask

    # ---------------- C: TF ----------------
    def _build_tf(self, df: pd.DataFrame) -> None:
        n = len(df)
        self.tf_row = np.full(n, -1, dtype=np.int64)
        self.tf_valid = np.zeros(n, dtype=bool)
        if "tf_row" not in df.columns:
            self._tf_dim = 0
            return
        tr = pd.to_numeric(df["tf_row"], errors="raise").to_numpy(dtype=np.float64)
        finite = np.isfinite(tr)
        if not finite.any():
            self._tf_dim = 0
            return
        if tr[finite].min() < 0 or (tr[finite] != np.floor(tr[finite])).any():
            raise ValueError("tf_row invalid")
        arr = np.load(self.tf_path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError("tf matrix not 2d")
        self._tf_dim = int(arr.shape[1])
        if self._tf_dim > 400:
            raise ValueError(
                f"tf matrix has {self._tf_dim} columns > 400; needs TRAIN top400 selector"
            )
        rows = tr[finite].astype(np.int64)
        if rows.max() >= arr.shape[0]:
            raise ValueError("tf_row out of range")
        # validate in chunks: all finite and 0/1
        for i in range(0, len(rows), 8192):
            chunk = rows[i : i + 8192]
            data = np.asarray(arr[chunk])
            if not np.isfinite(data).all():
                raise ValueError("tf matrix contains non-finite values")
            if not np.isin(data, [0, 1]).all():
                raise ValueError("tf matrix values must be 0 or 1")
        idx = np.where(finite)[0]
        self.tf_row[idx] = rows
        self.tf_valid[idx] = True
        self._tf_arr = arr

        # validate tf_row against features/omics/chip_tf_row_index_v1.tsv if present
        self._tf_index_check = "unavailable"
        self._tf_index_count = 0
        self._tf_index_path = None
        tf_index_path = project_path(self.root, "features/omics/chip_tf_row_index_v1.tsv")
        if os.path.exists(tf_index_path):
            idx_df = pd.read_csv(tf_index_path, sep="\t", low_memory=False)
            if "record_id" not in idx_df.columns or "row_index" not in idx_df.columns:
                raise ValueError("tf row index file missing record_id/row_index")
            idx_df["record_id"] = idx_df["record_id"].astype(str)
            idx_df["row_index"] = pd.to_numeric(idx_df["row_index"], errors="raise").astype(np.int64)
            if idx_df["record_id"].duplicated().any():
                dup = idx_df.loc[idx_df["record_id"].duplicated(keep=False), "record_id"].unique()
                raise ValueError(f"tf row index duplicate record_id: {dup[:5].tolist()}")
            idx_map = pd.Series(idx_df["row_index"].to_numpy(), index=idx_df["record_id"].to_numpy())
            valid_idx = np.where(self.tf_valid)[0]
            if valid_idx.size:
                rids = df["record_id"].astype(str).to_numpy()[valid_idx]
                mapped = idx_map.reindex(rids)
                if mapped.isna().any():
                    missing = rids[mapped.isna().to_numpy()]
                    raise ValueError(f"tf row index missing record_id {missing[0]}")
                expected = mapped.to_numpy(dtype=np.int64)
                actual = self.tf_row[valid_idx]
                bad = expected != actual
                if bad.any():
                    j = int(np.argmax(bad))
                    raise ValueError(
                        f"tf_row mismatch for {rids[j]}: {actual[j]} != {expected[j]}"
                    )
            self._tf_index_check = "passed"
            self._tf_index_count = int(valid_idx.size)
            self._tf_index_path = tf_index_path

    # ---------------- D: sequences ----------------
    def _seq_paths(self, key: str, species: str) -> Tuple[str, str]:
        if species == "human":
            up = project_path(self.root, "promoters/memmap/promoter_up2000_uint8.memmap")
            sy = project_path(self.root, "promoters/memmap/promoter_sym400_uint8.memmap")
        else:
            up = project_path(self.root, f"data/multispecies/memmap/{key}_up2000_uint8.memmap")
            sy = project_path(self.root, f"data/multispecies/memmap/{key}_sym400_uint8.memmap")
        return up, sy

    def _open_seq(self, df: pd.DataFrame) -> None:
        self._seq_up: Dict[str, np.ndarray] = {}
        self._seq_sy: Dict[str, np.ndarray] = {}
        keys = df["memmap_key"].astype(str).unique()
        for key in keys:
            sub = df[df["memmap_key"].astype(str) == key]
            sp = sub["species_norm"].iloc[0]
            up, sy = self._seq_paths(key, sp)
            if not os.path.exists(up) or not os.path.exists(sy):
                raise FileNotFoundError(f"missing sequence memmap for {key}")
            # sequences are headerless raw uint8
            up_size = os.path.getsize(up)
            sy_size = os.path.getsize(sy)
            if up_size % 2000 != 0:
                raise ValueError("up2000 file size not multiple of 2000")
            if sy_size % 400 != 0:
                raise ValueError("sym400 file size not multiple of 400")
            a_up = np.memmap(up, dtype=np.uint8, mode="r", shape=(up_size // 2000, 2000))
            a_sy = np.memmap(sy, dtype=np.uint8, mode="r", shape=(sy_size // 400, 400))
            r2 = sub["r2000"].to_numpy()
            r4 = sub["r400"].to_numpy()
            if r2.max() >= a_up.shape[0] or r4.max() >= a_sy.shape[0]:
                raise ValueError("sequence row out of range")
            self._seq_up[key] = a_up
            self._seq_sy[key] = a_sy

    # ---------------- E: profile ----------------
    def _build_profile(self, df: pd.DataFrame) -> None:
        prof = pd.read_csv(self.profile_path, sep="\t", usecols=["gene_id", "tissue", "mean_tpm"], low_memory=False)
        prof["gene_base"] = _strip_gene(prof["gene_id"])
        prof["tissue"] = prof["tissue"].astype(str)
        self.tissue_names = sorted(prof["tissue"].unique().tolist())
        self.n_profile = len(self.tissue_names)
        tmap = {t: i for i, t in enumerate(self.tissue_names)}

        tpm = pd.to_numeric(prof["mean_tpm"], errors="raise").to_numpy(dtype=np.float64)
        bad = np.isfinite(tpm) & (tpm < 0)
        if bad.any():
            raise ValueError("negative TPM")
        # +inf reject
        if np.isinf(tpm).any():
            raise ValueError("profile contains +inf")
        prof["_tpm"] = np.where(np.isfinite(tpm), tpm, np.nan)

        # duplicate gene x tissue -> exclude whole gene
        dup = prof.duplicated(subset=["gene_base", "tissue"], keep=False)
        bad_genes = set(prof.loc[dup, "gene_base"].unique())
        self.profile_ambiguous_genes = sorted(bad_genes)
        self.profile_ambiguous_count = len(bad_genes)
        prof = prof[~prof["gene_base"].isin(bad_genes)]

        piv = prof.pivot(index="gene_base", columns="tissue", values="_tpm")
        piv = piv.reindex(columns=self.tissue_names)
        mat = np.log2(1.0 + piv.to_numpy(dtype=np.float64)).astype(np.float32)
        self.profile_matrix = mat
        self._profile_genes = list(piv.index)
        self._profile_gene_to_idx = {g: i for i, g in enumerate(self._profile_genes)}

        n = len(df)
        self.profile_lookup = np.full(n, -1, dtype=np.int64)
        is_human = (df["species_norm"] == "human").to_numpy()
        gb = df["gene_base"].to_numpy()
        for i in np.where(is_human)[0]:
            j = self._profile_gene_to_idx.get(gb[i])
            if j is not None:
                self.profile_lookup[i] = j

    # ---------------- F: assemble ----------------
    def _build(self) -> None:
        df = self._read_table()
        core = self._read_core()
        self._build_ortholog(df, core)
        self._build_tf(df)
        self._open_seq(df)
        self._build_profile(df)

        # drop nonhuman DEV/TEST with no signal
        nonhum = (df["species_norm"] != "human").to_numpy()
        eval_split = df["split_canon"].isin(["DEV", "TEST"]).to_numpy()
        # has_signal includes finite ESB with label_mask=1
        esb_signal = np.zeros(len(df), dtype=bool)
        lm = df["_label_mask"].to_numpy() == 1
        for c in ESB_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=np.float64)
                esb_signal |= np.isfinite(v) & lm
        has_signal = (
            (df["_ortholog_mask"].to_numpy() > 0)
            | self.tf_valid
            | df[[c for c in OMICS_COLS if c in df.columns]].notna().any(axis=1).to_numpy()
            | esb_signal
        )
        drop = nonhum & eval_split & ~has_signal
        # count actual human EVAL with no heads (profile finite included)
        human_eval = (df["species_norm"] == "human").to_numpy() & eval_split
        profile_signal = np.zeros(len(df), dtype=bool)
        valid_prof = self.profile_lookup >= 0
        if valid_prof.any():
            sub = self.profile_matrix[self.profile_lookup[valid_prof]]
            profile_signal[valid_prof] = np.isfinite(sub).any(axis=1)
        human_no_signal = human_eval & ~has_signal & ~profile_signal
        self._unlabeled_human_eval = int(human_no_signal.sum())
        df = df.loc[~drop].copy()
        # slice arrays to match df
        self.tf_valid = self.tf_valid[~drop]
        self.tf_row = self.tf_row[~drop]
        self.profile_lookup = self.profile_lookup[~drop]

        n = len(df)
        self.n = n
        self.df = df.reset_index(drop=True)
        self.stable_id = self.df["record_id"].astype(str).to_numpy()
        self.gene_id = self.df["gene_base"].to_numpy()
        self.split = self.df["split_canon"].to_numpy()
        self.species = self.df["species_id"].to_numpy(dtype=np.int64)
        self.species_norm = self.df["species_norm"].to_numpy()
        self.keys = self.df["memmap_key"].astype(str).to_numpy()
        self.r400 = self.df["r400"].to_numpy(dtype=np.int64)
        self.r2000 = self.df["r2000"].to_numpy(dtype=np.int64)
        self.source_rows = self.df["_source_row"].to_numpy(dtype=np.int64)
        self.n_species = max(self._species_to_id.values()) + 1
        self.n_tf = self._tf_dim

        # ESB
        y_raw = np.full((n, 3), np.nan, dtype=np.float64)
        y_mask = np.zeros((n, 3), dtype=np.float64)
        lm = self.df["_label_mask"].to_numpy() == 1
        for j, c in enumerate(ESB_COLS):
            if c in self.df.columns:
                v = pd.to_numeric(self.df[c], errors="raise").to_numpy(dtype=np.float64)
                fin = np.isfinite(v) & lm
                y_raw[fin, j] = v[fin]
                y_mask[fin, j] = 1.0
        self.y_raw = y_raw
        self.y_mask = y_mask
        self.mask = y_mask

        # omics
        omics_raw = np.full((n, 9), np.nan, dtype=np.float64)
        omics_mask = np.zeros((n, 9), dtype=np.float64)
        self.omics_present_cols = [c for c in OMICS_COLS if c in self.df.columns]
        self.omics_missing_cols = [c for c in OMICS_COLS if c not in self.df.columns]
        for j, c in enumerate(OMICS_COLS):
            if c in self.df.columns:
                v = pd.to_numeric(self.df[c], errors="raise").to_numpy(dtype=np.float64)
                fin = np.isfinite(v)
                omics_raw[fin, j] = v[fin]
                omics_mask[fin, j] = 1.0
        self.omics_raw = omics_raw
        self.omics_mask = omics_mask

        self.ortholog_raw = self.df["_ortholog_raw"].to_numpy(dtype=np.float64)
        self.ortholog_mask = self.df["_ortholog_mask"].to_numpy(dtype=np.float64)
        self.species_to_id = self._species_to_id

        # normalizers
        train = self.split == "TRAIN"
        self.normalizers = {
            "esb": self._norm_stats(y_raw, y_mask, train),
            "omics": self._norm_stats(omics_raw, omics_mask, train),
            "ortholog": self._norm_stats(
                self.ortholog_raw.reshape(-1, 1), self.ortholog_mask.reshape(-1, 1), train
            ),
        }
        # profile normalizer on unique human TRAIN genes
        hum_train = train & (self.species_norm == "human")
        uniq_idx = np.unique(self.profile_lookup[hum_train & (self.profile_lookup >= 0)])
        self.profile_training_unique_genes = int(len(uniq_idx))
        if len(uniq_idx) > 0:
            sub = self.profile_matrix[uniq_idx]
            cnt = np.isfinite(sub).sum(axis=0).astype(np.int64)
            mean = np.nanmean(sub, axis=0)
            std = np.nanstd(sub, axis=0)
            mean = np.where(cnt > 0, mean, 0.0)
            std = np.where(cnt > 0, np.maximum(std, 1e-6), 1.0)
        else:
            cnt = np.zeros(self.n_profile, dtype=np.int64)
            mean = np.zeros(self.n_profile, dtype=np.float64)
            std = np.ones(self.n_profile, dtype=np.float64)
        self.normalizers["profile"] = {
            "count": cnt.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "tissue_names": list(self.tissue_names),
            "columns": list(self.tissue_names),
            "source_sha256": self._profile_hash,
        }

        # residual conflicts
        self._residual_conflicts = 0
        sp_train = train & (self.species_norm != "human")
        for sp, hset in self._heldout_spgene.items():
            if not hset:
                continue
            m = sp_train & (self.species_norm == sp) & pd.Series(self.gene_id).isin(set(hset)).to_numpy()
            self._residual_conflicts += int(m.sum())

        # profile coverage
        self._profile_coverage = self._compute_profile_coverage()
        # source_paths attribute
        self.source_paths = {
            "table": self.table_path,
            "ortholog": self.ortholog_path,
            "core": self.core_path,
            "tf": self.tf_path,
            "profile": self.profile_path,
        }
        for key in np.unique(self.keys):
            sub = self.df[self.df["memmap_key"].astype(str) == key]
            sp = sub["species_norm"].iloc[0]
            up, sy = self._seq_paths(key, sp)
            self.source_paths[f"seq_up_{key}"] = up
            self.source_paths[f"seq_sy_{key}"] = sy
        if getattr(self, "_tf_index_path", None):
            self.source_paths["tf_index"] = self._tf_index_path

    def _norm_stats(self, raw: np.ndarray, mask: np.ndarray, train: np.ndarray) -> Dict[str, Any]:
        cnt = np.zeros(raw.shape[1], dtype=np.int64)
        mean = np.zeros(raw.shape[1], dtype=np.float64)
        std = np.ones(raw.shape[1], dtype=np.float64)
        for j in range(raw.shape[1]):
            m = train & (mask[:, j] > 0) & np.isfinite(raw[:, j])
            c = int(m.sum())
            cnt[j] = c
            if c > 0:
                v = raw[m, j]
                mean[j] = float(v.mean())
                s = float(v.std())
                std[j] = max(s, 1e-6)
        return {"count": cnt.tolist(), "mean": mean.tolist(), "std": std.tolist()}

    def _compute_profile_coverage(self) -> Dict[str, Any]:
        lookup = self.profile_lookup
        valid = lookup >= 0
        finite = np.zeros(len(lookup), dtype=bool)
        if valid.any():
            sub = self.profile_matrix[lookup[valid]]
            finite[valid] = np.isfinite(sub).any(axis=1)
        out = {}
        for sp in ["TRAIN", "DEV", "TEST"]:
            m = self.split == sp
            out[sp] = {
                "rows_with_any_profile": int((m & finite).sum()),
                "unique_genes": int(len(np.unique(self.gene_id[m & finite]))),
            }
        return out

    # ---------------- public API ----------------
    def __len__(self) -> int:
        return self.n

    def split_indices(self, split: str) -> np.ndarray:
        s = _SPLIT_MAP.get(split.strip().lower(), split.strip().upper())
        return np.where(self.split == s)[0]

    def records(self, indices: Optional[Iterable[int]] = None) -> Iterator[Dict[str, Any]]:
        if indices is None:
            indices = range(self.n)
        for i in indices:
            yield self.record(int(i))

    def record(self, i: int) -> Dict[str, Any]:
        key = self.keys[i]
        up = self._seq_up[key]
        sy = self._seq_sy[key]
        context = np.asarray(up[self.r2000[i]], dtype=np.uint8)
        local = np.asarray(sy[self.r400[i]], dtype=np.uint8)

        esb = np.where(self.y_mask[i] > 0, self.y_raw[i], 0.0)
        esb_mask = self.y_mask[i].copy()
        esb_z = self._z(self.y_raw[i], self.y_mask[i], self.normalizers["esb"])

        om = np.where(self.omics_mask[i] > 0, self.omics_raw[i], 0.0)
        om_mask = self.omics_mask[i].copy()
        om_z = self._z(self.omics_raw[i], self.omics_mask[i], self.normalizers["omics"])

        orth = float(self.ortholog_raw[i]) if self.ortholog_mask[i] > 0 else 0.0
        orth_m = float(self.ortholog_mask[i])
        orth_z = 0.0
        if orth_m > 0:
            nz = self.normalizers["ortholog"]
            orth_z = (orth - nz["mean"][0]) / nz["std"][0]

        if self.tf_valid[i]:
            tf = np.asarray(self._tf_arr[self.tf_row[i]], dtype=np.float64)
            tf_mask = np.ones(self.n_tf, dtype=np.float64)
        else:
            tf = np.zeros(self.n_tf, dtype=np.float64)
            tf_mask = np.zeros(self.n_tf, dtype=np.float64)

        j = self.profile_lookup[i]
        if j >= 0:
            prof = self.profile_matrix[j].astype(np.float64)
            prof_mask = np.isfinite(prof).astype(np.float64)
            prof = np.where(prof_mask > 0, prof, 0.0)
            nz = self.normalizers["profile"]
            prof_z = np.where(
                prof_mask > 0,
                (prof - np.asarray(nz["mean"])) / np.asarray(nz["std"]),
                0.0,
            )
        else:
            prof = np.zeros(self.n_profile, dtype=np.float64)
            prof_mask = np.zeros(self.n_profile, dtype=np.float64)
            prof_z = np.zeros(self.n_profile, dtype=np.float64)

        is_train = 1.0 if self.split[i] == "TRAIN" else 0.0
        return {
            "local_codes400": local,
            "context_codes2000": context,
            "species": int(self.species[i]),
            "targets_ESB": esb,
            "finite_mask": esb_mask,
            "targets_ESB_z": esb_z,
            "ortholog_E": orth,
            "ortholog_mask": orth_m,
            "ortholog_E_z": float(orth_z),
            "targets_omics": om,
            "omics_mask": om_mask,
            "targets_omics_z": om_z,
            "targets_tf": tf,
            "tf_mask": tf_mask,
            "targets_profile": prof,
            "profile_mask": prof_mask,
            "targets_profile_z": prof_z,
            "mlm_loss_mask": is_train,
            "species_loss_mask": is_train,
            "stableID": str(self.stable_id[i]),
            "gene_id": str(self.gene_id[i]),
            "split": str(self.split[i]),
        }

    @staticmethod
    def _z(raw: np.ndarray, mask: np.ndarray, nz: Dict[str, Any]) -> np.ndarray:
        mean = np.asarray(nz["mean"], dtype=np.float64)
        std = np.asarray(nz["std"], dtype=np.float64)
        out = np.zeros_like(raw, dtype=np.float64)
        m = mask > 0
        out[m] = (raw[m] - mean[m]) / std[m]
        return out

    # ---------------- audit ----------------
    def audit(self, hash_sources: bool = True) -> Dict[str, Any]:
        by_split = {s: int((self.split == s).sum()) for s in ["TRAIN", "DEV", "TEST"]}
        masks = {}
        for name, m in [
            ("ESB", self.y_mask),
            ("omics", self.omics_mask),
        ]:
            masks[name] = {
                s: int((m[self.split == s] > 0).sum()) for s in ["TRAIN", "DEV", "TEST"]
            }
            masks[name]["per_col"] = {
                s: (m[self.split == s] > 0).sum(axis=0).tolist() for s in ["TRAIN", "DEV", "TEST"]
            }
        masks["ortholog"] = {s: int((self.ortholog_mask[self.split == s] > 0).sum()) for s in ["TRAIN", "DEV", "TEST"]}
        masks["TF"] = {s: int((self.tf_valid[self.split == s]).sum()) for s in ["TRAIN", "DEV", "TEST"]}
        prof_any = np.zeros(self.n, dtype=bool)
        valid = self.profile_lookup >= 0
        if valid.any():
            sub = self.profile_matrix[self.profile_lookup[valid]]
            prof_any[valid] = np.isfinite(sub).any(axis=1)
        masks["profile"] = {s: int((prof_any[self.split == s]).sum()) for s in ["TRAIN", "DEV", "TEST"]}
        masks["species"] = {"TRAIN": int((self.split == "TRAIN").sum()), "DEV": 0, "TEST": 0}
        masks["MLM"] = {"TRAIN": int((self.split == "TRAIN").sum()), "DEV": 0, "TEST": 0}

        source_paths = self.source_paths

        source_hashes = {}
        if hash_sources:
            for k, p in source_paths.items():
                if not os.path.exists(p):
                    continue
                if k == "profile" and self._profile_hash is not None:
                    source_hashes[k] = {"path": p, "size": os.path.getsize(p), "sha256": self._profile_hash}
                    continue
                if p in self._hash_cache:
                    source_hashes[k] = self._hash_cache[p]
                    continue
                h = _sha256(p)
                rec = {"path": p, "size": os.path.getsize(p), "sha256": h}
                self._hash_cache[p] = rec
                if k == "profile":
                    self._profile_hash = h
                source_hashes[k] = rec
        else:
            for k, p in source_paths.items():
                if os.path.exists(p):
                    source_hashes[k] = {"path": p, "size": os.path.getsize(p), "sha256": "pending"}

        result = {
            "source_table_rows": self._source_table_rows,
            "source_by_split": self._source_counts,
            "retained": self.n,
            "by_split": by_split,
            "leakage_excluded_rows": self._leakage_excluded_rows,
            "dropped_leak_per_species": self._dropped_leak_per_species,
            "residual_conflicts": self._residual_conflicts,
            "profile_ambiguous_count": self.profile_ambiguous_count,
            "profile_ambiguous_genes": self.profile_ambiguous_genes,
            "profile_training_unique_genes": self.profile_training_unique_genes,
            "excluded_unknown_split": self._excluded_unknown_split,
            "unlabeled_human_eval": self._unlabeled_human_eval,
            "masks": masks,
            "profile_coverage": self._profile_coverage,
            "omics_present_cols": self.omics_present_cols,
            "omics_missing_cols": self.omics_missing_cols,
            "normalizers": self.normalizers,
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "tf_index_check": self._tf_index_check,
            "tf_index_count": self._tf_index_count,
        }
        result["counts"] = {
            "source_table_rows": self._source_table_rows,
            "retained": self.n,
            "by_split": by_split,
            "leakage_excluded_rows": self._leakage_excluded_rows,
            "dropped_leak_per_species": self._dropped_leak_per_species,
            "residual_conflicts": self._residual_conflicts,
            "profile_ambiguous_genes": self.profile_ambiguous_genes,
            "profile_ambiguous_count": self.profile_ambiguous_count,
            "profile_training_unique_genes": self.profile_training_unique_genes,
            "blocked_human_genes": sorted(self._blocked_human_genes),
        }
        return result