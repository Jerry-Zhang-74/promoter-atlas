#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch12 v6.13.0-siface.rc1 frozen gene-level raw-S interface CLI.

Read-only local CLI (no hosted service). Three subcommands:

  predict   conditional raw-S prediction for explicit canonical Ensembl human
            gene IDs (args or batch file). Genuine saved-model inference on
            frozen feature lookup. Not a sequence promoter scorer.
  trace     label-trace lookup: observed E_raw/S_raw/B_raw + legacy scores,
            eligibility/missing reasons and source-file/hash links. Never
            substitutes label lookup for prediction.
  catalog   emit the complete 18769-gene prediction catalog (split roles +
            trace metadata) and the explicit 178-gene exclusion list.

Output is UTF-8 JSON (finite numeric serialization, no NaN) or TSV, with
deterministic ordering and row-level structured reason codes. Artifact
integrity (curated known-path + expected SHA256) is verified before inference;
missing/tampered artifacts fail closed. No arbitrary pickle path is ever
accepted from user input.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from s12_common import (  # noqa: E402
    BATCH, CANDIDATE_VERSION, DATA_OUT, FEATURE_NAMES_ALL, FROZEN_ARTIFACTS,
    HGB_CONFIG, MODEL_KEY, MODEL_OUT, N_CANONICAL, N_ELIGIBLE, N_EXCLUDED,
    REPORT_OUT, S_TARGET_DEF, S_THEORETICAL_MAX, TARGET_TRAIN_MEAN,
    TARGET_TRAIN_STD, atomic_write_json, base_gene_id, load_feature_matrix,
    load_gene_universe, load_label_catalog, now_local, reconstruct_s_raw_full,
    sha256_file, verify_artifacts)

# Human gene / transcript / non-human Ensembl formats (documented, anchored).
_HUMAN_GENE_RE = re.compile(r"^ENSG\d{11}$")
_TRANSCRIPT_RE = re.compile(r"^ENST\d{11}$")
_NONHUMAN_GENE_RE = re.compile(r"^ENS[A-Z]{2,}G\d+$")   # e.g. ENSMUSG, ENSRNOG
_VERSION_SUFFIX_RE = re.compile(r"^(.+?)\.\d+$")

LEGACY_TRACE_UNION = REPORT_OUT  # batch12-owned dir holds the trace manifest

# Applicability status strings (frozen).
APPLICABILITY_OK = "applicable_conditional_catalog"
APPLICABILITY_IN_SAMPLE = "applicable_conditional_catalog_in_sample"


def strip_version(gid):
    """Strip only a documented terminal dot-integer (e.g. ENSG...3.14 -> ENSG...3).
    Returns (normalized, stripped: bool)."""
    m = _VERSION_SUFFIX_RE.match(str(gid or ""))
    if m and m.group(1):
        return m.group(1), True
    return str(gid or ""), False


def classify_id(raw, catalog, universe_set):
    """Classify a (normalized) gene id into a structured result prefix.

    Returns (status, reason_code, reason_detail). status in
    {"predicted", "rejected"}. Only canonical S-defined feature-covered human
    gene IDs are predicted; everything else is rejected with a reason code.
    """
    g = str(raw or "").strip()
    if not g:
        return "rejected", "empty_input", "empty gene id"
    if _TRANSCRIPT_RE.match(g):
        return ("rejected", "transcript_id",
                "Ensembl transcript ID (ENST) is not a gene ID; no transcript->gene mapping performed")
    if _NONHUMAN_GENE_RE.match(g) and not _HUMAN_GENE_RE.match(g):
        return ("rejected", "non_human",
                "non-human Ensembl gene ID (species prefix != ENSG); out of scope")
    if g not in catalog:
        if _HUMAN_GENE_RE.match(g):
            return ("rejected", "unknown",
                    "canonical-format human ENSG not present in the 58988-gene label catalog")
        return ("rejected", "unknown",
                "not a canonical Ensembl human gene ID (no fuzzy/symbol mapping performed)")
    row = catalog[g]
    if not row["S_defined_main"]:
        return ("rejected", "S_undefined",
                row["S_eligibility_reason"] or "S undefined (B_raw<0.30 or <3 detectable tissues)")
    if g not in universe_set:
        return ("rejected", "S_defined_feature_missing",
                "canonical S-defined but absent from the 57827-gene feature universe (178-gene exclusion)")
    return "predicted", None, None


class FrozenPredictor:
    """Frozen batch10 full-1069 HGB raw-S predictor (no fit, read-only)."""

    def __init__(self, verify=True):
        if verify:
            ok, problems = verify_artifacts(FROZEN_ARTIFACTS)
            if not ok:
                raise RuntimeError(
                    "artifact integrity check FAILED (fail closed): "
                    + json.dumps(problems, ensure_ascii=False))
        import joblib  # deferred: only after integrity passes
        self.gene_gid, self.gene_split, self.comp_dense, self.n_prom = load_gene_universe()
        self.X = load_feature_matrix()
        self.catalog = load_label_catalog()
        self.universe_set = set(self.gene_gid)
        self._universe_index = {g: i for i, g in enumerate(self.gene_gid)}
        # full-precision observed S (18947 S-defined genes)
        self.observed_s = reconstruct_s_raw_full()
        self.scaler = joblib.load(Path(FROZEN_ARTIFACTS[1]["path"]))
        self.models = joblib.load(Path(FROZEN_ARTIFACTS[0]["path"]))
        if MODEL_KEY not in self.models:
            raise RuntimeError("frozen model dict missing key %s" % MODEL_KEY)
        self.model = self.models[MODEL_KEY]
        self.model_sha = FROZEN_ARTIFACTS[0]["sha256"]
        self.features1024_sha = FROZEN_ARTIFACTS[4]["sha256"]
        self.features45_sha = FROZEN_ARTIFACTS[3]["sha256"]
        self.split_sha = FROZEN_ARTIFACTS[5]["sha256"]
        self.tracka_sha = FROZEN_ARTIFACTS[6]["sha256"]

    def _eligible_mask(self):
        return np.array([g in self.observed_s for g in self.gene_gid])

    def _predict_rows(self, row_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        Xe = self.X[row_indices]
        Xs = self.scaler.transform(Xe)
        z = self.model.predict(Xs)
        return z * TARGET_TRAIN_STD + TARGET_TRAIN_MEAN

    def _build_prediction(self, gid):
        idx = self._universe_index[gid]
        pred = float(self._predict_rows([idx])[0])
        row = self.catalog[gid]
        split = str(self.gene_split[idx])
        out_of_range = (pred > S_THEORETICAL_MAX)
        return {
            "status": "predicted",
            "normalized_id": gid,
            "predicted_raw_S": pred,
            "observed_raw_S": float(self.observed_s[gid]),
            "legacy_Score_S": row["Score_S"],
            "observed_E_raw": row["E_raw"],
            "observed_B_raw": row["B_raw"],
            "legacy_Score_E": row["Score_E"],
            "legacy_Score_B": row["Score_B"],
            "split": split,
            "component_id": int(self.comp_dense[idx]),
            "n_promoters": int(self.n_prom[idx]),
            "in_sample": (split == "train"),
            "applicability_status": (APPLICABILITY_IN_SAMPLE if split == "train"
                                     else APPLICABILITY_OK),
            "outside_theoretical_range": bool(out_of_range),
            "promoter_aggregation": "arithmetic mean across ALL promoters of the gene (no label-driven promoter selection)",
            "target_definition": S_TARGET_DEF,
            "provenance": {
                "candidate_version": CANDIDATE_VERSION,
                "batch": BATCH,
                "model": HGB_CONFIG,
                "model_sha256": self.model_sha,
                "features45_sha256": self.features45_sha,
                "features1024_sha256": self.features1024_sha,
                "split_sha256": self.split_sha,
                "trackA_sha256": self.tracka_sha,
                "n_features": len(FEATURE_NAMES_ALL),
            },
        }

    def predict(self, inputs, normalize_version=False):
        results = []
        seen = {}  # normalized -> count (for collision disclosure)
        for i, raw in enumerate(inputs, start=1):
            orig = str(raw)
            norm, stripped = strip_version(orig) if normalize_version else (orig, False)
            status, reason_code, reason_detail = classify_id(norm, self.catalog,
                                                             self.universe_set)
            rec = {
                "row_id": i,
                "input_id": orig,
                "normalized_id": norm if (norm != orig or normalize_version) else None,
                "version_suffix_stripped": bool(stripped),
                "status": status,
            }
            if status == "predicted":
                rec.update(self._build_prediction(norm))
            else:
                rec.update({"reason_code": reason_code, "reason_detail": reason_detail})
                if norm in self.catalog:
                    row = self.catalog[norm]
                    rec.update({
                        "observed_E_raw": row["E_raw"],
                        "observed_B_raw": row["B_raw"],
                        "legacy_Score_E": row["Score_E"],
                        "legacy_Score_B": row["Score_B"],
                        "legacy_Score_S": row["Score_S"],
                    })
                rec["predicted_raw_S"] = None
            results.append(rec)
            # collision disclosure after the fact
        return results

    def trace(self, inputs, normalize_version=False):
        results = []
        for i, raw in enumerate(inputs, start=1):
            orig = str(raw)
            norm, stripped = strip_version(orig) if normalize_version else (orig, False)
            rec = {
                "row_id": i,
                "input_id": orig,
                "normalized_id": norm if (norm != orig or normalize_version) else None,
                "version_suffix_stripped": bool(stripped),
            }
            if norm in self.catalog:
                row = self.catalog[norm]
                rec.update({
                    "status": "found",
                    "gene_id": norm,
                    "S_defined_main": row["S_defined_main"],
                    "observed_E_raw": row["E_raw"],
                    "observed_S_raw": row["S_raw_full"],
                    "observed_B_raw": row["B_raw"],
                    "legacy_Score_E": row["Score_E"],
                    "legacy_Score_S": row["Score_S"],
                    "legacy_Score_B": row["Score_B"],
                    "detectable_tissue_count": row["detectable_tissue_count"],
                    "tissue_count": row["tissue_count"],
                    "eligibility_reason": row["S_eligibility_reason"] or ("" if row["S_defined_main"] else "S_undefined"),
                    "source_files": {
                        "canonical_label_catalog": {
                            "file": "data/breakthrough_20261002/batch08/label_audit_all_genes.tsv",
                            "sha256": FROZEN_ARTIFACTS[9]["sha256"],
                            "role": "full-precision S_raw + E_raw/B_raw + Score_E/S/B + eligibility",
                        },
                        "raw_expression_table": {
                            "file": "labels/build/gene_expression_table.tsv",
                            "sha256": FROZEN_ARTIFACTS[7]["sha256"],
                            "role": "raw tissue mean_tpm (S_raw formula source)",
                        },
                        "label_table": {
                            "file": "labels/build/gene_expression_label_table.tsv",
                            "sha256": FROZEN_ARTIFACTS[8]["sha256"],
                            "role": "canonical E/S/B raw + scores",
                        },
                    },
                    "formula": S_TARGET_DEF["formula"],
                })
            elif _TRANSCRIPT_RE.match(norm):
                rec.update({"status": "rejected", "reason_code": "transcript_id",
                            "reason_detail": "ENST transcript ID; no gene-level label"})
            elif _NONHUMAN_GENE_RE.match(norm) and not _HUMAN_GENE_RE.match(norm):
                rec.update({"status": "rejected", "reason_code": "non_human",
                            "reason_detail": "non-human Ensembl gene; out of scope"})
            else:
                rec.update({"status": "rejected", "reason_code": "unknown",
                            "reason_detail": "not in canonical label catalog"})
            results.append(rec)
        return results

    def build_catalog(self):
        mask = self._eligible_mask()
        n = int(mask.sum())
        assert n == N_ELIGIBLE, (n, N_ELIGIBLE)
        row_indices = np.flatnonzero(mask)
        preds = self._predict_rows(row_indices)
        rows = []
        for i, gid in enumerate(self.gene_gid[mask]):
            row = self.catalog[gid]
            split = str(self.gene_split[row_indices[i]])
            pred = float(preds[i])
            rows.append({
                "gene_id": gid,
                "split": split,
                "component_id": int(self.comp_dense[row_indices[i]]),
                "n_promoters": int(self.n_prom[row_indices[i]]),
                "observed_raw_S": float(self.observed_s[gid]),
                "predicted_raw_S": pred,
                "legacy_Score_S": row["Score_S"],
                "observed_E_raw": row["E_raw"],
                "observed_B_raw": row["B_raw"],
                "legacy_Score_E": row["Score_E"],
                "legacy_Score_B": row["Score_B"],
                "in_sample": (split == "train"),
                "outside_theoretical_range": bool(pred > S_THEORETICAL_MAX),
            })
        return rows

    def build_exclusion(self):
        sdef_ids = {g for g, r in self.catalog.items() if r["S_defined_main"]}
        excl = sorted(sdef_ids - self.universe_set)
        assert len(excl) == N_EXCLUDED, (len(excl), N_EXCLUDED)
        out = []
        for g in excl:
            row = self.catalog[g]
            out.append({
                "gene_id": g,
                "reason": "S_defined_feature_missing",
                "reason_detail": "canonical S-defined but absent from the 57827-gene feature universe",
                "observed_E_raw": row["E_raw"],
                "observed_S_raw": row["S_raw_full"],
                "observed_B_raw": row["B_raw"],
                "legacy_Score_S": row["Score_S"],
            })
        return out


# ---------------------------------------------------------------------------
# Serialization helpers.
# ---------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _finite(v):
    if v is None:
        return None
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return None  # finite serialization; NaN/Inf are never emitted
    return v


def _emit_json(results):
    clean = []
    for r in results:
        rr = dict(r)
        for k in list(rr):
            v = rr[k]
            if isinstance(v, float):
                rr[k] = _finite(v)
            elif isinstance(v, dict):
                rr[k] = json.loads(json.dumps(v, default=_json_default, allow_nan=False))
        clean.append(rr)
    sys.stdout.write(json.dumps(clean, ensure_ascii=False, indent=2,
                                default=_json_default, allow_nan=False) + "\n")


def _emit_tsv(rows, columns):
    sys.stdout.write("\t".join(columns) + "\n")
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, float):
                v = _finite(v)
            if v is None:
                v = ""
            elif isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False, default=_json_default)
            vals.append(str(v))
        sys.stdout.write("\t".join(vals) + "\n")


PRED_COLUMNS = [
    "row_id", "input_id", "normalized_id", "version_suffix_stripped", "status",
    "reason_code", "reason_detail", "predicted_raw_S", "observed_raw_S",
    "legacy_Score_S", "observed_E_raw", "observed_B_raw", "legacy_Score_E",
    "legacy_Score_B", "split", "component_id", "n_promoters", "in_sample",
    "applicability_status", "outside_theoretical_range",
]

TRACE_COLUMNS = [
    "row_id", "input_id", "normalized_id", "gene_id", "version_suffix_stripped",
    "status", "reason_code", "reason_detail", "S_defined_main", "observed_E_raw",
    "observed_S_raw", "observed_B_raw", "legacy_Score_E", "legacy_Score_S",
    "legacy_Score_B", "detectable_tissue_count", "tissue_count",
    "eligibility_reason", "formula", "source_files",
]


def read_inputs(args):
    inputs = []
    if args.genes:
        for g in args.genes:
            inputs.append(g)
    if args.file:
        p = Path(args.file)
        if not p.is_file():
            raise SystemExit("input file not found: %s" % args.file)
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    inputs.append(s)
    if not inputs:
        raise SystemExit("no gene IDs supplied (use --genes or --file)")
    return inputs


def cmd_predict(args):
    p = FrozenPredictor(verify=True)
    inputs = read_inputs(args)
    results = p.predict(inputs, normalize_version=args.normalize_version)
    if args.format == "tsv":
        _emit_tsv(results, PRED_COLUMNS)
    else:
        _emit_json(results)
    return 0


def cmd_trace(args):
    p = FrozenPredictor(verify=True)
    inputs = read_inputs(args)
    results = p.trace(inputs, normalize_version=args.normalize_version)
    if args.format == "tsv":
        _emit_tsv(results, TRACE_COLUMNS)
    else:
        _emit_json(results)
    return 0


def cmd_catalog(args):
    p = FrozenPredictor(verify=True)
    rows = p.build_catalog()
    excl = p.build_exclusion()
    cat_tsv = DATA_OUT / "catalog.tsv"
    excl_tsv = DATA_OUT / "exclusion_178.tsv"
    with open(cat_tsv, "w", encoding="utf-8") as fh:
        cols = ["gene_id", "split", "component_id", "n_promoters",
                "observed_raw_S", "predicted_raw_S", "legacy_Score_S",
                "observed_E_raw", "observed_B_raw", "legacy_Score_E",
                "legacy_Score_B", "in_sample", "outside_theoretical_range"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(excl_tsv, "w", encoding="utf-8") as fh:
        cols = ["gene_id", "reason", "reason_detail", "observed_E_raw",
                "observed_S_raw", "observed_B_raw", "legacy_Score_S"]
        fh.write("\t".join(cols) + "\n")
        for r in excl:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    manifest = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": now_local(),
        "n_eligible": len(rows),
        "n_excluded": len(excl),
        "catalog_tsv": str(cat_tsv),
        "exclusion_tsv": str(excl_tsv),
        "target_definition": S_TARGET_DEF,
        "model": HGB_CONFIG,
        "model_sha256": p.model_sha,
    }
    atomic_write_json(DATA_OUT / "catalog_manifest.json", manifest)
    print(json.dumps({"catalog": str(cat_tsv), "exclusion": str(excl_tsv),
                      "n_eligible": len(rows), "n_excluded": len(excl)},
                     ensure_ascii=False, indent=2))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="s12_predict.py",
        description="v6.13.0-siface.rc1 frozen gene-level raw-S interface (read-only).")
    sub = ap.add_subparsers(dest="cmd")

    for name in ("predict", "trace"):
        sp = sub.add_parser(name)
        sp.add_argument("--genes", nargs="*", metavar="ENSG",
                        help="explicit canonical Ensembl human gene IDs")
        sp.add_argument("--file", metavar="PATH",
                        help="batch file (one gene ID per line, # comments allowed)")
        sp.add_argument("--format", choices=["json", "tsv"], default="json")
        sp.add_argument("--normalize-version", action="store_true",
                        help="strip only a documented terminal dot-integer (e.g. .14)")
        sp.set_defaults(func={"predict": cmd_predict, "trace": cmd_trace}[name])

    sp = sub.add_parser("catalog",
                        help="emit complete 18769-gene catalog + 178-gene exclusion list")
    sp.set_defaults(func=cmd_catalog)

    sp = sub.add_parser("version")
    sp.set_defaults(func=lambda a: (print(CANDIDATE_VERSION), 0)[1])
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    if args.cmd == "version":
        print(CANDIDATE_VERSION)
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

