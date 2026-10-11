#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promoter Atlas batch20 unified raw-sequence CLI (v6.16.0-sequence.rc1).

Modes
  mpra-fasta      raw FASTA -> K562 MPRA 164 bp activity (previously selected legnet ensemble)
  gene            gene symbols / Ensembl IDs -> OBSERVED catalog E/S/B + PREDICTED E/S/B
  gene-bag-fasta  user promoter-bag FASTA -> EXACT 1069 features + E/S/B prediction
  constructs      built-in ACTB/GAPDH/CMV experimental constructs (observed vs prediction)

Every mode writes CSV + JSON + a self-contained HTML report and prints the paths.
No silent truncation, no silent rescaling, no fake labels: unsupported inputs are
reported with explicit reasons and no numeric prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from b20_common import (B20InputError, CANDIDATE_VERSION, MPRA_LEN, N_SLOTS, L400, L2000,
                        MODEL_OUT, REPORT_OUT, atomic_write_json, encode_uint8, read_fasta,
                        sha256_file, sha256_text, set_thread_limits, BASE)  # noqa: E402
import b20_features as F  # noqa: E402
import b20_gene as G  # noqa: E402
import b20_report as R  # noqa: E402

set_thread_limits(4)

SCOPE_MPRA = ("K562 MPRA (GSE118242, 164 bp oligos) reporter activity only. NOT HEK293T, "
              "NOT gene E/S/B. Robustness of this model was NOT confirmed (batch02 grouped-CV margin missed).")
SCOPE_GENE = ("Human gene-level NATURAL EXPRESSION E/S/B from a genome-promoter bag. "
              "NOT arbitrary fragment activity, NOT HEK293T constructs, human only.")
SCOPE_BAG = ("User-supplied promoter bag: exact 1069 features computed from the supplied sequence. "
             "The 4-slot sequence rule of the trained model is only exactly reproducible for "
             "in-catalog genes; here it is a DECLARED rule and marked exploratory.")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_csv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gc(seq):
    s = seq.upper()
    n = len(s)
    if n == 0:
        return float("nan")
    return (s.count("G") + s.count("C")) / n


def cmd_mpra_fasta(args):
    import b20_mpra as M
    recs = read_fasta(args.fasta)
    input_sha = sha256_file(args.fasta)
    rows, errors, warnings = [], [], []
    seen = {}
    ok_seq, ok_idx = [], []
    for i, (name, seq) in enumerate(recs):
        s = seq.strip().upper()
        bad = sorted(set(s) - set("ACGTN"))
        dup = s in seen
        seen.setdefault(s, name)
        n_amb = sum(1 for c in s if c not in "ACGT")
        rows.append({"_i": i, "name": name, "length": len(s), "gc": round(_gc(s), 4),
                     "n_ambiguous": n_amb,
                     "sequence_sha256": sha256_text(s), "duplicate_of": seen[s] if dup else "",
                     "observed_k562_mpra": "", "observed_source": "",
                     "_bad": ",".join(bad), "supported": "", "predicted_activity": "",
                     "ensemble_sd": "", "exploratory_window": "", "note": ""})
        if not s:
            errors.append({"query": name, "error": "empty sequence"})
            rows[-1]["note"] = "ERROR empty"
            continue
        if bad:
            errors.append({"query": name, "error": "invalid base(s): %s" % ",".join(bad)})
            rows[-1]["note"] = "ERROR invalid bases; not scored"
            continue
        if len(s) == MPRA_LEN:
            rows[-1]["supported"] = "native_164bp"
            nab = int(rows[-1]["n_ambiguous"])
            rows[-1]["note"] = ("native supported length" if nab == 0 else
                "native supported length BUT %d ambiguous base(s): encoded as N->4 (all-zero one-hot at those "
                "positions); prediction is the model's documented N-handling and must be interpreted with caution" % nab)
            ok_seq.append(s)
            ok_idx.append(i)
        elif len(s) > MPRA_LEN and args.exploratory_window:
            rows[-1]["supported"] = "unsupported_length"
            rows[-1]["exploratory_window"] = "last164"
            rows[-1]["note"] = "whole fragment UNVALIDATED; exploratory TSS-proximal last-164 window only"
            ok_seq.append(s[-MPRA_LEN:])
            ok_idx.append(i)
        else:
            rows[-1]["supported"] = "unsupported_length"
            rows[-1]["note"] = ("no numeric prediction: length %d != %d and windowing %s"
                                % (len(s), MPRA_LEN, "disabled" if not args.exploratory_window else "inapplicable (shorter)"))
            errors.append({"query": name, "error": "unsupported length %d (model requires %d bp)" % (len(s), MPRA_LEN)})
    obs_map = _mpra_observed_map()
    seen_sha = {}
    for r in rows:
        prev = seen_sha.get(r["sequence_sha256"])
        if prev is not None:
            r["duplicate_of"] = prev
            r["note"] = ((r["note"] + "; ") if r["note"] else "") +                 "DUPLICATE of %s (identical sequence SHA); scored but ranks are not independent" % prev
        else:
            seen_sha[r["sequence_sha256"]] = r["name"]
        hit = obs_map.get(r["sequence_sha256"])
        if hit is not None:
            r["observed_k562_mpra"] = round(hit, 5)
            r["observed_source"] = "features/mpra/mpra_supervised.tsv (GSE118242 K562, Mean_log10)"
    t0 = time.time()
    if ok_seq:
        pred = M.predict_legnet(ok_seq)
        for k, i in enumerate(ok_idx):
            rows[i]["predicted_activity"] = round(float(pred["ensemble_mean"][k]), 5)
            rows[i]["ensemble_sd"] = round(float(pred["ensemble_std_over_models"][k]), 5)
    latency = time.time() - t0
    if any(r["duplicate_of"] for r in rows):
        warnings.append({"level": "warn", "text": "Duplicate sequences detected; each is scored but ranks are not independent."})
    warnings.append({"level": "warn", "text": "Model robustness was NOT confirmed in batch02 grouped CV; treat all values as UNCONFIRMED."})
    cols = ["name", "length", "gc", "n_ambiguous", "supported", "predicted_activity", "ensemble_sd",
            "observed_k562_mpra", "observed_source", "exploratory_window", "duplicate_of",
            "sequence_sha256", "note"]
    return {
        "mode": "mpra-fasta", "generated_at_local": _now(), "input_name": str(args.fasta),
        "scope_note": SCOPE_MPRA, "columns": cols,
        "rows": [{k: r[k] for k in cols} for r in rows], "errors": errors, "warnings": warnings,
        "provenance": {
            "candidate_version": CANDIDATE_VERSION, "input_sha256": input_sha,
            "n_records": len(recs), "n_scored": len(ok_seq),
            "model": "K562 MPRA legnet ensemble (5 grouped folds x 3 seeds)",
            "model_source_sha": M.source_sha(),
            "preprocessing": "encode_uint8 A/C/G/T->0..3, N/other->4; length 164 exactly; per-fold train-only destandardization; mean over 15 models",
            "length_routing": {"native": 164, "longer": "exploratory last-164 window" if args.exploratory_window else "refused",
                               "shorter": "refused (no fake zero)"},
            "latency_s": round(latency, 3), "device": "cuda" if _cuda() else "cpu",
        },
    }


def _cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cmd_gene(args):
    import b20_gene_model as GM
    cat = G.load_catalog()
    pred = GM.GeneESBPredictor()
    queries = []
    for tok in args.genes:
        queries.extend([q for q in tok.split(",") if q.strip()])
    rows, errors, parity = [], [], {}
    seqs_all, feats_all, meta = [], [], []
    for q in queries:
        try:
            gi, eid = G.resolve_gene(q)
        except B20InputError as e:
            errors.append({"query": q, "error": str(e)})
            continue
        s4, s2 = G.gene_bag_sequences(gi)
        feat_rt = F.gene_1069(s4, s2)
        feat_cat = cat["X"][gi].astype(np.float64)
        md = float(np.max(np.abs(feat_rt - feat_cat)))
        parity[eid] = md
        slots = G.slot_sequences(gi)
        seqs_all.append(slots)
        feats_all.append(feat_rt)
        meta.append((q, gi, eid, md))
    if seqs_all:
        out = pred.predict(np.stack(seqs_all), np.stack(feats_all))
    for k, (q, gi, eid, md) in enumerate(meta):
        sdef = bool(cat["obs_S_defined"][gi]) if not np.isnan(cat["obs_S_defined"][gi]) else False
        rows.append({
            "query": q, "gene_id": eid, "symbol": cat["gene_sym"][gi] or "",
            "n_promoters": int(cat["starts"][gi + 1] - cat["starts"][gi]),
            "obs_E_ScoreE": _r6(cat["obs_E"][gi]),
            "obs_S_raw": _r6(cat["obs_S_raw"][gi]), "obs_B_raw": _r6(cat["obs_B_raw"][gi]),
            "obs_Score_S": _r6(cat["obs_Score_S"][gi]), "obs_Score_B": _r6(cat["obs_Score_B"][gi]),
            "obs_S_defined": ("1" if sdef else "0"),
            "pred_E": _r6(out["E"][k]), "pred_S": _r6(out["S"][k]) if sdef else "not_defined",
            "pred_B": _r6(out["B"][k]),
            "split": cat["gene_split"][gi],
            "feature_parity_max_abs": "%.3g" % md,
            "note": ("" if md == 0.0 else "PARITY!=0 check") + ("" if sdef else " S not defined for this gene (mask); S prediction suppressed"),
        })
    rows.sort(key=lambda r: (-float(r["pred_E"]) if r["pred_E"] not in ("",) else 0.0))
    cols = ["query", "gene_id", "symbol", "n_promoters",
            "obs_E_ScoreE", "obs_S_raw", "obs_B_raw", "obs_Score_S", "obs_Score_B", "obs_S_defined",
            "pred_E", "pred_S", "pred_B", "split", "feature_parity_max_abs", "note"]
    return {
        "mode": "gene", "generated_at_local": _now(),
        "input_name": ",".join(queries[:8]) + ("..." if len(queries) > 8 else ""),
        "scope_note": SCOPE_GENE, "rows": rows, "cols": cols, "errors": errors,
        "warnings": [{"level": "ok", "text": "1069 features recomputed from raw promoter-bag sequence; "
                      "parity vs frozen catalog matrix verified (max abs diff %s)."
                      % (max(parity.values()) if parity else "n/a")},
                     {"level": "warn", "text": "OBSERVED labels are catalog measurements; PREDICTED are model outputs. "
                      "S is only defined for genes passing the S mask. Rankings are gene-level natural expression, "
                      "not HEK293T activity."}],
        "definitions": {
            "obs_E_ScoreE": "OBSERVED Score_E (rank-scaled [0,1]); SAME UNITS as pred_E. Source: labels/build/gene_expression_label_table.tsv",
            "obs_S_raw": "OBSERVED S_raw=-MAD(log2(1+detectable TPM)); SAME UNITS as pred_S; defined only where obs_S_defined=1",
            "obs_B_raw": "OBSERVED B_raw=detectable_tissue_count/tissue_count; SAME UNITS as pred_B",
            "obs_Score_S": "OBSERVED Score_S (rank-scaled, informational; NOT same units as pred_S)",
            "obs_Score_B": "OBSERVED Score_B (rank-scaled, informational; NOT same units as pred_B)",
            "pred_E": "MODEL E, raw-unit Score_E-like [0,1]; independent CNN seed 20261016",
            "pred_S": "MODEL S, raw-unit S_raw; suppressed when the gene's S mask is not defined",
            "pred_B": "MODEL B, raw-unit B_raw [0,1]",
            "catalog": "57827-gene human feature universe; symbols joined from promoter_analysis_set.tsv",
        },
        "provenance": {
            "candidate_version": CANDIDATE_VERSION,
            "model": "v6.15.0-multitask.rc1 arm A (independent), reference seed 20261016, packaged v6.15.1-release.rc2",
            "model_sha256": pred.model_sha,
            "feature_source": "b20_features.gene_1069 (exact re-implementation of batch03+batch10+batch16)",
            "slot_rule": "diverse_slots(n_prom, gene_idx) = [(gene_idx %% n)+k for k in 0..3]",
            "n_queries": len(queries), "n_resolved": len(rows),
        },
    }


_OBS_CACHE = {}


def _mpra_observed_map():
    """sequence -> observed K562 MPRA activity (Mean_log10) from the traceable source."""
    if _OBS_CACHE:
        return _OBS_CACHE
    import csv as _csv
    from b20_common import MPRA_SUPERVISED
    m = {}
    with open(MPRA_SUPERVISED, "r", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh, delimiter="	"):
            m[sha256_text(row["sequence"].strip().upper())] = float(row["activity"])
    _OBS_CACHE.update(m)
    return _OBS_CACHE


def _r6(x):
    try:
        x = float(x)
    except Exception:
        return ""
    return "" if (x != x) else ("%.6f" % x)


def _parse_bag_fasta(path):
    recs = read_fasta(path)
    bags = {}
    for name, seq in recs:
        parts = name.split("|")
        if len(parts) != 3:
            raise B20InputError("bag FASTA header must be 'query|promoter_index|window(sym400|up2000)'; got %r" % name)
        q, idx, win = parts[0].strip(), parts[1].strip(), parts[2].strip().lower()
        if win not in ("sym400", "up2000"):
            raise B20InputError("window must be sym400 or up2000; got %r" % win)
        bags.setdefault(q, {}).setdefault(int(idx), {})[win] = seq.strip().upper()
    out = {}
    for q, prom in bags.items():
        idxs = sorted(prom)
        s4, s2 = [], []
        for i in idxs:
            d = prom[i]
            if "sym400" not in d or "up2000" not in d:
                raise B20InputError("gene %s promoter %d missing sym400 or up2000 record" % (q, i))
            if len(d["sym400"]) != L400:
                raise B20InputError("sym400 for %s promoter %d is %d bp, expected %d" % (q, i, len(d["sym400"]), L400))
            if len(d["up2000"]) != L2000:
                raise B20InputError("up2000 for %s promoter %d is %d bp, expected %d" % (q, i, len(d["up2000"]), L2000))
            s4.append(encode_uint8(d["sym400"], L400, name="sym400"))
            s2.append(encode_uint8(d["up2000"], L2000, name="up2000"))
        out[q] = (np.array(s4), np.array(s2))
    return out


def cmd_gene_bag_fasta(args):
    import b20_gene_model as GM
    bags = _parse_bag_fasta(args.fasta)
    pred = GM.GeneESBPredictor()
    cat = G.load_catalog()
    rows, warnings = [], []
    seqs_all, feats_all, meta = [], [], []
    for q, (s4, s2) in bags.items():
        feat = F.gene_1069(s4, s2)
        n = len(s4)
        in_cat, gi, parity = False, None, None
        try:
            gi, eid = G.resolve_gene(q)
            in_cat = True
            parity = float(np.max(np.abs(feat - cat["X"][gi].astype(np.float64))))
        except B20InputError:
            eid = q
        if in_cat and args.use_catalog_slots:
            slots = G.slot_sequences(gi)
            slot_rule = "catalog_exact_diverse_slots"
            exact = True
        else:
            slots = s4[:min(N_SLOTS, n)]
            if len(slots) < N_SLOTS:
                pad = np.zeros((N_SLOTS - len(slots), L400), dtype=np.uint8)
                slots = np.concatenate([slots, pad], axis=0)
            slot_rule = "declared_first_min4_file_order"
            exact = bool(in_cat and parity == 0.0 and n >= N_SLOTS)
        seqs_all.append(slots)
        feats_all.append(feat)
        meta.append((q, eid, n, in_cat, parity, slot_rule, exact))
        if not exact:
            warnings.append({"level": "warn", "text": "Gene %s: 4-slot rule is DECLARED (not the exact training "
                             "rule); E/S/B is exploratory for this input." % q})
    out = pred.predict(np.stack(seqs_all), np.stack(feats_all))
    for k, (q, eid, n, in_cat, parity, slot_rule, exact) in enumerate(meta):
        rows.append({
            "query": q, "resolved_gene_id": eid if in_cat else "", "n_promoters": n,
            "in_catalog": int(in_cat),
            "feature_parity_max_abs": ("%.3g" % parity) if parity is not None else "n/a (not in catalog)",
            "slot_rule": slot_rule, "exact_supported": int(exact),
            "pred_E": _r6(out["E"][k]), "pred_S": _r6(out["S"][k]), "pred_B": _r6(out["B"][k]),
            "note": "" if exact else "EXPLORATORY: not the exact trained slot rule / out-of-catalog",
        })
    cols = ["query", "resolved_gene_id", "n_promoters", "in_catalog", "feature_parity_max_abs",
            "slot_rule", "exact_supported", "pred_E", "pred_S", "pred_B", "note"]
    return {
        "mode": "gene-bag-fasta", "generated_at_local": _now(), "input_name": str(args.fasta),
        "scope_note": SCOPE_BAG, "columns": cols, "rows": rows, "errors": [], "warnings": warnings,
        "provenance": {
            "candidate_version": CANDIDATE_VERSION, "input_sha256": sha256_file(args.fasta),
            "n_bags": len(bags),
            "feature_source": "1069 features recomputed from supplied raw sequence (exact formula parity)",
            "model_sha256": pred.model_sha,
        },
    }


CELLLINE_V2_DIR = (Path(__file__).resolve().parents[4] / "outputs" /
                   "01a0f783-d754-7803-abcf-34aa761a6a6b" /
                   "全库标签与细胞系模型_含CMVcore_20261001_文件夹" /
                   "reports" / "cellline_v2_20261001")
CONSTRUCT_TSV = CELLLINE_V2_DIR / "construct_endpoints_D28.tsv"


def _int_or0(v):
    try:
        if v is None:
            return 0
        f = float(v)
        if f != f:
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


def cmd_constructs(args):
    import b20_mpra as M
    import b20_gene_model as GM  # noqa: F401
    import pandas as pd
    if not CONSTRUCT_TSV.exists():
        raise B20InputError("construct endpoints not found at %s" % CONSTRUCT_TSV)
    df = pd.read_csv(CONSTRUCT_TSV, sep="\t", low_memory=False)
    rows, warnings = [], []
    seqs, idxs = [], []
    for _, r in df.iterrows():
        cid = str(r.get("construct_id", "")).strip()
        fam = str(r.get("family", ""))
        s = str(r.get("sequence", "")).strip().upper()
        tv = r.get("target_value", "")
        obs_val = ""
        try:
            if tv is not None and str(tv).strip() != "" and not pd.isna(tv):
                obs_val = round(float(tv), 5)
        except (TypeError, ValueError):
            obs_val = ""
        rows.append({
            "construct_id": cid, "family": fam,
            "length": _int_or0(r.get("sequence_length_bp", 0)) or len(s),
            "observed_target_name": str(r.get("target_name", "")),
            "observed_value": obs_val,
            "observed_units": "log10(GFP-positive median / same-batch WT-negative median), HEK293T ROSA26-4, D28 second sort",
            "observable": ("1" if obs_val != "" else "0"),
            "n_independent_bio_reps": _int_or0(r.get("independent_biological_replicates", 0)),
            "barcode_verified": _int_or0(r.get("assay_to_plasmid_barcode_verified", 0)),
            "endpoint_eligible": _int_or0(r.get("endpoint_eligible", 0)),
            "quality_flags": str(r.get("quality_flags", ""))[:200],
            "sequence_sha256": sha256_text(s), "mpra_exploratory_window": "", "mpra_pred": "", "note": "",
        })
        if len(s) >= MPRA_LEN:
            seqs.append(s[-MPRA_LEN:]); idxs.append(len(rows) - 1)
            rows[-1]["mpra_exploratory_window"] = "last164"
        else:
            rows[-1]["note"] = ("shorter than 164 bp: no K562 MPRA window; no numeric MPRA prediction "
                                "(observed HEK293T value still reported)")
    if seqs:
        pred = M.predict_legnet(seqs)
        for k, i in enumerate(idxs):
            rows[i]["mpra_pred"] = round(float(pred["ensemble_mean"][k]), 5)
    for r in rows:
        if r["mpra_pred"] != "":
            r["note"] = (r["note"] + " EXPLORATORY last-164 K562 MPRA window; whole-fragment HEK293T "
                         "activity is NOT predicted by any validated model.").strip()
    gene_ctx = cmd_gene(argparse.Namespace(genes=["ACTB", "GAPDH"]))
    warnings.append({"level": "warn", "text": "OBSERVED HEK293T D28 log10-ratio values and MODEL predictions are "
                     "DIFFERENT quantities and are NOT comparable; the observed column is the honest measurement."})
    warnings.append({"level": "warn", "text": "This construct set has ZERO independent biological replicates "
                     "(technical same-tube repeats only; independent_biological_replicates is blank in source)."})
    warnings.append({"level": "warn", "text": "CMVcore/CMVfull carry no human gene context; NO gene-level E/S/B is "
                     "fabricated for them."})
    cols = ["construct_id", "family", "length", "observed_target_name", "observed_value", "observed_units",
            "observable", "n_independent_bio_reps", "barcode_verified", "endpoint_eligible", "quality_flags",
            "mpra_exploratory_window", "mpra_pred", "sequence_sha256", "note"]
    return {
        "mode": "constructs", "generated_at_local": _now(),
        "input_name": str(CONSTRUCT_TSV), "scope_note": SCOPE_MPRA + " | " + SCOPE_GENE,
        "columns": cols, "rows": rows, "errors": [], "warnings": warnings,
        "gene_level_context": {"columns": gene_ctx["cols"], "rows": gene_ctx["rows"]},
        "provenance": {
            "candidate_version": CANDIDATE_VERSION,
            "observed_source": str(CONSTRUCT_TSV), "observed_source_sha256": sha256_file(CONSTRUCT_TSV),
            "mpra_model_source_sha": M.source_sha(),
            "note": "MPRA prediction column is an EXPLORATORY last-164 window; not a validated construct-activity predictor",
        },
    }


def _emit(args, result):
    outdir = Path(args.outdir) if args.outdir else REPORT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or result["mode"].replace("-", "_")
    csv_cols = result.get("columns") or result.get("cols") or (list(result["rows"][0].keys()) if result["rows"] else [])
    csv_path = outdir / ("%s.csv" % tag)
    json_path = outdir / ("%s.json" % tag)
    html_path = outdir / ("%s.html" % tag)
    _write_csv(csv_path, result.get("rows", []), csv_cols)
    result["outputs"] = {"csv": str(csv_path), "json": str(json_path), "html": str(html_path)}
    R.write_json(json_path, result)
    R.write_html(html_path, result)
    print(json.dumps({"mode": result["mode"], "n_rows": len(result.get("rows", [])),
                      "n_errors": len(result.get("errors", [])),
                      "csv": str(csv_path), "json": str(json_path), "html": str(html_path)},
                     ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promoter_atlas_batch20",
                                 description="Promoter Atlas raw-sequence CLI v6.16.0-sequence.rc1")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("mpra-fasta", help="raw FASTA -> K562 MPRA 164 bp activity (legnet ensemble)")
    p1.add_argument("--fasta", required=True)
    p1.add_argument("--exploratory-window", action="store_true",
                    help="for >164bp fragments also emit an EXPLORATORY TSS-proximal last-164 window")
    p1.add_argument("--outdir", default=None)
    p1.add_argument("--tag", default=None)

    p2 = sub.add_parser("gene", help="gene symbols/IDs -> observed + predicted E/S/B")
    p2.add_argument("--genes", nargs="+", required=True, help="symbols or Ensembl IDs, space/comma separated")
    p2.add_argument("--outdir", default=None)
    p2.add_argument("--tag", default=None)

    p3 = sub.add_parser("gene-bag-fasta", help="promoter-bag FASTA -> exact 1069 features + E/S/B")
    p3.add_argument("--fasta", required=True)
    p3.add_argument("--use-catalog-slots", action="store_true",
                    help="for in-catalog genes use the exact trained 4-slot rule")
    p3.add_argument("--outdir", default=None)
    p3.add_argument("--tag", default=None)

    p4 = sub.add_parser("constructs", help="ACTB/GAPDH/CMV experimental constructs: observed vs exploratory prediction")
    p4.add_argument("--outdir", default=None)
    p4.add_argument("--tag", default=None)

    args = ap.parse_args(argv)
    if args.mode == "mpra-fasta":
        res = cmd_mpra_fasta(args)
    elif args.mode == "gene":
        res = cmd_gene(args)
    elif args.mode == "gene-bag-fasta":
        res = cmd_gene_bag_fasta(args)
    elif args.mode == "constructs":
        res = cmd_constructs(args)
    else:
        ap.error("unknown mode")
    _emit(args, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
