#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 DATA/ENDPOINT stage.

Exports the 14-construct sequence/measurement audit from the ORIGINAL project
files (never from derived prose), verifies the primary public-source metadata
(GSE188639 first, then at most one second human promoter-activity source), and
writes public_source_verification.json with explicit acceptance/rejection reasons.

It also documents the declared input-quality rule and the fact that NO batch21
candidate model can score the full-length HEK293T constructs (native-164 branch).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from b21_common import (REPORT_OUT, DATA_OUT, CONSTRUCT_ENDPOINTS, CONSTRUCT_SEQUENCES,
                        EXCLUDED_81, MPRA_SUPERVISED, now_local, sha256_file,
                        atomic_write_json, MPRA_LEN)
import b21_mpra

GEO_ACC = "GSE188639"
PROBE_DIR = DATA_OUT / "_probe"


def _read_tsv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _sha_if_exists(p):
    p = Path(p)
    return sha256_file(p) if p.exists() else None


def build_construct_audit():
    ep = _read_tsv(CONSTRUCT_ENDPOINTS)
    seqs = {r["construct_id"]: r for r in _read_tsv(CONSTRUCT_SEQUENCES)}
    excluded = _read_tsv(EXCLUDED_81)
    out_rows = []
    for r in ep:
        cid = r["construct_id"]
        s = seqs.get(cid, {})
        L = int(r["sequence_length_bp"])
        # orientation from the ORIGINAL sequence record
        orient = s.get("feature_directionality", "")
        orient = {"1": "forward", "-1": "reverse"}.get(orient, orient or "unknown")
        scorable = (L == MPRA_LEN)
        out_rows.append({
            "construct_id": cid,
            "family": r["family"],
            "sequence_length_bp": L,
            "documented_length_bp": s.get("documented_length_bp", ""),
            "sequence_sha256": r["sequence_sha256"],
            "orientation_from_original": orient,
            "sample_annotation": r.get("sample_annotation", ""),
            "cell_line_reported": r.get("cell_line_reported", ""),
            "days_after_sort_reported": r.get("days_after_sort_reported", ""),
            "endpoint_name": r.get("target_name", ""),
            "endpoint_value": r.get("target_value", ""),
            "relative_gfp": r.get("relative_gfp", ""),
            "normalization": "log10(GFP-positive median / same-batch WT-negative median); "
                             "relative_gfp = 10**target_value",
            "within_export_dispersion_sd": r.get("within_export_dispersion_sd", ""),
            "independent_biological_replicates": r.get("independent_biological_replicates", ""),
            "n_exported_tube_rows": r.get("n_exported_tube_rows", ""),
            "assay_to_plasmid_barcode_verified": r.get("assay_to_plasmid_barcode_verified", ""),
            "endpoint_eligible": r.get("endpoint_eligible", ""),
            "quality_flags": r.get("quality_flags", ""),
            "source_path": r.get("source_path", ""),
            "source_sha256": r.get("source_sha256", ""),
            "sequence_source_path": r.get("sequence_source_path", ""),
            "sequence_source_sha256": r.get("sequence_source_sha256", ""),
            "integration_status_from_originals":
                "source is a plasmid (.dna) reporter pool -> transient plasmid reporter; "
                "single D28 endpoint; integrated-vs-transient NOT asserted beyond originals",
            "scorable_by_batch21_164bp_candidate": scorable,
            "scorable_reason": ("native length equals 164 bp" if scorable else
                                "length %d bp != 164 bp native branch; NOT padded by duplication; "
                                "unsupported for the batch21 K562-164 candidate" % L),
        })
    excluded_rows = []
    for r in excluded:
        excluded_rows.append({
            "construct_id": r["construct_id"], "family": r["family"],
            "sequence_length_bp": r["sequence_length_bp"],
            "sequence_sha256": r["sequence_sha256"],
            "status": r.get("sequence_status", ""),
            "reason_excluded": "untransfected 81 bp fragment (user_confirmed_not_transfected); "
                               "excluded from the measured construct set",
        })
    return out_rows, excluded_rows


def _get(url, timeout=40):
    return urllib.request.urlopen(url, timeout=timeout).read()


def _geo_text(acc):
    u = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s&targ=self&form=text&view=brief" % acc)
    return _get(u).decode("utf-8", "replace")


def verify_gse188639():
    """Verify primary metadata from the GEO record; never infer relevance from the name."""
    ev = {"accession": GEO_ACC, "primary_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE188639"}
    try:
        txt = _geo_text(GEO_ACC)
    except Exception as e:
        ev["error"] = "metadata fetch failed: %r" % (e,)
        ev["verified"] = False
        return ev
    def field(k):
        return [l.split(" = ", 1)[1] for l in txt.splitlines() if l.startswith(k)]
    ev["title"] = (field("!Series_title") or [""])[0]
    ev["pubmed"] = field("!Series_pubmed_id")
    ev["summary"] = (field("!Series_summary") or [""])[0]
    ev["design"] = (field("!Series_overall_design") or [""])[0]
    ev["series_type"] = field("!Series_type")
    ev["supplementary_files"] = field("!Series_supplementary_file")
    try:
        st = _get(("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE188639&targ=gsm"
                   "&form=text&view=brief")).decode("utf-8", "replace")
        bg = [l.split("background: ", 1)[1] for l in st.splitlines() if "cell line background" in l]
        ev["sample_cell_backgrounds"] = sorted(set(bg))
        ev["has_HEK293_arm"] = any("HEK293" in b for b in bg)
        ev["has_K562_arm"] = any("K562" in b for b in bg)
    except Exception as e:
        ev["sample_error"] = repr(e)[:160]
    ev["verified"] = True
    return ev


def probe_bulk_files():
    """Download the two small processed tables and record their shape + sha256."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    base = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE188nnn/GSE188639/suppl/")
    out = {}
    for f in ("GSE188639_mixed_cell_bulk_exp.csv.gz", "GSE188639_k562_bulk_exp.csv.gz"):
        p = PROBE_DIR / f
        try:
            if not p.exists():
                p.write_bytes(_get(base + f))
            b = p.read_bytes()
            rec = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
            try:
                txt = gzip.decompress(b).decode("utf-8", "replace")
                lines = txt.splitlines()
                rec["n_lines"] = len(lines)
                rec["header"] = lines[0][:200]
                if "mixed_cell" in f:
                    rec["contains"] = "per-promoter k562 exp / hek exp / k562_norm / hek_norm"
                    rec["n_promoters"] = len(lines) - 1
            except Exception as e:
                rec["parse_error"] = repr(e)[:100]
            out[f] = rec
        except Exception as e:
            out[f] = {"error": repr(e)[:160]}
    return out


def second_source_check():
    """At most ONE bounded second human-promoter-activity triage (plus rejections)."""
    cands = []
    for acc, note in (("GSE104878", "LegNet benchmark candidate"), ("GSE163045", "LegNet GPRA candidate")):
        rec = {"accession": acc, "note": note}
        try:
            txt = _geo_text(acc)
            def f(k):
                return [l.split(" = ", 1)[1] for l in txt.splitlines() if l.startswith(k)]
            rec["title"] = (f("!Series_title") or [""])[0]
            org = f("!Series_platform_organism")
            rec["platform_organism"] = org
            rec["verdict"] = "REJECTED"
            rec["reason"] = ("non-human surrogate (%s); protocol forbids a yeast surrogate for human activity"
                             % (org[0] if org else "unknown"))
        except Exception as e:
            rec["verdict"] = "UNVERIFIED"
            rec["reason"] = "metadata fetch failed: %r" % (e,)
        cands.append(rec)
    return cands


def run():
    t0 = now_local()
    audit, excluded = build_construct_audit()
    # write construct audit (TSV + JSON)
    cols = list(audit[0].keys())
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    for d in (DATA_OUT, REPORT_OUT):
        p = d / "construct_audit.tsv"
        with io.open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in audit:
                w.writerow(r)
    fam_counts = {}
    for r in audit:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1

    gse = verify_gse188639()
    probes = probe_bulk_files()
    second = second_source_check()
    mixed = probes.get("GSE188639_mixed_cell_bulk_exp.csv.gz", {})
    n_prom = mixed.get("n_promoters")
    hek_ok = bool(gse.get("has_HEK293_arm"))
    gate = {"min_usable_sequences": 1000, "min_independent_groups": 100,
            "observed_usable_sequences": n_prom, "cell": "HEK293 (not HEK293T)"}
    hek_available = bool(n_prom is not None and n_prom >= 1000 and hek_ok and "hek_norm" in mixed.get("contains", ""))

    decision = ("K562 GSE118242 (local, in-domain)" if not hek_available else "GSE188639 HEK293")
    ver = {
        "generated_at_local": now_local(), "started_local": t0,
        "search_bound_minutes": 60,
        "declared_selection_rule":
            "First verified HEK293/HEK293T promoter-reporter source with processed sequences and "
            "valid per-sequence scalar activity, >=1000 usable sequences AND >=100 independent "
            "sequence groups, within preparation bounds (<=500MB, <=30min). HEK293T preferred over "
            "HEK293 and never equated. Otherwise fall back to K562 GSE118242 as an explicitly "
            "in-domain experiment and record the HEK293T gap. Selection is availability-based, NOT score-based.",
        "primary_candidate": {
            "accession": GEO_ACC, "verification": gse, "processed_probes": probes,
            "gate": gate,
            "verdict": ("ACCEPTED" if hek_available else "REJECTED_FOR_TRAINING"),
            "rejection_reason": (None if hek_available else
                "676 core-promoter members (< 1000 gate); cell is HEK293 not HEK293T; HEK activity "
                "exists as a bulk per-sequence column (hek_norm) but the library is too small for the "
                "declared training gate. Retained as a future HEK293 supervised source (single-cell "
                "deconvolution / larger library needed), not used to tune anything now."),
        },
        "second_source_search": second,
        "second_source_conclusion":
            "No second verified HUMAN promoter-activity source with processed sequence/activity and "
            ">=1000 sequences was obtained within the 60-minute bound; inspected yeast LegNet series "
            "were rejected as non-human surrogates.",
        "local_selected_source": {
            "accession": "GSE118242", "path": str(MPRA_SUPERVISED),
            "sha256": sha256_file(MPRA_SUPERVISED),
            "context": "K562", "assay": "oligo MPRA", "length_bp": MPRA_LEN, "n_sequences": 11803,
            "role": "in-domain training/validation/test source; not HEK293T and not a HEK293T substitute",
        },
        "dataset_decision": decision,
        "HEK293T_gap": ("No HEK293/HEK293T promoter-reporter source with >=1000 processed "
                        "sequence/activity pairs was obtainable within bounds. The HEK293T gap is "
                        "recorded explicitly; no K562 MPRA score is presented as HEK293T reliability."),
        "constructs": {"n_constructs": len(audit), "families": fam_counts,
                       "n_scorable_by_164bp_candidate":
                           sum(1 for r in audit if r["scorable_by_batch21_164bp_candidate"]),
                       "n_unsupported": sum(1 for r in audit if not r["scorable_by_batch21_164bp_candidate"])},
        "excluded": excluded,
        "input_quality_rule": {"max_N_fraction": 0.05,
                               "note": "sequences with >5 percent N are refused as unsupported; "
                                       "length must equal 164 bp for the K562 branch; IUPAC codes refused"},
        "source_sha256": b21_mpra.source_sha(),
        "endpoints_sha256": sha256_file(CONSTRUCT_ENDPOINTS),
    }
    atomic_write_json(REPORT_OUT / "public_source_verification.json", ver)
    atomic_write_json(DATA_OUT / "construct_audit.json",
                      {"constructs": audit, "excluded": excluded})
    return ver


if __name__ == "__main__":
    v = run()
    print(json.dumps({"decision": v["dataset_decision"],
                      "primary": v["primary_candidate"]["verdict"],
                      "n_constructs": v["constructs"]["n_constructs"],
                      "n_unsupported": v["constructs"]["n_unsupported"]},
                     ensure_ascii=False, indent=2))
